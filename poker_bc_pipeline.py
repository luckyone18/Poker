"""
Stage 4: RL Fine-tune BC Policy vs SmartBot (Clean Implementation)

Key design:
- Collect N full episodes into a replay buffer
- Each episode: store (features, action, log_prob, value) per step
- After hand: set reward on ALL steps in episode (MC return = chip_delta)
- Every batch_size episodes: PPO update using all stored steps
  - advantage = return - value_estimate
  - policy loss = -min(ratio * advantage, clip * advantage)
  - value loss = MSE(return, value_estimate)
- Alternate RL position (SB/BB) each episode
- Log: per-500-ep WR, per-1000-ep mean reward

Usage:
  modal run poker_bc_pipeline.py::stage4_v2_train --episodes 10000 --chips 2000
"""
import os, sys

POKER_ROOT = os.path.dirname(os.path.abspath(__file__))
VOL_MODELS = "/root/models"
MODAL_APP   = "poker-bc-pipeline"
IMAGE_NAME  = "python:3.11-slim"

# ── helpers ──────────────────────────────────────────────────────────────────

def _setup_symlink():
    """Ensure /root/models → poker-models volume."""
    if not os.path.exists("/root/models"):
        os.makedirs("/root/models", exist_ok=True)
        os.symlink("/root/poker/models", "/root/models")


# ── Stage 4 v2 ───────────────────────────────────────────────────────────────

def stage4_v2_train(
    episodes: int = 10_000,
    chips: int = 2000,
    lr: float = 3e-5,
    batch_size: int = 512,    # episodes per PPO update
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    hidden: int = 256,
):
    """
    Clean RL fine-tune using MC returns (no GAE complexity).
    Reward = normalised chip delta. Advantage = return - V(s).
    All steps in an episode share the same MC return (standard for discrete PB).
    """
    _setup_symlink()
    sys.path.insert(0, POKER_ROOT)

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from collections import deque

    from bots.rl_bot import RLBot
    from bots.poker_mind_bot import SmartBot
    from core.engine import Table, Seat, InProcessBot

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Stage4v2] device={device} | eps={episodes} | chips={chips} | lr={lr} | batch={batch_size}")

    # ── Load BC starting policy ────────────────────────────────────────────
    rl_bot = RLBot(
        training_mode=True,
        learning_rate=lr,
        batch_size=batch_size,
        starting_chips=chips,
        exploration_rate=0.0,   # pure exploitation during training
        use_fallback=True,
    )

    # Try Stage-3 output first, then fallback to bc_policy
    for ckpt_path, key in [
        (f"{VOL_MODELS}/bc_model_final.pt",  "policy"),
        (f"{VOL_MODELS}/rl_model_bc_policy.pt",  None),
    ]:
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = ckpt if key is None else ckpt.get(key, ckpt)
            rl_bot.policy_net.load_state_dict(state_dict)
            rl_bot.policy_net.to(device).train()
            rl_bot.value_net.to(device).train()
            print(f"✅ Starting policy loaded: {ckpt_path}")
            break

    smartbot = InProcessBot(SmartBot())
    optimizer = optim.Adam(
        list(rl_bot.policy_net.parameters()) +
        list(rl_bot.value_net.parameters()),
        lr=lr,
    )

    # ── Replay buffer: list of episodes, each episode = list of step dicts ─
    replay: list = []
    recent_wr = deque(maxlen=100)
    recent_rewards = deque(maxlen=500)
    total_steps = 0

    ep = 0
    updates = 0

    while ep < episodes:
        ep += 1

        # Alternate RL seat: odd=P1(BTN/SB/BB), even=P2(BB/SB/BTN)
        rl_is_p1 = (ep % 2 == 1)
        rl_seat_idx = 0 if rl_is_p1 else 1

        seats = [
            Seat(player_id="P1", chips=chips),
            Seat(player_id="P2", chips=chips),
        ]
        bots = {
            "P1": InProcessBot(rl_bot) if rl_is_p1 else smartbot,
            "P2": smartbot if rl_is_p1 else InProcessBot(rl_bot),
        }

        # Reset episode buffer for new hand
        rl_bot.current_episode = []
        rl_bot.episode_buffer = []

        table = Table()
        table.play_hand(
            seats,
            small_blind=10,
            big_blind=20,
            dealer_index=0,
            bot_for=bots,
        )

        # Compute normalised reward (chip delta relative to start)
        final_chips = seats[rl_seat_idx].chips
        reward = (final_chips - chips) / chips   # e.g. +0.005 = win 1 BB
        won = 1.0 if final_chips > chips else 0.5 if final_chips == chips else 0.0
        recent_wr.append(won)
        recent_rewards.append(reward)

        # Attach reward to ALL steps in this episode (MC return = same for all)
        for step in rl_bot.current_episode:
            step["reward"] = reward
            step["return"] = reward   # MC return (same for every step)

        # Append episode to replay
        replay.extend(rl_bot.current_episode)
        total_steps += len(rl_bot.current_episode)

        # ── PPO update every batch_size episodes ──────────────────────────
        if len(replay) >= batch_size * 10:   # at least 10 steps per ep on avg
            # Sample a batch of steps
            batch_steps = replay[:batch_size * 10]
            replay = replay[batch_size * 10:]

            # Build tensors
            states   = torch.stack([s["state"].squeeze(0) for s in batch_steps]).to(device)
            actions  = torch.tensor([s["action"]  for s in batch_steps], dtype=torch.long, device=device)
            old_lp   = torch.stack([s["log_prob"].detach() for s in batch_steps]).to(device)
            old_val  = torch.stack([s["value"].detach()    for s in batch_steps]).to(device)
            returns  = torch.tensor([s["return"]  for s in batch_steps], dtype=torch.float, device=device)

            # Recompute values and policy probs
            probs, values = rl_bot.policy_net(states), rl_bot.value_net(states).squeeze(-1)
            dist    = torch.distributions.Categorical(probs)
            log_p   = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # PPO ratio & clipped objective
            ratio       = torch.exp(log_p - old_lp)
            surr1       = ratio * (returns - old_val.detach())
            surr2       = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * (returns - old_val.detach())
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss  = nn.functional.mse_loss(values, returns.detach())

            # Total loss
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(rl_bot.policy_net.parameters()) +
                list(rl_bot.value_net.parameters()),
                max_norm=1.0,
            )
            optimizer.step()
            updates += 1

        # ── Logging ──────────────────────────────────────────────────────────
        if ep % 500 == 0:
            rolling_wr = sum(recent_wr) / len(recent_wr) * 100
            rolling_r  = sum(recent_rewards) / len(recent_rewards) * 100  # in BB/100
            print(f"  ep {ep}/{episodes} | RL=P{'1' if rl_is_p1 else '2'} | "
                  f"chips={final_chips} | wr50={rolling_wr:.1f}% | "
                  f"r/100={rolling_r:.2f} | buf={len(replay)} | updates={updates}")

    # ── Final PPO update with remaining replay ─────────────────────────────
    if replay:
        batch_steps = replay
        states   = torch.stack([s["state"].squeeze(0) for s in batch_steps]).to(device)
        actions  = torch.tensor([s["action"]  for s in batch_steps], dtype=torch.long, device=device)
        old_lp   = torch.stack([s["log_prob"].detach() for s in batch_steps]).to(device)
        old_val  = torch.stack([s["value"].detach()    for s in batch_steps]).to(device)
        returns  = torch.tensor([s["return"]  for s in batch_steps], dtype=torch.float, device=device)

        probs, values = rl_bot.policy_net(states), rl_bot.value_net(states).squeeze(-1)
        dist    = torch.distributions.Categorical(probs)
        log_p   = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio       = torch.exp(log_p - old_lp)
        surr1       = ratio * (returns - old_val.detach())
        surr2       = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * (returns - old_val.detach())
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss  = nn.functional.mse_loss(values, returns.detach())
        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        updates += 1

    # ── Save ────────────────────────────────────────────────────────────────
    final_path = f"{VOL_MODELS}/bc_smartbot_v2.pt"
    torch.save({
        "policy": rl_bot.policy_net.state_dict(),
        "value":  rl_bot.value_net.state_dict(),
    }, final_path)

    final_wr = sum(recent_wr) / len(recent_wr) * 100
    final_r  = sum(recent_rewards) / len(recent_rewards) * 100
    print(f"\n✅ Stage4v2 done | {ep} eps | wr={final_wr:.1f}% | r/100={final_r:.2f} | "
          f"updates={updates} | total_steps={total_steps}")
    print(f"   → {final_path}")


# ── CLI entry point (used by modal run) ──────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int,  default=10_000)
    parser.add_argument("--chips",    type=int,  default=2000)
    parser.add_argument("--lr",       type=float, default=3e-5)
    parser.add_argument("--batch",    type=int,  default=512)
    args = parser.parse_args()

    stage4_v2_train(
        episodes=args.episodes,
        chips=args.chips,
        lr=args.lr,
        batch_size=args.batch,
    )