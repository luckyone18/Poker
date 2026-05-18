"""
Stage 4: RL Fine-tune BC Policy vs SmartBot (Modal v1.4.2)
Deploy: modal deploy poker/stage4_run.py
Run:    modal run poker/stage4_run.py::train --episodes 10000

Key path facts:
  - Modal mounts ~/Poker/ → /root/poker/ (case preserved from host path)
  - This script is INSIDE ~/Poker/, so __file__-based path resolves correctly
  - bots/ at /root/poker/bots/
  - models/ at /root/poker/models/
  - poker-models volume mounted at /root/models_vol/
"""
import os, sys

# Resolve POKER_ROOT from this script's location (must be inside ~/Poker/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POKER_ROOT  = _SCRIPT_DIR          # e.g. /root/poker on Modal (lowercase 'p')
VOL_MODELS  = "/root/models_vol"

import modal

app = modal.App("poker-stage4")
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch")
       .add_local_dir(".", remote_path="/root/poker", copy=True)  # copy files AT BUILD TIME
       .run_commands("cd /root/poker && pip install . --no-deps"))  # now setup.py exists
vol = modal.Volume.from_name("poker-models")


@app.function(
    image=img,
    gpu="T4",
    volumes={VOL_MODELS: vol},
    timeout=600,
    retries=0,
)
def train(episodes: int = 10_000, chips: int = 2000, lr: float = 3e-5, batch_size: int = 512):
    """PPO fine-tune BC policy vs SmartBot with proper MC-return batching."""
    import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
    from collections import deque

    # POKER_ROOT resolved from __file__ at module level (see top of file)
    POKER_ROOT = "/root/poker"
    if POKER_ROOT not in sys.path:
        sys.path.insert(0, POKER_ROOT)
    os.chdir(POKER_ROOT)

    # Debug paths
    print(f"[DEBUG] cwd={os.getcwd()}", flush=True)
    print(f"[DEBUG] ls .: {os.listdir('.')[:8]}", flush=True)
    print(f"[DEBUG] ls bots/: {os.listdir('bots/')[:6] if os.path.exists('bots/') else 'MISSING'}", flush=True)
    print(f"[DEBUG] ls models/: {os.listdir('models/') if os.path.exists('models/') else 'MISSING'}", flush=True)
    print(f"[DEBUG] sys.path[:3]={sys.path[:3]}", flush=True)

    os.makedirs(VOL_MODELS, exist_ok=True)

    from bots.rl_bot import RLBot
    from bots.poker_mind_bot import SmartBot
    from core.engine import Table, Seat, InProcessBot

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Stage4] device={device} | eps={episodes} | chips={chips} | lr={lr} | batch={batch_size}", flush=True)

    # ── Create RL bot & load BC starting weights ─────────────────────────
    rl_bot = RLBot(
        model_path="models/rl_model_bc_policy.pt",
        device="cuda",
        training_mode=True,
        learning_rate=lr,
        batch_size=batch_size,
        starting_chips=chips,
        exploration_rate=0.1,
        use_fallback=True,
    )

    for ckpt_path, key in [
        ("models/rl_model_bc_policy.pt", None),
        ("models/bc_model_final.pt",     "policy"),
    ]:
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            sd = ckpt if key is None else ckpt.get(key, ckpt)
            rl_bot.policy_net.load_state_dict(sd)
            rl_bot.policy_net.to(device).train()
            rl_bot.value_net.to(device).train()
            print(f"✅ Loaded: {ckpt_path}", flush=True)
            break
    else:
        print("⚠️  No checkpoint found — training from scratch", flush=True)

    smartbot = InProcessBot(SmartBot())
    optimizer = optim.Adam(
        list(rl_bot.policy_net.parameters()) + list(rl_bot.value_net.parameters()),
        lr=lr,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    replay: list = []
    recent_wr    = deque(maxlen=100)
    recent_r     = deque(maxlen=500)
    total_steps  = 0
    updates      = 0
    ep           = 0

    while ep < episodes:
        ep += 1
        rl_is_p1  = (ep % 2 == 1)
        seat_idx  = 0 if rl_is_p1 else 1

        # IMPORTANT: In heads-up, ring[0] acts FIRST preflop.
        # If we put RLBot at position 1 (P2), the opponent at position 0
        # acts first and can fold, ending the hand before RLBot ever acts.
        # Fix: ALWAYS put RLBot at seat 0 (first to act preflop).
        # When rl_is_p1=False, swap P1<->P2 so RLBot ends up at index 0.
        if rl_is_p1:
            seats = [Seat(player_id="P1", chips=chips), Seat(player_id="P2", chips=chips)]
            bots  = {"P1": InProcessBot(rl_bot), "P2": smartbot}
        else:
            # Swap player IDs so RLBot (now "P1") is at index 0
            seats = [Seat(player_id="P2", chips=chips), Seat(player_id="P1", chips=chips)]
            bots  = {"P1": InProcessBot(rl_bot), "P2": smartbot}
            rl_is_p1 = True   # RL now effectively at P1 (index 0)

        print(f"[EP] ep={ep} rl_is_p1={rl_is_p1}", flush=True)
        # CRITICAL: reset starting_chips so feature normalization stays consistent
        rl_bot.starting_chips = chips
        rl_bot.current_episode = []
        table = Table()
        table.play_hand(seats, 10, 20, 0, bot_for=bots)

        final_chips = seats[seat_idx].chips
        reward      = (final_chips - chips) / chips
        won         = 1.0 if final_chips > chips else 0.5 if final_chips == chips else 0.0
        recent_wr.append(won)
        recent_r.append(reward)

        ep_len = len(rl_bot.current_episode)
        if ep_len == 0:
            # RLBot folded without acting → small penalty so it learns not to fold always
            shaping = -0.05
        else:
            # Reward shaping: +bonus for reaching each street, +big bonus for winning at showdown
            streets_reached = set()
            for step in rl_bot.current_episode:
                st = int(step["state"][0, 0].item())
                streets_reached.add(st)
            shaping = 0.0
            if 1 in streets_reached: shaping += 0.03   # reached flop
            if 2 in streets_reached: shaping += 0.03   # reached turn
            if 3 in streets_reached: shaping += 0.05   # reached river
            shaping += 0.10 if won == 1.0 else 0.0     # bonus for winning

        for i, step in enumerate(rl_bot.current_episode):
            if ep_len == 0:
                step["reward"] = shaping
                step["return"] = shaping
            else:
                # Linear discount shaping: early steps get less final reward
                frac = (i + 1) / ep_len
                step["reward"] = shaping + reward * frac
                step["return"] = shaping + reward * frac

        replay.extend(rl_bot.current_episode)
        total_steps += len(rl_bot.current_episode)

        # PPO update every ~256 steps (frequent small updates for faster learning)
        # Use gradient accumulation: accumulate 4 mini-batches of 64 before one optimizer step
        MINI_BATCH = 64
        if len(replay) >= MINI_BATCH:
            batch = replay[:MINI_BATCH * 4]   # take up to 256 steps
            replay = replay[MINI_BATCH * 4:]

            states  = torch.stack([s["state"].squeeze(0) for s in batch]).to(device)
            actions = torch.tensor([s["action"] for s in batch], dtype=torch.long, device=device)
            old_lp  = torch.stack([s["log_prob"].detach() for s in batch]).to(device)
            old_v   = torch.stack([s["value"].detach()    for s in batch]).to(device)
            returns = torch.tensor([s["return"] for s in batch], dtype=torch.float, device=device)

            probs_raw, values = rl_bot.policy_net(states), rl_bot.value_net(states.detach()).squeeze(-1)
            probs    = F.softmax(probs_raw, dim=-1).clamp(min=1e-8)
            dist     = torch.distributions.Categorical(probs)
            log_p    = dist.log_prob(actions)
            entropy  = dist.entropy().mean()

            ratio       = torch.exp(log_p - old_lp)
            advantage   = (returns - old_v.detach())
            surr1       = ratio * advantage
            surr2       = torch.clamp(ratio, 0.8, 1.2) * advantage
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss  = nn.functional.mse_loss(values, returns.detach())
            loss        = policy_loss + 0.5 * value_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(rl_bot.policy_net.parameters()) +
                                    list(rl_bot.value_net.parameters()), 1.0)
            optimizer.step()
            updates += 1

        if ep % 500 == 0:
            wr50 = sum(recent_wr) / len(recent_wr) * 100
            r100 = sum(recent_r)  / len(recent_r)  * 100
            print(f"  ep {ep}/{episodes} | RL=P{'1' if rl_is_p1 else '2'} | "
                  f"wr50={wr50:.1f}% | r/100={r100:.2f} | buf={len(replay)} | upd={updates}", flush=True)

    # Final flush
    if replay:
        batch    = replay
        states   = torch.stack([s["state"].squeeze(0) for s in batch]).to(device)
        actions  = torch.tensor([s["action"] for s in batch], dtype=torch.long, device=device)
        old_lp   = torch.stack([s["log_prob"].detach() for s in batch]).to(device)
        old_v    = torch.stack([s["value"].detach()    for s in batch]).to(device)
        returns  = torch.tensor([s["return"] for s in batch], dtype=torch.float, device=device)
        probs_raw, values = rl_bot.policy_net(states), rl_bot.value_net(states.detach()).squeeze(-1)
        probs   = F.softmax(probs_raw, dim=-1).clamp(min=1e-8)
        dist    = torch.distributions.Categorical(probs)
        ratio    = torch.exp(dist.log_prob(actions) - old_lp)
        advantage = (returns - old_v.detach())
        loss = (-torch.min(ratio * advantage,
                           torch.clamp(ratio, 0.8, 1.2) * advantage).mean()
                + 0.5 * nn.functional.mse_loss(values, returns.detach())
                - 0.01 * dist.entropy().mean())
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        updates += 1

    # Save to volume
    final_path = f"{VOL_MODELS}/bc_smartbot_v2.pt"
    torch.save({
        "policy": rl_bot.policy_net.state_dict(),
        "value":  rl_bot.value_net.state_dict(),
    }, final_path)

    wr  = sum(recent_wr) / len(recent_wr) * 100
    r   = sum(recent_r)  / len(recent_r)  * 100
    print(f"\n✅ Done | {ep} eps | wr={wr:.1f}% | r/100={r:.2f} | upd={updates} | steps={total_steps}", flush=True)
    print(f"   → {final_path}", flush=True)


@app.local_entrypoint()
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--chips",    type=int, default=2000)
    parser.add_argument("--lr",       type=float, default=3e-5)
    parser.add_argument("--batch",    type=int,   default=512)
    args = parser.parse_args()
    train.call(args.episodes, args.chips, args.lr, args.batch)