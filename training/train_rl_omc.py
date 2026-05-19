"""
RL training with graded curriculum and checkpointing.

Curriculum (easier → harder):
  Stage 1: RandomBot       (WR > 60% → promote)
  Stage 2: SmartBot         (WR > 50% → promote)
  Stage 3: MonteCarlo(200)  (WR > 45% → promote)
  Stage 4: MonteCarlo(500)  (WR > 40% → promote)
  Stage 5: Self-play

Auto-saves checkpoint every SAVE_EVERY episodes so training can be resumed.
"""

import os, csv, json, argparse, sys
from collections import deque

import torch

# Project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bots.rl_bot import RLBot
from core.engine import RandomBot
from bots.poker_mind_bot import SmartBot
from bots.monte_carlo_bot import MonteCarloBot
from core.bot_api import BotAdapter
from core.engine import InProcessBot, RandomBot, Seat, Table


# ── PlayerViewAdapter ─────────────────────────────────────────────────
class PlayerViewAdapter(BotAdapter):
    def __init__(self, bot):
        self.bot = bot

    def act(self, view) -> 'Action':
        return self.bot.act(view)


# ── Optimized Monte Carlo loading (optional) ──────────────────────────
OPT_CONFIG_PATH = "models/optimized_mc_config.json"
OPT_BOT_CACHE   = None

def load_optimized_mc_bot():
    global OPT_BOT_CACHE
    if OPT_BOT_CACHE is not None:
        return OPT_BOT_CACHE
    if os.path.exists(OPT_CONFIG_PATH):
        with open(OPT_CONFIG_PATH) as f:
            cfg_data = json.load(f)
        cfg = cfg_data["config"]
        from scripts.optimize_monte_carlo import OptimizedMonteCarloBot
        OPT_BOT_CACHE = OptimizedMonteCarloBot(**cfg)
        print(f"[O-MC] Loaded optimized MonteCarloBot (WR={cfg_data.get('wr_vs_smartbot', '?')}% vs SmartBot)")
    else:
        print(f"[O-MC] No optimized config found — using default MonteCarloBot(1000)")
        OPT_BOT_CACHE = InProcessBot(MonteCarloBot(simulations=1000))
    return OPT_BOT_CACHE


# ── Curriculum ────────────────────────────────────────────────────────
CURRICULUM = [
    {
        "name":       "random",
        "make_bot":   lambda: InProcessBot(RandomBot()),
        "promote_wr": 0.55,
    },
    {
        "name":       "smartbot",
        "make_bot":   lambda: InProcessBot(SmartBot()),
        "promote_wr": 0.50,
    },
    {
        "name":       "mc50",
        "make_bot":   lambda: InProcessBot(MonteCarloBot(simulations=50)),
        "promote_wr": 0.45,
    },
    {
        "name":       "mc100",
        "make_bot":   lambda: InProcessBot(MonteCarloBot(simulations=100)),
        "promote_wr": 0.42,
    },
    {
        "name":       "mc200",
        "make_bot":   lambda: InProcessBot(MonteCarloBot(simulations=200)),
        "promote_wr": 0.40,
    },
    {
        "name":       "mc500",
        "make_bot":   lambda: InProcessBot(MonteCarloBot(simulations=500)),
        "promote_wr": 0.38,
    },
    {
        "name":       "selfplay",
        "make_bot":   None,
    },
]

PROMOTE_WINDOW   = 500
SNAPSHOT_PATH    = "models/rl_selfplay_snapshot.pt"
SNAPSHOT_EVERY   = 200
START_CHECKPOINT = "models/rl_model_omc.pt"
FINAL_MODEL_PATH = "models/rl_model_omc_final.pt"
DEFAULT_CSV_PATH = "output/rl_omc_training_log.csv"
HIDDEN_SIZE      = 256
SAVE_EVERY       = 500          # save checkpoint every N episodes


# ── Training ──────────────────────────────────────────────────────────
def train_rl_bot(num_episodes=10_000, chips_per_player=500,
                 csv_path=None, lr_step_episodes=50_000,
                 resume_from=None):
    """Train RL bot with graded curriculum."""

    print("=" * 70)
    print("TRAINING RL BOT — Graded Curriculum")
    print("=" * 70)
    print(f"Episodes:            {num_episodes}")
    print(f"Chips per player:    {chips_per_player}")
    print(f"Hidden size:         {HIDDEN_SIZE}")
    print(f"Curriculum:          {' → '.join(s['name'] for s in CURRICULUM)}")
    thresholds = ", ".join(
        f"{s['name']}={s['promote_wr']:.0%}"
        for s in CURRICULUM if "promote_wr" in s
    )
    print(f"Promotion thresholds: {thresholds}  (over {PROMOTE_WINDOW} episodes)")
    print(f"Checkpoint interval:  {SAVE_EVERY} episodes")
    print("=" * 70)
    print()

    rl_bot = RLBot(
        model_path="",
        training_mode=True,
        learning_rate=5e-6,     # Conservative: BC policy is already strong, RL should be fine-tuning
        starting_chips=chips_per_player,
        batch_size=512,  # Increased from 8 for faster training
    )

    # Resume from checkpoint if available
    start_episode = 1
    if resume_from and os.path.exists(resume_from):
        try:
            checkpoint = torch.load(resume_from, map_location=rl_bot.device)
            if isinstance(checkpoint, dict) and 'policy' in checkpoint:
                rl_bot.policy_net.load_state_dict(checkpoint['policy'])
                rl_bot.value_net.load_state_dict(checkpoint['value'])
                start_episode = checkpoint.get('episode', 1)
            else:
                rl_bot.policy_net.load_state_dict(checkpoint)
            rl_bot.policy_net.train()
            rl_bot.value_net.train()
            print(f"Resumed from checkpoint {resume_from} at episode {start_episode}")
        except Exception as e:
            print(f"[checkpoint] Could not load {resume_from}: {e} — starting fresh")
    elif os.path.exists(START_CHECKPOINT):
        try:
            checkpoint = torch.load(START_CHECKPOINT, map_location=rl_bot.device)
            if isinstance(checkpoint, dict) and 'policy' in checkpoint:
                rl_bot.policy_net.load_state_dict(checkpoint['policy'])
                rl_bot.value_net.load_state_dict(checkpoint['value'])
                start_episode = checkpoint.get('episode', 1)
            else:
                rl_bot.policy_net.load_state_dict(checkpoint)
            rl_bot.policy_net.train()
            rl_bot.value_net.train()
            print(f"Loaded checkpoint from {START_CHECKPOINT} at episode {start_episode}")
        except Exception as e:
            print(f"[checkpoint] Could not load {START_CHECKPOINT}: {e} — starting fresh")
    else:
        print(f"[checkpoint] {START_CHECKPOINT} not found — starting fresh")

    initial_lr      = 5e-6   # Conservative: BC policy is already strong (99.2%), RL fine-tuning
    lr_decay_factor = 0.5
    first_lr_drop   = lr_step_episodes // 2   # first halving earlier

    # ── State ──────────────────────────────────────────────────────────
    wins             = 0
    recent_rewards: deque[float] = deque(maxlen=100)
    selfplay_opponent = None

    stage_idx           = 0
    stage_wins          = deque(maxlen=PROMOTE_WINDOW)
    stage_episode_count = 0

    csv_file   = None
    csv_writer = None
    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        csv_file   = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["episode", "won", "reward", "rolling_wr", "avg_reward", "lr", "stage"]
        )

    print(f"[curriculum] Starting stage 1/{len(CURRICULUM)}: "
          f"{CURRICULUM[stage_idx]['name']}\n")

    for episode in range(start_episode, start_episode + num_episodes):
        rl_bot.end_episode()
        rl_bot.opponent_stats = {}

        # LR scheduling — first drop earlier, then every lr_step_episodes
        if episode > 1:
            if episode == first_lr_drop + 1:
                for pg in rl_bot.optimizer.param_groups:
                    pg["lr"] = initial_lr * lr_decay_factor
                print(f"  [LR] First reduction to {initial_lr * lr_decay_factor:.2e} at episode {episode}")
            elif (episode - 1) % lr_step_episodes == 0:
                num_decays = (episode - 1) // lr_step_episodes
                new_lr = initial_lr * (lr_decay_factor ** num_decays)
                for pg in rl_bot.optimizer.param_groups:
                    pg["lr"] = new_lr
                print(f"  [LR] Reduced to {new_lr:.2e} at episode {episode}")

        # Self-play snapshot management
        in_selfplay = stage_idx == len(CURRICULUM) - 1
        if in_selfplay:
            if episode % SNAPSHOT_EVERY == 0:
                os.makedirs("models", exist_ok=True)
                rl_bot.save_model(SNAPSHOT_PATH)
                selfplay_opponent = None

            if selfplay_opponent is None and os.path.exists(SNAPSHOT_PATH):
                selfplay_opponent = InProcessBot(
                    RLBot(
                        model_path=SNAPSHOT_PATH,
                        training_mode=False,
                        use_fallback=True,
                    )
                )

        seats = [
            Seat(player_id="P1", chips=chips_per_player),
            Seat(player_id="P2", chips=chips_per_player),
        ]
        if in_selfplay and selfplay_opponent is not None:
            opponent_bot = selfplay_opponent
        else:
            opponent_bot = CURRICULUM[stage_idx]["make_bot"]()
        bots = {
            "P1": opponent_bot,
            "P2": InProcessBot(rl_bot),
        }

# Tournament
        table        = Table()
        dealer_index = 0
        initial_chips_p2 = chips_per_player

        while True:
            active_seats = [s for s in seats if s.chips > 0]
            if len(active_seats) <= 1:
                winner = active_seats[0].player_id if active_seats else None
                break

            result = table.play_hand(
                seats=active_seats,
                small_blind=1, big_blind=2,
                dealer_index=dealer_index % len(active_seats),
                bot_for={s.player_id: bots[s.player_id] for s in active_seats},
                on_event=None,
                log_decisions=False,
            )
            # Per-hand chip delta is NOW handled automatically inside rl_bot.record_reward()
            # via per-round chip tracking in each step. No per-hand reward needed here.

            dealer_index = (dealer_index + 1) % len(seats)
            if dealer_index > 10_000:
                winner = max(seats, key=lambda s: s.chips).player_id
                break

        final_chips_p2 = sum(s.chips for s in seats if s.player_id == "P2")
        won            = winner == "P2"
        final_reward   = (final_chips_p2 - initial_chips_p2) / max(initial_chips_p2, 1)
        final_bonus    = 1.0 if won else -0.5
        rl_bot.record_reward(final_bonus)

        if won:
            wins += 1
        recent_rewards.append(final_reward)

        # CSV
        if csv_writer:
            rolling_wr_val = wins / (episode - start_episode + 1)
            avg_reward     = sum(recent_rewards) / len(recent_rewards)
            current_lr     = rl_bot.optimizer.param_groups[0]["lr"]
            csv_writer.writerow([
                episode, int(won), final_reward,
                rolling_wr_val, avg_reward,
                current_lr, CURRICULUM[stage_idx]["name"]
            ])
            if episode % 50 == 0:
                csv_file.flush()

        # Curriculum promotion
        stage_wins.append(1 if won else 0)
        stage_episode_count += 1

        if stage_idx < len(CURRICULUM) - 1 and len(stage_wins) >= PROMOTE_WINDOW:
            rolling_wr     = sum(stage_wins) / len(stage_wins)
            promote_thresh = CURRICULUM[stage_idx]["promote_wr"]
            if rolling_wr >= promote_thresh:
                stage_idx          += 1
                stage_episode_count = 0
                stage_wins.clear()
                print(f"\n{'=' * 70}")
                print(f"[curriculum] PROMOTED to stage {stage_idx + 1}/{len(CURRICULUM)}: "
                      f"{CURRICULUM[stage_idx]['name']}  "
                      f"(episode {episode}, rolling WR {rolling_wr:.1%})")
                print(f"{'=' * 70}\n")
                if stage_idx == len(CURRICULUM) - 1:
                    os.makedirs("models", exist_ok=True)
                    rl_bot.save_model(SNAPSHOT_PATH)
                    selfplay_opponent = None
                    print(f"  [snapshot] Initial self-play snapshot saved.\n")
            elif episode % 1_000 == 0:
                needed = CURRICULUM[stage_idx]["promote_wr"]
                print(f"  [curriculum] stage={CURRICULUM[stage_idx]['name']}  "
                      f"rolling_wr={rolling_wr:.1%}/{needed:.0%}  "
                      f"stage_eps={stage_episode_count}")

        # Progress
        if episode % 100 == 0:
            rolling_wr_display = wins / (episode - start_episode + 1)
            avg_reward         = sum(recent_rewards) / len(recent_rewards)
            current_lr         = rl_bot.optimizer.param_groups[0]["lr"]
            stage_name         = CURRICULUM[stage_idx]["name"]
            print(f"  ep={episode:>6}  wins={wins:>5}  "
                  f"wr={rolling_wr_display:.1%}  avg_r={avg_reward:+.3f}  "
                  f"lr={current_lr:.1e}  stage={stage_name}")

        # Checkpoint save
        if episode % SAVE_EVERY == 0:
            os.makedirs("models", exist_ok=True)
            ckpt = {
                'policy': rl_bot.policy_net.state_dict(),
                'value':  rl_bot.value_net.state_dict(),
                'episode': episode,
                'stage':   stage_idx,
                'wins':    wins,
            }
            torch.save(ckpt, START_CHECKPOINT)
            print(f"  [checkpoint] Saved at episode {episode}")

        # Early exit if completely stuck for too long
        if stage_idx > 0 and stage_idx < len(CURRICULUM) - 1:
            if stage_episode_count > 3_000:
                rolling_wr = sum(stage_wins) / len(stage_wins)
                if rolling_wr < 0.15:
                    print(f"\n⚠️  STUCK at stage {CURRICULUM[stage_idx]['name']} — "
                          f"WR={rolling_wr:.1%} after {stage_episode_count} eps. "
                          f"Saving checkpoint and stopping early.\n")
                    break

    # End
    rl_bot.flush_buffer()
    os.makedirs("models", exist_ok=True)
    rl_bot.save_model(FINAL_MODEL_PATH)
    print(f"\nModel saved to {FINAL_MODEL_PATH}")

    if csv_file:
        csv_file.close()
        if csv_path:
            print(f"Training log saved to {csv_path}")

    episodes_run = episode - start_episode + 1
    final_wr  = wins / episodes_run if episodes_run > 0 else 0
    avg_final = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
    print(f"\n{'=' * 70}")
    print(f"Training complete.")
    print(f"  Episodes:            {episodes_run}")
    print(f"  Wins:                {wins} / {episodes_run}  ({final_wr:.1%})")
    print(f"  Avg reward (last 100): {avg_final:+.3f}")
    print(f"  Final stage:         {CURRICULUM[stage_idx]['name']}")
    print(f"{'=' * 70}")
    return rl_bot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5_000)
    parser.add_argument("--chips", type=int, default=500)
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH)
    parser.add_argument("--lr_step", type=int, default=25_000)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)
    train_rl_bot(
        num_episodes=args.episodes,
        chips_per_player=args.chips,
        csv_path=args.csv,
        lr_step_episodes=args.lr_step,
        resume_from=args.resume,
    )