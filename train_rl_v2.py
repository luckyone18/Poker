#!/usr/bin/env python3
"""
Train RL Poker bot against Monte Carlo + Smart opponents using PPO.

Key improvements over train_rl_vs_mc.py:
- Runs fixed 6-hand sessions (not elimination) for consistent episode length
- Reward per hand = chip delta normalized by starting stack
- Multiple opponents for diverse training
"""

import sys, os, time, random, math
import torch
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.engine import Table, Seat, eval_hand, EVAL_HAND_MAX
from bots.rl_bot import RLBot
from bots import parse_players, escalate_blinds


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train RL Poker bot")
    parser.add_argument("--episodes", type=int, default=20000,
                        help="Total sessions (default: 20000)")
    parser.add_argument("--hands-per-session", type=int, default=10,
                        help="Hands per session (default: 10)")
    parser.add_argument("--chips", type=int, default=500)
    parser.add_argument("--sb", type=int, default=5)
    parser.add_argument("--bb", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--report-every", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--model-path", type=str, default="models/rl_model_v3.pt")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--opponents", type=str, default="smart,mc100")
    parser.add_argument("--hidden-size", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Training: {args.episodes} sessions × {args.hands_per_session} hands")
    print(f"Opponents: {args.opponents}")
    print(f"Save to: {args.model_path}")
    print()

    rl = RLBot(
        model_path=args.resume or args.model_path,
        device=device,
        training_mode=True,
        learning_rate=args.lr,
        exploration_rate=0.25,
        use_fallback=True,
        starting_chips=args.chips,
        hidden_size=args.hidden_size,
    )

    # Stats tracking
    win_rates = deque(maxlen=200)        # % sessions where RL gained chips
    avg_rewards = deque(maxlen=200)      # avg per-hand reward
    episode_times = deque(maxlen=100)

    start_time = time.time()
    total_hands_played = 0

    # Pre-compute opponent bots (reuse)
    opp_types = args.opponents.split(",")

    for ep in range(1, args.episodes + 1):
        ep_start = time.time()

        # Build bot dict for this session
        bots = {"rl": rl}
        player_specs = [("rl", "rl", rl)]

        for i, ot in enumerate(opp_types):
            from bots import create_bot
            opponent = create_bot(ot)
            pid = f"opp{i+1}"
            bots[pid] = opponent
            player_specs.append((pid, ot, opponent))

        # Randomize seating
        random.shuffle(player_specs)

        seats = [Seat(player_id=pid, chips=args.chips) for pid, _, _ in player_specs]
        by_pid = {s.player_id: s for s in seats}

        dealer = 0
        active_seats = list(seats)
        rl_chip_start = by_pid["rl"].chips
        hands_played = 0
        rl_eliminated = False

        for hand_num in range(args.hands_per_session):
            # Clean up eliminated players
            active = [s for s in seats if s.chips > 0]
            if by_pid["rl"] not in active:
                rl_eliminated = True
                break
            if len(active) <= 1:
                break

            sb, bb = escalate_blinds(hand_num + 1, args.sb, args.bb, 1000)  # near-constant blinds
            di = dealer % len(active)
            active_bots = {s.player_id: bots[s.player_id] for s in active}

            table = Table()
            try:
                table.play_hand(active, sb, bb, di, active_bots, on_event=None)
            except Exception as e:
                # Skip broken hand, reset seats
                for s in seats:
                    s.reset_for_new_hand()
                continue

            hands_played += 1
            dealer = (dealer + 1) % max(len(active), 1)

            # Reward = normalized chip change for RL
            rl_chip_delta = by_pid["rl"].chips - rl_chip_start
            reward = rl_chip_delta / max(1, args.chips)
            rl.record_reward(reward)

        # End session
        total_hands_played += hands_played
        won = by_pid["rl"].chips > args.chips
        win_rates.append(1 if won else 0)

        session_reward = (by_pid["rl"].chips - args.chips) / max(1, args.chips)
        avg_rewards.append(session_reward)

        rl.end_episode()

        # Decay exploration
        if ep % 2000 == 0 and rl.exploration_rate > 0.02:
            rl.exploration_rate = max(0.02, rl.exploration_rate * 0.90)

        ep_time = time.time() - ep_start
        episode_times.append(ep_time)

        if ep % args.report_every == 0:
            elapsed = time.time() - start_time
            win_percent = sum(win_rates) / max(1, len(win_rates))
            avg_r = sum(avg_rewards) / max(1, len(avg_rewards))
            avg_time = sum(episode_times) / max(1, len(episode_times))
            est_remaining = (args.episodes - ep) * avg_time / 60
            print(f"[{ep}/{args.episodes}] win={win_percent:.1%} "
                  f"avg_r={avg_r:+.3f} "
                  f"h={total_hands_played//ep} "
                  f"elapsed={elapsed/60:.1f}m "
                  f"eta={est_remaining:.0f}m "
                  f"ε={rl.exploration_rate:.3f}")

        if ep % args.save_every == 0:
            dir_ = os.path.dirname(args.model_path)
            if dir_:
                os.makedirs(dir_, exist_ok=True)
            torch.save({
                'policy': rl.policy_net.state_dict(),
                'value': rl.value_net.state_dict(),
            }, args.model_path)
            print(f"  → Saved {args.model_path}")

    # Final save
    rl.flush_buffer()
    torch.save({
        'policy': rl.policy_net.state_dict(),
        'value': rl.value_net.state_dict(),
    }, args.model_path)
    total_time = (time.time() - start_time) / 60
    print(f"\nDone! {args.episodes} sessions in {total_time:.1f}m")
    print(f"Model: {args.model_path}")


if __name__ == "__main__":
    main()