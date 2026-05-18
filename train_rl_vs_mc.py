#!/usr/bin/env python3
"""
Train RL Poker bot against Monte Carlo opponents using self-play.

Trains PPO agent against Monte Carlo opponents with periodic saving
and progress reporting. Designed for long-running local training sessions.
"""

import sys, os, time, random
import torch
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.engine import Table, Seat
from bots.rl_bot import RLBot
from bots import parse_players, create_bot, escalate_blinds


def run_episode(table, seats, bots, sb, bb, dealer_index):
    """Run a single hand. Returns the RL bot's instance from the bots dict."""
    table.play_hand(
        seats=seats, small_blind=sb, big_blind=bb,
        dealer_index=dealer_index, bot_for=bots, on_event=None)

    # Find the RL bot (player_id starts with 'rl')
    for pid, bot in bots.items():
        if pid.startswith('rl'):
            return bot
    return None


def find_rl_id(seats):
    """Find the seat with player_id starting with 'rl'."""
    for s in seats:
        if s.player_id.startswith('rl'):
            return s
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train RL Poker bot vs MC opponents")
    parser.add_argument("--episodes", type=int, default=100000,
                        help="Total episodes to train (default: 100000)")
    parser.add_argument("--chips", type=int, default=500,
                        help="Starting chips (default: 500)")
    parser.add_argument("--sb", type=int, default=5,
                        help="Small blind (default: 5)")
    parser.add_argument("--bb", type=int, default=10,
                        help="Big blind (default: 10)")
    parser.add_argument("--save-every", type=int, default=1000,
                        help="Save model every N episodes (default: 1000)")
    parser.add_argument("--report-every", type=int, default=100,
                        help="Report progress every N episodes (default: 100)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--model-path", type=str, default="models/rl_model_train.pt",
                        help="Model save path")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from existing model")
    parser.add_argument("--opponents", type=str, default="smart,mc200,mc100",
                        help="Comma-separated opponent types")
    parser.add_argument("--hidden-size", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Training for {args.episodes} episodes, save every {args.save_every}")
    print(f"Opponents: {args.opponents}")

    # Create RL bot in training mode
    rl = RLBot(
        model_path=args.resume or "models/rl_model_train.pt",
        device=device,
        training_mode=True,
        learning_rate=args.lr,
        exploration_rate=0.2,
        use_fallback=True,
        starting_chips=args.chips,
        hidden_size=args.hidden_size,
    )

    # Rolling stats
    win_rate = deque(maxlen=100)
    recent_rewards = deque(maxlen=100)
    episode_times = deque(maxlen=100)

    start_time = time.time()
    total_hands = 0

    for ep in range(1, args.episodes + 1):
        ep_start = time.time()

        # Parse opponents + our RL bot
        opponent_specs = parse_players(args.opponents)
        player_specs = [("rl", "rl", rl)] + opponent_specs
        random.shuffle(player_specs)

        seats = [Seat(player_id=pid, chips=args.chips) for pid, bt, _ in player_specs]
        bots = {pid: adapter for pid, bt, adapter in player_specs}

        dealer = 0
        hand_count = 0
        rl_seat = find_rl_id(seats)
        rl_starting_chips = rl_seat.chips

        # Play until RL is eliminated or only 1 player left
        while True:
            active = [s for s in seats if s.chips > 0]
            if not rl_seat or rl_seat.chips <= 0:
                break
            if len(active) <= 1:
                break

            sb, bb = escalate_blinds(hand_count + 1, args.sb, args.bb, 100)
            di = dealer % len(active)
            active_bots = {s.player_id: bots[s.player_id] for s in active}

            try:
                run_episode(Table(), active, active_bots, sb, bb, di)
            except Exception as e:
                # Skip problematic hands
                print(f"  [ERR ep{ep} hand{hand_count}] {e}")
                break

            hand_count += 1
            dealer = (dealer + 1) % max(len(active), 1)

            # Record reward for this hand
            current_chips = rl_seat.chips
            reward = (current_chips - rl_starting_chips) / max(1, args.chips)
            rl.record_reward(reward)

        # Episode done
        total_hands += hand_count

        # Determine if RL won (has chips > starting)
        rl_final = rl_seat.chips
        won = rl_final > args.chips
        win_rate.append(1 if won else 0)

        ep_reward = (rl_final - args.chips) / max(1, args.chips)
        recent_rewards.append(ep_reward)

        # End episode in RL bot → triggers batch PPO update
        rl.end_episode()

        # Decay exploration
        if ep % 500 == 0 and rl.exploration_rate > 0.05:
            rl.exploration_rate = max(0.05, rl.exploration_rate * 0.95)

        ep_time = time.time() - ep_start
        episode_times.append(ep_time)

        # Report progress
        if ep % args.report_every == 0:
            elapsed = time.time() - start_time
            avg_win = sum(win_rate) / max(1, len(win_rate))
            avg_reward = sum(recent_rewards) / max(1, len(recent_rewards))
            avg_time = sum(episode_times) / max(1, len(episode_times))
            est_remaining = (args.episodes - ep) * avg_time / 60
            print(f"[{ep}/{args.episodes}] "
                  f"win={avg_win:.1%} "
                  f"avg_reward={avg_reward:+.3f} "
                  f"avg_hands={total_hands//ep} "
                  f"time={elapsed/60:.1f}m "
                  f"eta={est_remaining:.0f}m "
                  f"explore={rl.exploration_rate:.3f}")

        # Save periodic
        if ep % args.save_every == 0:
            torch.save({
                'policy': rl.policy_net.state_dict(),
                'value': rl.value_net.state_dict(),
            }, args.model_path)
            print(f"  → Saved model to {args.model_path}")

    # After all episodes
    rl.flush_buffer()
    torch.save({
        'policy': rl.policy_net.state_dict(),
        'value': rl.value_net.state_dict(),
    }, args.model_path)
    total_time = (time.time() - start_time) / 60
    print(f"\nTraining complete! {args.episodes} episodes in {total_time:.1f}m")
    print(f"Model saved to {args.model_path}")


if __name__ == "__main__":
    main()