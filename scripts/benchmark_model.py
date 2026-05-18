"""
Benchmark RL model vs RandomBot and SmartBot.
Evaluates model across multiple chip stacks and reports detailed stats.
Usage:
    python scripts/benchmark_model.py --model models/rl_model_run3.pt --episodes 2000
"""
import os
import sys
import time
import json
import argparse
from collections import defaultdict

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from core.engine import Table, Seat, InProcessBot, RandomBot, eval_hand
from core.bot_api import Action, BotAdapter, PlayerView
from bots.rl_bot import RLBot
from bots.poker_mind_bot import SmartBot


class PlayerViewAdapter(BotAdapter):
    def __init__(self, bot):
        self.bot = bot

    def act(self, view: PlayerView) -> Action:
        return self.bot.act(view)


def run_tournament(rl_bot, opponent_bot_factory, chips=500, rl_position="P2"):
    """Run a single heads-up tournament and return results."""
    table = Table()
    rl_seat = Seat(player_id=rl_position, chips=chips)
    opp_seat = Seat(player_id="P2" if rl_position == "P1" else "P1", chips=chips)
    seats = [rl_seat, opp_seat] if rl_position == "P1" else [opp_seat, rl_seat]

    rl_inprocess = InProcessBot(rl_bot)
    opponent = opponent_bot_factory()

    bots = {}
    for s in seats:
        bots[s.player_id] = rl_inprocess if s.player_id == rl_position else opponent

    # Play tournament
    hand_count = 0
    dealer_idx = 0

    while True:
        players_alive = [s for s in seats if s.chips > 0]
        if len(players_alive) <= 1:
            break

        hand_count += 1
        table.play_hand(seats, small_blind=10, big_blind=20, dealer_index=dealer_idx, bot_for=bots)

        # Advance dealer
        dealer_idx = (dealer_idx + 1) % len(seats)
        # Skip dead players for next dealer
        for _ in range(len(seats)):
            if seats[dealer_idx].chips > 0:
                break
            dealer_idx = (dealer_idx + 1) % len(seats)

    rl_won = seats[0].chips > 0 if rl_position == "P1" else seats[1].chips > 0
    rl_final_chips = rl_seat.chips

    return {
        "rl_won": rl_won,
        "rl_final_chips": rl_final_chips,
        "opp_final_chips": opp_seat.chips,
        "hands_played": hand_count,
    }


def benchmark(model_path, episodes=2000, chips=500, device="cpu"):
    """Run benchmarks against RandomBot and SmartBot."""
    print("=" * 70)
    print(f"BENCHMARK: {model_path}")
    print(f"Episodes: {episodes} per opponent")
    print(f"Starting chips: {chips}")
    print("=" * 70)
    print()

    # Load model
    rl_bot = RLBot(
        model_path=model_path,
        training_mode=False,
        use_fallback=True,
        device=device,
    )
    print(f"Model loaded: {model_path}")
    print(f"Device: {device}")
    print(f"Exploration rate: {rl_bot.exploration_rate}")
    print()

    opponents = [
        ("RandomBot", lambda: InProcessBot(RandomBot())),
        ("SmartBot", lambda: PlayerViewAdapter(SmartBot())),
    ]

    results = {}

    for name, factory in opponents:
        print(f"\n{'─' * 60}")
        print(f"  vs {name}")
        print(f"{'─' * 60}")

        wins = 0
        total_chips = 0
        total_hands = 0
        chip_distribution = defaultdict(int)
        t_start = time.time()

        for ep in range(1, episodes + 1):
            rl_pos = "P2" if ep % 2 == 0 else "P1"  # Alternate position
            result = run_tournament(rl_bot, factory, chips=chips, rl_position=rl_pos)
            wins += result["rl_won"]
            total_chips += result["rl_final_chips"]
            total_hands += result["hands_played"]

            # Track chip distribution
            stack_bucket = (result["rl_final_chips"] // 100) * 100
            chip_distribution[stack_bucket] += 1

            if ep % 500 == 0:
                elapsed = time.time() - t_start
                wr = wins / ep * 100
                avg_chips = total_chips / ep
                avg_hands = total_hands / ep
                speed = ep / elapsed
                print(f"  [{ep:5d}/{episodes}]  WR={wr:5.1f}%  avg_chips={avg_chips:6.0f}  "
                      f"avg_hands={avg_hands:4.1f}  speed={speed:5.1f} eps/s")

        elapsed = time.time() - t_start
        wr = wins / episodes * 100
        avg_chips = total_chips / episodes
        avg_hands = total_hands / episodes

        print(f"\n  ✓ FINAL: WR={wr:.1f}%  avg_chips={avg_chips:.0f}  "
              f"avg_hands={avg_hands:.1f}  time={elapsed:.0f}s")

        results[name] = {
            "win_rate": round(wr, 1),
            "avg_final_chips": round(avg_chips, 0),
            "avg_hands_per_game": round(avg_hands, 1),
            "total_episodes": episodes,
            "total_time_s": round(elapsed, 1),
            "chip_distribution": dict(sorted(chip_distribution.items())),
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, r in results.items():
        print(f"  vs {name:12s}  WR={r['win_rate']:5.1f}%  "
              f"avg_chips={r['avg_final_chips']:6.0f}  "
              f"avg_hands={r['avg_hands_per_game']:4.1f}")

    # Save results
    os.makedirs("benchmarks", exist_ok=True)
    result_path = f"benchmarks/benchmark_{os.path.basename(model_path)}_ep{episodes}.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {result_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/rl_model_run3.pt")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--chips", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    benchmark(args.model, args.episodes, args.chips, args.device)