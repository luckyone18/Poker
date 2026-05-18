#!/usr/bin/env python3
"""
Benchmark semua bot bawaan vs RandomBot dan vs SmartBot.
Masing-masing 200 hands heads-up, posisi alternating.
Output: tabel WR -> ~/Poker/scripts/baseline_bots.md
"""
import os
import sys
import time
import random

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.engine import Table, Seat, InProcessBot, RandomBot
from core.bot_api import Action, PlayerView, BotAdapter

# ── Bot imports ───────────────────────────────────────────────────────────────
from bots.poker_mind_bot import SmartBot
from bots.monte_carlo_bot import MonteCarloBot
from bots.cfr_bot import CFRBot
from bots.gto_bot import GTOBot
from bots.icm_bot import ICMBot
from bots.exploitative_bot import ExploitativeBot
from bots.rl_bot import RLBot
from bots.ml_bot import MLBot
from bots.opponent_model_bot import OpponentModelBot

# ── Adapter wrapper (mirip _PlayerViewAdapter di __init__.py) ──────────────────
class PlayerViewAdapter(BotAdapter):
    def __init__(self, bot):
        self.bot = bot
    def act(self, view: PlayerView) -> Action:
        result = self.bot.act(view)
        if isinstance(result, dict):
            return Action(result.get("type", "fold"), result.get("amount"))
        return result

# ── Bot registry ──────────────────────────────────────────────────────────────
BOTS = {
    "SmartBot": lambda: SmartBot(),
    "MonteCarloBot": lambda: MonteCarloBot(simulations=500),
    "CFRBot": lambda: CFRBot(iterations=100, use_average=True),
    "GTOBot": lambda: GTOBot(),
    "ICMBot": lambda: ICMBot(simulations=300),
    "ExploitativeBot": lambda: ExploitativeBot(),
    "RLBot": lambda: RLBot(training_mode=False, use_fallback=True),
    "MLBot": lambda: MLBot(use_fallback=True),
    "OpponentModelBot": lambda: OpponentModelBot(),
}

RANDOM_FACTORY = lambda: RandomBot()
SMART_FACTORY = lambda: SmartBot()

HANDS_PER_MATCH = 200
STARTING_CHIPS = 500
SMALL_BLIND = 5
BIG_BLIND = 10

# ── Run heads-up match (fresh chips each hand) ────────────────────────────────
def run_match(bot_a_factory, bot_b_factory, n_hands=HANDS_PER_MATCH, rng=None):
    """
    Play n_hands heads-up with alternating dealer.
    Reset chips to STARTING_CHIPS each hand.
    Returns: (wins_a, chips_won_a, chips_won_b)
    """
    rng = rng or random.Random()
    wins_a = 0
    total_chips_a = 0
    total_chips_b = 0

    for i in range(n_hands):
        # Alternate position
        if i % 2 == 0:
            seat_a = Seat(player_id="P1", chips=STARTING_CHIPS)
            seat_b = Seat(player_id="P2", chips=STARTING_CHIPS)
            dealer_idx = 0  # P1 = BTN, P2 = BB (heads-up)
        else:
            seat_a = Seat(player_id="P2", chips=STARTING_CHIPS)
            seat_b = Seat(player_id="P1", chips=STARTING_CHIPS)
            dealer_idx = 0  # P1 = BTN, tetap index 0 dari perspektif seats

        seats = [seat_a, seat_b]

        # Wrap bots fresh (CFRBot, ExploitativeBot have state)
        bot_a = bot_a_factory()
        bot_b = bot_b_factory()

        bot_for = {}
        for s in seats:
            if s.player_id == seat_a.player_id:
                if isinstance(bot_a, RandomBot):
                    bot_for[s.player_id] = InProcessBot(bot_a)
                else:
                    bot_for[s.player_id] = PlayerViewAdapter(bot_a)
            else:
                if isinstance(bot_b, RandomBot):
                    bot_for[s.player_id] = InProcessBot(bot_b)
                else:
                    bot_for[s.player_id] = PlayerViewAdapter(bot_b)

        table = Table(rng=rng)
        try:
            net = table.play_hand(
                seats,
                small_blind=SMALL_BLIND,
                big_blind=BIG_BLIND,
                dealer_index=dealer_idx,
                bot_for=bot_for,
            )
        except Exception as e:
            print(f"  ERROR hand {i}: {e}")
            continue

        # Determine who won based on net chip change
        # (seats[0] is always seat_a as we constructed)
        net_a = net.get(seat_a.player_id, 0)
        net_b = net.get(seat_b.player_id, 0)

        if net_a > net_b:
            wins_a += 1
        # tie -> no win counted

        total_chips_a += net_a
        total_chips_b += net_b

    wr = wins_a / n_hands * 100 if n_hands > 0 else 0
    avg_chips_a = total_chips_a / n_hands if n_hands > 0 else 0
    avg_chips_b = total_chips_b / n_hands if n_hands > 0 else 0

    return {
        "win_rate": round(wr, 2),
        "avg_chips_won": round(avg_chips_a, 2),
        "avg_chips_lost": round(avg_chips_b, 2),
        "hands": n_hands,
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("BASELINE BOT BENCHMARK")
    print(f"Hands per match: {HANDS_PER_MATCH}")
    print(f"Starting chips: {STARTING_CHIPS}")
    print(f"Blinds: {SMALL_BLIND}/{BIG_BLIND}")
    print("=" * 70)

    results = {}

    # Use a fixed seed for reproducibility
    main_rng = random.Random(42)

    for bot_name, bot_factory in BOTS.items():
        print(f"\n{'─' * 60}")
        print(f"  Testing {bot_name}...")
        print(f"{'─' * 60}")

        # Test vs RandomBot
        print(f"  vs RandomBot ({HANDS_PER_MATCH} hands)...")
        t0 = time.time()
        res_random = run_match(bot_factory, RANDOM_FACTORY, HANDS_PER_MATCH, rng=main_rng)
        elapsed = time.time() - t0
        print(f"    WR={res_random['win_rate']:.1f}%  "
              f"avg_chips={res_random['avg_chips_won']:+.1f}  "
              f"time={elapsed:.1f}s")

        # Test vs SmartBot
        print(f"  vs SmartBot ({HANDS_PER_MATCH} hands)...")
        t0 = time.time()
        res_smart = run_match(bot_factory, SMART_FACTORY, HANDS_PER_MATCH, rng=main_rng)
        elapsed = time.time() - t0
        print(f"    WR={res_smart['win_rate']:.1f}%  "
              f"avg_chips={res_smart['avg_chips_won']:+.1f}  "
              f"time={elapsed:.1f}s")

        results[bot_name] = {
            "vs_RandomBot": res_random,
            "vs_SmartBot": res_smart,
        }

    # ── Print summary table ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    header = f"| {'Bot':<20s} | {'vs RandomBot WR':>14s} | {'vs SmartBot WR':>14s} | {'Avg Hands':>10s} |"
    sep = "|" + "-" * 22 + "|" + "-" * 16 + "|" + "-" * 16 + "|" + "-" * 12 + "|"

    print(header)
    print(sep)

    for bot_name in BOTS:
        r = results[bot_name]
        wr_rand = f"{r['vs_RandomBot']['win_rate']:.1f}%"
        wr_smart = f"{r['vs_SmartBot']['win_rate']:.1f}%"
        avg_hands_str = f"{r['vs_RandomBot']['hands']}"
        print(f"| {bot_name:<20s} | {wr_rand:>14s} | {wr_smart:>14s} | {avg_hands_str:>10s} |")

    # ── Write to markdown ─────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_bots.md")
    with open(out_path, "w") as f:
        f.write("# Baseline Bot Benchmark\n\n")
        f.write(f"**Hands per match:** {HANDS_PER_MATCH}  \n")
        f.write(f"**Starting chips:** {STARTING_CHIPS}  \n")
        f.write(f"**Blinds:** {SMALL_BLIND}/{BIG_BLIND}  \n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  \n\n")
        f.write("## Results\n\n")
        f.write(header + "\n")
        f.write(sep + "\n")
        for bot_name in BOTS:
            r = results[bot_name]
            wr_rand = f"{r['vs_RandomBot']['win_rate']:.1f}%"
            wr_smart = f"{r['vs_SmartBot']['win_rate']:.1f}%"
            avg_hands_str = f"{r['vs_RandomBot']['hands']}"
            f.write(f"| {bot_name:<20s} | {wr_rand:>14s} | {wr_smart:>14s} | {avg_hands_str:>10s} |\n")

        f.write("\n\n## Detailed Breakdown\n\n")
        for bot_name, r in results.items():
            f.write(f"### {bot_name}\n\n")
            f.write(f"- **vs RandomBot**: WR={r['vs_RandomBot']['win_rate']:.1f}%, "
                    f"avg_chips={r['vs_RandomBot']['avg_chips_won']:+.1f} "
                    f"({r['vs_RandomBot']['hands']} hands)\n")
            f.write(f"- **vs SmartBot**: WR={r['vs_SmartBot']['win_rate']:.1f}%, "
                    f"avg_chips={r['vs_SmartBot']['avg_chips_won']:+.1f} "
                    f"({r['vs_SmartBot']['hands']} hands)\n\n")

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()