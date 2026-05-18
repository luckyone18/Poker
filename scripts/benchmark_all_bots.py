"""Benchmark all bots vs RandomBot and SmartBot for baseline comparison."""
import os, sys, time
from collections import defaultdict

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.engine import Table, Seat, InProcessBot, RandomBot
from core.bot_api import BotAdapter, PlayerView, Action
from bots.rl_bot import RLBot
from bots.poker_mind_bot import SmartBot
from bots.monte_carlo_bot import MonteCarloBot
from bots.cfr_bot import CFRBot
from bots.gto_bot import GTOBot
from bots.icm_bot import ICMBot
from bots.exploitative_bot import ExploitativeBot


BOTS = {
    "RLBot":          lambda: RLBot(training_mode=False, exploration_rate=0.0),
    "SmartBot":       lambda: SmartBot(),
    "MonteCarloBot":  lambda: MonteCarloBot(),
    "CFRBot":         lambda: CFRBot(),
    "GTOBot":         lambda: GTOBot(),
    "ICMBot":         lambda: ICMBot(),
    "ExploitativeBot": lambda: ExploitativeBot(),
}


class SimpleBotAdapter(BotAdapter):
    """Adapter: wraps any bot with .act(PlayerView) -> Action."""
    def __init__(self, bot):
        self.bot = bot

    def act(self, view: PlayerView) -> Action:
        return self.bot.act(view)


def run_matchup(bot_a_name, make_bot_a, bot_b_name, make_bot_b, num_hands=200):
    """Run heads-up matches, alternating positions."""
    wins_a = 0
    wins_b = 0
    total_hands = 0
    t0 = time.time()

    bot_a = make_bot_a()
    bot_b = make_bot_b()

    adapter_a = SimpleBotAdapter(bot_a)
    adapter_b = SimpleBotAdapter(bot_b)

    table = Table()

    for h in range(num_hands):
        # Alternating positions
        dealer_idx = h % 2

        seats = [
            {"player_id": "P1", "chips": 500},
            {"player_id": "P2", "chips": 500},
        ]

        if dealer_idx == 0:
            bots = {"P1": adapter_a, "P2": adapter_b}
        else:
            bots = {"P1": adapter_b, "P2": adapter_a}

        try:
            result = table.play_hand(
                seats=seats,
                small_blind=1,
                big_blind=2,
                dealer_index=dealer_idx,
                bot_for=bots,
                on_event=None,
                log_decisions=False,
            )

            winner = None
            for pid, net in result.items():
                if net > 0:
                    winner = pid
                    break

            if dealer_idx == 0:
                if winner == "P1":
                    wins_a += 1
                elif winner == "P2":
                    wins_b += 1
            else:
                if winner == "P1":
                    wins_b += 1
                elif winner == "P2":
                    wins_a += 1

            total_hands += 1
        except Exception as e:
            continue

        if (h + 1) % 50 == 0:
            elapsed = time.time() - t0
            wr = wins_a / max(total_hands, 1) * 100
            print(f"  {h+1}/{num_hands}  {bot_a_name} WR={wr:.0f}%  ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    wr_a = wins_a / max(total_hands, 1) * 100
    wr_b = wins_b / max(total_hands, 1) * 100

    return {
        "bot_a": bot_a_name,
        "bot_b": bot_b_name,
        "wr_a": wr_a,
        "wr_b": wr_b,
        "total_hands": total_hands,
        "time": elapsed,
    }


def main():
    print("=" * 70)
    print("BOT BENCHMARK — Baseline Win Rates")
    print("=" * 70)

    results = []

    for name, make_bot in BOTS.items():
        if name == "SmartBot":
            continue  # Skip self-matchup

        print(f"\n{name} vs RandomBot (200 hands)...")
        r = run_matchup(name, make_bot, "RandomBot", lambda: InProcessBot(RandomBot()), 200)
        results.append(r)
        print(f"  Result: {name} WR={r['wr_a']:.1f}%")

        print(f"{name} vs SmartBot (200 hands)...")
        r = run_matchup(name, make_bot, "SmartBot", SmartBot, 200)
        results.append(r)
        print(f"  Result: {name} WR={r['wr_a']:.1f}%")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY — All Matchups")
    print("=" * 70)
    print(f"{'Bot':<18} {'vs RandomBot':<15} {'vs SmartBot':<15} {'Time'}")
    print("-" * 60)

    # Group by bot
    bot_stats = {}
    for r in results:
        bot = r['bot_a']
        if bot not in bot_stats:
            bot_stats[bot] = {}
        opponent_type = "RandomBot" if r['bot_b'] == "RandomBot" else "SmartBot"
        bot_stats[bot][opponent_type] = r

    for bot_name, stats in bot_stats.items():
        vs_random = stats.get("RandomBot", {}).get("wr_a", "N/A")
        vs_smart = stats.get("SmartBot", {}).get("wr_a", "N/A")
        total_time = sum(s.get("time", 0) for s in stats.values())
        print(f"{bot_name:<18} {f'{vs_random:.1f}%':>8}       {f'{vs_smart:.1f}%':>8}       {total_time:.0f}s")


if __name__ == "__main__":
    main()