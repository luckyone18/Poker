"""Optimize MonteCarloBot thresholds via random search, then create an optimized sparring partner for RL training.

The optimized MonteCarloBot is tested vs SmartBot (200 hands per config).
Best config gets saved and used as the RL training opponent instead of SmartBot.
"""

import random, time, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.bot_api import PlayerView, Action
from bots.monte_carlo_bot import MonteCarloBot

# ── Random search space ──────────────────────────────────────────
SEARCH_SPACE = {
    "simulations":       [300, 500, 1000],
    "bet_threshold":     [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
    "raise_threshold":   [0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
    "late_adjust":       [-0.10, -0.05, 0.00, 0.05],  # subtract from threshold in late position
    "stack_cap_raise":   [0.20, 0.25, 0.30, 0.35, 0.40],
    "stack_cap_bet":     [0.15, 0.20, 0.25, 0.30],
    "pot_mult_raise":    [0.50, 0.75, 1.00, 1.25],
    "pot_mult_bet":      [0.33, 0.50, 0.66, 0.75],
}

NUM_TRIALS = 200
HANDS_PER_TRIAL = 200

# ── Optimized MonteCarloBot (parameterized) ───────────────────────
class OptimizedMonteCarloBot:
    """MonteCarloBot with tunable thresholds."""

    def __init__(self, simulations=500, bet_threshold=0.70, raise_threshold=0.65,
                 late_adjust=-0.05,  # more aggressive in late position
                 stack_cap_raise=0.30, stack_cap_bet=0.25,
                 pot_mult_raise=0.75, pot_mult_bet=0.50):
        self.simulations = simulations
        self.bet_threshold = bet_threshold
        self.raise_threshold = raise_threshold
        self.late_adjust = late_adjust
        self.stack_cap_raise = stack_cap_raise
        self.stack_cap_bet = stack_cap_bet
        self.pot_mult_raise = pot_mult_raise
        self.pot_mult_bet = pot_mult_bet
        # Reuse MonteCarloBot for equity calculation
        self._equity_bot = MonteCarloBot(simulations=simulations)

    def act(self, state: PlayerView):
        hole = state.hole_cards
        board = state.board
        pot = state.pot
        to_call = state.to_call
        legal = state.legal_actions
        position = state.position

        if not hole:
            for a in legal:
                if a["type"] == "check": return Action("check")
            return Action("fold")

        n_opps = len(state.opponents) or 1
        winrate = self._equity_bot._estimate_equity(hole, board, n_opps, sims=self.simulations)

        # Position adjustment
        position_order = {"UTG": 1.0, "UTG+1": 0.9, "MP": 0.7, "LJ": 0.6,
                          "HJ": 0.4, "CO": 0.2, "BTN": 0.0, "SB": 0.5, "BB": 0.7}
        pos_factor = position_order.get(position, 0.5)

        if pos_factor < 0.5:  # late position -> more aggressive
            bet_th = self.bet_threshold + self.late_adjust
            raise_th = self.raise_threshold + self.late_adjust
        else:
            bet_th = self.bet_threshold
            raise_th = self.raise_threshold

        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0

        # Facing a bet
        if to_call > 0:
            if winrate < pot_odds:
                return self._choose("fold", legal)
            if winrate < raise_th:
                return self._choose("call", legal)
            return self._raise(pot, legal)
        else:
            # No bet yet
            if winrate > bet_th:
                return self._bet(pot, legal)
            for a in legal:
                if a["type"] == "check":
                    return Action("check")
            return self._choose("fold", legal)

    def _choose(self, typ, legal):
        for a in legal:
            if a["type"] == typ: return Action(typ)
        for a in legal:
            if a["type"] in ("call", "check"): return Action(a["type"])
        return Action("fold")

    def _raise(self, pot, legal):
        for a in legal:
            if a["type"] == "raise":
                stack_cap = a["max"] * self.stack_cap_raise
                amt = max(a["min"], min(a["max"], pot * self.pot_mult_raise, stack_cap))
                return Action("raise", int(amt))
        return self._choose("call", legal)

    def _bet(self, pot, legal):
        for a in legal:
            if a["type"] == "bet":
                stack_cap = a["max"] * self.stack_cap_bet
                amt = max(a["min"], min(a["max"], pot * self.pot_mult_bet, stack_cap))
                return Action("bet", int(amt))
        return self._choose("check", legal)


# ── Quick matchup runner (lightweight) ────────────────────────────
def quick_matchup(bot, opponent_bot, num_hands=200):
    """Run heads-up match, both positions alternating. Returns (wins, total)."""
    from core.engine import Table, Seat
    from bots.poker_mind_bot import SmartBot
    from core.engine import RandomBot

    wins = 0
    for h in range(num_hands):
        # Alternate positions
        if h % 2 == 0:
            bot_for = {0: bot}
            opp_for = {1: opponent_bot}
            dealer = 0
        else:
            bot_for = {1: bot}
            opp_for = {0: opponent_bot}
            dealer = 1

        # BUGFIX: fresh seats each hand
        seats = [
            Seat(player_id=0, chips=500),
            Seat(player_id=1, chips=500),
        ]
        bots = {}
        for k in range(2):
            bots[k] = bot_for.get(k) or opp_for.get(k)

        table = Table()
        try:
            table.play_hand(seats, small_blind=1, big_blind=2, dealer_index=dealer,
                           bot_for=bots, on_event=None, log_decisions=False)
        except Exception:
            pass

        # Determine winner (who gained chips)
        if seats[0].chips > 500:
            winner_id = 0
        elif seats[1].chips > 500:
            winner_id = 1
        else:
            # Tie or no change — skip
            continue

        bot_seat = 0 if bot_for.get(0) is bot else 1
        if winner_id == bot_seat:
            wins += 1

    return wins, num_hands


# ── Main: Random Search ──────────────────────────────────────────
def main():
    print(f"🚀 Random Search: {NUM_TRIALS} trials × {HANDS_PER_TRIAL} hands")
    print(f"   Search space: {len(SEARCH_SPACE)} params, {total_combinations()} combinations")
    print()
    
    best_wr = 0
    best_config = None
    results = []

    from bots.poker_mind_bot import SmartBot
    smart_bot = SmartBot()

    t0 = time.time()
    for trial in range(NUM_TRIALS):
        cfg = {k: random.choice(v) for k, v in SEARCH_SPACE.items()}
        bot = OptimizedMonteCarloBot(**cfg)
        wins, total = quick_matchup(bot, smart_bot, HANDS_PER_TRIAL)
        wr = wins / total * 100

        results.append((wr, cfg))
        if wr > best_wr:
            best_wr = wr
            best_config = cfg
            print(f"  🔥 Trial {trial+1}: WR={wr:.1f}% ← NEW BEST | cfg={cfg}")
        elif (trial + 1) % 20 == 0:
            print(f"  Trial {trial+1}: best so far WR={best_wr:.1f}%")
    
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"🏆 BEST CONFIG: WR={best_wr:.1f}% vs SmartBot")
    print(f"   {best_config}")
    print(f"   Time: {elapsed:.0f}s for {NUM_TRIALS} trials")
    print()

    # Save best config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'optimized_mc_config.json')
    import json
    with open(config_path, 'w') as f:
        json.dump({"wr_vs_smartbot": best_wr, "config": best_config, "trials": NUM_TRIALS}, f, indent=2)
    print(f"✅ Saved to {config_path}")

    # Top 5
    results.sort(key=lambda x: -x[0])
    print(f"\nTop 5 configs:")
    for wr, cfg in results[:5]:
        print(f"  {wr:.1f}% — {cfg}")


def total_combinations():
    t = 1
    for v in SEARCH_SPACE.values():
        t *= len(v)
    return t


if __name__ == '__main__':
    main()
