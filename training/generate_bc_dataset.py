"""
Generate behavioral cloning dataset: (state_features, expert_action) pairs.
Records decisions from SmartBot and MonteCarloBot during heads-up matches.

Uses the same 26-dim feature encoding as RLBot._make_features().
"""
import os, sys, pickle, argparse, random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bots.rl_bot import RLBot
from bots.poker_mind_bot import SmartBot
from bots.monte_carlo_bot import MonteCarloBot
from core.engine import Table, Seat, InProcessBot, RandomBot
from core.bot_api import BotAdapter, Action


# ── Action → index mapping (matches RLBot._action_idx_to_action) ──
ACTION_TYPE_TO_IDX = {
    "fold": 0,
    "check": 1,
    "call": 2,
}

def action_to_idx(action: Action, legal_actions) -> int:
    """Convert Action to 0-5 index matching RLBot's encoding."""
    typ = action.type
    # If it's a raise/bet, determine bucket
    if typ in ("raise", "bet"):
        # Get range
        raises = [a for a in legal_actions if a["type"] == "raise"]
        bets = [a for a in legal_actions if a["type"] == "bet"]
        candidates = raises if raises else bets
        if not candidates:
            return 2  # fallback to call
        lo, hi = candidates[0]["min"], candidates[0]["max"]
        if hi <= lo:
            return 3  # small
        amt = action.amount if action.amount else 0
        frac = (amt - lo) / (hi - lo) if hi > lo else 0
        if frac < 0.33:
            return 3  # small raise
        elif frac < 0.66:
            return 4  # medium raise
        else:
            return 5  # large raise
    # Fold/check/call
    return ACTION_TYPE_TO_IDX.get(typ, 0)


class ExpertWrapper(BotAdapter):
    """Wraps an expert bot and records (features, action) for BC."""
    def __init__(self, expert_bot, recorder):
        self.bot = expert_bot
        self.recorder = recorder  # RLBot instance for _make_features

    def act(self, view):
        # SmartBot/MCBot expect PlayerView-like object, not dict
        if isinstance(view, dict):
            class DictView:
                def __init__(self, d):
                    for k, v in d.items():
                        setattr(self, k, v)
            view = DictView(view)
        action = self.bot.act(view)
        # Record the expert decision
        features = self.recorder._make_features(view)  # 1x26 tensor
        idx = action_to_idx(action, view.legal_actions)
        self.recorder.current_episode.append({
            'features': features.detach().cpu(),
            'action': idx,
        })
        return action


def generate_dataset(
    expert_type: str = "smartbot",
    num_episodes: int = 500,
    chips: int = 2000,
    output_path: str = "models/bc_dataset.pkl",
):
    """Generate BC dataset from expert vs random/smartbot matches."""
    print(f"Generating BC dataset: {expert_type} expert, {num_episodes} episodes")

    # Create RLBot just for the feature encoder
    recorder = RLBot(model_path="", training_mode=False, use_fallback=False)

    # Create expert
    if expert_type == "smartbot":
        raw_expert = SmartBot()
    elif expert_type == "mc200":
        raw_expert = MonteCarloBot(simulations=200)
    elif expert_type == "mc500":
        raw_expert = MonteCarloBot(simulations=500)
    else:
        raise ValueError(f"Unknown expert: {expert_type}")

    expert = ExpertWrapper(raw_expert, recorder)

    # Opponent: mix of random and smartbot for diversity
    all_samples = []

    for ep in range(1, num_episodes + 1):
        recorder.current_episode = []

        if ep % 3 == 0:
            opponent = InProcessBot(RandomBot())
            opp_name = "random"
        elif ep % 3 == 1:
            opponent = InProcessBot(SmartBot())
            opp_name = "smartbot"
        else:
            opponent = InProcessBot(MonteCarloBot(simulations=100))
            opp_name = "mc100"

        seats = [
            Seat(player_id="P1", chips=chips),
            Seat(player_id="EXPERT", chips=chips),
        ]
        bots = {
            "P1": opponent,
            "EXPERT": InProcessBot(expert),
        }

        table = Table()
        dealer_idx = 0

        while True:
            active = [s for s in seats if s.chips > 0]
            if len(active) <= 1:
                break

            table.play_hand(
                seats=active,
                small_blind=1, big_blind=2,
                dealer_index=dealer_idx % len(active),
                bot_for={s.player_id: bots[s.player_id] for s in active},
                on_event=None,
                log_decisions=False,
            )
            dealer_idx = (dealer_idx + 1) % len(seats)
            if dealer_idx > 5000:
                break

        # Extract recorded samples
        for step in recorder.current_episode:
            all_samples.append({
                'features': step['features'].numpy().flatten(),
                'action': step['action'],
            })

        if ep % 100 == 0:
            print(f"  ep {ep}/{num_episodes}: {len(all_samples)} samples collected")

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(all_samples, f)

    print(f"\n✅ Saved {len(all_samples)} samples to {output_path}")

    # Stats
    action_counts = defaultdict(int)
    for s in all_samples:
        action_counts[s['action']] += 1
    action_names = ["fold", "check", "call", "raise_s", "raise_m", "raise_l"]
    print("Action distribution:")
    for i in range(6):
        print(f"  {action_names[i]}: {action_counts[i]} ({action_counts[i]/len(all_samples)*100:.1f}%)")

    return all_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", type=str, default="smartbot",
                        choices=["smartbot", "mc200", "mc500"])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--chips", type=int, default=2000)
    parser.add_argument("--output", type=str, default="models/bc_dataset.pkl")
    args = parser.parse_args()

    generate_dataset(
        expert_type=args.expert,
        num_episodes=args.episodes,
        chips=args.chips,
        output_path=args.output,
    )