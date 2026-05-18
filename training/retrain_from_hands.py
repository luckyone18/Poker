"""
Re-train RL bot using hand histories collected from live play.

This uses offline (batch) learning from recorded hands — the bot
learns from past decisions and their outcomes without playing live.
Great for:
- Learning from real-money play data
- Fine-tuning without risking chips
- Periodic model updates from accumulated data
"""
import os
import sys
import json
import argparse
from collections import defaultdict

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from bots.rl_bot import RLBot
from core.engine import eval_hand, EVAL_HAND_MAX
from core.bot_api import Action


def load_hand_histories(jsonl_path: str):
    """Load hand histories from JSONL file."""
    hands = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                hands.append(json.loads(line))
    return hands


def train_from_hands(
    hands: list,
    base_model_path: str = None,
    output_path: str = "models/rl_model_finetuned.pt",
    epochs: int = 10,
    learning_rate: float = 1e-4,
):
    """
    Re-train RL bot from recorded hand histories.

    For each decision point:
    - Reconstruct the feature vector from saved state
    - Compute advantage = actual outcome - expected outcome
    - Update policy via PPO

    Args:
        hands: List of hand record dicts from HandHistoryCollector
        base_model_path: Path to base model (or None for fresh start)
        output_path: Where to save the fine-tuned model
        epochs: Number of passes through the data
        learning_rate: Learning rate for fine-tuning
    """
    print("=" * 70)
    print("RE-TRAINING FROM HAND HISTORIES")
    print("=" * 70)
    print(f"Hands loaded:    {len(hands)}")
    print(f"Base model:      {base_model_path or 'fresh start'}")
    print(f"Epochs:          {epochs}")
    print(f"Learning rate:   {learning_rate}")
    print(f"Output:          {output_path}")
    print()

    # Initialize bot
    rl_bot = RLBot(
        model_path=base_model_path or "",
        training_mode=True,
        learning_rate=learning_rate,
    )

    # Parse all decision points
    all_decisions = []
    for hand in hands:
        winner = hand.get("winner")
        for dp in hand.get("decisions", []):
            # Label: 1.0 if this player won the hand, -0.3 if lost, 0 if unknown
            if winner and dp["player_id"] == winner:
                outcome = 1.0
            elif winner:
                outcome = -0.3
            else:
                outcome = 0.0

            all_decisions.append({
                "street": dp["street"],
                "hole_cards": dp["hole_cards"],
                "community_cards": dp["community_cards"],
                "stack": dp["stack"],
                "pot": dp["pot"],
                "current_bet": dp["current_bet"],
                "to_call": dp["to_call"],
                "action_taken": dp["action"],
                "action_amount": dp["action_amount"],
                "outcome": outcome,
            })

    print(f"Total decision points: {len(all_decisions)}")

    # Map action names to indices
    ACTION_MAP = {
        "fold": 0, "check": 1, "call": 2,
        "raise_small": 3, "raise_medium": 4, "raise_large": 5,
    }

    # Training loop
    rl_bot.policy_net.train()
    rl_bot.value_net.train()

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for dp in all_decisions:
            # Reconstruct features (simplified — same as _make_features in rl_bot.py)
            try:
                features = _reconstruct_features(dp)
            except Exception:
                continue

            features_t = torch.FloatTensor(features).unsqueeze(0)

            # Forward pass
            logits = rl_bot.policy_net(features_t)
            value = rl_bot.value_net(features_t)

            # Target: action that was taken, weighted by outcome
            target_action = ACTION_MAP.get(dp["action_taken"])
            if target_action is None:
                continue

            # Advantage = actual outcome
            advantage = dp["outcome"]

            # Policy loss: cross-entropy weighted by advantage sign
            target_t = torch.tensor([target_action])
            ce_loss = torch.nn.functional.cross_entropy(logits, target_t, reduction="none")
            if advantage > 0:
                policy_loss = ce_loss * (-advantage)  # reinforce good actions
            else:
                policy_loss = ce_loss * (-advantage) * 0.3  # dampen bad actions

            policy_loss = policy_loss.mean()

            # Value loss: MSE vs actual outcome
            value_loss = torch.nn.functional.mse_loss(
                value, torch.tensor([advantage]).float()
            )

            # Combined loss
            loss = policy_loss + 0.5 * value_loss

            # Update
            rl_bot.optimizer.zero_grad()
            rl_bot.value_optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(rl_bot.policy_net.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(rl_bot.value_net.parameters(), 1.0)

            rl_bot.optimizer.step()
            rl_bot.value_optimizer.step()

            total_loss += loss.item()

            # Track accuracy
            pred = logits.argmax(dim=1).item()
            if pred == target_action:
                correct += 1
            total += 1

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1) * 100
        print(f"  Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  acc={accuracy:.1f}%  "
              f"samples={total}")

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    rl_bot.save_model(output_path)
    print(f"\n✅ Model saved to {output_path}")

    return rl_bot


def _reconstruct_features(dp: dict):
    """
    Reconstruct the same 26-dim feature vector as _make_features in rl_bot.py.
    This is needed because we're not running through the live engine.
    """
    features = [0.0] * 26

    # Hand strength (0): encode hole cards rank
    if dp.get("hole_cards") and len(dp["hole_cards"]) >= 2:
        ranks = "23456789TJQKA"
        try:
            r1 = ranks.index(dp["hole_cards"][0][0]) / 12.0
            r2 = ranks.index(dp["hole_cards"][1][0]) / 12.0
            features[0] = max(r1, r2)
            features[1] = min(r1, r2)
            # Suited
            suited = dp["hole_cards"][0][1] == dp["hole_cards"][1][1]
            features[2] = 1.0 if suited else 0.0
            # Paired
            paired = dp["hole_cards"][0][0] == dp["hole_cards"][1][0]
            features[3] = 1.0 if paired else 0.0
        except (IndexError, ValueError):
            pass
    else:
        features[0] = features[1] = 0.5

    # Stack: normalized
    features[4] = min(dp.get("stack", 500) / 500.0, 1.0)

    # Pot: normalized
    features[5] = min(dp.get("pot", 0) / 100.0, 1.0)

    # Current bet: normalized
    features[6] = min(dp.get("current_bet", 0) / 100.0, 1.0)

    # To call: normalized
    features[7] = min(dp.get("to_call", 0) / 100.0, 1.0)

    # Street one-hot
    streets = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
    st = streets.get(dp.get("street", "preflop"), 0)
    for i in range(4):
        features[8 + i] = 1.0 if i == st else 0.0

    # Community card features (indices 12-19)
    if dp.get("community_cards"):
        ranks = "23456789TJQKA"
        for i, card in enumerate(dp["community_cards"][:4]):
            try:
                features[12 + i * 2] = ranks.index(card[0]) / 12.0
                features[12 + i * 2 + 1] = 1.0 if "hd".find(card[1]) >= 0 else 0.0
            except (IndexError, ValueError):
                pass

    # Pot odds (feature 20)
    if dp.get("to_call", 0) > 0 and dp.get("pot", 0) > 0:
        features[20] = min(dp["to_call"] / (dp["pot"] + dp["to_call"]), 1.0)

    # Position (feature 21): simplified — use remaining features as 0
    # Effective stack ratio (feature 22)
    features[22] = min(dp.get("stack", 500) / max(dp.get("pot", 1), 1), 10.0) / 10.0

    return features


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL hand history file")
    parser.add_argument("--base-model", default="models/rl_model_run3.pt")
    parser.add_argument("--output", default="models/rl_model_finetuned.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    hands = load_hand_histories(args.input)
    if not hands:
        print("No hands found in input file!")
        sys.exit(1)

    train_from_hands(
        hands=hands,
        base_model_path=args.base_model,
        output_path=args.output,
        epochs=args.epochs,
        learning_rate=args.lr,
    )