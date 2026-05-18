"""
Hand History Collector — records every hand played by the bot for later re-training.

Captures:
- Full game state at each decision point (cards, chips, pot, street)
- Bot's action taken
- Outcome (who won, final pot)
- Timestamp and table info

Two modes:
1. Local self-play (during RL training)
2. SWC live play (via WebSocket capture)

Output: JSONL files (one line per hand) → ready for re-training
"""
import os
import json
import time
import sys
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


@dataclass
class DecisionPoint:
    """One action taken by a bot at a decision point."""
    player_id: str
    street: str           # preflop, flop, turn, river
    hole_cards: List[str] # player's private cards
    community_cards: List[str]
    stack: int
    pot: int
    current_bet: int
    to_call: int
    action: str           # fold, check, call, raise_small, raise_medium, raise_large, allin
    action_amount: int    # chips put in
    equity: Optional[float] = None  # calculated hand strength


@dataclass
class HandRecord:
    """Complete record of one poker hand."""
    hand_id: str
    timestamp: str          # ISO 8601
    game_type: str          # "holdem", "omaha", etc.
    variant: str            # "cash", "tournament", "sitngo"
    table_id: str
    seats: List[Dict[str, Any]]  # [{player_id, chips, position}]
    decisions: List[DecisionPoint]
    winner: Optional[str] = None
    final_pot: int = 0
    final_community: List[str] = field(default_factory=list)
    source: str = "selfplay"  # "selfplay" or "swc_live"


class HandHistoryCollector:
    """Collects and persists hand histories."""

    def __init__(self, output_dir: str = "hand_histories"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._current_hand: Optional[HandRecord] = None
        self._total_hands = 0
        self._init_output_file()

    def _init_output_file(self):
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.output_path = os.path.join(self.output_dir, f"hands_{date_str}.jsonl")
        print(f"[HandHistory] Writing to {self.output_path}")

    def start_hand(
        self,
        table_id: str = "default",
        seats: List[Dict[str, Any]] = None,
        source: str = "selfplay",
        game_type: str = "holdem",
        variant: str = "tournament",
    ):
        """Start recording a new hand."""
        hand_id = hashlib.md5(
            f"{datetime.now(timezone.utc).isoformat()}_{self._total_hands}".encode()
        ).hexdigest()[:12]

        self._current_hand = HandRecord(
            hand_id=hand_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            game_type=game_type,
            variant=variant,
            table_id=table_id,
            seats=seats or [],
            decisions=[],
            source=source,
        )

    def record_decision(
        self,
        player_id: str,
        street: str,
        hole_cards: List[str],
        community_cards: List[str],
        stack: int,
        pot: int,
        current_bet: int,
        to_call: int,
        action: str,
        action_amount: int,
        equity: Optional[float] = None,
    ):
        """Record a bot's action at a decision point."""
        if self._current_hand is None:
            return

        dp = DecisionPoint(
            player_id=player_id,
            street=street,
            hole_cards=hole_cards,
            community_cards=community_cards,
            stack=stack,
            pot=pot,
            current_bet=current_bet,
            to_call=to_call,
            action=action,
            action_amount=action_amount,
            equity=equity,
        )
        self._current_hand.decisions.append(dp)

    def end_hand(
        self,
        winner: Optional[str] = None,
        final_pot: int = 0,
        final_community: List[str] = None,
    ):
        """Finalize and save the hand."""
        if self._current_hand is None:
            return

        self._current_hand.winner = winner
        self._current_hand.final_pot = final_pot
        self._current_hand.final_community = final_community or []
        self._current_hand.timestamp = datetime.now(timezone.utc).isoformat()

        # Write as JSONL
        with open(self.output_path, "a") as f:
            f.write(json.dumps(asdict(self._current_hand)) + "\n")

        self._total_hands += 1
        self._current_hand = None

    def get_stats(self) -> Dict[str, Any]:
        """Return collection statistics."""
        return {
            "total_hands": self._total_hands,
            "output_file": self.output_path,
            "output_dir": self.output_dir,
        }


class SWCHandCapture:
    """
    Captures hand histories from live SWC Poker sessions.
    Parses WebSocket messages and converts to HandRecords.

    Protocol reference (from reverse-engineering):
    - nodeProxyUrl: https://game.swcpoker.club:443/poker/
    - Messages: JSON with callback IDs
    - Events: PlayerCommand, round updates, showCards, pot updates
    """

    # Known SWC message types (from reverse-engineering)
    MESSAGE_TYPES = {
        "hand_start": ["newHand", "handStart", "dealCards"],
        "player_action": ["playerAction", "act"],
        "community_cards": ["flop", "turn", "river", "boardUpdate"],
        "pot_update": ["potUpdate", "chipUpdate"],
        "winner": ["showdown", "winner", "handWinner"],
        "seat_update": ["seatUpdate", "playerJoin", "playerLeave"],
    }

    def __init__(self, collector: HandHistoryCollector):
        self.collector = collector
        self._current_table: Dict[str, Any] = {}
        self._pending_hands: Dict[str, HandRecord] = {}

    def process_message(self, raw_msg: Dict[str, Any]) -> Optional[HandRecord]:
        """
        Process a raw SWC WebSocket message.
        Returns completed HandRecord if hand finished.
        """
        msg_type = raw_msg.get("t", raw_msg.get("type", ""))

        if msg_type in self.MESSAGE_TYPES["hand_start"]:
            self._on_hand_start(raw_msg)
        elif msg_type in self.MESSAGE_TYPES["player_action"]:
            self._on_player_action(raw_msg)
        elif msg_type in self.MESSAGE_TYPES["community_cards"]:
            self._on_community_cards(raw_msg)
        elif msg_type in self.MESSAGE_TYPES["winner"]:
            return self._on_winner(raw_msg)

        return None

    def _on_hand_start(self, msg: Dict[str, Any]):
        hand_id = msg.get("handId", msg.get("id", str(int(time.time() * 1000))))
        self.collector.start_hand(
            table_id=msg.get("tableId", "unknown"),
            source="swc_live",
            game_type="holdem",
            variant=msg.get("gameType", "cash"),
        )

    def _on_player_action(self, msg: Dict[str, Any]):
        if self.collector._current_hand is None:
            return

        self.collector.record_decision(
            player_id=msg.get("playerId", "unknown"),
            street=msg.get("street", "unknown"),
            hole_cards=msg.get("holeCards", []),
            community_cards=msg.get("communityCards", []),
            stack=msg.get("stack", 0),
            pot=msg.get("pot", 0),
            current_bet=msg.get("currentBet", 0),
            to_call=msg.get("toCall", 0),
            action=msg.get("action", "unknown"),
            action_amount=msg.get("amount", 0),
        )

    def _on_community_cards(self, msg: Dict[str, Any]):
        # Update current hand's community cards
        if self.collector._current_hand:
            self.collector._current_hand.final_community = msg.get("cards", [])

    def _on_winner(self, msg: Dict[str, Any]):
        if self.collector._current_hand is None:
            return None

        self.collector.end_hand(
            winner=msg.get("winnerId"),
            final_pot=msg.get("pot", 0),
            final_community=msg.get("board", []),
        )
        return self.collector._current_hand


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test: record a few mock self-play hands
    collector = HandHistoryCollector()

    for i in range(5):
        collector.start_hand(
            table_id="test_table",
            seats=[
                {"player_id": "P1", "chips": 500, "position": "BTN"},
                {"player_id": "P2", "chips": 500, "position": "BB"},
            ],
        )

        collector.record_decision(
            player_id="P1",
            street="preflop",
            hole_cards=["Ah", "Kh"],
            community_cards=[],
            stack=495,
            pot=3,
            current_bet=2,
            to_call=2,
            action="raise_medium",
            action_amount=12,
        )

        collector.record_decision(
            player_id="P2",
            street="preflop",
            hole_cards=["Qd", "Jd"],
            community_cards=[],
            stack=488,
            pot=15,
            current_bet=12,
            to_call=10,
            action="call",
            action_amount=12,
        )

        collector.end_hand(
            winner="P1" if i % 2 == 0 else "P2",
            final_pot=27 if i % 2 == 0 else 19,
        )

    print(collector.get_stats())