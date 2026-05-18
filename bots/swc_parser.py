"""SWC Poker WebSocket message parser — converts SWC messages to internal PlayerView format."""
import os, sys
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


@dataclass
class SWCEvent:
    """Base event from SWC WebSocket."""
    type: str
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HandStart(SWCEvent):
    """A new hand is dealt."""
    hand_id: str = ""
    table_id: str = ""
    dealer_index: int = 0
    small_blind: int = 1
    big_blind: int = 2


@dataclass
class PlayerAction(SWCEvent):
    """A player took an action."""
    player_id: str = ""
    action: str = ""           # fold, check, call, raise, allin
    amount: int = 0
    street: str = "preflop"


@dataclass
class CommunityCards(SWCEvent):
    """Community cards were revealed."""
    street: str = ""            # flop, turn, river
    cards: List[str] = field(default_factory=list)


@dataclass
class PotUpdate(SWCEvent):
    """Pot or chip amounts changed."""
    pot: int = 0
    player_id: str = ""
    chips: int = 0


@dataclass
class HandEnd(SWCEvent):
    """Hand reached showdown or all folded."""
    winner_id: str = ""
    pot: int = 0
    board: List[str] = field(default_factory=list)


@dataclass
class SeatUpdate(SWCEvent):
    """Seat/player status changed."""
    player_id: str = ""
    seat_index: int = 0
    chips: int = 0
    action: str = ""            # join, leave, sit_out


class SWCParser:
    """
    Parses SWC Poker WebSocket messages into typed events.

    SWC uses custom message format (not standard poker protocol).
    Messages have a 't' or 'type' field indicating the event type.
    """

    # Known SWC message type keywords (case-insensitive match)
    HAND_START_KEYWORDS = ["newhand", "handstart", "dealcards", "deal_cards"]
    PLAYER_ACTION_KEYWORDS = ["playeraction", "act", "player_action"]
    COMMUNITY_KEYWORDS = ["flop", "turn", "river", "boardupdate", "board_update"]
    POT_KEYWORDS = ["potupdate", "pot_update", "chipupdate", "chip_update"]
    WINNER_KEYWORDS = ["showdown", "winner", "handwinner", "hand_winner"]
    SEAT_KEYWORDS = ["seatupdate", "seat_update", "playerjoin", "playerleave"]

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset parser state for a new hand."""
        self._current_street = "preflop"
        self._community_cards: List[str] = []
        self._pot = 0

    def parse(self, message: Dict[str, Any]) -> Optional[SWCEvent]:
        """
        Parse a raw SWC WebSocket message.

        Args:
            message: Raw JSON dict from socket.io

        Returns:
            SWCEvent or None if unrecognized
        """
        if not isinstance(message, dict):
            return None

        msg_type = (message.get("t") or message.get("type") or "").lower()
        if not msg_type:
            return None

        # Hand start
        if any(kw in msg_type for kw in self.HAND_START_KEYWORDS):
            self.reset()
            return HandStart(
                type="hand_start",
                raw=message,
                hand_id=str(message.get("handId", message.get("id", ""))),
                table_id=str(message.get("tableId", "")),
                dealer_index=message.get("dealer", 0),
                small_blind=message.get("smallBlind", message.get("sb", 1)),
                big_blind=message.get("bigBlind", message.get("bb", 2)),
            )

        # Player action
        if any(kw in msg_type for kw in self.PLAYER_ACTION_KEYWORDS):
            action = message.get("action", message.get("act", ""))
            street = message.get("street", self._current_street)
            self._current_street = street
            if message.get("pot"):
                self._pot = message["pot"]
            return PlayerAction(
                type="player_action",
                raw=message,
                player_id=str(message.get("playerId", message.get("pid", ""))),
                action=action,
                amount=message.get("amount", message.get("bet", 0)),
                street=street,
            )

        # Community cards
        if any(kw in msg_type for kw in self.COMMUNITY_KEYWORDS):
            cards = message.get("cards", message.get("board", []))
            if isinstance(cards, list):
                self._community_cards = cards
            street = msg_type if msg_type in ("flop", "turn", "river") else "community"
            self._current_street = street
            return CommunityCards(
                type="community_cards",
                raw=message,
                street=street,
                cards=self._community_cards.copy(),
            )

        # Pot update
        if any(kw in msg_type for kw in self.POT_KEYWORDS):
            pot = message.get("pot", message.get("total", self._pot))
            if pot:
                self._pot = int(pot)
            return PotUpdate(
                type="pot_update",
                raw=message,
                pot=self._pot,
                player_id=str(message.get("playerId", "")),
                chips=message.get("chips", message.get("stack", 0)),
            )

        # Hand end
        if any(kw in msg_type for kw in self.WINNER_KEYWORDS):
            winner = message.get("winnerId", message.get("winner", ""))
            pot = message.get("pot", message.get("totalPot", self._pot))
            board = message.get("board", message.get("cards", self._community_cards))
            event = HandEnd(
                type="hand_end",
                raw=message,
                winner_id=str(winner),
                pot=int(pot) if pot else 0,
                board=board if isinstance(board, list) else [],
            )
            self.reset()
            return event

        # Seat update
        if any(kw in msg_type for kw in self.SEAT_KEYWORDS):
            return SeatUpdate(
                type="seat_update",
                raw=message,
                player_id=str(message.get("playerId", "")),
                seat_index=message.get("seat", message.get("seatIndex", 0)),
                chips=message.get("chips", message.get("stack", 0)),
                action=message.get("action", msg_type),
            )

        # Unknown message type — still useful for debugging
        return SWCEvent(type=msg_type, raw=message)


# ── Test with mock messages ─────────────────────────────────────────────────
if __name__ == "__main__":
    parser = SWCParser()

    mock_messages = [
        {"t": "newHand", "handId": "12345", "tableId": "T1", "dealer": 0, "sb": 1, "bb": 2},
        {"t": "act", "playerId": "hero", "action": "call", "amount": 2, "street": "preflop"},
        {"t": "flop", "cards": ["2d", "7c", "Js"]},
        {"t": "act", "playerId": "villain", "action": "raise", "amount": 10, "street": "flop"},
        {"t": "turn", "cards": ["2d", "7c", "Js", "Ah"]},
        {"t": "act", "playerId": "hero", "action": "call", "amount": 10, "street": "turn"},
        {"t": "river", "cards": ["2d", "7c", "Js", "Ah", "Kc"]},
        {"t": "showdown", "winnerId": "hero", "pot": 25, "board": ["2d", "7c", "Js", "Ah", "Kc"]},
    ]

    for msg in mock_messages:
        event = parser.parse(msg)
        if event:
            print(f"  [{event.type:>15}] {event}")
        else:
            print(f"  [UNPARSED] {msg}")