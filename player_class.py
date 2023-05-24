from typing import List

import eval7


class Player_State:
    def __init__(
        self,
        hand,
        board,
        stack,
        pot,
        phase,
        raise_amt,
        put_in_pot,
        players_in_hand,
        position,
        equity,
        ex_eq
    ) -> None:
        self.hand = hand
        self.board = board
        self.stack = stack
        self.pot = pot
        self.phase = phase
        self.raise_amt = raise_amt
        self.put_in_pot = put_in_pot
        self.players_in_hand = players_in_hand
        self.position = position
        self.equity = equity
        self.exact_equity = ex_eq

    def get_attr_as_list(self) -> List:
        items_as_list: List = []
        for attribute, value in vars(self).items():
            if attribute not in ["hand", "board", "exact_equity"]:
                items_as_list.append(value)
        return items_as_list


class Player:
    def __init__(self) -> None:
        self.hand: List[eval7.Card]
        self.stack: float
        self.starting_stack: float
        self.position: float
        self.put_in_pot: float
        self.folded: bool = False

    def set_stack(self, stack: float) -> None:
        self.stack = stack

    def adjust_stack(self, change: float) -> None:
        self.stack += change
        if self.stack < 0:
            self.stack = 0

    def get_dealt(self, hand: List[eval7.Card], position: int, blind: float) -> None:
        self.hand = hand
        self.position = position
        self.starting_stack = self.stack
        self.put_in_pot = blind
        self.stack -= blind
