"""
Script containing the main game classes, gets decision making from solver.py
"""
import pprint
from copy import copy
from collections import deque
from typing import List

import eval7
import pandas as pd

from player_class import Player, Player_State
from solver import DQNPokerSolver, Solver


class Holdem:
    def __init__(self, test=False) -> None:
        self.board: List[eval7.Card]
        self.players: deque[Player] = deque(maxlen=10)
        self.pot: float
        self.players_in_hand: float
        self.solver = DQNPokerSolver()
        self.raiser: int = 1
        self.testing: bool = test
        pass

    def display(self, text: str):
        # used for testing/debugging
        if self.testing:
            print(text)

    def set_players(self, stack_sizes: List[float]) -> None:
        for s in stack_sizes:
            p = Player()
            p.set_stack(s)
            self.players.append(p)
        self.players_in_hand = len(self.players)

    def set_up_next_hand(self) -> None:
        # set up players for the next hand
        self.pot = 0
        moved_plr = self.players.popleft()
        self.players.append(moved_plr)
        self.players_in_hand = len(self.players)
        self.raiser = 1
        for ind, p in enumerate(self.players):
            # reset stack if it gets too short or large
            if p.stack < 10 or p.stack > 1000:
                p.stack = 100
            p.folded = False
            p.position = ind

    def set_up_next_round(self) -> None:
        # reset PIP val for the next round
        for p in self.players:
            p.put_in_pot = 0
        self.raiser = -1

    def save_nn_model(self) -> None:
        self.solver.save_model()

    def load_nn_model(self) -> None:
        self.solver.load_model()
        # if trained make sure solver knows not to randomize selection
        self.solver.episode = 1000

    def play_hand(self) -> None:
        deck = eval7.Deck()
        deck.shuffle()
        self.board = []
        phase = 0
        # TODO: multiple pots for when player is all in
        self.pot = 1.5
        # initialize hand
        cards_dealt = 0
        for i, p in enumerate(self.players):
            if i == 0:
                blind = 0.5
            elif i == 1:
                blind = 1.0
            else:
                blind = 0.0
            p.get_dealt(deck.deal(2), i, blind)
            cards_dealt += 2
            self.display(str(p.hand))
            self.display(str(p.stack))
        # pre-flop action
        self.play_stage(1.0, phase)
        self.set_up_next_round()
        if self.players_in_hand == 1:
            self.get_winner()
            return
        self.board += deck.deal(3)
        phase += 1

        # flop
        self.play_stage(0, phase)
        self.set_up_next_round()
        if self.players_in_hand == 1:
            self.get_winner()
            return
        self.board += deck.deal(1)
        phase += 1

        # turn
        self.play_stage(0, phase)
        self.set_up_next_round()
        if self.players_in_hand == 1:
            self.get_winner()
            return
        self.board += deck.deal(1)
        phase += 1

        # river
        self.play_stage(0, phase)
        self.get_winner()

    def play_stage(self, raise_amount: float, phase: int) -> None:
        # play the pre, flop, turn, or river
        new_raise: float
        while True:
            new_raise = self.go_around(raise_amount, phase)
            if raise_amount == new_raise:
                break
            raise_amount = new_raise
        return 

    def go_around(self, raise_amt: float, phase: int) -> float:
        self.solver.round_start()
        actions_taken = []
        start_index: int = 0
        villain_range = eval7.HandRange("AA, A3o, 32s")
        if phase == 0:
            # start from UTG
            start_index = 2
            actions_taken = [-1, -1]
        for player in range(len(self.players)):
            # first check win condition
            if self.players_in_hand == 1:
                actions_taken.append(-1)
                break
            # go through each player and get an action
            p_ind = (player + start_index) % len(self.players)
            if self.players[p_ind].folded or self.players[p_ind].stack == 0:
                actions_taken.append(-1)
                continue
            # get the actual hand of the most relevant villain for getting a reward value
            if self.raiser < 0 or self.raiser == p_ind:
                # if player is first to act most relevant range is next to act on their left
                for p in range(1, len(self.players)):
                    if not self.players[(p + p_ind) % len(self.players)].folded:
                        villain_hand = self.players[
                            (p + p_ind) % len(self.players)
                        ].hand
                        break
            else:
                villain_hand = tuple(self.players[self.raiser].hand)
            # fn can only read hands as strings so convert to str format
            v_hand = ""
            for item in villain_hand:
                v_hand += str(item)

            self.display(v_hand)
            # the player's hand whose equity we need
            h_hand = self.players[p_ind].hand

            # equity vs a range of hands
            equity = round(
                eval7.py_hand_vs_range_monte_carlo(
                    h_hand, villain_range, self.board, 10000000
                ),
                3
            )
            # the actual answer for reward calculations
            exact_equity = round(
                eval7.py_hand_vs_range_exact(
                    h_hand, eval7.HandRange(v_hand), self.board
                ),
                3,
            )
            state = Player_State(
                self.players[p_ind].hand,
                self.board,
                self.players[p_ind].stack,
                self.pot,
                phase,
                raise_amt,
                self.players[p_ind].put_in_pot,
                self.players_in_hand,
                self.players[p_ind].position,
                equity,
                exact_equity
            )
            self.display(state.get_attr_as_list())
            # get the action and bet amount from the solver
            action, action_amt = self.get_state_action(state)
            actions_taken.append(int(action))
            self.display(f"Action taken: {action}")
            self.display(action_amt)
            if action == 0 and self.players_in_hand > 1:
                self.players[p_ind].folded = True
                self.players_in_hand -= 1
            elif (
                action_amt - raise_amt > raise_amt or (action_amt == self.players[p_ind].stack and action_amt > raise_amt)
            ):
                # raise has to be greater than min-raise (or all in), else check
                raise_amt = action_amt
                self.raiser = p_ind
            if not self.players[p_ind].folded and self.players_in_hand > 1:
                # adjust player's stack and pot size if still in hand
                # TODO: acct for all ins
                self.pot += min(
                    raise_amt - self.players[p_ind].put_in_pot,
                    self.players[p_ind].stack,
                )
                self.players[p_ind].adjust_stack(
                    self.players[p_ind].put_in_pot - raise_amt
                )
                self.players[p_ind].put_in_pot = raise_amt
        if phase == 0:
            # shift actions taken when in preflop
            # TODO: make this variable dependent for when we can change # of players
            actions_taken[0] = actions_taken[4]
            actions_taken[1] = actions_taken[5]
            actions_taken.pop()
            actions_taken.pop()
        # used to get fold equity calculations
        self.display("Actions taken for the current round " + str(actions_taken))
        players_remaining = self.players_in_hand
        self.solver.update_states(actions_taken, players_remaining)
        return raise_amt
    
    def get_state_action(self, state: Player_State):
        # separated so it may be tested independently
        action, action_amt = self.solver.get_action(state)
        return action, action_amt

    def get_winner(self) -> None:
        # find who won if there's one player left in the hand or a showdown
        # adjust that player's stack
        winner: List[Player]
        if self.players_in_hand == 1:
            for p in self.players:
                if not p.folded:
                    winner = [p]
        else:
            # compare showdown hands to see which is best
            # assumption is that the evaluate function returns a higher number the better a hand is
            max_val: float = 0
            for p in self.players:
                if not p.folded:
                    hand_val: float = eval7.evaluate(p.hand)
                    if hand_val > max_val:
                        winner = [p]
                        max_val = hand_val
                    elif hand_val == max_val:
                        winner.append(p)
        win_val = self.pot / len(winner)
        for w in winner:
            w.adjust_stack(win_val)
            self.display(f"Winner Stack: {w.stack}")


def create_dataset(n_hands: int) -> None:
    game = Holdem()
    game.set_players([100] * 6)
    for n in range(n_hands):
        game.play_hand()
        game.set_up_next_hand()
        print(n)
    game.save_nn_model()


def get_dataset() -> dict:
    df = pd.read_csv("hand_data/solver_data.csv", index_col=0)
    d = df.to_dict("split")
    d = dict(zip(d["index"], d["data"]))
    return d


def normalize_rewards(data: dict, key: str) -> None:
    minval = min(data[key])
    maxval = max(data[key])
    normal_list = []
    for datapoint in data[key]:
        datapoint = (datapoint - minval) / (maxval - minval)
        normal_list.append(round(datapoint, 3))
    data[key] = normal_list


if __name__ == "__main__":
    create_dataset(1000)

    
