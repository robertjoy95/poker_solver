"""
Main script for using the solver
"""
import math
import random
from collections import deque
from copy import deepcopy, copy
from typing import List

import eval7
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from player_class import Player, Player_State


class Solver:
    def __init__(self) -> None:
        self.hand_data: dict = {
            "position": [],
            "choice": [],
            "pot": [],
            "phase": [],
            "equity": [],
            "stack": [],
            "players_in_hand": [],
            "raise_amount": [],
            "hand_reward": [],
        }
        self.current_hand: dict = {
            "position": [],
            "choice": [],
            "pot": [],
            "phase": [],
            "equity": [],
            "stack": [],
            "players_in_hand": [],
            "raise_amount": [],
            "hand_reward": [],
        }

    def update_hand_data(self, players: List[Player]) -> None:
        # update the current hand with reward data and merge current hand into hand data
        for item in range(len(self.current_hand["position"])):
            reward_val = (
                players[self.current_hand["position"][item]].stack
                - players[self.current_hand["position"][item]].starting_stack
            )
            self.current_hand["hand_reward"].append(reward_val)
        for key in self.hand_data.keys():
            self.hand_data[key] += self.current_hand[key]
        self.current_hand: dict = {
            "position": [],
            "choice": [],
            "pot": [],
            "phase": [],
            "equity": [],
            "stack": [],
            "players_in_hand": [],
            "raise_amount": [],
            "hand_reward": [],
        }

    def save_choice(
        self,
        choice: int,
        pot: float,
        phase: int,
        equity: float,
        stack: float,
        plyrs_in_hand: int,
        raise_amt: float,
        position: int,
    ) -> None:
        # save the current choice as a data point
        self.current_hand["position"].append(position)
        self.current_hand["choice"].append(choice)
        self.current_hand["pot"].append(pot)
        self.current_hand["phase"].append(phase)
        self.current_hand["equity"].append(equity)
        self.current_hand["stack"].append(stack)
        self.current_hand["players_in_hand"].append(plyrs_in_hand)
        self.current_hand["raise_amount"].append(raise_amt)

    def get_action(
        self,
        hand: List[eval7.Card],
        board: List[eval7.Card],
        stack: float,
        pot: float,
        phase: int,
        raise_amt: float,
        plyrs_in_hand: float,
        position: int,
        villain: eval7.HandRange,
    ) -> float:
        """
        how we'll get an action for the player
        """
        actions: List[float] = [-1, 0, 0.25, 0.33, 0.5, 0.66, 0.75, 1.0, 1.5, stack]
        # TODO: get actual villain ranges
        equity = round(
            eval7.py_hand_vs_range_monte_carlo(hand, villain, board, 10000000), 3
        )

        choice = random.randint(0, len(actions) - 1)
        bet_amt = pot * actions[choice]
        if raise_amt == 0 and bet_amt < 0:
            bet_amt = 0
            choice = 1
        if bet_amt > stack:
            bet_amt = stack
        self.save_choice(
            choice, pot, phase, equity, stack, plyrs_in_hand, raise_amt, position
        )

        return bet_amt


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 24)
        self.fc2 = nn.Linear(24, 48)
        self.fc3 = nn.Linear(48, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class DQNPokerSolver:
    def __init__(
        self,
        n_episodes=1000,
        gamma=1.0,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_log_decay=0.05,
        alpha=0.01,
        alpha_decay=0.005,
        batch_size=64,
        state_vars=8,
        quiet=False,
    ):
        self.memory = deque(maxlen=100000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.number_of_players = 6  # TODO - change this to variable
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.quiet = quiet
        self.state_variables = state_vars
        self.hand_states: List[Player_State] = []
        self.episode = 1
        #self.actions: List[float] = [-1, 0, 0.25, 0.33, 0.5, 0.66, 0.75, 1.0, 1.5, 1000]
        self.actions: List[float] = [-1, 0, 0.33, 0.66, 1.0, 2]
        # Init model
        self.dqn = DQN()
        self.criterion = torch.nn.MSELoss()
        self.opt = torch.optim.Adam(self.dqn.parameters(), lr=0.01)

    def round_start(self) -> None:
        # reset hand states
        self.hand_states = []

    def update_states(self, actions_taken, players_remaining: int):
        # update each state with a given reward value for the action and store in memory
        for i, state in enumerate(self.hand_states):
            if actions_taken[i] < 0:
                # player did not take an action, don't add to dataset
                continue
            reward = self.get_reward(state, actions_taken[i], players_remaining)
            next_state: Player_State = copy(state)
            # next state
            # only increase pot if raise amount is valid
            done: bool = False
            if self.actions[actions_taken[i]] * state.pot > 2 * state.raise_amt:
                # next state is after raising a bet
                next_state.pot += state.pot * self.actions[actions_taken[i]]
                next_state.raise_amt = state.pot * self.actions[actions_taken[i]] - state.raise_amt
                next_state.put_in_pot = next_state.raise_amt
                next_state.equity = 2 / players_remaining * state.equity
                if players_remaining == 1:
                    next_state.equity = 1.0
            elif i > 0:
                # after calling a bet
                next_state.pot += state.raise_amt
                next_state.raise_amt = state.pot * self.actions[actions_taken[i]] - state.raise_amt
                next_state.put_in_pot = next_state.raise_amt
                next_state.equity = 2 / players_remaining * state.equity
            else:
                # after a fold
                next_state.equity = 0
                done = True
            proccessed_state = self.preprocess_state(state.get_attr_as_list())
            proccessed_nxt_state = self.preprocess_state(next_state.get_attr_as_list())
            self.remember(
                proccessed_state,
                actions_taken[i],
                reward,
                proccessed_nxt_state,
                done,
            )

    def get_reward(self, state: Player_State, action, players_remaining) -> float:
        # get an approximate reward at the end of each rotation
        if action == 0:
            # loss is what has been put in pot
            reward = -1 * state.put_in_pot
        elif players_remaining == 1 and action > 1:
            # if the action resulted in all others folding, reward is pot
            reward = state.pot
        else:
            # reward is EV - call amount
            reward = 2 / players_remaining * state.exact_equity * state.pot - (
                state.raise_amt - state.put_in_pot
            )
        # normalize reward (rw-min)/(max-min)
        reward = (reward - (-1000)) / (self.number_of_players * 1000 - (-1000))
        return reward

    def get_epsilon(self, t):
        return max(
            self.epsilon_min,
            min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)),
        )

    def preprocess_state(self, state: Player_State):
        # process the current situation's info into a tensor
        return torch.tensor(
            np.reshape(state, [1, self.state_variables]), dtype=torch.float32
        )

    def choose_action(self, state: Player_State, epsilon):
        if np.random.random() <= epsilon:
            choice = random.randint(0, len(self.actions) - 1)
            #print(f"choice {choice}")
            return choice
        else:
            with torch.no_grad():
                return torch.argmax(self.dqn(state)).numpy()
            
    def save_model(self) -> None:
        torch.save(self.dqn, "hand_data/DQN_model.pt")

    def load_model(self) -> None:
        self.dqn = torch.load("hand_data/DQN_model.pt")

    def remember(self, state, action, reward, next_state, done):
        reward = torch.tensor(reward)
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        y_batch, y_target_batch = [], []
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y = self.dqn(state)
            y_target = y.clone().detach()
            with torch.no_grad():
                y_target[0][action] = (
                    reward
                    if done
                    else reward + self.gamma * torch.max(self.dqn(next_state)[0])
                )
            y_batch.append(y[0])
            y_target_batch.append(y_target[0])

        y_batch = torch.cat(y_batch)
        y_target_batch = torch.cat(y_target_batch)

        self.opt.zero_grad()
        loss = self.criterion(y_batch, y_target_batch)
        loss.backward()
        self.opt.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_action(self, state: Player_State):
        """
        how we'll get an action for the player
        """
        # add the current state to our stored matrix
        self.hand_states.append(state)
        pre_state = self.preprocess_state(state.get_attr_as_list())
        action = self.choose_action(pre_state, self.get_epsilon(self.episode))
        # if there is no negative EV to folding, ensure we don't fold
        if action == 0 and state.put_in_pot == state.raise_amt:
            action = 1
        self.episode += 1
        return action, round(min(self.actions[action]*state.pot, state.stack), 3)


if __name__ == "__main__":
    dqs = DQNPokerSolver()
