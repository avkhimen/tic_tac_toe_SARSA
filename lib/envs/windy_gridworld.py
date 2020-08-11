import gym
import numpy as np
import sys
from gym.envs.toy_text import discrete

#   7 0 4
#    \|/
#  3 - - 1
#    /|\
#   6 2 5

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
# Modification 1: Change directions to correspond
# to King's moves and change the index for 
# those directions. Add an option of doing nothing
UP_RIGHT = 4
RIGHT_DOWN = 5
DOWN_LEFT = 6
LEFT_UP = 7
DO_NOTHING = 8

class WindyGridworldEnv(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta, winds):
        new_position = np.array(current) + np.array(delta) + np.array([-1, 0]) * winds[tuple(current)]
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        is_done = tuple(new_position) == (3, 7)
        return [(1.0, new_state, -1.0, is_done)]

    def __init__(self):
        self.shape = (7, 10)

        nS = np.prod(self.shape)
        nA = 4

        # Wind strength
        winds = np.zeros(self.shape)
        winds[:,[3,4,5,8]] = 1
        winds[:,[6,7]] = 2

        # Calculate transition probabilities
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = { a : [] for a in range(nA) }
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0], winds)
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1], winds)
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0], winds)
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1], winds)

        # We always start in state (3, 0)
        isd = np.zeros(nS)
        isd[np.ravel_multi_index((3,0), self.shape)] = 1.0

        super(WindyGridworldEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human', close=False):
        self._render(mode, close)

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            # print(self.s)
            if self.s == s:
                output = " x "
            elif position == (3,7):
                output = " T "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")

class WindyGridworldWithKingsMovesEnv(discrete.DiscreteEnv):
    
    metadata = {'render.modes': ['human', 'ansi']}

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    # Modification 3: if stochastic wind is preesnt, the effect of 
    # the wind is supplemented with the addition of a modification 
    # (-1, 0, or 1)
    def _calculate_transition_prob(self, current, delta, winds):
        if self.stochastic_wind:
            probs_state_reward_done_list = []
            for modification in [-1, 0, 1]:
                new_position = (np.array(current) + np.array(delta) + np.array([-1 + modification, 0]) * winds[tuple(current)])
                new_position = self._limit_coordinates(new_position).astype(int)
                new_state = np.ravel_multi_index(tuple(new_position), self.shape)
                is_done = tuple(new_position) == (3, 7)
                probs_state_reward_done_list.append((float(1/3), new_state, -1.0, is_done))

            return probs_state_reward_done_list
        
        else:    
            new_position = np.array(current) + np.array(delta) + np.array([-1, 0]) * winds[tuple(current)]
            new_position = self._limit_coordinates(new_position).astype(int)
            new_state = np.ravel_multi_index(tuple(new_position), self.shape)
            is_done = tuple(new_position) == (3, 7)

            return [(1.0, new_state, -1.0, is_done)]

    def __init__(self, include_do_nothing = False, stochastic_wind = False):
        self.shape = (7, 10)

        nS = np.prod(self.shape)
        
        # Modification 2: Change the number of actions
        # to 8 to correspond to King's moves. Number of
        # actions will be 9 if we include doing nothing
        self.include_do_nothing = include_do_nothing
        nA = 8
        if self.include_do_nothing:
            nA = 9
        
        # Modification 3: if stochastic wind is present
        # the strength of the wind varies by 1
        self.stochastic_wind = stochastic_wind

        # Wind strength
        winds = np.zeros(self.shape)
        winds[:,[3,4,5,8]] = 1
        winds[:,[6,7]] = 2

        # Calculate transition probabilities
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = { a : [] for a in range(nA) }
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0], winds)
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1], winds)
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0], winds)
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1], winds)
            
            # Modification 4: include King's moves
            P[s][UP_RIGHT] = self._calculate_transition_prob(position, [-1, 1], winds)
            P[s][RIGHT_DOWN] = self._calculate_transition_prob(position, [1, 1], winds)
            P[s][DOWN_LEFT] = self._calculate_transition_prob(position, [1, -1], winds)
            P[s][LEFT_UP] = self._calculate_transition_prob(position, [-1, -1], winds)
            
            # Modification 2: define probaility for the
            # action of doing nothing, if applicable
            if self.include_do_nothing:
                P[s][DO_NOTHING] = self._calculate_transition_prob(position, [0, 0], winds)

        # We always start in state (3, 0)
        isd = np.zeros(nS)
        isd[np.ravel_multi_index((3,0), self.shape)] = 1.0
        
        # Modification 5: change the super class to 
        # WindyGridworldWithKingsMovesEnv
        super(WindyGridworldWithKingsMovesEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human', close=False):
        self._render(mode, close)

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            # print(self.s)
            if self.s == s:
                output = " x "
            elif position == (3,7):
                output = " T "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")