from collections import defaultdict
import numpy as np
import pandas as pd
import numpy as np
import sys
import ast
import itertools
import matplotlib
from io import StringIO
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# TicTacToeEnv class defines the environment and the 
# functions for the agent to learn the best policy 
# in playing Tic Tac Toe
class TicTacToeEnv():
    def __init__(self):
        self.shape = (3, 3)
        self.isd = np.ones(9) / 9
    # Checks whether the agent has won, drew, or lost the game
    # and returns the is_done flag and the rewards parameter
    def is_done(self):
        if (self.state[0] == 'x' and self.state[1] == 'x' and self.state[2] == 'x' or 
            self.state[3] == 'x' and self.state[4] == 'x' and self.state[5] == 'x' or
            self.state[6] == 'x' and self.state[7] == 'x' and self.state[8] == 'x' or
            self.state[0] == 'x' and self.state[3] == 'x' and self.state[6] == 'x' or
            self.state[1] == 'x' and self.state[4] == 'x' and self.state[7] == 'x' or
            self.state[2] == 'x' and self.state[5] == 'x' and self.state[8] == 'x' or
            self.state[0] == 'x' and self.state[4] == 'x' and self.state[8] == 'x' or
            self.state[2] == 'x' and self.state[4] == 'x' and self.state[6] == 'x'):
            is_done = True
            reward = 5
            print('YOU LOST')
        elif (self.state[0] == 'o' and self.state[1] == 'o' and self.state[2] == 'o' or 
              self.state[3] == 'o' and self.state[4] == 'o' and self.state[5] == 'o' or
              self.state[6] == 'o' and self.state[7] == 'o' and self.state[8] == 'o' or
              self.state[0] == 'o' and self.state[3] == 'o' and self.state[6] == 'o' or
              self.state[1] == 'o' and self.state[4] == 'o' and self.state[7] == 'o' or
              self.state[2] == 'o' and self.state[5] == 'o' and self.state[8] == 'o' or
              self.state[0] == 'o' and self.state[4] == 'o' and self.state[8] == 'o' or
              self.state[2] == 'o' and self.state[4] == 'o' and self.state[6] == 'o'):
            is_done = True
            reward = - 10
            print('YOU WON')
        elif self.no_empty_fields_left():
            is_done = True
            reward = -1
            print('ITS A DRAW')
        else:
            is_done = False
            reward = -1
            
        return is_done, reward
    
    # Resets the initial conditions for the start of each new game
    def reset(self):
        initial_agent_move = self.categorical_sample()
        empty_state = {i:'a' for i in range(9)}
        empty_state[initial_agent_move] = 'x'
        print('Computer move is: ')
        self.state = empty_state
        self.render_board()
        print('\n')
        self.render_board_states()
        print('\n')
        computer_move = self.computer_random_step()
        self.state[computer_move] = 'o'
        print('Your move is: ')
        self.render_board()
        print('\n')
        self.lastaction = None
        return self.state
    
    # Defines the initial agent action
    def categorical_sample(self):
        """
        Sample from categorical distribution
        Each row specifies class probabilities
        """
        prob_n = np.asarray(self.isd)
        csprob_n = np.cumsum(prob_n)
        return (csprob_n > np.random.rand()).argmax()
    
    # Performs the random step of the computer
    def computer_random_step(self):
        # available_actions = []
        # for key in self.state.keys():
        #     if self.state[key] == 'a':
        #         available_actions.append(key)
        computer_move = int(input('Please enter where you would place a mark '))
        return computer_move
    
    # Completes two steps, one for the agent and
    # one for the environment and returns the next_state,
    # reward, is_done flag, as well when no more fields are
    # left to play
    def step(self, a):
        self.state[a] = 'x'
        print('Computer move is: ')
        self.render_board()
        print('\n')
        no_fields_left = False
        if self.no_empty_fields_left():
            no_fields_left = True
        else:
        	self.render_board_states()
        	print('\n')
        	computer_move = self.computer_random_step()
        	self.state[computer_move] = 'o'
        	print('Your move is: ')
        	print('\n')
        	self.render_board()
        	print('\n')
        next_state = self.state
        self.state = next_state
        self.lastaction = a
        done, reward = self.is_done()
        return (next_state, reward, done, no_fields_left)
    
    # Checks if the no more squares
    # are available to play on
    def no_empty_fields_left(self):
        fields = []
        for field in self.state:
            if self.state[field] == 'a':
                fields.append(self.state[field])
        if 'a' in fields:
            return False
        else:
            return True
        
    # Checks if the Q dictionary contains the state
    # and adds it in with default action values if
    # the state is not present
    @staticmethod
    def check_if_state_in_Q(state, Q):
        if str(state) not in Q.keys():
            available_actions = []
            for key in state.keys():
                if state[key] == 'a':
                    available_actions.append(key)
            value = {i:0 for i in available_actions}
            key = str(state)
            Q[key] = value
        return Q
    
    # Displays the board
    def render_board(self):
    	values = self.state.values()
    	print(np.array(list(values)).reshape((3, 3)))

    # Displays the board states
    @staticmethod
    def render_board_states():
    	print('Board positions for reference')
    	values = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    	print(np.array(list(values)).reshape((3, 3)))

def make_epsilon_greedy_policy(Q, epsilon):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        available_actions = []
        for key in observation.keys():
            if observation[key] == 'a':
                available_actions.append(key)
        nA = len(available_actions)
        A = np.ones(nA, dtype=float) * epsilon / nA
        observation = str(observation)
        max_key_value = max(Q[observation], key = Q[observation].get)
        #max_key_value = get_max_key_value(Q[observation])
        # best_action = np.where(np.array(available_actions) == max_key_value)[0].tolist()[0]
        best_action = 0
        for ii, item in enumerate(available_actions):
        	if item == max_key_value:
        		best_action = ii
        A[best_action] += (1.0 - epsilon)
        return available_actions, A
    
    return policy_fn

def main(env, num_episodes = 1, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    with open("Q_sarsa_10M_episodes.json") as f:
    	Q = json.load(f)

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon)
    
    for i_episode in range(num_episodes):
        
        # Reset the environment and pick the first action
        state = env.reset().copy()
        
        # Check whether the state is in the action-value function
        Q = env.check_if_state_in_Q(state, Q)
        
        # Calculate the action from the state
        available_actions, action_probs = policy(state)
        action = np.random.choice(available_actions, p=action_probs)
        
        # One step in the environment
        for t in itertools.count():
            # Take a step
            next_state, reward, done, no_fields_left = env.step(action)
            
            # Check if no more fields are left to play
            if no_fields_left:
                break
            
            # Check whether the state is in the action-value function
            Q = env.check_if_state_in_Q(next_state, Q)
            
            # Pick the next action
            next_available_actions, next_action_probs = policy(next_state)
            next_action = np.random.choice(next_available_actions, p=next_action_probs)
    
            if done:
                break
                
            action = next_action
            state = next_state

if __name__ == '__main__':
	env = TicTacToeEnv()
	main(env)