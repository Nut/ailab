"""
Q-Learning implementation.
"""

import math

import numpy as np
from typing import Tuple

from lib import AbstractAgent
            
class AdvancedQLearning(AbstractAgent):

    def __init__(self, action_size: Tuple[int], buckets: Tuple[int, int, int, int],
                 gamma: float = None, epsilon: float = None, epsilon_min: float = None, 
                 alpha: float = None, alpha_min: float = None):
        self.action_size = action_size

        self.gamma = gamma  # discount factor (how much discount future reward)
        self.epsilon = epsilon  # exploration rate for the agent
        self.alpha = alpha  # learning rate
        
        self.epsilon_min = epsilon_min
        self.alpha_min = alpha_min
        
        # Initialize Q[s,a] table
        # TODO
        self.Q = np.zeros(buckets + (action_size,))
        self.epochs = 0 # played episodes
        self.epsilon_decay = 0.95
        self.alpha_decay = 0.95

    def act(self, state: Tuple[int, int, int, int]) -> int:
        """Selects the action to be executed based on the given state.
        
        Implements epsilon greedy exploration strategy, i.e. with a probability of
        epsilon, a random action is selected.
        
        Args:
            state: Tuple of agent and target position, representing the state.
        
        Returns:
            Action.
        """
        # TODO
        exploration_factor = np.random.random_sample()
        if exploration_factor < self.epsilon:
            action = np.random.randint(0,self.action_size)
        else:
            action = np.argmax(self.Q[state])
            
        return action


    def train(self, experience: Tuple[Tuple[int, int, int, int], int, Tuple[int, int, int, int], float, bool]) -> None:
        """Learns the Q-values based on experience.
                

        Args:
            experience: Tuple of state, action, next state, reward, done.
        
        Returns:
            None
        """
        state = experience[0]
        action = (experience[1],)
        next_state = experience[2]
        reward = experience[3]
        done = experience[4]
        a = (np.argmax(self.Q[next_state]),)
        
        
        
        if done:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)  # exploration rate for the agent
            self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)  # learning rate
        
        self.Q[state + action] = self.Q[state + action] + self.alpha * ( reward + self.gamma * self.Q[next_state + a] -  self.Q[state + action])