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
        
        self.epsilon_min = 0.1
        self.alpha_min = 0.1
        
        # Initialize Q[s,a] table
        # TODO
        self.Q = None
        self.t = 0 # played episodes

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
        return 0

    def train(self, experience: Tuple[Tuple[int, int, int, int], int, Tuple[int, int, int, int], float, bool]) -> None:
        """Learns the Q-values based on experience.
        
        Args:
            experience: Tuple of state, action, next state, reward, done.
        
        Returns:
            None
        """
        # TODO
        