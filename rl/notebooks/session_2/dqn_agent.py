"""
DQN-Learning implementation with keras.
"""

import random
from collections import deque
from typing import Tuple

import numpy as np
from tensorflow.keras.layers import Dense, multiply, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from lib import AbstractAgent

import random

class DQN(AbstractAgent):

    def __init__(self, action_size: int, state_size: int,
                 gamma: float = None, epsilon: float = None, epsilon_decay: float = None, epsilon_min: float = None,
                 alpha: float = None, batch_size=None, memory_size=None, start_replay_step=None,
                 target_model_update_interval=None):
        self.action_size = action_size
        self.state_size = state_size

        # randomly remembered states and rewards (used because of efficiency and correlation)
        self.memory = deque(maxlen=memory_size)

        # discount factor (how much discount future reward)
        self.gamma = gamma

        # initial exploration rate of the agent (exploitation vs. exploration)
        self.epsilon = epsilon

        # decay epsilon over time to shift from exploration to exploitation
        self.epsilon_decay = epsilon_decay

        # minimal epsilon: x% of the time take random action
        self.epsilon_min = epsilon_min

        # step size also called learning rate alpha
        self.alpha = alpha

        # can be any multiple of 32 (smaller mini-batch size usually leads to higher accuracy/ NN performs better)
        self.batch_size = batch_size

        # number of steps played
        self.step = 0

        # after how many played steps the experience replay should start
        self.start_replay_step = start_replay_step

        # after how many steps should the weights of the target model be updated
        self.target_model_update_interval = target_model_update_interval

        assert self.start_replay_step >= self.batch_size, \
            "The number of steps to start replay must be at least as large as the batch size."

        self.action_mask = np.ones((1, self.action_size))
        self.action_mask_batch = np.ones((self.batch_size, self.action_size))

        self.model = self._build_model()
        self.target_model = self._build_model()

    def _build_model(self):
        # With the functional API we need to define the inputs. Sequential API no longer works because of merge mask
        states_input = Input((self.state_size,), name='states')
        action_mask = Input((self.action_size,), name='action_mask')

        hidden_1 = Dense(units=32, activation='relu')(states_input)
        hidden_2 = Dense(units=32, activation='relu')(hidden_1)
        # "The output layer is a fully-connected linear layer with a single output for each valid action."
        output = Dense(self.action_size, activation='linear')(hidden_2)

        # Finally, we multiply the output by the mask!
        # "The main drawback of [passing the action as an input] is that a separate forward pass is required
        # to compute the Q-value of each action, resulting in a cost that scales linearly with the number of
        # actions. We instead use an architecture in which there is a separate output unit for each possible
        # action, and only the state representation is an input to the neural network.
        # The outputs correspond to the predicted Q-values of the individual action for the input state.
        # The main advantage of this type of architecture is the ability to compute Q-values for
        # all possible actions in a given state with only a single forward pass through the network.
        filtered_output = multiply([output, action_mask])

        model = Model(inputs=[states_input, action_mask], outputs=filtered_output)
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha), metrics=None)

        return model

    def _remember(self, experience: Tuple[np.ndarray, int, np.ndarray, float, bool]) -> None:
        # Todo: Store experience in memory
        self.memory.append(experience)

    def _replay(self) -> None:
        # Todo: Get a random mini batch from memory and create numpy arrays for each part of this experience.
        rnd_mem_idx = random.sample(range(0, len(self.memory)), self.batch_size)
        
        states, actions, next_states, rewards, dones = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
        for i in rnd_mem_idx:
            states = np.append(states, self.memory[i][0])
            actions = np.append(actions, self.memory[i][1])
            next_states = np.append(next_states, self.memory[i][2])
            rewards = np.append(rewards, self.memory[i][3])
            dones = np.append(dones, self.memory[i][4])

        # The following assert statements are intended to support further implementation,
        # but can also be removed/adjusted if necessary.
        assert all(isinstance(x, np.ndarray) for x in (states, actions, rewards, next_states, dones)), \
            "All experience batches should be of type np.ndarray."
        assert states.shape == (self.batch_size, self.state_size), \
            f"States shape should be: {(self.batch_size, self.state_size)}"
        assert actions.shape == (self.batch_size,), f"Actions shape should be: {(self.batch_size,)}"
        assert rewards.shape == (self.batch_size,), f"Rewards shape should be: {(self.batch_size,)}"
        assert next_states.shape == (self.batch_size, self.state_size), \
            f"Next states shape should be: {(self.batch_size, self.state_size)}"
        assert dones.shape == (self.batch_size,), f"Dones shape should be: {(self.batch_size,)}"

        # Todo: Predict the Q values of the next states. Passing ones as the action mask.
        next_q_values = self.model.predict([states, np.ones(self.action_size)]) 

       
        
        # Todo: Calculate the Q values, remember
        #  the Q values of each non-terminal state is the reward + gamma * the max next state Q value
        # Depending on the implementation, the axis must be specified to get the max q-value for EACH batch element!
        q_values = rewards + gamma * [max(next_q_val) for next_q_val in next_q_values]
         
        # Todo: Set the Q values of terminal states to 0 (by definition)
        done_idx = np.where(dones == 1)
        for idx in done_idx:
            q_values[idx] = 0

        # Todo: Create a one hot encoding of the actions (the selected action is 1 all others 0)
        one_hot_actions = to_categorical(actions, num_classes=self.action_size)
        

        # Todo: Create the target Q values based on the one hot encoding of the actions and the calculated q-values
        target_q_values = None

        # Todo: fit the model with the right x and y values
        self.model.fit(
            x=None,  # states and mask
            y=None,  # target Q values
            batch_size=self.batch_size,
            verbose=0
        )

    def act(self, state: np.ndarray) -> int:
        """Selects the action to be executed based on the given state.

        Implements epsilon greedy exploration strategy, i.e. with a probability of
        epsilon, a random action is selected.

        Args:
            state: Numpy array with shape (1,4) representing the state based on 4 float values:
                - cart position,
                - cart velocity,
                - angle,
                - angle velocity

        Returns:
            Action.
        """
        if random.random() < self.epsilon:
            # Todo: return random valid action
            action = 0
        else:
            # Todo: Use the model to get the Q values for the state and determine the action based on the max Q value.
            action = 1
        return action

    def train(self, experience: Tuple[np.ndarray, int, np.ndarray, float, bool]) -> None:
        """Stores the experience in memory. If memory is full trains network by replay.

        Args:
            experience: Tuple of state, action, next state, reward, done.

        Returns:
            None
        """
        self._remember(experience)

        # Todo: As soon as enough steps are played:
        #  - Update epsilon as long as it is not minimal
        #  - update weights of the target model (syn of the two models)
        #  - execute replay

        self.step += 1
