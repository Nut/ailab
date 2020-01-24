"""
Atari DQN-Learning implementation with keras.
"""

import random
from collections import deque
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.backend import set_session
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Lambda, multiply, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.losses import huber_loss

from lib import AbstractAgent
from lib.atari_helpers import LazyFrames


class AtariDQN(AbstractAgent):

    def __init__(self, action_size: int, state_size: int,
                 gamma: float = None, epsilon: float = None, epsilon_decay: float = None, epsilon_min: float = None,
                 alpha: float = None, batch_size=None, memory_size=None, start_replay_step=None,
                 target_model_update_interval=None, train_freq=None):
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

        # at which frequency (interval) the model should be trained (steps)
        self.train_freq = train_freq

        assert self.start_replay_step >= self.batch_size, \
            "The number of steps to start replay must be at least as large as the batch size."

        self.action_mask = np.ones((1, self.action_size))
        self.action_mask_batch = np.ones((self.batch_size, self.action_size))

        config = tf.ConfigProto(intra_op_parallelism_threads=8,
                                inter_op_parallelism_threads=4,
                                allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        set_session(session)  # set this TensorFlow session as the default session for Keras

        self.model = self._build_model()
        self.target_model = self._build_model()

    def _build_model(self):
        atari_shape = (84, 84, 4)
        # With the functional API we need to define the inputs. Sequential API no longer works because of merge mask
        frames_input = Input(atari_shape, name='frames')
        action_mask = Input((self.action_size,), name='action_mask')

        # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
        normalized = Lambda(lambda x: x / 255.0, name='normalization')(frames_input)

        conv1 = Conv2D(filters=32,
                       kernel_size=(8, 8),
                       strides=(4, 4),
                       activation='relu',
                       kernel_initializer=tf.variance_scaling_initializer(scale=2)
                       )(normalized)

        conv2 = Conv2D(filters=64,
                       kernel_size=(4, 4),
                       strides=(2, 2),
                       activation='relu',
                       kernel_initializer=tf.variance_scaling_initializer(scale=2)
                       )(conv1)

        conv3 = Conv2D(filters=64,
                       kernel_size=(3, 3),
                       strides=(1, 1),
                       activation='relu',
                       kernel_initializer=tf.variance_scaling_initializer(scale=2)
                       )(conv2)

        # Flattening the last convolutional layer.
        conv_flattened = Flatten()(conv3)

        # "The final hidden layer is fully-connected and consists of 256 rectifier units."
        hidden = Dense(units=256, activation='relu')(conv_flattened)

        # "The output layer is a fully-connected linear layer with a single output for each valid action."
        output = Dense(self.action_size)(hidden)

        # Finally, we multiply the output by the mask!
        # "The main drawback of [passing the action as an input] is that a separate forward pass is required
        # to compute the Q-value of each action, resulting in a cost that scales linearly with the number of
        # actions. We instead use an architecture in which there is a separate output unit for each possible
        # action, and only the state representation is an input to the neural network.
        # The outputs correspond to the predicted Q-values of the individual action for the input state.
        # The main advantage of this type of architecture is the ability to compute Q-values for
        # all possible actions in a given state with only a single forward pass through the network.
        filtered_output = multiply([output, action_mask])

        model = Model(inputs=[frames_input, action_mask], outputs=filtered_output)
        model.compile(loss=huber_loss, optimizer=Adam(lr=self.alpha, clipnorm=10), metrics=None)

        return model

    def _remember(self, experience: Tuple[LazyFrames, int, LazyFrames, float, bool]) -> None:
        self.memory.append(experience)

    def _replay(self) -> None:
        mini_batch = random.sample(self.memory, self.batch_size)
        states, actions, next_states, rewards, dones = zip(*mini_batch)

        # Todo: Convert the parts of the mini-batch into corresponding numpy arrays.
        #  Note that the states are of type 'LazyFrames' due to memory efficiency
        #  and must therefore be converted individually.
        states = None
        next_states = None
        actions = None
        rewards = None
        dones = None

        # The following assert statements are intended to support further implementation,
        #  but can also be removed/adjusted if necessary.
        assert all(isinstance(x, np.ndarray) for x in (states, actions, rewards, next_states, dones)), \
            "All experience batches should be of type np.ndarray."
        assert states.shape == (self.batch_size, 84, 84, 4), \
            f"States shape should be: {(self.batch_size, 84, 84, 4)}"
        assert actions.shape == (self.batch_size,), f"Actions shape should be: {(self.batch_size,)}"
        assert rewards.shape == (self.batch_size,), f"Rewards shape should be: {(self.batch_size,)}"
        assert next_states.shape == (self.batch_size, 84, 84, 4), \
            f"Next states shape should be: {(self.batch_size, 84, 84, 4)}"
        assert dones.shape == (self.batch_size,), f"Dones shape should be: {(self.batch_size,)}"

        # Todo: Predict the Q values of the next states (choose the right model!). Passing ones as the action mask
        #  Note that a suitable mask has already been created in '__init__'.
        next_q_values = None

        # Todo: Calculate the Q values, remember
        #  - the Q values of each non-terminal state is the reward + gamma * the max next state Q value
        #  - and the Q values of terminal states should be the reward (Hint: 1.0 - dones) makes sure that if the game is
        #    over, targetQ = rewards
        # Depending on the implementation, the axis must be specified to get the max q-value for EACH batch element!
        q_values = None

        # Todo: Create a one hot encoding of the actions (the selected action is 1 all others 0)
        #  Hint look at the imports. A Keras help function will be imported there.
        one_hot_actions = None

        # Todo: Create the target Q values based on the one hot encoding of the actions and the calculated q-values
        #  Hint you have to "reshape" the q_values to match the shape
        target_q_values = None

        self.model.fit(
            x=None,  # states and mask
            y=None,  # target Q values
            batch_size=self.batch_size,
            verbose=0
        )

    def act(self, state: LazyFrames) -> int:
        """Selects the action to be executed based on the given state.

        Implements epsilon greedy exploration strategy, i.e. with a probability of
        epsilon, a random action is selected.

        Args:
            state: LazyFrames object representing the state based on 4 stacked observations (images)

        Returns:
            Action.
        """
        if random.random() < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            # Todo: Use the model to get the Q values for the state and determine the action based on the max Q value.
            #  Hint: You have to convert the state to a list of numpy arrays before you can pass it to the model
            q_values = None
            action = 1
        return action

    def train(self, experience: Tuple[LazyFrames, int, LazyFrames, float, bool]) -> None:
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


def main():
    import time
    from contextlib import suppress
    from datetime import datetime
    from lib.loggers import TensorBoardLogger, tf_summary_image
    from lib.atari_helpers import wrap_deepmind, make_atari

    ############
    # Start of dirty monkey patching
    # ignore the following section
    # monkey patch start method of ImageEncoder class. Please don't do this at home!
    import os
    from gym import logger
    import subprocess
    from gym.wrappers.monitoring.video_recorder import ImageEncoder

    def vp9_mokey_patched_start(self):
        print('vp9 version used')
        self.cmdline = (self.backend,
                        '-nostats',
                        '-loglevel', 'error', # suppress warnings
                        '-y',
                        '-r', '%d' % self.frames_per_sec,

                        # input
                        '-f', 'rawvideo',
                        '-s:v', '{}x{}'.format(*self.wh),
                        '-pix_fmt',('rgb32' if self.includes_alpha else 'rgb24'),
                        '-i', '-', # this used to be /dev/stdin, which is not Windows-friendly

                        # output
                        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                        '-vcodec', 'libvpx-vp9',
                        '-pix_fmt', 'yuv420p',
                        self.output_path[:-3] + 'webm'
                        )

        logger.debug('Starting ffmpeg with "%s"', ' '.join(self.cmdline))
        if hasattr(os,'setsid'): #setsid not present on Windows
            self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE, preexec_fn=os.setsid)
        else:
            self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE)

    ImageEncoder.start = vp9_mokey_patched_start
    # End of dirty monkey patching
    ############

    from gym.wrappers import Monitor

    env = make_atari('PongNoFrameskip-v4')
    env = wrap_deepmind(env, frame_stack=True)

    # train for number of episodes or number of steps (what happens first)
    n_episodes = 600
    max_steps = 1000000
    verbose = True

    tb_logger = TensorBoardLogger(f'./logs/run-{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}')

    action_size = env.action_space.n
    state_size = env.observation_space.shape[0]

    # Hyperparams
    annealing_steps = 100000  # not episodes!
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = (epsilon - epsilon_min) / annealing_steps
    alpha = 0.0001
    batch_size = 64
    memory_size = 10000
    start_replay_step = 10000
    target_model_update_interval = 1000
    train_freq = 4

    agent = AtariDQN(action_size=action_size, state_size=state_size, gamma=gamma,
                     epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
                     alpha=alpha, batch_size=batch_size, memory_size=memory_size,
                     start_replay_step=start_replay_step,
                     target_model_update_interval=target_model_update_interval, train_freq=train_freq)

    total_step = 0
    with suppress(KeyboardInterrupt):
        for episode in range(n_episodes):
            done = False
            episode_reward = 0
            state = env.reset()
            episode_start_time = time.time()

            episode_step = 0
            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)

                agent.train((state, action, next_state, reward, done))

                if episode == 0:
                    # for debug purpose log every state of first episode
                    for obs in state:
                        tb_logger.log_image(f'state_t{episode_step}:', tf_summary_image(np.array(obs, copy=False)),
                                            global_step=total_step)

                state = next_state
                episode_reward += reward

                episode_step += 1

            total_step += episode_step

            if episode % 10 == 0:
                speed = episode_step / (time.time() - episode_start_time)
                tb_logger.log_scalar('score', episode_reward, global_step=total_step)
                tb_logger.log_scalar('epsilon', agent.epsilon, global_step=total_step)
                tb_logger.log_scalar('speed', speed, global_step=total_step)
                if verbose:
                    print(f'episode: {episode}/{n_episodes}, score: {episode_reward}, steps: {episode_step}, '
                          f'total steps: {total_step}, e: {agent.epsilon:.3f}, speed: {speed:.2f} steps/s')

            if total_step >= max_steps:
                break
    env.close()

    print('start evaluation...')

    # capture every episode and clean 'video' folder before each run
    env = make_atari('PongNoFrameskip-v4')
    env = Monitor(env, './video', video_callable=lambda episode_id: True, force=True, uid='id')
    env = wrap_deepmind(env, frame_stack=True)

    best_episode = 0
    best_episode_score = -100
    for episode in range(10):
        done = False
        state = env.reset()
        episode_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward

        print(f'episode: {episode}/{10}, score: {episode_reward}')

        if best_episode_score < episode_reward:
            best_episode = episode
            best_episode_score = episode_reward

    print(f'best episode: {best_episode} score: {best_episode_score}')
    env.close()

    # write html file with best episode embedded
    from pathlib import Path
    player_file_path = Path().resolve() / 'play_best_episode.html'
    player_file_path.write_text(f"""
    <html>
        <div align="middle">
            <video controls>
                  <source src="./video/openaigym.video.0.id.video00000{best_episode}.webm" type="video/webm">
            </video>
        </div>
    </html>
    """)


if __name__ == '__main__':
    main()
