import random

import gym
import numpy as np
from gym import spaces


class TradingEnv(gym.Env):
    """
    Define a simple Banana environment.
    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self, data_source, random_mode=True):
        self.__version__ = "0.1.0"
        print("TradingEnv - Version {}".format(self.__version__))
        self.data_source = data_source
        self.training_data = data_source.train_set
        self.random_mode = random_mode
        # GYM attributes (important)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-100, 100, [self.training_data.shape[2]])

        self.current_day_data = np.empty(1)
        self.current_day_index = 0
        self.intraday_index = 0
        self.current_reward = 0
        self.previous_action = 0
        self.current_delta_price = 0
        self.num_trades = 0

        self.trading_cost = 2.40

        self.z_min_max = data_source.z_min_max
        self.day_index = 0

    def render(self, mode='human'):
        pass

    def reset(self):
        return self._reset()

    def step(self, action):
        return self._step(action)

    def _step(self, action):
        self.current_delta_price = self.current_day_data[self.intraday_index, -1]
        self._take_action(action)
        reward = self._get_reward()

        self.intraday_index += 1
        ob = self._get_state()
        return ob, reward, self.is_done(), self.num_trades

    def _take_action(self, action):
        self.current_reward = self.compute_reward(action)

    def compute_reward(self, action):
        action = action - 1
        delta_price_tick = self.data_source.denorm(self.current_delta_price).squeeze() * 100
        abs_delta_action = np.ceil(abs(action - self.previous_action) / 2)
        r_t = self.previous_action * delta_price_tick - abs_delta_action * self.trading_cost

        # if isinstance(r_t,np.float64):
        #     if r_t > 0:
        #         r_t = 1
        #     else:
        #         r_t = -1
        # else:
        #     r_t =[1.0 if rew > 0 else -1.0 for rew in r_t]

        if action != self.previous_action and self.previous_action != 1:
            self.num_trades += 1

        self.previous_action = action
        return r_t

    def is_done(self):
        return self.intraday_index == self.current_day_data.shape[0] - 1

    def _get_reward(self):
        return self.current_reward

    def _reset(self):
        self.current_day_data = self.get_next_day()
        self.current_day_index = 0
        self.intraday_index = 0
        self.current_reward = 0
        self.previous_action = 0
        self.current_delta_price = 0
        self.num_trades = 0

        return self._get_state()

    def get_next_day(self):
        if self.random_mode is True:
            number_of_days = self.training_data.shape[0]
            day_index = random.randrange(number_of_days)
        else:
            day_index = self.day_index
            self.day_index = (self.day_index + 1) % self.training_data.shape[0]
        return self.training_data[day_index]

    def _render(self, mode='human', close=False):
        return

    def _get_state(self):
        """Get the observation."""
        ob = self.current_day_data[self.intraday_index]
        return ob

    def _seed(self, seed):
        random.seed(seed)
        np.random.seed()
