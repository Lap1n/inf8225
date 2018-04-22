# Le projet est inspiré de l'implémentation pytorch de  A2c de ikostrikov :   https://github.com/ikostrikov/pytorch-a2c-ppo-acktr

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import pandas as pd
import matplotlib.pyplot as plt

from TradingEnv import TradingEnv
from envs import make_env
from gold_data_loader import GoldDataLoader
from model import DirectRLModel
from storage import RolloutStorage
from test import test_model

input_size = 45
n_input_momentum = 5
tick_size = 1
ticks_per_day = int(450 / tick_size)

data_set = GoldDataLoader(input_size=input_size, n_input_momentum=n_input_momentum, tick_size=tick_size)

num_frames = 1
num_steps = 405

num_processes = 10

num_epoch = 50
num_updates = num_epoch * data_set.train_set.shape[0] * data_set.train_set.shape[1] // num_steps // num_processes
print("{} updates will be used to trained the model".format(num_updates))

use_cuda = False
log_interval = 10

# training constants
lr = 0.0001
eps = 1e-8,
alpha = 0.99
entropy_coef = 0.001
max_grad_norm = 0.5
value_loss_coef = 0.001

# compute return constants
use_gae = False
gamma = 0.99
tau = 0.005


batch_size = 1

test_env = TradingEnv(data_set, False)


def main():
    best_score = 0
    os.environ['OMP_NUM_THREADS'] = '1'
    scores = []
    test_scores = []
    envs = [make_env(data_set)
            for i in range(num_processes)]

    if num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    obs_shape = envs.observation_space.shape

    actor_critic = DirectRLModel(input_size=data_set.train_set.shape[2], hidden_size=128,
                                 action_space=envs.action_space, num_layers=1)

    if envs.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs.action_space.shape[0]

    if use_cuda:
        actor_critic.cuda()

    optimizer = optim.Adam(actor_critic.parameters(), lr)

    rollouts = RolloutStorage(num_steps, num_processes, obs_shape, envs.action_space, actor_critic.state_size)
    current_obs = torch.zeros(num_processes, *obs_shape)

    num_trades = []

    def update_current_obs(obs):
        shape_dim0 = envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        current_obs[:, -shape_dim0:] = obs

    obs = envs.reset()
    update_current_obs(obs)

    rollouts.observations[0].copy_(current_obs)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([num_processes, 1])
    final_rewards = torch.zeros([num_processes, 1])

    if use_cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    start = time.time()
    for j in range(num_updates):
        for step in range(num_steps):
            # Sample actions
            value, action, action_log_prob, states = actor_critic.act(
                Variable(rollouts.observations[step], volatile=True),
                Variable(rollouts.states[step], volatile=True),
                Variable(rollouts.masks[step], volatile=True),
            deterministic=False)
            cpu_actions = action.data.squeeze(1).cpu().numpy()

            # Obser reward and next obs
            obs, reward, done, current_num_of_trades = envs.step(cpu_actions)
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            if use_cuda:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks

            if done[0] == True:
                num_trades.append(current_num_of_trades)

            update_current_obs(obs)
            rollouts.insert(current_obs, states.data, action.data, action_log_prob.data, value.data, reward, masks)

        next_value = actor_critic.get_value(Variable(rollouts.observations[-1], volatile=True),
                                            Variable(rollouts.states[-1], volatile=True),
                                            Variable(rollouts.masks[-1], volatile=True)).data

        rollouts.compute_returns(next_value, use_gae, gamma, tau)

        values, action_log_probs, dist_entropy, states = actor_critic.evaluate_actions(
            Variable(rollouts.observations[:-1].view(-1, *obs_shape)),
            Variable(rollouts.states[0].view(-1, actor_critic.state_size)),
            Variable(rollouts.masks[:-1].view(-1, 1)),
            Variable(rollouts.actions.view(-1, action_shape)))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = Variable(rollouts.returns[:-1]) - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(Variable(advantages.data) * action_log_probs).mean()

        optimizer.zero_grad()
        total_loss = (value_loss * value_loss_coef + action_loss - dist_entropy * entropy_coef)
        total_loss.backward()
        nn.utils.clip_grad_norm(actor_critic.parameters(), max_grad_norm)
        optimizer.step()

        rollouts.after_update()

        if j % log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * num_processes * num_steps
            print(
                "Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f} num of trades:L{}".
                    format(j, total_num_steps,
                           int(total_num_steps / (end - start)),
                           final_rewards.mean(),
                           final_rewards.median(),
                           final_rewards.min(),
                           final_rewards.max(), dist_entropy.data[0],
                           value_loss.data[0], action_loss.data[0], np.mean(num_trades))

            )
            test_score = test_model(actor_critic)
            test_scores.append(test_score)

        scores.append(final_rewards.mean())

    pd.DataFrame(scores).plot()
    plt.xlabel('Episode')
    plt.ylabel('Average Score ($)')
    plt.legend()
    plt.title(
        'Average score after each training episode\n nbre step={} lr={} entropy coef={} value loss coef={} '.format(
            num_steps, lr, entropy_coef,
            value_loss_coef))
    plt.show()

    pd.DataFrame(test_scores).plot()
    plt.xlabel('Episode')
    plt.ylabel('Average Test Score ($)')
    plt.legend()
    plt.title(
        'Average score on test data after each training episode \n nbre step={} lr={} entropy coef={} value loss coef={} '.format(
            num_steps, lr, entropy_coef,
            value_loss_coef))
    plt.show()


if __name__ == "__main__":
    main()
