import torch
import numpy as np
from torch.autograd import Variable

from projet.a2cGold.TradingEnv import TradingEnv
from projet.a2cGold.gold_data_loader import GoldDataLoader
from projet.a2cGold.model import DirectRLModel

input_size = 45
tick_size = 1
n_input_momentum = 5
batch_size = 1
num_processes = 1
data_set = GoldDataLoader(input_size=input_size, n_input_momentum=n_input_momentum, tick_size=tick_size)

test_env = TradingEnv(data_set, False)

best_score = 0
ticks_per_day = int(450 / tick_size)


use_cuda = False


def test_model(actor_critic):
    scores = []
    nb_of_trades = []
    n_batch = data_set.train_set.shape[0] / batch_size

    for a_random_day in range(int(n_batch)):
        score = 0
        day_trades = 0
        last_action = 1
        actions = []
        episode_rewards = 0

        num_step = ticks_per_day - input_size
        obs_shape = test_env.observation_space.shape
        current_obs = torch.zeros(1, *obs_shape)

        def update_current_obs(obs):
            shape_dim0 = test_env.observation_space.shape[0]
            obs = torch.from_numpy(obs).float()
            current_obs[:, -shape_dim0:] = obs

        obs = test_env.reset()
        update_current_obs(obs)
        states = Variable(torch.zeros(1, actor_critic.state_size))
        mask = Variable(torch.FloatTensor([1.0]))
        final_rewards = torch.zeros([num_processes, 1])
        day_rewards = []

        for test_time_step in range(num_step):
            value, action, action_log_prob, states = actor_critic.act(
                Variable(current_obs, volatile=True),
                Variable(states.data, volatile=True),
                Variable(mask.data, volatile=True))

            if test_time_step == ticks_per_day - input_size - 1 - 1:
                cpu_actions = (torch.FloatTensor(np.ones(1))).cpu().numpy()
            else:
                cpu_actions = action.data.squeeze(1).cpu().numpy()

            # Obser reward and next obs
            obs, reward, done, info = test_env.step(cpu_actions)
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            if cpu_actions != last_action and last_action != 1:
                day_trades += 1

            # If done then clean the history of observations.
            if done is True:
                masks = torch.FloatTensor([0.0])
                break
            else:
                masks = torch.FloatTensor([1.0])

            last_action = cpu_actions

            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            if use_cuda:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks

            update_current_obs(obs)
        scores.append(episode_rewards)
        nb_of_trades.append(day_trades)

    total_reward = np.sum(scores)
    mean_scores = np.mean(scores)
    mean_trades = np.mean(nb_of_trades)
    scores_variance = np.var(scores)
    print(
        "mean score:{} variance: {} nbOfTrades:{} total_reward:{}".format(mean_scores,
                                                                          scores_variance,
                                                                          mean_trades, total_reward))
    max = np.max(scores)
    min = np.min(scores)
    print("highest score: {} lowest score:{}".format(max, min))
    return mean_scores