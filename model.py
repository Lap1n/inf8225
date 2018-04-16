# Le projet est inspiré de l'implémentation pytorch de  A2c de ikostrikov :   https://github.com/ikostrikov/pytorch-a2c-ppo-acktr

import torch
from torch import nn
import numpy as np

from projet.a2cGold.distributions import get_distribution
from projet.a2cGold.utils import orthogonal


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        """
        All classes that inheret from Policy are expected to have
        a feature exctractor for actor and critic (see examples below)
        and modules called linear_critic and dist. Where linear_critic
        takes critic features and maps them to value and dist
        represents a distribution of actions.        
        """

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks, deterministic=False):
        hidden_critic, hidden_actor, states = self(inputs, states, masks)

        action = self.dist.sample(hidden_actor, deterministic=deterministic)

        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(hidden_actor, action)
        value = self.critic_linear(hidden_critic)

        return value, action, action_log_probs, states

    def get_value(self, inputs, states, masks):
        hidden_critic, _, states = self(inputs, states, masks)
        value = self.critic_linear(hidden_critic)
        return value

    def evaluate_actions(self, inputs, states, masks, actions):
        hidden_critic, hidden_actor, states = self(inputs, states, masks)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(hidden_actor, actions)
        value = self.critic_linear(hidden_critic)
        return value, action_log_probs, dist_entropy, states


class DirectRLModel(Policy):
    def __init__(self, input_size, hidden_size, action_space, num_layers=1):
        super(DirectRLModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.action_space = action_space

        self.seq_no_dropout = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 20),
            nn.ReLU(),
        )
        self.apply(self.weights_init)

        # Initialisation de la partie rnn du réseau
        self.rnn = nn.GRUCell(20, hidden_size)

        # Initialisation de la critique v(s) dans A2c
        self.critic_linear = nn.Linear(hidden_size, 1)

        # # Initialisation de l'acteur qui décidera des actions entre Short (0), Neutral (1) ou Buy (2) dans A2c
        self.dist = get_distribution(hidden_size, self.action_space)

    @property
    def state_size(self):
        return self.hidden_size

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            weight_shape = list(m.weight.data.size())
            fan_in = np.prod(weight_shape[1:4])
            fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
            w_bound = np.sqrt(6. / (fan_in + fan_out))
            m.weight.data.uniform_(-w_bound, w_bound)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            weight_shape = list(m.weight.data.size())
            fan_in = weight_shape[1]
            fan_out = weight_shape[0]
            w_bound = np.sqrt(6. / (fan_in + fan_out))
            m.weight.data.uniform_(-w_bound, w_bound)
            m.bias.data.fill_(0)

    def reset_parameters(self):
        orthogonal(self.gru.weight_ih.data)
        orthogonal(self.gru.weight_hh.data)
        self.gru.bias_ih.data.fill_(0)
        self.gru.bias_hh.data.fill_(0)
        self.critic_linear.bias.data.fill_(0)

    def forward(self, inputs, states, masks):
        x = self.seq_no_dropout(inputs)
        if inputs.size(0) == states.size(0):
            x = states = self.rnn(x, states * masks)
        else:
            x = x.view(-1, states.size(0), x.size(1))
            masks = masks.view(-1, states.size(0), 1)
            outputs = []
            for i in range(x.size(0)):
                hx = states = self.rnn(x[i], states * masks[i])
                outputs.append(hx)
            x = torch.cat(outputs, 0)
        return x, x, states
