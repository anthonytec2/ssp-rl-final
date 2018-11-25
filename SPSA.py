import gym
import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorboard
import os
import argparse
from noisyopt import minimizeSPSA
import wandb


def discount_and_normalize(reward_list, gamma=.99):
    discounted_reward_list = torch.zeros(len(reward_list))
    cumulative = 0
    for i in reversed(range(len(reward_list))):
        cumulative = cumulative * gamma + reward_list[i]
        discounted_reward_list[i] = cumulative
    # discounted_reward_list = (
    #    discounted_reward_list-torch.mean(discounted_reward_list))/torch.std###(discounted_reward_list)
    return discounted_reward_list


def run_network(w1, w2, w3, x):
    h1 = x.mm(w1)
    h_relu1 = h1.clamp(min=0)
    h2 = h_relu1.mm(w2)
    h_relu2 = h2.clamp(min=0)
    h3 = h_relu2.mm(w3)
    output = F.softmax(h3)
    return output


def run_batch(w):
    env.reset()
    w1 = torch.reshape(torch.tensor(w[0:40]).float(), (4, 10))
    w2 = torch.reshape(torch.tensor(w[40:60]).float(), (10, 2))
    w3 = torch.reshape(torch.tensor(w[60:]).float(), (2, 2))
    done = False
    state_holder = []  # Holds all the states for a single iteration
    reward_holder = []  # Holds all the rewards for an episodes
    action_holder = []  # Hold all the action for an episode
    observation = torch.tensor([0, 0, 0, 0]).float()
    rounds = 0
    while not done:
        # env.render(mode='rgb_array')
        # Run Observation through network, get probs
        action = run_network(w1, w2, w3, observation.unsqueeze(0))
        # Sample probs to find what action to take
        state_holder.append(observation)
        act = torch.distributions.categorical.Categorical(action).sample()
        action_holder.append(act.item())
        # Perform a step with given action and continue trajectory
        observation, reward, done, info = env.step(action_holder[-1])
        observation = torch.tensor(observation).float()
        reward_holder.append(reward)
        rounds += 1
     # Discount and Normalize the reward returns
    dr = discount_and_normalize(reward_holder, gamma=.99)

    return -float(dr[0]), rounds, sum(reward_holder)


wandb.init()
env = gym.make('CartPole-v1')
w = torch.randn(64, 1).numpy()
niter = 1000
A = 0.01 * niter
c = 1.0
a = 1.0
gamma = 0.101
alpha = 0.602
hisory = []
for i in range(niter):
    ak = a/(i+1.0+A)**alpha
    ck = c/(i+1.0)**gamma
    theta = torch.randint(low=0, high=2, size=(64, 1))
    theta[theta == 0] = -1
    l1, r1, rr1 = run_batch(w+ck*theta)
    l2, r2, rr2 = run_batch(w-ck*theta)
    g = (l1-l2)/(2*ck*theta)
    wandb.log({'cumm_reward': (rr1+rr2)/2, 'eps_len': (r1+r2)/2})
    w -= ak*g
print(run_batch(w))
