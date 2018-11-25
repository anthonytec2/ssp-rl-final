import gym
import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorboard
import os
import argparse
from noisyopt import minimizeSPSA


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

    return -float(dr[0])  # , rounds, sum(reward_holder)


w = torch.randn(64).numpy()
lr = 1e-3
env = gym.make('CartPole-v1')
Grad = .1
'''
for i in range(100):
    theta = torch.randint(low=0, high=1, size=(64, 1))
    theta[theta == 0] = -1
    l1, rounds1, rew1 = run_batch(w+Grad*theta, env)
    l2, rounds2, rew2 = run_batch(w-Grad*theta, env)
    print(rounds1, rew1, rounds2, rew2)
    del_j = l1-l2
    g = del_j/(2*Grad)
    w += lr*g_fd
'''
res = minimizeSPSA(run_batch, w, paired=False, niter=1000)
print(res)
w = res.x
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
    env.render()
    observation = torch.tensor(observation).float()
    reward_holder.append(reward)
    rounds += 1
env.close()
