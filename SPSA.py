import gym
import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorboard
import os
import argparse


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


def run_batch(w1, w2, w3, env):
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
    # Run previous states through model, pick action, take log prob, mutiply by reward
    state_batch_run = torch.stack(state_holder)
    action_batch = torch.tensor(action_holder)
    probs = run_network(w1, w2, w3, state_batch_run)
    res = torch.distributions.categorical.Categorical(
        probs).log_prob(action_batch)
    final_loss = torch.sum(res*dr)
    return final_loss, rounds, sum(reward_holder)


def perb():
    d1 = torch.randint(low=0, high=1, size=(4, 10))
    d2 = torch.randint(low=0, high=1, size=(10, 2))
    d3 = torch.randint(low=0, high=1, size=(2, 2))
    d1[d1 == 0] = -1
    d2[d2 == 0] = -1
    d3[d3 == 0] = -1
    return d1, d2, d3


w1 = torch.randn(4, 10)
w2 = torch.randn(10, 2)
w3 = torch.randn(2, 2)
lr = 1e-1
Grad = 1
gam = .99
Epsilon = .99
env = gym.make('CartPole-v1')
for i in range(100):
    d1, d2, d3 = perb()
    env.reset()
    l1, rounds1, rew1 = run_batch(w1+Grad*d1, w2+Grad*d2, w3+Grad*d3, env)
    env.reset()
    l2, rounds2, rew2 = run_batch(w1-Grad*d1, w2-Grad*d2, w3-Grad*d3, env)
    print(rounds1, rew1, rounds2, rew2)
    alpha = ((l2-l1)/(2*Grad))
    g1 = alpha*d1
    g2 = alpha*d2
    g3 = alpha*d3
    print(Grad, Epsilon)
    Grad = Grad/(i+1)**gam
    w1 -= lr*g1
    w2 -= lr*g2
    w3 -= lr*g3
    lr = lr/(i+1)**Epsilon
env.close()
