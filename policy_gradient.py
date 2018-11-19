import gym
import model
import torch
import numpy as np

net = model.pg_model()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.8)
loss_fn = torch.nn.CrossEntropyLoss()
NUM_EPSIODES = 1000

env = gym.make('CartPole-v0')
for i in range(NUM_EPSIODES):
    env.reset()
    not_done = False
    env.reset()
    state_holder = []
    reward_holder = []
    observation = np.array([0, 0, 0, 0])
    while not_done:
        env.render()
        action = net(observation)
        state_holder.append(observation)
        observation, reward, not_done, info = env.step(simulate_action(action))
        reward_holder.append(reward)
    dr = discount_and_normalize(reward_list, gamma=.9)
    loss_fn()


def simulate_action(action_prob):
    action_take = torch.bernoulli(action_prob[0])
    return action_take


def discount_and_normalize(reward_list, gamma=.9):
    discounted_reward_list = np.zeros(len(reward_list))
    cumulative = 0
    for i in reversed(rande(len(reward_list))):
        cumulative = cumulative * gamma + reward_list[i]
        discounted_reward_list[i] = cumulative
    discounted_reward_list = (
        discounted_reward_list-np.mean(discounted_reward_list))/np.std(discounted_reward_list)
    return discounted_reward_list
