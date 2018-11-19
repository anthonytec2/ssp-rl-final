import gym
import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import os


def simulate_action(action_prob):
    # print(action_prob)
    action_take = torch.bernoulli(action_prob[0][0])
    return action_take


def discount_and_normalize(reward_list, gamma=.99):
    discounted_reward_list = torch.zeros(len(reward_list))
    cumulative = 0
    for i in reversed(range(len(reward_list))):
        cumulative = cumulative * gamma + reward_list[i]
        discounted_reward_list[i] = cumulative
    discounted_reward_list = (
        discounted_reward_list-torch.mean(discounted_reward_list))/torch.std(discounted_reward_list)
    return discounted_reward_list


net = model.pg_model()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
#loss_fn = torch.nn.CrossEntropyLoss(reduce=False)
NUM_EPSIODES = 10000
SAVE_TIME = 10
#writer = SummaryWriter()
env = gym.make('CartPole-v1')
wandb.init()
wandb.hook_torch(net)
for i in range(NUM_EPSIODES):
    env.reset()
    done = False
    state_holder = []
    reward_holder = []
    action_holder = []
    observation = torch.tensor([0, 0, 0, 0]).float()
    w = 0
    k = 0
    while not done:
        env.render(mode='rgb_array')
        # print(observation)
        action = net(observation.unsqueeze(0))
        state_holder.append(observation)
        act = torch.distributions.categorical.Categorical(action).sample()
        # torch.tensor([1, 0] if act else [0, 1])
        action_holder.append(act.item())
        observation, reward, done, info = env.step(action_holder[-1])
        observation = torch.tensor(observation).float()
        reward_holder.append(reward)
        k += 1
    if w % 100:
        torch.save(net, os.path.join(wandb.run.dir, "model.pt"))
    wandb.log({'cumm_reward': sum(reward_holder), 'eps_len': k})
    dr = discount_and_normalize(reward_holder, gamma=.9)
    state_batch_run = torch.stack(state_holder)
    action_batch = torch.tensor(action_holder)
    y_pred = net(state_batch_run)
    res = torch.distributions.categorical.Categorical(
        y_pred).log_prob(action_batch)
    optimizer.zero_grad()
    final_loss = torch.sum(-res*dr)
    final_loss.backward()
    optimizer.step()
    w += 1
