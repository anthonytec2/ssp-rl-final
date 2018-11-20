import gym
import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import os
import argparse

def discount_and_normalize(reward_list, gamma=.99):
    discounted_reward_list = torch.zeros(len(reward_list))
    cumulative = 0
    for i in reversed(range(len(reward_list))):
        cumulative = cumulative * gamma + reward_list[i]
        discounted_reward_list[i] = cumulative
    discounted_reward_list = (
        discounted_reward_list-torch.mean(discounted_reward_list))/torch.std(discounted_reward_list)
    return discounted_reward_list

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=.001, metavar='N',
                    help='Learning Rate for optimizer')
parser.add_argument('--gamma', type=float, default=.99, metavar='N',
                    help='Learning Rate for optimizer')
parser.add_argument('--epsiodes', type=int, default=3000, metavar='N',
                    help='Number of Epsiodes to roll out over')
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = model.pg_model().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
NUM_EPSIODES = args.epsiodes
SAVE_TIME = 10
env = gym.make('CartPole-v1')
wandb.init()
wandb.config.update(args)
wandb.hook_torch(net)
weight_updates = 0
for i in range(NUM_EPSIODES):
    env.reset()
    done = False
    state_holder = []
    reward_holder = []
    action_holder = []
    observation = torch.tensor([0, 0, 0, 0]).float()
    epsiode_length=0
    while not done:
        # env.render(mode='rgb_array')
        action = net(observation.unsqueeze(0).to(device))
        state_holder.append(observation)
        act = torch.distributions.categorical.Categorical(action).sample()
        action_holder.append(act.item())
        observation, reward, done, info = env.step(action_holder[-1])
        observation = torch.tensor(observation).float()
        reward_holder.append(reward)
        epsiode_length+=1
    if weight_updates % 100:
        torch.save(net.state_dict(), os.path.join(wandb.run.dir, "model.pt"))
    wandb.log({'cumm_reward': sum(reward_holder), 'eps_len': epsiode_length})
    dr = discount_and_normalize(reward_holder, gamma=args.gamma).to(device)
    state_batch_run = torch.stack(state_holder).to(device)
    action_batch = torch.tensor(action_holder).to(device)
    y_pred = net(state_batch_run)
    res = torch.distributions.categorical.Categorical(
        y_pred).log_prob(action_batch)
    optimizer.zero_grad()
    final_loss = torch.sum(-res*dr)
    final_loss.backward()
    optimizer.step()
    weight_updates += 1