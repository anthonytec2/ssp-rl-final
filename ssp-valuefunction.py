import gym
import model
import model_value
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import os
import argparse

# Calculates discounted future rewards and normalizes it


def discount_and_normalize(reward_list, gamma=.99):
    discounted_reward_list = torch.zeros(len(reward_list))
    cumulative = 0
    for i in reversed(range(len(reward_list))):
        cumulative = cumulative * gamma + reward_list[i]
        discounted_reward_list[i] = cumulative
    return discounted_reward_list


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=.001, metavar='N',
                    help='Learning Rate for optimizer')
parser.add_argument('--gamma', type=float, default=.99, metavar='N',
                    help='Learning Rate for optimizer')
parser.add_argument('--episodes', type=int, default=3000, metavar='N',
                    help='Number of Episodes to roll out over')
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = model.pg_model().to(device)
value_fn = model_value.value_model().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
optimizer_val = torch.optim.Adam(value_fn.parameters(), lr=args.lr)
loss_val = nn.MSELoss()
NUM_EPISODES = args.episodes
SAVE_TIME = 100
env = gym.make('CartPole-v1')
wandb.init()
wandb.config.update(args)
wandb.watch((value_fn, net))

weight_updates = 0
for i in range(NUM_EPISODES):
    env.reset()
    done = False
    state_holder = []  # Holds all the states for a single iteration
    reward_holder = []  # Holds all the rewards for an episodes
    action_holder = []  # Hold all the action for an episode
    observation = torch.tensor([0, 0, 0, 0]).float()
    epsiode_length = 0
    with torch.no_grad():
        while not done:
            # env.render(mode='rgb_array')
            # Run Observation through network, get probs
            action = net(observation.unsqueeze(0).to(device))
            # Sample probs to find what action to take
            state_holder.append(observation.data.numpy().tolist())
            act = torch.distributions.categorical.Categorical(action).sample()
            action_holder.append(act.item())
            # Perform a step with given action and continue trajectory
            observation, reward, done, info = env.step(action_holder[-1])
            observation = torch.tensor(observation).float()
            reward_holder.append(reward)
            epsiode_length += 1

    # Save the model every SAVE_TIME Iter
    if weight_updates % SAVE_TIME:
        torch.save(net.state_dict(), os.path.join(wandb.run.dir, "model.pt"))

    state_batch_run = torch.tensor(state_holder).to(device)
    action_batch = torch.tensor(action_holder).to(device)

    # Discount and Normalize the reward returns
    dr = discount_and_normalize(
        reward_holder, gamma=args.gamma).to(device)
    # Run previous states through model, pick action, take log prob, mutiply by reward
    baseline = value_fn(state_batch_run)
    loss_a = loss_val(baseline.squeeze(1), dr)
    optimizer_val.zero_grad()
    loss_a.backward()
    optimizer_val.step()
    dr = (dr-baseline)  # discounted_reward_list)

    probs = net(state_batch_run)
    res = torch.distributions.categorical.Categorical(
        probs).log_prob(action_batch)

    # Perform gradient update step
    final_loss = torch.sum(-res*dr.detach())
    optimizer.zero_grad()
    final_loss.backward()
    optimizer.step()

    wandb.log({'policy/cumm_reward': sum(reward_holder),
               'policy/eps_len': epsiode_length,
               'valuefn/loss': loss_a})
    weight_updates += 1
env.close()
