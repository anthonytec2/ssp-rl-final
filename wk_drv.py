import gym
import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import os
import argparse
import numpy as np
p1 = np.array([[0.9, 0.1], [0.2, 0.8]])
p2 = np.array([[0.3, 0.7], [0.6, 0.4]])
p3 = np.array([[0.5, 0.5], [0.1, 0.9]])
cost_mat = -1*np.array([[50, 200, 10], [3, 500, 0]])
theta_init = np.array([[0.2, 0.6, 0.2], [0.4, 0.4, 0.2]])
NUM_BATCHES = 10000
BATCH_SIZE = 10000
NUM_STATES = 2
NUM_ACTIONS = 3
state = 0
reward_ls_bef = []
reward_ls_af = []
state1 = None
state2 = None
C = 0
for k in range(NUM_BATCHES):
    for i in range(BATCH_SIZE):
        if i < 500:
            action_chose = np.random.choice([0, 1, 2], p=theta_init[state])
            reward_ls_bef.append(cost_mat[state, action_chose])
            if action_chose == 0:
                state = np.random.choice([0, 1], p=p1[state])
            elif action_chose == 1:
                state = np.random.choice([0, 1], p=p2[state])
            elif action_chose == 2:
                state = np.random.choice([0, 1], p=p3[state])
        elif i == 500:
            action_chose = np.random.choice([0, 1, 2], p=theta_init[state])
            if action_chose == 0:
                state1 = np.random.choice([0, 1], p=p1[state])
            elif action_chose == 1:
                state1 = np.random.choice([0, 1], p=p2[state])
            elif action_chose == 2:
                state1 = np.random.choice([0, 1], p=p3[state])
            state2 = np.random.choice([0, 1], p=[0.5, 0.5])
        else:
            action_chose1 = np.random.choice([0, 1, 2], p=theta_init[state1])
            action_chose2 = np.random.choice([0, 1, 2], p=theta_init[state2])
            reward_ls_af.append(
                (cost_mat[state1, action_chose1], cost_mat[state2, action_chose2]))
            if action_chose1 == 0:
                state1 = np.random.choice([0, 1], p=p1[state1])
            elif action_chose1 == 1:
                state1 = np.random.choice([0, 1], p=p2[state1])
            elif action_chose1 == 2:
                state1 = np.random.choice([0, 1], p=p3[state1])
            if action_chose2 == 0:
                state2 = np.random.choice([0, 1], p=p1[state2])
            elif action_chose2 == 1:
                state2 = np.random.choice([0, 1], p=p2[state2])
            elif action_chose2 == 2:
                state2 = np.random.choice([0, 1], p=p3[state2])
            C += cost_mat[state1, action_chose1] - \
                cost_mat[state2, action_chose2]
    eval_drv = (theta_init*(1-theta_init))*C
    print(eval_drv)
