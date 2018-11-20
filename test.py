import torch
import gym
import model
env = gym.make('CartPole-v0')
device = torch.device('cpu')
net = model.pg_model()
net.load_state_dict(torch.load('model.pt', map_location=device))
env = gym.wrappers.Monitor(env, 'recordings', force=True)
for i in range(5):
    done = False
    env.reset()
    cumm_reward = 0
    observation = torch.tensor([0, 0, 0, 0]).float()
    while not done:
        action = net(observation.unsqueeze(0))
        act = torch.distributions.categorical.Categorical(action).sample()
        observation, reward, done, info = env.step(act.item())
        env.render()
        cumm_reward += reward
        observation = torch.tensor(observation).float()
    print(f"Iter: {i} Reward: {cumm_reward}")
env.env.close()
