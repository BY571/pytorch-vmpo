import gym
import torch
from torch import Tensor

class TorchWrapper(gym.Wrapper):
    def __init__(self, env, device="cpu"):
        self.env = env
        self.device = device
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    
    def reset(self, ):
        obs = self.env.reset()
        t_obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
        return t_obs
    
    def step(self, action: Tensor):
        next_obs, reward, done, info = self.env.step(action.detach().cpu().squeeze().numpy())
        next_t_obs = torch.from_numpy(next_obs).float().to(self.device).unsqueeze(0)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.tensor([done]).int().to(self.device)
        return next_t_obs, reward, done, info