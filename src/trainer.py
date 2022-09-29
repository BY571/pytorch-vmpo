from cmath import log
import torch
from torch import nn, Tensor
import gym
import gym_cartpole_swingup
from gym import Env
from gym import spaces
from typing import Tuple
import sys

from actor_critic import ActorCriticDiscrete, ActorCriticContinuous
from wrapper import TorchWrapper
from tqdm import tqdm

def bt(m):
    return m.transpose(dim0=-2, dim1=-1)


def btr(m):
    return m.diagonal(dim1=-2, dim2=-1).sum(-1)

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def add(self, state, action, logprob, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(done)
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class Trainer():
    def __init__(self, config):
        self.update_steps = 0
        self.interaction_steps = 0
        self.config = config
        self.gamma = config.common.gamma
        self.K_epochs = 8

        self.train_env, self.test_env, obs_space, act_space = self.get_env(config.common.env_name,
                                                                           device=self.config.common.device)
        
        self.set_eta_alpha(act_space, config)

        self.policy, self.old_policy = self.get_policies(obs_space, act_space)
                
        self.policy.to(config.common.device)
        self.old_policy.to(config.common.device)
        
        if type(act_space) == gym.spaces.discrete.Discrete:
            params = [{'params': self.policy.parameters()},
                    {'params': self.eta},
                    {'params': self.alpha}]
        else:
            params = [{'params': self.policy.parameters()},
                    {'params': self.eta},
                    {'params': self.alpha_mean},
                    {'params': self.alpha_sig}]

        self.set_update(act_space=act_space)
        self.optimizer = torch.optim.Adam(params, lr=1e-4)
        self.mseloss = nn.MSELoss()
        self.memory = Memory()
    
    def set_eta_alpha(self, action_space: gym.spaces, config: dict)-> None:
        if type(action_space) == gym.spaces.discrete.Discrete:
            self.alpha = torch.tensor([config.discrete.init_alpha], requires_grad=True)
            self.eps_alpha = torch.FloatTensor(config.discrete.eps_alpha).to(config.common.device).log()
            
            self.eta = torch.tensor([config.discrete.init_eta], requires_grad=True)
            self.eps_eta = torch.FloatTensor([config.discrete.eps_eta]).to(config.common.device)
        else:
            self.alpha_mean = torch.tensor([config.continuous.init_alpha_mean], requires_grad=True)
            self.alpha_sig = torch.tensor([config.continuous.init_alpha_sig], requires_grad=True)
            self.eps_alpha_mean = torch.FloatTensor(config.continuous.eps_alpha_mean).to(config.common.device).log()
            self.eps_alpha_sigma= torch.FloatTensor(config.continuous.eps_alpha_sig).to(config.common.device).log()
            
            self.eta = torch.tensor([config.discrete.init_eta], requires_grad=True)
            self.eps_eta = torch.FloatTensor([config.discrete.eps_eta]).to(config.common.device)

    @staticmethod
    def get_env(env_name: str, device: str): # -> Tuple[Env, Env, gym.spaces, gym.spaces]:
        train_env = TorchWrapper(gym.make(env_name), device=device)
        test_env = TorchWrapper(gym.make(env_name), device=device)
        return train_env, test_env, train_env.observation_space, train_env.action_space
    
    @staticmethod
    def get_policies(obs_space: gym.spaces, act_space: gym.spaces)-> Tuple[nn.Module, nn.Module]:
        if type(act_space) == gym.spaces.discrete.Discrete:
            policy = ActorCriticDiscrete(obs_space.shape[0], act_space.n, hidden_dim=256)
            old_policy = ActorCriticDiscrete(obs_space.shape[0], act_space.n, hidden_dim=256)
            old_policy.load_state_dict(policy.state_dict())
        else:
            policy = ActorCriticContinuous(obs_space.shape[0], act_space.shape[0], hidden_dim=256)
            old_policy = ActorCriticContinuous(obs_space.shape[0], act_space.shape[0], hidden_dim=256)
            old_policy.load_state_dict(policy.state_dict())
            
        return policy, old_policy
    
    @staticmethod
    def get_KL(prob1: Tensor, logprob1: Tensor, logprob2: Tensor)-> Tensor:
        kl = prob1 * (logprob1 - logprob2)
        return kl.sum(1, keepdim=True)
    
    @staticmethod
    def get_conti_kl(mean: Tensor, mean_old: Tensor, sigma: Tensor, sigma_old: Tensor):
        """
        decoupled KL between two multivariate gaussian distribution
        C_μ = KL(f(x|μi,Σi)||f(x|μ,Σi))
        C_Σ = KL(f(x|μi,Σi)||f(x|μi,Σ))
        :param μi: (B, n)
        :param μ: (B, n)
        :param Ai: (B, n, n)
        :param A: (B, n, n)
        :return: C_μ, C_Σ: scalar
            mean and covariance terms of the KL
        :return: mean of determinanats of Σi, Σ
        ref : https://stanford.edu/~jduchi/projects/general_notes.pdf page.13
        """
        d = sigma.size(-1)
        mean_old = mean_old.unsqueeze(-1)  # (B, n, 1)
        mean = mean.unsqueeze(-1)  # (B, n, 1)
        sigma_old = sigma_old @ bt(sigma_old)  # (B, n, n)
        sigma = sigma @ bt(sigma)  # (B, n, n)
        sigma_old_det = sigma_old.det()  # (B,)
        sigma_det = sigma.det()  # (B,)
        # determinant can be minus due to numerical calculation error
        # https://github.com/daisatojp/mpo/issues/11
        sigma_old_det = torch.clamp_min(sigma_old_det, 1e-6)
        sigma_det = torch.clamp_min(sigma_det, 1e-6)
        sigma_old_inv = sigma_old.inverse()  # (B, n, n)
        sigma_inv = sigma.inverse()  # (B, n, n)
        kl_mean = 0.5 * ((mean - mean_old).transpose(-2, -1) @ sigma_old_inv @ (mean - mean_old)).squeeze()  # (B,) eq: 25
        kl_sigma = 0.5 * (btr(sigma_inv @ sigma_old) - d + torch.log(sigma_det / sigma_old_det)) # (B,) eq: 26
        kl_mean = 0.5 * torch.mean(kl_mean)
        kl_sigma = 0.5 * torch.mean(kl_sigma)
        return kl_mean, kl_sigma
    
    @staticmethod
    def discouted_returns(rewards: Tensor, dones: Tensor, gamma: float=0.99)-> Tensor:
        disk_return = 0
        discounted = []
        
        for idx in reversed(range(len(rewards))):
            disk_return = rewards[idx] + disk_return * gamma
            discounted.insert(0, disk_return)
        return torch.stack(discounted)
    
    @staticmethod
    def normalize(x: Tensor)-> Tensor:
        return (x - x.mean()) / (x.std() + 1e-12)

    def set_update(self, act_space: gym.spaces)-> None:
        if type(act_space) == gym.spaces.discrete.Discrete:
            self.update = self.update_discrete
        else:
            self.update = self.update_continuous
    
    def evaluate(self, ):
        returns = []
        for i in range(self.config.common.eval_runs):
            test_return = 0
            state = self.test_env.reset()
            while True:
                action, _ = self.policy.get_action(state)
                state, reward, done, info = self.test_env.step(action)
                test_return += reward
                if done:
                    break
            returns.append(test_return)
        return torch.stack(returns).cpu().mean().numpy()
    
    def collect_data(self, ):
        state = self.train_env.reset()
        for step in range(self.config.common.max_steps_per_epoch):
            # TODO: really old policy?
            action, log_prob = self.old_policy.get_action(state)
            next_state, reward, done, info = self.train_env.step(action)
            self.memory.add(state, action, log_prob, reward, done)
            state = next_state
            self.interaction_steps += 1
            if done:
                break

    @torch.no_grad()
    def get_advantage(self, states: Tensor, actions: Tensor, rewards: Tensor)-> Tuple[Tensor, Tensor]:
        _, state_values, dist_probs = self.old_policy.evaluate(states, actions)
        advantages = rewards - state_values.detach()
        return advantages, dist_probs

    @torch.no_grad()
    def get_advantage_conti(self, states: Tensor, actions: Tensor, rewards: Tensor)-> Tuple[Tensor, Tensor]:
        _, state_values, mean, cholesky = self.old_policy.evaluate(states, actions)
        advantages = rewards - state_values.detach()
        return advantages, mean, cholesky
        
    def load_ckpt(self, ckpt_path)-> None:
        self.policy.load_state_dict(torch.load(ckpt_path))
        
    def save_ckpt(self, path)-> None:
        torch.save(self.policy.state_dict(), path)
    
    def update_discrete(self, ):
        states = torch.vstack(self.memory.states).detach()
        actions = torch.vstack(self.memory.actions).detach()
        disc_reward = self.discouted_returns(self.memory.rewards, self.memory.is_terminals, gamma=self.gamma)
        norm_disc_reward = self.normalize(disc_reward)
        advantages, old_dist_probs = self.get_advantage(states, actions, norm_disc_reward)
        for i in range(self.K_epochs):
            logprobs, state_values, dist_probs = self.policy.evaluate(states, actions)
            
            # Get samples with top half advantages
            advprobs = torch.stack((advantages.squeeze(), logprobs))
            advprobs = advprobs[:,torch.sort(advprobs[0], descending=True).indices]
            good_advantages = advprobs[0, :len(states)//2]
            good_logprobs = advprobs[1, :len(states)//2]
            
            # Get losses
            phis = torch.exp(good_advantages/self.eta.detach().to(self.config.common.device))/torch.sum(torch.exp(good_advantages/self.eta.detach().to(self.config.common.device)))
            loss_pi = (-phis * good_logprobs).mean()
            loss_eta = self.eta.to(self.config.common.device) * self.eps_eta + self.eta.to(self.config.common.device) * (good_advantages/self.eta.to(self.config.common.device)).exp().mean().log()
            
            kl = self.get_KL(old_dist_probs.detach(),torch.log(old_dist_probs).detach(),torch.log(dist_probs))
            
            coef_alpha = torch.distributions.Uniform(self.eps_alpha[0], self.eps_alpha[1]).sample().exp()
            loss_alpha = torch.mean(self.alpha.to(self.config.common.device) * (coef_alpha - kl.detach()) + self.alpha.detach().to(self.config.common.device) * kl)

            value_loss = self.mseloss(state_values, norm_disc_reward)
            
            loss = (loss_pi + loss_eta + loss_alpha + 0.5 * value_loss).mean()

            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.update_steps += 1
            with torch.no_grad():
                self.eta.copy_(torch.clamp(self.eta, min=1e-8))
                self.alpha.copy_(torch.clamp(self.alpha, min=1e-8))
        
        # Copy new weights into old policy:
        self.old_policy.load_state_dict(self.policy.state_dict())
        loss_dict = {"loss": loss.item(),
                     "policy_loss": loss_pi.item(),
                     "loss_eta": loss_eta.item(),
                     "kl": kl.mean().item(),
                     "alpha": self.alpha.item(),
                     "alpa_loss": loss_alpha.item()}
        return loss_dict

    def update_continuous(self, ):
            states = torch.vstack(self.memory.states).detach()
            actions = torch.vstack(self.memory.actions).detach()
            disc_reward = self.discouted_returns(self.memory.rewards, self.memory.is_terminals, gamma=self.gamma)
            norm_disc_reward = self.normalize(disc_reward)
            advantages, mean_old, sigma_old = self.get_advantage_conti(states, actions, norm_disc_reward)
            for i in range(self.K_epochs):
                logprobs, state_values, mean, sigma = self.policy.evaluate(states, actions)
                
                # Get samples with top half advantages
                advprobs = torch.stack((advantages.squeeze(), logprobs))
                advprobs = advprobs[:,torch.sort(advprobs[0], descending=True).indices]
                good_advantages = advprobs[0, :len(states)//2]
                good_logprobs = advprobs[1, :len(states)//2]
                
                # Get losses
                phis = torch.exp(good_advantages/self.eta.detach().to(self.config.common.device))/torch.sum(torch.exp(good_advantages/self.eta.detach().to(self.config.common.device)))
                loss_pi = (-phis * good_logprobs).mean()
                loss_eta = self.eta.to(self.config.common.device) * self.eps_eta + self.eta.to(self.config.common.device) * (good_advantages/self.eta.to(self.config.common.device)).exp().mean().log()
                
                kl_mean, kl_sigma = self.get_conti_kl(mean, mean_old, sigma, sigma_old)
                
                coef_alpha_mean = torch.distributions.Uniform(self.eps_alpha_mean[0], self.eps_alpha_mean[1]).sample().exp()
                coef_alpha_sigma = torch.distributions.Uniform(self.eps_alpha_sigma[0], self.eps_alpha_sigma[1]).sample().exp()
                loss_alpha_mean = torch.mean(self.alpha_mean.to(self.config.common.device) * (coef_alpha_mean - kl_mean.detach()) + self.alpha_mean.detach().to(self.config.common.device) * kl_mean)
                loss_alpha_sigma = torch.mean(self.alpha_sig.to(self.config.common.device) * (coef_alpha_sigma - kl_sigma.detach()) + self.alpha_sig.detach().to(self.config.common.device) * kl_sigma)

                value_loss = self.mseloss(state_values, norm_disc_reward)
                
                loss = (loss_pi + loss_eta + loss_alpha_mean + loss_alpha_sigma + 0.5 * value_loss).mean()

                # take gradient step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.update_steps += 1
                with torch.no_grad():
                    self.eta.copy_(torch.clamp(self.eta, min=1e-8))
                    self.alpha_mean.copy_(torch.clamp(self.alpha_mean, min=1e-8))
                    self.alpha_sig.copy_(torch.clamp(self.alpha_sig, min=1e-8))
            
            # Copy new weights into old policy:
            self.old_policy.load_state_dict(self.policy.state_dict())
            loss_dict = {"loss": loss.item(),
                         "policy_loss": loss_pi.item(),
                         "value_loss (w/o coeff: 0.5)": value_loss.item(),
                         "eta_loss": loss_eta.item(),
                         "kl_mean": kl_mean.mean().item(),
                         "kl_sigma": kl_sigma.mean().item(),
                         "alpha_mean_loss": loss_alpha_mean.item(),
                         "alpha_sig_loss": loss_alpha_sigma.item(),
                         "alpha_mean":self.alpha_mean.item(),
                         "alpha_sig":self.alpha_sig.item()}
            return loss_dict

    def train(self, wandb):
        
        pre_training_reward = self.evaluate()
        wandb.log({"reward": pre_training_reward,
                   "epoch": 0,
                   "gradient_updates": self.update_steps}, step=self.interaction_steps)
        
        for e in tqdm(range(1, self.config.common.epochs+1), desc="Training", file=sys.stdout):
            self.collect_data()
            log_info = self.update()
            wandb.log(log_info, step=self.interaction_steps)
            self.memory.clear_memory()
            if e % self.config.common.eval_every == 0:
                eval_reward = self.evaluate()
                wandb.log({"reward": eval_reward,
                           "epoch": e,
                           "gradient_updates": self.update_steps}, step=self.interaction_steps)
                # TODO: save model    
            
                
