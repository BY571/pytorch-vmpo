import torch
from torch import nn, Tensor
from torch.distributions import MultivariateNormal
import torch.nn.functional as F 
from typing import Tuple


# TODO:
class ActorCriticContinuous(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, log_std_min=-20, log_std_max=2):
        super(ActorCriticContinuous, self).__init__()
        self.action_dim = action_dim

        self.policy_layer = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh())
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.cholesky_layer = nn.Linear(256, (action_dim * (action_dim + 1)) // 2)
        
        

        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
                )
        
    def forward(self, x: Tensor)-> Tensor:
        batch_size = x.shape[0]
        x = self.policy_layer(x)
        mean = torch.tanh(self.mean_layer(x))
        cholesky_vector = self.cholesky_layer(x)
        #cholesky = torch.cholesky(F.softplus(cholesky_vector.unsqueeze(0)))
        cholesky_diag_index = torch.arange(self.action_dim, dtype=torch.long).to(x.device) + 1
        cholesky_diag_index = torch.div((cholesky_diag_index * (cholesky_diag_index + 1)), 2, rounding_mode='trunc') - 1
        cholesky_vector[:, cholesky_diag_index] = F.softplus(cholesky_vector[:, cholesky_diag_index])
        tril_indices = torch.tril_indices(row=self.action_dim, col=self.action_dim, offset=0)
        cholesky = torch.zeros(size=(batch_size, self.action_dim, self.action_dim), dtype=torch.float32).to(x.device)
        cholesky[:, tril_indices[0], tril_indices[1]] = cholesky_vector
        return mean, cholesky
    
    def get_action(self, state: Tensor)-> Tuple[Tensor, Tensor]:
        mean, cholesky = self.forward(state)
        dist = MultivariateNormal(mean, scale_tril=cholesky)
        action = dist.sample()        
        return action, dist.log_prob(action)
    
    def evaluate(self, state: Tensor, action: Tensor)-> Tuple[Tensor, Tensor, Tensor]:
        mean, cholesky = self.forward(state)
        dist = MultivariateNormal(mean, scale_tril=cholesky)
 
        action_logprobs = dist.log_prob(action)
        
        state_value = self.value_layer(state)
        
        return action_logprobs, state_value, mean, cholesky