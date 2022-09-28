import torch
from torch import nn, Tensor
from torch.distributions import Categorical
from typing import Tuple

class ActorCriticDiscrete(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCriticDiscrete, self).__init__()

        self.policy_layer = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=-1)
                )
        

        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
                )
        
    def forward(self, x: Tensor)-> Tensor:
        action_probs = self.policy_layer(x)
        return action_probs
    
    def get_action(self, state: Tensor)-> Tuple[Tensor, Tensor]:
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()        
        return action, dist.log_prob(action)
    
    def evaluate(self, state: Tensor, action: Tensor)-> Tuple[Tensor, Tensor, Tensor]:
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action.squeeze())
        dist_probs = dist.probs
        
        state_value = self.value_layer(state)
        
        return action_logprobs, state_value, dist_probs