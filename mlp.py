import torch
import torch.nn as nn
import torch.optim as optim
from configuration import MLPConfig



class ActorNetwork(nn.Module):
    def __init__(self, d_in, d_out, d_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out)
        )

    def forward(self, observation):
        logits = self.net(observation)
        output_dist = torch.distributions.Categorical(logits=logits)
        return output_dist 
    
        
class CriticNetwork(nn.Module):
    def __init__(self, d_in, d_out, d_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out)
        )
        
    def forward(self, observation):
        return self.net(observation)


class MLP(nn.Module):
    def __init__(self, config:MLPConfig):
        super().__init__()
        self.actor = ActorNetwork(d_in=config.d_in, d_hidden=config.d_model, 
                                d_out=config.d_out_actor)
        self.critic = CriticNetwork(d_in=config.d_in, d_hidden=config.d_model_critic, 
                                    d_out=config.d_out_critic)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.learning_rate)

    def optim_step(self):
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def optim_zero_grad(self):
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

    def forward(self, x):
        action_dist:torch.distributions.Categorical = self.actor.forward(x)
        value = self.critic.forward(x)
        return action_dist, value