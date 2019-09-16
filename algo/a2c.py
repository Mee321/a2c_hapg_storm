import torch
import torch.nn as nn
import torch.optim as optim
import math

class A2C():
    def __init__(self,
                 actor,
                 critic,
                 actor_lr=3e-3,
                 critic_lr=3e-3,
                 max_grad_norm=None):

        self.actor = actor
        self.critic = critic
        self.max_grad_norm = max_grad_norm

        self.actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        obs = rollouts.obs[:-1].view(-1, *obs_shape)
        action = rollouts.actions.view(-1, action_shape)
        action_mean, log_std = self.actor(obs)
        var = torch.exp(log_std) ** 2
        action_log_probs = -((action - action_mean) ** 2) / (2 * var) - log_std - math.log(math.sqrt(2 * math.pi))
        action_log_probs = action_log_probs.sum(1, keepdim=True)
        values = self.critic(obs)

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)
        advantages = rollouts.returns[:-1] - values

        value_loss = advantages.pow(2).mean()
        action_loss = -(advantages.detach() * action_log_probs).mean()

        self.actor_optimizer.zero_grad()
        action_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        return value_loss.item(), action_loss.item()