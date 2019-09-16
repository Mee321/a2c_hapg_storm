import torch.optim as optim
from utils import *
import math

# LVC version, DiCE with a bug since denominator becomes 0
class STORM_LVC():
    def __init__(self,
                 actor,
                 critic,
                 actor_lr=3e-3,
                 critic_lr=3e-3,
                 alpha_initial=1,
                 max_grad_norm=None):

        self.actor = actor
        self.critic = critic
        self.actor_lr = actor_lr
        self.alpha_initial = alpha_initial
        self.max_grad_norm = max_grad_norm
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)
        self.iteration = 1

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
        # value loss
        value_loss = advantages.pow(2).mean()
        # action loss
        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)
        probs = torch.exp(action_log_probs)
        magic_box = probs/probs.detach()
        action_loss = -(advantages.detach() * magic_box).mean()
        grad = torch.autograd.grad(action_loss, self.actor.parameters(), retain_graph=True)
        grad = flatten(grad)
        # actor_update
        prev_params = get_flat_params_from(self.actor)
        direction = grad / torch.norm(grad)
        updated_params = prev_params - self.actor_lr * direction
        d_theta = updated_params - prev_params
        set_flat_params_to(self.actor, updated_params)
        # critic update
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        return value_loss.item(), action_loss.item(), grad, d_theta

    def inner_update(self, rollouts, prev_grad, d_theta):
        alpha = self.alpha_initial / self.iteration ** (2 / 3)
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
        # value loss
        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()
        # action loss
        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)
        probs = torch.exp(action_log_probs)
        magic_box = probs / probs.detach()
        action_loss = -(magic_box * advantages.detach()).mean()
        # LVC Hessian
        jacob = torch.autograd.grad(action_loss, self.actor.parameters(), retain_graph=True, create_graph=True)
        jacob = flatten(jacob)
        product = torch.dot(jacob, d_theta)
        d_grad = torch.autograd.grad(product, self.actor.parameters(), retain_graph=True)
        # storm update
        grad = (1 - alpha) * prev_grad + alpha * jacob + (1 - alpha) * flatten(d_grad)
        # update params
        prev_params = get_flat_params_from(self.actor)
        direction = grad / torch.norm(grad)
        updated_params = prev_params - self.actor_lr * direction
        d_theta = updated_params - prev_params
        set_flat_params_to(self.actor, updated_params)

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        return value_loss.item(), action_loss.item(), grad, d_theta