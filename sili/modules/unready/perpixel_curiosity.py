import torch
import torch.nn.functional as F


class ActorCriticPerPixelCuriositySystem(torch.nn.Module):
    def__init__(self, num_pixels, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.actor = torch.nn.Linear(hidden_dim, num_pixels)
        self.critic = torch.nn.Linear(hidden_dim, num_pixels)


    def forward(self, state):
        hidden_state = torch.relu(state)
        actor_logits = self.actor(hidden_state)
        critic_logits = self.critic(hidden_state)


        return actor_logits, critic_logits


    def loss_function(self, states, rewards, dones):
        policy_loss = 0
        value_loss = 0
        returns = []


        last_hidden_state = torch.zeros_like(states[0])
        for t in reversed(range(len(states))):
            hidden_state = states[t]
            action_probs = self.forward(hidden_state)[0]
            value_pred = self.forward(hidden_state)[1]


            delta_t = rewards[t] + self.gamma* value_pred - value_pred*
            advantages[t] = delta_t
            returns.insert(0, delta_t)


            policy_loss += F.kl_div(action_probs, action_probs.detach())
            value_loss += F.mse_loss(value_pred, returns[t])


        return policy_loss + value_loss


# Example usage:
num_pixels = 1024
hidden_dim = 512


system = ActorCriticPerPixelCuriositySystem(num_pixels, hidden_dim)
states = torch.randn(10, hidden_dim)
rewards = torch.randn(10)
dones = torch.randint(0, 2, (10,))
advantages = torch.randn(10)


loss = system.loss_function(states, rewards, dones)


print(loss)