import jax
import numpy as np
import optax
from flax import linen as nn
from flax import serialization
from jax import numpy as jnp
from jax import random as jrandom
from typing import Tuple, Optional

from . import utils


class Actor(nn.Module):
    action_dim: int
    max_action: float
    hidden_dim: int = 256

    def setup(self):
        self.l1 = nn.Dense(self.hidden_dim)
        self.l2 = nn.Dense(self.hidden_dim)
        self.l3 = nn.Dense(self.action_dim)

    def __call__(self, state):
        a = nn.relu(self.l1(state))
        a = nn.relu(self.l2(a))
        return self.max_action * nn.tanh(self.l3(a))


class Critic(nn.Module):
    """Standard TD3 critic with two Q-networks"""

    hidden_dim: int = 256

    def setup(self):
        # Q1 architecture
        self.q1_l1 = nn.Dense(self.hidden_dim)
        self.q1_l2 = nn.Dense(self.hidden_dim)
        self.q1_l3 = nn.Dense(1)

        # Q2 architecture
        self.q2_l1 = nn.Dense(self.hidden_dim)
        self.q2_l2 = nn.Dense(self.hidden_dim)
        self.q2_l3 = nn.Dense(1)

    def __call__(self, state, action):
        sa = jnp.concatenate([state, action], axis=-1)

        q1 = nn.relu(self.q1_l1(sa))
        q1 = nn.relu(self.q1_l2(q1))
        q1 = self.q1_l3(q1)

        q2 = nn.relu(self.q2_l1(sa))
        q2 = nn.relu(self.q2_l2(q2))
        q2 = self.q2_l3(q2)

        return q1, q2

    def Q1(self, state, action):
        sa = jnp.concatenate([state, action], axis=-1)
        q1 = nn.relu(self.q1_l1(sa))
        q1 = nn.relu(self.q1_l2(q1))
        return self.q1_l3(q1)


class DistributionalCritic(nn.Module):
    """Distributional critic using categorical distribution (C51)"""

    hidden_dim: int = 256
    num_atoms: int = 101
    v_min: float = -250.0
    v_max: float = 250.0

    def setup(self):
        # Q1 architecture
        self.q1_l1 = nn.Dense(self.hidden_dim)
        self.q1_l2 = nn.Dense(self.hidden_dim)
        self.q1_l3 = nn.Dense(self.num_atoms)

        # Q2 architecture
        self.q2_l1 = nn.Dense(self.hidden_dim)
        self.q2_l2 = nn.Dense(self.hidden_dim)
        self.q2_l3 = nn.Dense(self.num_atoms)

    def __call__(self, state, action):
        sa = jnp.concatenate([state, action], axis=-1)

        # Q1 network
        q1 = nn.relu(self.q1_l1(sa))
        q1 = nn.relu(self.q1_l2(q1))
        q1_logits = self.q1_l3(q1)

        # Q2 network
        q2 = nn.relu(self.q2_l1(sa))
        q2 = nn.relu(self.q2_l2(q2))
        q2_logits = self.q2_l3(q2)

        return q1_logits, q2_logits

    def Q1(self, state, action):
        """Get Q1 logits for actor loss"""
        sa = jnp.concatenate([state, action], axis=-1)
        q1 = nn.relu(self.q1_l1(sa))
        q1 = nn.relu(self.q1_l2(q1))
        return self.q1_l3(q1)


class FastTD3:
    """FastTD3 with distributional critics implemented in JAX/Flax.

    This implementation includes:
    - Distributional critics using categorical distribution (C51)
    - Learning rate scheduling (linear, cosine)
    - Efficient vectorized operations
    - All core FastTD3 features from the original implementation
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        hidden_dim: int = 256,
        distributional: bool = True,
        num_atoms: int = 101,
        v_min: float = -250.0,
        v_max: float = 250.0,
        lr_schedule: Optional[str] = None,  # Options: None, 'linear', 'cosine'
        total_timesteps: int = int(1e6),
        seed: int = 0,
    ):
        # Initialize random number generators
        self.rngs = utils.PRNGKeys(seed)

        # Create dummy inputs for network initialization
        dummy_state = jnp.ones([1, state_dim], dtype=jnp.float32)
        dummy_action = jnp.ones([1, action_dim], dtype=jnp.float32)

        # Initialize actor network and parameters
        self.actor = Actor(action_dim, max_action, hidden_dim)
        self.actor_params = self.actor.init(self.rngs.get_key(), dummy_state)
        self.actor_target_params = self.actor.init(self.rngs.get_key(), dummy_state)

        # Configure learning rate schedules
        actor_schedule, critic_schedule = self._create_lr_schedules(
            lr_schedule, actor_lr, critic_lr, total_timesteps
        )

        self.actor_optimizer = optax.adam(learning_rate=actor_schedule)
        self.actor_opt_state = self.actor_optimizer.init(self.actor_params)

        # Initialize critic network (distributional or standard)
        self.distributional = distributional
        if distributional:
            self.critic = DistributionalCritic(hidden_dim, num_atoms, v_min, v_max)
            self._setup_distributional_params(num_atoms, v_min, v_max)
        else:
            self.critic = Critic(hidden_dim)

        self.critic_params = self.critic.init(self.rngs.get_key(), dummy_state, dummy_action)
        self.critic_target_params = self.critic.init(self.rngs.get_key(), dummy_state, dummy_action)
        self.critic_optimizer = optax.adam(learning_rate=critic_schedule)
        self.critic_opt_state = self.critic_optimizer.init(self.critic_params)

        # Store hyperparameters
        self._store_hyperparameters(
            max_action, discount, tau, policy_noise, noise_clip, policy_freq, action_dim
        )

        # Compile training functions for performance
        self._compile_training_functions()

        self.total_it = 0
        self.expl_noise = 0.1  # Exploration noise

    def _create_lr_schedules(self, lr_schedule, actor_lr, critic_lr, total_timesteps):
        """Create learning rate schedules based on configuration."""
        if lr_schedule == "linear":
            actor_schedule = optax.linear_schedule(
                init_value=actor_lr, end_value=actor_lr * 0.1, transition_steps=int(total_timesteps)
            )
            critic_schedule = optax.linear_schedule(
                init_value=critic_lr,
                end_value=critic_lr * 0.1,
                transition_steps=int(total_timesteps),
            )
        elif lr_schedule == "cosine":
            actor_schedule = optax.cosine_decay_schedule(
                init_value=actor_lr, decay_steps=int(total_timesteps)
            )
            critic_schedule = optax.cosine_decay_schedule(
                init_value=critic_lr, decay_steps=int(total_timesteps)
            )
        else:
            actor_schedule = actor_lr
            critic_schedule = critic_lr

        return actor_schedule, critic_schedule

    def _setup_distributional_params(self, num_atoms, v_min, v_max):
        """Setup parameters for distributional critic."""
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = jnp.linspace(v_min, v_max, num_atoms)
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

    def _store_hyperparameters(
        self, max_action, discount, tau, policy_noise, noise_clip, policy_freq, action_dim
    ):
        """Store algorithm hyperparameters."""
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.action_dim = action_dim

    def _compile_training_functions(self):
        """JIT compile training functions for performance."""
        if self.distributional:
            self.critic_step = jax.jit(self._critic_step_distributional)
            self.actor_step = jax.jit(self._actor_step_distributional)
        else:
            self.critic_step = jax.jit(self._critic_step)
            self.actor_step = jax.jit(self._actor_step)
        self.update_targets = jax.jit(self._update_targets)

    def select_action(self, state: np.ndarray, add_noise: bool = False) -> np.ndarray:
        """Select action for a single state.

        Args:
            state: Environment state
            add_noise: Whether to add exploration noise

        Returns:
            Selected action
        """
        state = jax.device_put(state)
        if state.ndim == 1:
            state = state.reshape(1, -1)

        action = self.actor.apply(self.actor_params, state)

        if add_noise:
            noise = jrandom.normal(self.rngs.get_key(), action.shape) * self.max_action * self.expl_noise
            action = jnp.clip(action + noise, -self.max_action, self.max_action)

        return np.array(action).squeeze()

    def select_actions_batch(self, states: np.ndarray, add_noise: bool = False) -> np.ndarray:
        """Select actions for multiple states efficiently in a single forward pass.

        Args:
            states: Batch of environment states [batch_size, state_dim]
            add_noise: Whether to add exploration noise

        Returns:
            Batch of selected actions [batch_size, action_dim]
        """
        states = jax.device_put(states)
        actions = self.actor.apply(self.actor_params, states)

        if add_noise:
            noise = jrandom.normal(self.rngs.get_key(), actions.shape) * self.max_action * self.expl_noise
            actions = jnp.clip(actions + noise, -self.max_action, self.max_action)

        return np.array(actions)

    def _categorical_projection(self, next_probs, rewards, not_dones):
        """Project the categorical distribution for distributional Bellman update"""
        batch_size = rewards.shape[0]

        # Compute projected support
        proj_support = rewards + not_dones * self.discount * self.support[None, :]

        # Clip to support range
        proj_support = jnp.clip(proj_support, self.v_min, self.v_max)

        # Compute projection indices
        b = (proj_support - self.v_min) / self.delta_z
        l = jnp.floor(b).astype(jnp.int32)
        u = l + 1

        # Handle edge cases
        l = jnp.clip(l, 0, self.num_atoms - 1)
        u = jnp.clip(u, 0, self.num_atoms - 1)

        # Compute projection weights
        d_l = (u.astype(jnp.float32) - b) * next_probs
        d_u = (b - l.astype(jnp.float32)) * next_probs

        # Aggregate probabilities
        target_probs = jnp.zeros((batch_size, self.num_atoms))

        # Use scatter operations for projection
        indices_l = l
        indices_u = u

        # Flatten for scatter operations
        batch_indices = jnp.arange(batch_size)[:, None]
        batch_indices = jnp.broadcast_to(batch_indices, (batch_size, self.num_atoms))

        # Scatter add the probabilities
        target_probs = target_probs.at[batch_indices.flatten(), indices_l.flatten()].add(
            d_l.flatten()
        )
        target_probs = target_probs.at[batch_indices.flatten(), indices_u.flatten()].add(
            d_u.flatten()
        )

        return target_probs.reshape(batch_size, self.num_atoms)

    def _critic_loss_distributional(self, critic_params, target_params, batch, rng):
        state, action, next_state, reward, not_done = batch

        # Select action according to policy and add clipped noise
        noise = jrandom.normal(rng, action.shape) * self.policy_noise
        noise = jnp.clip(noise, -self.noise_clip, self.noise_clip)

        next_action = self.actor.apply(target_params["actor"], next_state)
        next_action = jnp.clip(next_action + noise, -self.max_action, self.max_action)

        # Get target distribution
        target_logits_1, target_logits_2 = self.critic.apply(
            target_params["critic"], next_state, next_action
        )

        # Take minimum Q-value (in expectation)
        target_probs_1 = jax.nn.softmax(target_logits_1, axis=-1)
        target_probs_2 = jax.nn.softmax(target_logits_2, axis=-1)
        target_q1 = jnp.sum(target_probs_1 * self.support[None, :], axis=-1, keepdims=True)
        target_q2 = jnp.sum(target_probs_2 * self.support[None, :], axis=-1, keepdims=True)

        # Use distribution with lower expected value
        use_q1 = (target_q1 <= target_q2).astype(jnp.float32)
        target_probs = use_q1 * target_probs_1 + (1 - use_q1) * target_probs_2

        # Project distribution
        target_probs = self._categorical_projection(target_probs, reward, not_done)

        # Get current Q distributions
        current_logits_1, current_logits_2 = self.critic.apply(critic_params, state, action)

        # Cross-entropy loss
        loss_1 = -jnp.sum(target_probs * jax.nn.log_softmax(current_logits_1, axis=-1), axis=-1)
        loss_2 = -jnp.sum(target_probs * jax.nn.log_softmax(current_logits_2, axis=-1), axis=-1)

        return jnp.mean(loss_1 + loss_2)

    def _critic_step_distributional(
        self, critic_params, critic_opt_state, target_params, batch, rng
    ):
        loss, grads = jax.value_and_grad(self._critic_loss_distributional)(
            critic_params, target_params, batch, rng
        )
        updates, critic_opt_state = self.critic_optimizer.update(grads, critic_opt_state)
        critic_params = optax.apply_updates(critic_params, updates)
        return critic_params, critic_opt_state, loss

    def _actor_loss_distributional(self, actor_params, critic_params, state):
        action = self.actor.apply(actor_params, state)
        q_logits = self.critic.apply(critic_params, state, action, method=self.critic.Q1)
        q_probs = jax.nn.softmax(q_logits, axis=-1)
        q_values = jnp.sum(q_probs * self.support[None, :], axis=-1)
        return -jnp.mean(q_values)

    def _actor_step_distributional(self, actor_params, actor_opt_state, critic_params, state):
        loss, grads = jax.value_and_grad(self._actor_loss_distributional)(
            actor_params, critic_params, state
        )
        updates, actor_opt_state = self.actor_optimizer.update(grads, actor_opt_state)
        actor_params = optax.apply_updates(actor_params, updates)
        return actor_params, actor_opt_state, loss

    def _critic_loss(self, critic_params, target_params, batch, rng):
        state, action, next_state, reward, not_done = batch

        # Select action according to policy and add clipped noise
        noise = jrandom.normal(rng, action.shape) * self.policy_noise
        noise = jnp.clip(noise, -self.noise_clip, self.noise_clip)

        next_action = self.actor.apply(target_params["actor"], next_state)
        next_action = jnp.clip(next_action + noise, -self.max_action, self.max_action)

        # Compute the target Q value
        target_Q1, target_Q2 = self.critic.apply(target_params["critic"], next_state, next_action)
        target_Q = jnp.minimum(target_Q1, target_Q2)
        target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic.apply(critic_params, state, action)

        # Compute critic loss
        critic_loss = jnp.mean((current_Q1 - target_Q) ** 2) + jnp.mean(
            (current_Q2 - target_Q) ** 2
        )

        return critic_loss

    def _critic_step(self, critic_params, critic_opt_state, target_params, batch, rng):
        loss, grads = jax.value_and_grad(self._critic_loss)(
            critic_params, target_params, batch, rng
        )
        updates, critic_opt_state = self.critic_optimizer.update(grads, critic_opt_state)
        critic_params = optax.apply_updates(critic_params, updates)
        return critic_params, critic_opt_state, loss

    def _actor_loss(self, actor_params, critic_params, state):
        action = self.actor.apply(actor_params, state)
        q_value = self.critic.apply(critic_params, state, action, method=self.critic.Q1)
        return -jnp.mean(q_value)

    def _actor_step(self, actor_params, actor_opt_state, critic_params, state):
        loss, grads = jax.value_and_grad(self._actor_loss)(actor_params, critic_params, state)
        updates, actor_opt_state = self.actor_optimizer.update(grads, actor_opt_state)
        actor_params = optax.apply_updates(actor_params, updates)
        return actor_params, actor_opt_state, loss

    def _update_targets(self, params, target_params):
        return jax.tree.map(lambda p, tp: self.tau * p + (1 - self.tau) * tp, params, target_params)

    def _train_step(self, batch):
        """Internal training step with pre-sampled batch"""
        self.total_it += 1
        state, action, next_state, reward, not_done = batch

        # Update critic
        target_params = {"actor": self.actor_target_params, "critic": self.critic_target_params}
        self.critic_params, self.critic_opt_state, critic_loss = self.critic_step(
            self.critic_params, self.critic_opt_state, target_params, batch, self.rngs.get_key()
        )

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Update actor
            self.actor_params, self.actor_opt_state, actor_loss = self.actor_step(
                self.actor_params, self.actor_opt_state, self.critic_params, state
            )

            # Update target networks
            self.actor_target_params = self.update_targets(
                self.actor_params, self.actor_target_params
            )
            self.critic_target_params = self.update_targets(
                self.critic_params, self.critic_target_params
            )

    def train(self, replay_buffer, batch_size=256):
        """Original train method for backward compatibility"""
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        batch = (state, action, next_state, reward, not_done)
        self._train_step(batch)

    def set_exploration_noise(self, noise: float):
        """Set exploration noise level."""
        self.expl_noise = noise

    def save(self, filename):
        with open(f"{filename}_actor.pkl", "wb") as f:
            f.write(serialization.to_bytes(self.actor_params))
        with open(f"{filename}_critic.pkl", "wb") as f:
            f.write(serialization.to_bytes(self.critic_params))

    def load(self, filename):
        with open(f"{filename}_actor.pkl", "rb") as f:
            self.actor_params = serialization.from_bytes(self.actor_params, f.read())
        with open(f"{filename}_critic.pkl", "rb") as f:
            self.critic_params = serialization.from_bytes(self.critic_params, f.read())
        self.actor_target_params = self.actor_params
        self.critic_target_params = self.critic_params
