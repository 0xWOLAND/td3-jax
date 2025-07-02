import time
import numpy as np
import jax.numpy as jnp
from jax import random as jrandom
from typing import Optional, Tuple


class ReplayBuffer:
    """Simple replay buffer for single environment training.

    Stores transitions and samples random batches for training.
    """

    def __init__(self, state_dim: int, action_dim: int, max_size: int = int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # Preallocate arrays for efficient storage
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        done: float,
    ) -> None:
        """Add a transition to the replay buffer."""
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Tuple[jnp.ndarray, ...]:
        """Sample a random batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)

        return (
            jnp.array(self.state[indices]),
            jnp.array(self.action[indices]),
            jnp.array(self.next_state[indices]),
            jnp.array(self.reward[indices]),
            jnp.array(self.not_done[indices]),
        )


class VectorizedReplayBuffer:
    """Efficient replay buffer for vectorized environments.

    Matches FastTD3's design with separate storage per environment
    to maintain proper sampling distribution across parallel environments.
    """

    def __init__(self, n_envs: int, state_dim: int, action_dim: int, max_size: int = int(1e6)):
        self.n_envs = n_envs
        self.max_size = max_size // n_envs  # Per-environment buffer size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ptr = 0
        self.size = 0

        # Preallocate storage arrays for all environments
        self.states = np.zeros((n_envs, self.max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((n_envs, self.max_size, action_dim), dtype=np.float32)
        self.next_states = np.zeros((n_envs, self.max_size, state_dim), dtype=np.float32)
        self.rewards = np.zeros((n_envs, self.max_size), dtype=np.float32)
        self.dones = np.zeros((n_envs, self.max_size), dtype=np.float32)

    def add_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Add a batch of transitions from all environments.

        Args:
            states: Current states [n_envs, state_dim]
            actions: Actions taken [n_envs, action_dim]
            next_states: Next states [n_envs, state_dim]
            rewards: Rewards received [n_envs]
            dones: Done flags [n_envs]
        """
        # Store transitions for all environments simultaneously
        self.states[:, self.ptr] = states
        self.actions[:, self.ptr] = actions
        self.next_states[:, self.ptr] = next_states
        self.rewards[:, self.ptr] = rewards
        self.dones[:, self.ptr] = dones

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Tuple[jnp.ndarray, ...]:
        """Sample a batch of transitions across all environments.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, next_states, rewards, not_dones)
        """
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")

        # Sample random indices across all environments and timesteps
        env_indices = np.random.randint(0, self.n_envs, size=batch_size)
        time_indices = np.random.randint(0, self.size, size=batch_size)

        # Efficiently gather samples using advanced indexing
        states = self.states[env_indices, time_indices]
        actions = self.actions[env_indices, time_indices]
        next_states = self.next_states[env_indices, time_indices]
        rewards = self.rewards[env_indices, time_indices].reshape(-1, 1)
        not_dones = (1.0 - self.dones[env_indices, time_indices]).reshape(-1, 1)

        return (
            jnp.array(states),
            jnp.array(actions),
            jnp.array(next_states),
            jnp.array(rewards),
            jnp.array(not_dones),
        )


class EmpiricalNormalization:
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape: Tuple[int, ...], eps: float = 1e-2):
        """Initialize normalization with given shape.

        Args:
            shape: Shape of input values (without batch dimension)
            eps: Small value for numerical stability
        """
        self.eps = eps
        self.shape = shape if isinstance(shape, tuple) else (shape,)

        # Initialize statistics
        self._mean = np.zeros(self.shape)
        self._var = np.ones(self.shape)
        self._std = np.ones(self.shape)
        self.count = 0

    @property
    def mean(self):
        return self._mean.copy()

    @property
    def std(self):
        return self._std.copy()

    def update(self, x: np.ndarray):
        """Update running statistics with new batch of data."""
        if self.count == 0:
            self._mean = np.mean(x, axis=0)
            self._var = np.var(x, axis=0)
            self._std = np.sqrt(self._var)
            self.count = x.shape[0]
        else:
            batch_size = x.shape[0]
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)

            # Update count
            new_count = self.count + batch_size

            # Update mean using incremental formula
            delta = batch_mean - self._mean
            self._mean += (batch_size / new_count) * delta

            # Update variance using Welford's online algorithm
            m_a = self._var * self.count
            m_b = batch_var * batch_size
            M2 = m_a + m_b + (delta**2) * (self.count * batch_size / new_count)
            self._var = M2 / new_count
            self._std = np.sqrt(self._var)

            self.count = new_count

    def normalize(self, x: np.ndarray, center: bool = True) -> np.ndarray:
        """Normalize input using current statistics."""
        if center:
            return (x - self._mean) / (self._std + self.eps)
        else:
            return x / (self._std + self.eps)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Inverse of normalize."""
        return x * (self._std + self.eps) + self._mean


class RewardNormalizer:
    """Normalize rewards using running statistics of discounted returns."""

    def __init__(self, num_envs: int, gamma: float = 0.99, epsilon: float = 1e-8):
        """Initialize reward normalizer.

        Args:
            num_envs: Number of parallel environments
            gamma: Discount factor
            epsilon: Small value for numerical stability
        """
        self.num_envs = num_envs
        self.gamma = gamma
        self.epsilon = epsilon

        # Running estimate of discounted returns for each environment
        self.returns = np.zeros(num_envs)

        # Statistics normalizer
        self.return_rms = EmpiricalNormalization(shape=(1,))

    def update_returns(self, rewards: np.ndarray, dones: np.ndarray):
        """Update running returns and statistics."""
        # Update discounted returns
        self.returns = rewards + self.gamma * self.returns * (1 - dones)

        # Update running statistics
        self.return_rms.update(self.returns.reshape(-1, 1))

    def normalize_reward(self, reward: float) -> float:
        """Normalize a single reward value."""
        return reward / (self.return_rms.std[0] + self.epsilon)


class Timer:
    def __init__(self):
        self._start_time = time.time()

    def reset(self):
        self._start_time = time.time()

    def time_cost(self):
        return time.time() - self._start_time


class PRNGKeys:
    def __init__(self, seed=0):
        self._key = jrandom.PRNGKey(seed)

    def get_key(self):
        self._key, subkey = jrandom.split(self._key)
        return subkey
