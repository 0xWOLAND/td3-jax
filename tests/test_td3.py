"""Tests for the TD3 implementation."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from fasttd3 import Actor, Critic, FastTD3


class TestActor:
    """Test Actor network."""
    
    def test_actor_init(self):
        """Test actor initialization."""
        actor = Actor(action_dim=3, max_action=1.0, hidden_dim=256)
        assert actor.action_dim == 3
        assert actor.max_action == 1.0
        assert actor.hidden_dim == 256
    
    def test_actor_forward(self):
        """Test actor forward pass."""
        key = jax.random.PRNGKey(0)
        actor = Actor(action_dim=3, max_action=1.0)
        state = jnp.ones((1, 10))
        
        params = actor.init(key, state)
        actions = actor.apply(params, state)
        
        assert actions.shape == (1, 3)
        assert jnp.all(jnp.abs(actions) <= 1.0)


class TestCritic:
    """Test Critic network."""
    
    def test_critic_init(self):
        """Test critic initialization."""
        critic = Critic(hidden_dim=256)
        assert critic.hidden_dim == 256
    
    def test_critic_forward(self):
        """Test critic forward pass."""
        key = jax.random.PRNGKey(0)
        critic = Critic()
        state = jnp.ones((1, 10))
        action = jnp.ones((1, 3))
        
        params = critic.init(key, state, action)
        q1, q2 = critic.apply(params, state, action)
        
        assert q1.shape == (1, 1)
        assert q2.shape == (1, 1)


class TestReplayBuffer:
    """Test ReplayBuffer."""
    
    def test_buffer_init(self):
        """Test buffer initialization."""
        from fasttd3 import ReplayBuffer
        
        buffer = ReplayBuffer(state_dim=10, action_dim=3, max_size=1000)
        assert buffer.max_size == 1000
        assert buffer.ptr == 0
        assert buffer.size == 0
    
    def test_buffer_add(self):
        """Test adding to buffer."""
        from fasttd3 import ReplayBuffer
        
        buffer = ReplayBuffer(state_dim=10, action_dim=3, max_size=100)
        
        state = np.ones(10)
        action = np.ones(3)
        next_state = np.ones(10) * 2
        reward = 1.0
        done = False
        
        buffer.add(state, action, next_state, reward, done)
        
        assert buffer.size == 1
        assert buffer.ptr == 1
    
    def test_buffer_sample(self):
        """Test sampling from buffer."""
        from fasttd3 import ReplayBuffer
        
        buffer = ReplayBuffer(state_dim=10, action_dim=3, max_size=100)
        
        # Add some data
        for i in range(50):
            state = np.ones(10) * i
            action = np.ones(3) * i
            next_state = np.ones(10) * (i + 1)
            reward = float(i)
            done = False
            buffer.add(state, action, next_state, reward, done)
        
        # Sample batch
        batch = buffer.sample(batch_size=32)
        states, actions, next_states, rewards, dones = batch
        
        assert states.shape == (32, 10)
        assert actions.shape == (32, 3)
        assert next_states.shape == (32, 10)
        assert rewards.shape == (32,)
        assert dones.shape == (32,)