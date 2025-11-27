"""
Unit tests for modern DDPG implementation.
Run with: pytest tests/test_ddpg.py -v
"""

import pytest
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from unittest.mock import Mock

from ddpg_modern import (
    DDPGConfig,
    DDPGAgent,
    Actor,
    Critic,
    OUNoise,
    ReplayBuffer,
    evaluate_policy
)


class TestDDPGConfig:
    """Test configuration validation."""

    def test_valid_config(self):
        config = DDPGConfig()
        assert config.gamma == 0.99
        assert config.batch_size == 256

    def test_invalid_gamma(self):
        with pytest.raises(ValueError):
            DDPGConfig(gamma=1.5)

    def test_invalid_tau(self):
        with pytest.raises(ValueError):
            DDPGConfig(tau=1.1)

    def test_invalid_batch_size(self):
        with pytest.raises(ValueError):
            DDPGConfig(batch_size=0)


class TestOUNoise:
    """Test Ornstein-Uhlenbeck noise process."""

    def test_initialization(self):
        noise = OUNoise(size=3, mu=0.0, theta=0.1, sigma=0.2)
        assert noise.size == 3
        assert jnp.allclose(noise.state, jnp.zeros(3))

    def test_sample_shape(self):
        noise = OUNoise(size=2)
        key = jrandom.PRNGKey(42)
        sample = noise.sample(key)
        assert sample.shape == (2,)

    def test_temporal_correlation(self):
        noise = OUNoise(size=1, theta=0.9, sigma=0.1)
        key = jrandom.PRNGKey(42)

        samples = []
        for _ in range(10):
            key, subkey = jrandom.split(key)
            sample = noise.sample(subkey)
            samples.append(float(sample[0]))

        # Check that samples are correlated (not purely random)
        diffs = [abs(samples[i] - samples[i-1]) for i in range(1, len(samples))]
        avg_diff = sum(diffs) / len(diffs)
        assert avg_diff < 1.0  # Should be correlated due to OU process


class TestReplayBuffer:
    """Test replay buffer functionality."""

    def test_initialization(self):
        buffer = ReplayBuffer(state_dim=3, action_dim=2, max_size=100)
        assert buffer.max_size == 100
        assert buffer.size == 0
        assert buffer.ptr == 0

    def test_add_and_sample(self):
        buffer = ReplayBuffer(state_dim=2, action_dim=1, max_size=10)

        # Add some experiences
        for i in range(5):
            state = np.array([i, i+1], dtype=np.float32)
            action = np.array([i], dtype=np.float32)
            buffer.add(state, action, float(i), state + 1, i % 2 == 0)

        assert buffer.size == 5

        # Sample batch
        key = jrandom.PRNGKey(42)
        batch = buffer.sample(key, batch_size=3)

        states, actions, rewards, next_states, dones = batch
        assert states.shape == (3, 2)
        assert actions.shape == (3, 1)
        assert rewards.shape == (3, 1)
        assert next_states.shape == (3, 2)
        assert dones.shape == (3, 1)

    def test_circular_buffer(self):
        buffer = ReplayBuffer(state_dim=1, action_dim=1, max_size=3)

        # Fill buffer
        for i in range(5):
            buffer.add(np.array([i]), np.array([i]), float(i), np.array([i+1]), False)

        assert buffer.size == 3  # Should not exceed max_size
        assert buffer.ptr == 2   # Should wrap around

    def test_not_ready(self):
        buffer = ReplayBuffer(state_dim=1, action_dim=1, max_size=10)
        assert not buffer.is_ready(batch_size=5)

        # Add some experiences
        for i in range(3):
            buffer.add(np.array([i]), np.array([i]), float(i), np.array([i+1]), False)

        assert not buffer.is_ready(batch_size=5)

    def test_ready(self):
        buffer = ReplayBuffer(state_dim=1, action_dim=1, max_size=10)

        # Add enough experiences
        for i in range(6):
            buffer.add(np.array([i]), np.array([i]), float(i), np.array([i+1]), False)

        assert buffer.is_ready(batch_size=5)


class TestNetworks:
    """Test neural network components."""

    def test_actor_forward(self):
        actor = Actor(action_dim=2, max_action=1.0, hidden_dims=[10, 5])

        key = jrandom.PRNGKey(42)
        dummy_state = jnp.zeros((1, 3))

        params = actor.init(key, dummy_state)
        output = actor.apply(params, dummy_state)

        assert output.shape == (1, 2)
        assert jnp.all(output >= -1.0) and jnp.all(output <= 1.0)

    def test_critic_forward(self):
        critic = Critic(hidden_dims=[10, 5])

        key = jrandom.PRNGKey(42)
        dummy_state = jnp.zeros((1, 3))
        dummy_action = jnp.zeros((1, 2))

        params = critic.init(key, dummy_state, dummy_action)
        output = critic.apply(params, dummy_state, dummy_action)

        assert output.shape == (1, 1)

    def test_network_jit_compilation(self):
        """Test that networks can be JIT compiled."""
        from jax import jit

        actor = Actor(action_dim=1, max_action=2.0, hidden_dims=[5])
        critic = Critic(hidden_dims=[5])

        key = jrandom.PRNGKey(42)
        dummy_state = jnp.zeros((1, 2))
        dummy_action = jnp.zeros((1, 1))

        actor_params = actor.init(key, dummy_state)
        critic_params = critic.init(jrandom.split(key)[1], dummy_state, dummy_action)

        # JIT compile forward passes
        jit_actor = jit(actor.apply)
        jit_critic = jit(critic.apply)

        # Test JIT execution
        actor_out = jit_actor(actor_params, dummy_state)
        critic_out = jit_critic(critic_params, dummy_state, dummy_action)

        assert actor_out.shape == (1, 1)
        assert critic_out.shape == (1, 1)


class TestDDPGAgent:
    """Test DDPG agent functionality."""

    def test_agent_initialization(self):
        config = DDPGConfig(batch_size=4, buffer_size=100)  # Small for testing
        agent = DDPGAgent(config, state_dim=3, action_dim=2, max_action=1.0)

        assert agent.state_dim == 3
        assert agent.action_dim == 2
        assert agent.max_action == 1.0

    def test_select_action_deterministic(self):
        config = DDPGConfig()
        agent = DDPGAgent(config, state_dim=2, action_dim=1, max_action=1.0)

        state = np.array([0.5, -0.3])
        action = agent.select_action(state, deterministic=True)

        assert action.shape == (1,)
        assert -1.0 <= action[0] <= 1.0

    def test_select_action_stochastic(self):
        config = DDPGConfig(noise_scale=0.1)
        agent = DDPGAgent(config, state_dim=2, action_dim=1, max_action=1.0)

        state = np.array([0.0, 0.0])

        # Get multiple actions to check variation
        actions = []
        for _ in range(10):
            action = agent.select_action(state, deterministic=False)
            actions.append(action[0])

        # Actions should vary due to noise
        unique_actions = len(set(actions))
        assert unique_actions > 1

    def test_train_step_without_buffer(self):
        config = DDPGConfig(batch_size=4, buffer_size=10)
        agent = DDPGAgent(config, state_dim=2, action_dim=1, max_action=1.0)

        # Buffer should not be ready yet
        losses = agent.train_step()
        assert losses["critic_loss"] == 0.0
        assert losses["actor_loss"] == 0.0

    def test_save_load(self, tmp_path):
        config = DDPGConfig()
        agent = DDPGAgent(config, state_dim=2, action_dim=1, max_action=1.0)

        # Save agent
        save_path = tmp_path / "test_agent.pth"
        agent.save(str(save_path))
        assert save_path.exists()

        # Create new agent and load
        new_agent = DDPGAgent(config, state_dim=2, action_dim=1, max_action=1.0)
        new_agent.load(str(save_path))

        # Check that parameters were loaded (basic check)
        assert hasattr(new_agent, 'actor_params')
        assert hasattr(new_agent, 'critic_params')


class TestEvaluation:
    """Test evaluation functionality."""

    def test_evaluate_policy(self):
        # Mock environment
        mock_env = Mock()
        mock_env.reset.return_value = (np.array([0.0, 0.0]), {})
        mock_env.step.return_value = (np.array([0.1, 0.1]), 1.0, False, False, {})

        # Create agent
        config = DDPGConfig()
        agent = DDPGAgent(config, state_dim=2, action_dim=1, max_action=1.0)

        # Evaluate
        reward = evaluate_policy(agent, mock_env, num_episodes=2)

        # Check that environment methods were called
        assert mock_env.reset.call_count == 2  # Once per episode
        assert mock_env.step.call_count > 0   # At least some steps

        # Reward should be a float
        assert isinstance(reward, float)


if __name__ == "__main__":
    pytest.main([__file__])
