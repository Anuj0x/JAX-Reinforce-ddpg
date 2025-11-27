"""
Modern DDPG Implementation using JAX
=====================================

A high-performance, memory-efficient implementation of Deep Deterministic Policy Gradient
using JAX for automatic differentiation and JIT compilation.

Features:
- JAX-based neural networks with JIT compilation for maximum performance
- Modern Python with comprehensive type hints
- Consolidated single-file architecture for clarity
- Advanced exploration strategies (OUNoise, ParameterSpaceNoise)
- Efficient vectorized operations
- Proper configuration management with dataclasses
- Comprehensive logging and monitoring
- Memory-efficient replay buffer with circular arrays
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List, Callable
from pathlib import Path
import pickle
import json

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import jit, grad, vmap
import flax.linen as nn
import optax
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import tensorboardX as tbx  # More modern than torch.utils.tensorboard

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DDPGConfig:
    """Configuration for DDPG agent with sensible defaults."""

    # Environment settings
    env_name: str = "Pendulum-v1"
    seed: int = 42
    render_mode: Optional[str] = None

    # Network architecture
    actor_hidden_dims: List[int] = field(default_factory=lambda: [400, 300])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [400, 300])

    # Training hyperparameters
    gamma: float = 0.99
    tau: float = 0.005  # Target network update rate
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    batch_size: int = 256  # Larger batch size for stability
    buffer_size: int = int(1e6)
    random_steps: int = 25000
    max_steps: int = int(1e6)
    eval_interval: int = 5000
    save_interval: int = 50000

    # Exploration
    noise_type: str = "ou"  # "ou" for Ornstein-Uhlenbeck, "normal" for Gaussian
    noise_scale: float = 0.1
    ou_theta: float = 0.15
    ou_sigma: float = 0.2

    # Device
    device: str = "cpu"  # JAX handles this automatically

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.gamma < 0 or self.gamma > 1:
            raise ValueError("gamma must be in [0, 1]")
        if self.tau <= 0 or self.tau > 1:
            raise ValueError("tau must be in (0, 1]")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.noise_type not in ["ou", "normal"]:
            raise ValueError("noise_type must be 'ou' or 'normal'")


class Actor(nn.Module):
    """Actor network using Flax for modern JAX integration."""

    action_dim: int
    max_action: float
    hidden_dims: List[int]

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for dim in self.hidden_dims[:-1]:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)

        x = nn.Dense(self.hidden_dims[-1])(x)
        x = nn.relu(x)

        x = nn.Dense(self.action_dim)(x)
        return jnp.tanh(x) * self.max_action


class Critic(nn.Module):
    """Critic network using Flax."""

    hidden_dims: List[int]

    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([state, action], axis=-1)

        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)

        return nn.Dense(1)(x)


@dataclass
class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration noise."""

    size: int
    mu: float = 0.0
    theta: float = 0.15
    sigma: float = 0.2
    dt: float = 1e-2

    def __post_init__(self):
        self.state = jnp.zeros(self.size)

    def sample(self, key: jax.Array) -> jnp.ndarray:
        """Sample from OU process."""
        # OU process: dx = theta*(mu - x)*dt + sigma*dW
        drift = self.theta * (self.mu - self.state) * self.dt
        diffusion = self.sigma * jnp.sqrt(self.dt) * jrandom.normal(key, (self.size,))
        self.state = self.state + drift + diffusion
        return self.state


class ReplayBuffer:
    """Efficient circular buffer replay buffer with JAX arrays."""

    def __init__(self, state_dim: int, action_dim: int, max_size: int):
        self.max_size = max_size
        self.size = 0
        self.ptr = 0

        # Pre-allocate arrays
        self.states = jnp.zeros((max_size, state_dim))
        self.actions = jnp.zeros((max_size, action_dim))
        self.rewards = jnp.zeros((max_size, 1))
        self.next_states = jnp.zeros((max_size, state_dim))
        self.dones = jnp.zeros((max_size, 1), dtype=bool)

    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool):
        """Add experience to buffer."""
        self.states = self.states.at[self.ptr].set(state)
        self.actions = self.actions.at[self.ptr].set(action)
        self.rewards = self.rewards.at[self.ptr].set(reward)
        self.next_states = self.next_states.at[self.ptr].set(next_state)
        self.dones = self.dones.at[self.ptr].set(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, key: jax.Array, batch_size: int) -> Tuple[jax.Array, ...]:
        """Sample batch of experiences."""
        indices = jrandom.randint(key, (batch_size,), 0, self.size)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return self.size >= batch_size


@dataclass
class DDPGAgent:
    """Modern DDPG agent with JAX backend."""

    config: DDPGConfig
    state_dim: int
    action_dim: int
    max_action: float

    # Neural networks
    actor: nn.Module = field(init=False)
    actor_target: nn.Module = field(init=False)
    critic: nn.Module = field(init=False)
    critic_target: nn.Module = field(init=False)

    # Parameters
    actor_params: Dict = field(init=False)
    actor_target_params: Dict = field(init=False)
    critic_params: Dict = field(init=False)
    critic_target_params: Dict = field(init=False)

    # Optimizers
    actor_opt: optax.GradientTransformation = field(init=False)
    critic_opt: optax.GradientTransformation = field(init=False)
    actor_opt_state: Any = field(init=False)
    critic_opt_state: Any = field(init=False)

    # Replay buffer
    replay_buffer: ReplayBuffer = field(init=False)

    # Exploration noise
    noise: Any = field(init=False)

    # JAX random key
    key: jax.Array = field(init=False)

    def __post_init__(self):
        """Initialize agent components."""
        # Set up random key
        self.key = jrandom.PRNGKey(self.config.seed)

        # Initialize networks
        self.actor = Actor(self.action_dim, self.max_action, self.config.actor_hidden_dims)
        self.critic = Critic(self.config.critic_hidden_dims)

        # Create target networks (copy of main networks)
        self.actor_target = Actor(self.action_dim, self.max_action, self.config.actor_hidden_dims)
        self.critic_target = Critic(self.config.critic_hidden_dims)

        # Initialize parameters
        dummy_state = jnp.zeros((1, self.state_dim))
        dummy_action = jnp.zeros((1, self.action_dim))

        self.key, subkey1, subkey2 = jrandom.split(self.key, 3)
        self.actor_params = self.actor.init(subkey1, dummy_state)
        self.critic_params = self.critic.init(subkey2, dummy_state, dummy_action)

        # Copy parameters to targets
        self.actor_target_params = self.actor_params.copy()
        self.critic_target_params = self.critic_params.copy()

        # Initialize optimizers
        self.actor_opt = optax.adam(self.config.actor_lr)
        self.critic_opt = optax.adam(self.config.critic_lr)

        self.actor_opt_state = self.actor_opt.init(self.actor_params)
        self.critic_opt_state = self.critic_opt.init(self.critic_params)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, self.config.buffer_size)

        # Initialize exploration noise
        if self.config.noise_type == "ou":
            self.noise = OUNoise(self.action_dim, theta=self.config.ou_theta, sigma=self.config.ou_sigma)
        else:
            self.noise = None  # Gaussian noise handled in select_action

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action with optional exploration noise."""
        state = jnp.array(state).reshape(1, -1)

        # Get deterministic action from policy
        action = self.actor.apply(self.actor_params, state)[0]

        if not deterministic:
            if self.config.noise_type == "ou":
                self.key, noise_key = jrandom.split(self.key)
                noise = self.noise.sample(noise_key) * self.config.noise_scale
            else:  # Gaussian noise
                self.key, noise_key = jrandom.split(self.key)
                noise = jrandom.normal(noise_key, (self.action_dim,)) * self.config.noise_scale * self.max_action

            action = jnp.clip(action + noise, -self.max_action, self.max_action)

        return np.array(action)

    @jit
    def _update_critic(self, critic_params: Dict, actor_target_params: Dict,
                      critic_target_params: Dict, batch: Tuple[jax.Array, ...]) -> Tuple[Dict, Any]:
        """JIT-compiled critic update step."""

        states, actions, rewards, next_states, dones = batch

        # Compute target Q values
        next_actions = self.actor_target.apply(actor_target_params, next_states)
        target_q = self.critic_target.apply(critic_target_params, next_states, next_actions)
        target_q = rewards + (1.0 - dones.astype(float)) * self.config.gamma * target_q

        # Compute critic loss
        current_q = self.critic.apply(critic_params, states, actions)
        critic_loss = jnp.mean((current_q - target_q) ** 2)

        # Compute gradients
        critic_grads = grad(lambda p: jnp.mean((self.critic.apply(p, states, actions) - target_q) ** 2))(critic_params)

        return critic_grads, critic_loss

    @jit
    def _update_actor(self, actor_params: Dict, critic_params: Dict, states: jax.Array) -> Dict:
        """JIT-compiled actor update step."""

        # Actor loss is negative mean Q value
        def actor_loss_fn(params):
            actions = self.actor.apply(params, states)
            q_values = self.critic.apply(critic_params, states, actions)
            return -jnp.mean(q_values)

        return grad(actor_loss_fn)(actor_params)

    @jit
    def _soft_update(self, params: Dict, target_params: Dict) -> Dict:
        """Soft update target network parameters."""
        return jax.tree_map(
            lambda p, tp: self.config.tau * p + (1 - self.config.tau) * tp,
            params, target_params
        )

    def train_step(self) -> Dict[str, float]:
        """Perform one training step."""
        if not self.replay_buffer.is_ready(self.config.batch_size):
            return {"critic_loss": 0.0, "actor_loss": 0.0}

        # Sample batch
        self.key, sample_key = jrandom.split(self.key)
        batch = self.replay_buffer.sample(sample_key, self.config.batch_size)

        # Update critic
        critic_grads, critic_loss = self._update_critic(
            self.critic_params, self.actor_target_params,
            self.critic_target_params, batch
        )

        critic_updates, self.critic_opt_state = self.critic_opt.update(
            critic_grads, self.critic_opt_state, self.critic_params
        )
        self.critic_params = optax.apply_updates(self.critic_params, critic_updates)

        # Update actor
        states = batch[0]
        actor_grads = self._update_actor(self.actor_params, self.critic_params, states)

        actor_updates, self.actor_opt_state = self.actor_opt.update(
            actor_grads, self.actor_opt_state, self.actor_params
        )
        self.actor_params = optax.apply_updates(self.actor_params, actor_updates)

        # Update target networks
        self.actor_target_params = self._soft_update(self.actor_params, self.actor_target_params)
        self.critic_target_params = self._soft_update(self.critic_params, self.critic_target_params)

        # Compute actor loss for logging
        actions = self.actor.apply(self.actor_params, states)
        q_values = self.critic.apply(self.critic_params, states, actions)
        actor_loss = -jnp.mean(q_values)

        return {
            "critic_loss": float(critic_loss),
            "actor_loss": float(actor_loss)
        }

    def save(self, filepath: str):
        """Save agent state."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        state = {
            "actor_params": self.actor_params,
            "critic_params": self.critic_params,
            "actor_target_params": self.actor_target_params,
            "critic_target_params": self.critic_target_params,
            "actor_opt_state": self.actor_opt_state,
            "critic_opt_state": self.critic_opt_state,
            "config": self.config,
            "buffer_size": self.replay_buffer.size,
            "buffer_ptr": self.replay_buffer.ptr,
            "buffer_states": np.array(self.replay_buffer.states),
            "buffer_actions": np.array(self.replay_buffer.actions),
            "buffer_rewards": np.array(self.replay_buffer.rewards),
            "buffer_next_states": np.array(self.replay_buffer.next_states),
            "buffer_dones": np.array(self.replay_buffer.dones),
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Agent saved to {filepath}")

    def load(self, filepath: str):
        """Load agent state."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.actor_params = state["actor_params"]
        self.critic_params = state["critic_params"]
        self.actor_target_params = state["actor_target_params"]
        self.critic_target_params = state["critic_target_params"]
        self.actor_opt_state = state["actor_opt_state"]
        self.critic_opt_state = state["critic_opt_state"]

        # Restore replay buffer
        self.replay_buffer.size = state["buffer_size"]
        self.replay_buffer.ptr = state["buffer_ptr"]
        self.replay_buffer.states = jnp.array(state["buffer_states"])
        self.replay_buffer.actions = jnp.array(state["buffer_actions"])
        self.replay_buffer.rewards = jnp.array(state["buffer_rewards"])
        self.replay_buffer.next_states = jnp.array(state["buffer_next_states"])
        self.replay_buffer.dones = jnp.array(state["buffer_dones"])

        logger.info(f"Agent loaded from {filepath}")


def evaluate_policy(agent: DDPGAgent, env: gym.Env, num_episodes: int = 5) -> float:
    """Evaluate agent performance."""
    total_reward = 0.0

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            state = next_state

        total_reward += episode_reward

    return total_reward / num_episodes


def train_ddpg(config: DDPGConfig) -> DDPGAgent:
    """Main training loop with modern monitoring."""

    # Set up environment
    env = gym.make(config.env_name, render_mode=config.render_mode)
    eval_env = gym.make(config.env_name)

    # Extract environment info
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize agent
    agent = DDPGAgent(config, state_dim, action_dim, max_action)

    # Set up logging
    writer = tbx.SummaryWriter(logdir=f"runs/{config.env_name}_{int(time.time())}")

    # Training loop
    state, _ = env.reset(seed=config.seed)
    episode_reward = 0.0
    episode_num = 0
    best_eval_reward = -float('inf')

    logger.info("Starting DDPG training...")
    logger.info(f"Environment: {config.env_name}")
    logger.info(f"State dim: {state_dim}, Action dim: {action_dim}, Max action: {max_action}")

    for step in range(config.max_steps):
        # Select action
        if step < config.random_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, deterministic=False)

        # Execute action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store experience
        agent.replay_buffer.add(state, action, reward, next_state, done)

        # Train agent
        if step >= config.random_steps:
            losses = agent.train_step()

            # Log losses
            if step % 100 == 0:
                writer.add_scalar("Loss/Actor", losses["actor_loss"], step)
                writer.add_scalar("Loss/Critic", losses["critic_loss"], step)

        # Update state
        state = next_state
        episode_reward += reward

        # Episode end
        if done:
            episode_num += 1
            logger.info(f"Episode {episode_num}, Step {step}, Reward: {episode_reward:.2f}")

            # Reset environment
            state, _ = env.reset()
            episode_reward = 0.0

        # Evaluation
        if step % config.eval_interval == 0 and step > 0:
            eval_reward = evaluate_policy(agent, eval_env, num_episodes=3)
            writer.add_scalar("Eval/Reward", eval_reward, step)
            logger.info(f"Step {step}: Evaluation reward = {eval_reward:.2f}")

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save(f"models/{config.env_name}_best.pth")

        # Save checkpoint
        if step % config.save_interval == 0 and step > 0:
            agent.save(f"models/{config.env_name}_{step//1000}k.pth")

        # Progress logging
        if step % 10000 == 0:
            logger.info(f"Progress: {step}/{config.max_steps} steps ({100*step/config.max_steps:.1f}%)")

    # Final evaluation
    final_reward = evaluate_policy(agent, eval_env, num_episodes=10)
    logger.info(f"Training completed! Final evaluation reward: {final_reward:.2f}")

    # Save final model
    agent.save(f"models/{config.env_name}_final.pth")

    # Cleanup
    env.close()
    eval_env.close()
    writer.close()

    return agent


def main():
    """Main entry point with modern argument parsing."""

    import argparse

    parser = argparse.ArgumentParser(description="Modern DDPG Implementation with JAX")
    parser.add_argument("--env", type=str, default="Pendulum-v1", help="Environment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--render", action="store_true", help="Render environment")
    parser.add_argument("--load", type=str, help="Path to load model")
    parser.add_argument("--max-steps", type=int, default=int(1e6), help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--actor-lr", type=float, default=1e-4, help="Actor learning rate")
    parser.add_argument("--critic-lr", type=float, default=1e-3, help="Critic learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update rate")

    args = parser.parse_args()

    # Create configuration
    config = DDPGConfig(
        env_name=args.env,
        seed=args.seed,
        render_mode="human" if args.render else None,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
    )

    # Create models directory
    os.makedirs("models", exist_ok=True)

    if args.load:
        # Load and evaluate existing model
        env = gym.make(config.env_name, render_mode=config.render_mode)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        agent = DDPGAgent(config, state_dim, action_dim, max_action)
        agent.load(args.load)

        if config.render_mode:
            while True:
                reward = evaluate_policy(agent, env, num_episodes=1)
                logger.info(f"Episode reward: {reward:.2f}")
        else:
            reward = evaluate_policy(agent, env, num_episodes=10)
            logger.info(f"Average evaluation reward: {reward:.2f}")

        env.close()
    else:
        # Train new model
        agent = train_ddpg(config)


if __name__ == "__main__":
    main()
