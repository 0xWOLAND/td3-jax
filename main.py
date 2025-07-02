import numpy as np
import gymnasium as gym
import argparse
import os
import time
from typing import Optional

import utils
import TD3


def parse_arguments():
    """Parse command line arguments for FastTD3 training."""
    parser = argparse.ArgumentParser(description="FastTD3 - Distributional TD3 with JAX")

    # Environment settings
    parser.add_argument("--env", default="HalfCheetah-v5", help="Environment name")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--num_envs", default=128, type=int, help="Number of parallel environments")

    # Training configuration
    parser.add_argument("--max_timesteps", default=1e6, type=int, help="Total training timesteps")
    parser.add_argument(
        "--start_timesteps", default=5e3, type=int, help="Random exploration timesteps"
    )
    parser.add_argument("--eval_freq", default=10e2, type=int, help="Evaluation frequency")
    parser.add_argument("--batch_size", default=1024, type=int, help="Batch size for training")
    parser.add_argument("--num_updates", default=32, type=int, help="Gradient updates per env step")

    # TD3 hyperparameters
    parser.add_argument("--discount", default=0.99, type=float, help="Discount factor")
    parser.add_argument("--tau", default=0.01, type=float, help="Target network update rate")
    parser.add_argument(
        "--policy_noise", default=0.1, type=float, help="Target policy smoothing noise"
    )
    parser.add_argument("--noise_clip", default=0.5, type=float, help="Noise clipping range")
    parser.add_argument(
        "--policy_freq", default=2, type=int, help="Delayed policy update frequency"
    )

    # Network configuration
    parser.add_argument("--actor_lr", default=1e-3, type=float, help="Actor learning rate")
    parser.add_argument("--critic_lr", default=1e-3, type=float, help="Critic learning rate")
    parser.add_argument("--hidden_dim", default=256, type=int, help="Hidden layer dimension")

    # FastTD3 specific features
    parser.add_argument(
        "--distributional",
        default=True,
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        help="Use distributional critic",
    )
    parser.add_argument(
        "--num_atoms", default=101, type=int, help="Number of atoms for distributional critic"
    )
    parser.add_argument(
        "--v_min", default=-250.0, type=float, help="Min value for distributional critic"
    )
    parser.add_argument(
        "--v_max", default=250.0, type=float, help="Max value for distributional critic"
    )
    parser.add_argument(
        "--normalize_obs",
        default=True,
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        help="Normalize observations",
    )
    parser.add_argument(
        "--lr_schedule",
        default=None,
        choices=[None, "linear", "cosine"],
        help="Learning rate schedule",
    )

    # Model persistence
    parser.add_argument("--policy", default="FastTD3", help="Policy name for saving")
    parser.add_argument("--save_model", action="store_true", help="Save model checkpoints")
    parser.add_argument("--load_model", default="", help="Load pretrained model")

    return parser.parse_args()


def setup_directories(save_model: bool) -> None:
    """Create necessary directories for results and models."""
    os.makedirs("./results", exist_ok=True)
    if save_model:
        os.makedirs("./models", exist_ok=True)


def evaluate_policy(policy, eval_env, eval_episodes: int = 10, obs_normalizer=None) -> float:
    """Evaluate policy performance using vectorized environments.

    Args:
        policy: The policy to evaluate
        eval_env: Vectorized evaluation environment
        eval_episodes: Number of episodes to evaluate
        obs_normalizer: Optional observation normalizer

    Returns:
        Average episode return across all evaluation episodes
    """
    num_eval_envs = eval_env.num_envs
    max_episode_steps = 1000  # Standard for MuJoCo environments

    # Initialize tracking variables
    episode_returns = np.zeros(num_eval_envs)
    done_masks = np.zeros(num_eval_envs, dtype=bool)
    episodes_completed = 0

    states, _ = eval_env.reset()

    # Run evaluation until we complete enough episodes
    for step in range(max_episode_steps * eval_episodes):
        # Get actions for all environments in batch
        if obs_normalizer is not None:
            normalized_states = obs_normalizer.normalize(states)
            actions = policy.select_actions_batch(normalized_states)
        else:
            actions = policy.select_actions_batch(states)

        # Step all environments simultaneously
        next_states, rewards, terminations, truncations, _ = eval_env.step(actions)
        dones = np.logical_or(terminations, truncations)

        # Update episode returns for active environments
        episode_returns = np.where(~done_masks, episode_returns + rewards, episode_returns)

        # Track newly completed episodes
        newly_done = dones & ~done_masks
        episodes_completed += np.sum(newly_done)
        done_masks = np.logical_or(done_masks, dones)

        # Early termination if we have enough completed episodes
        if episodes_completed >= eval_episodes:
            break

        # Reset all environments if all are done
        if done_masks.all():
            episode_returns.fill(0)
            done_masks.fill(False)
            states, _ = eval_env.reset()
        else:
            states = next_states

    # Return average performance of completed episodes
    completed_episodes = episode_returns[done_masks]
    return np.mean(completed_episodes) if len(completed_episodes) > 0 else 0.0


def setup_environment(args):
    """Setup training and evaluation environments."""
    if args.num_envs > 1:
        # Vectorized environments
        env = gym.make_vec(args.env, args.num_envs)
        eval_env = gym.make_vec(args.env, min(args.num_envs, 10))  # Limit eval envs
        env.reset(seed=args.seed)
        eval_env.reset(seed=args.seed + 1000)

        state_dim = env.single_observation_space.shape[0]
        action_dim = env.single_action_space.shape[0]
        max_action = float(env.single_action_space.high[0])
    else:
        # Single environments
        env = gym.make(args.env)
        eval_env = gym.make(args.env)
        env.reset(seed=args.seed)
        eval_env.reset(seed=args.seed + 1000)
        env.action_space.seed(args.seed)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

    return env, eval_env, state_dim, action_dim, max_action


def create_policy(args, state_dim: int, action_dim: int, max_action: float):
    """Create FastTD3 policy with specified configuration."""
    policy_config = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "hidden_dim": args.hidden_dim,
        "distributional": args.distributional,
        "num_atoms": args.num_atoms,
        "v_min": args.v_min,
        "v_max": args.v_max,
        "lr_schedule": args.lr_schedule,
        "total_timesteps": args.max_timesteps,
        "seed": args.seed,
    }
    return TD3.FastTD3(**policy_config)


def train_vectorized(args, env, eval_env, policy, replay_buffer, obs_normalizer):
    """Main vectorized training loop."""
    start_time = time.time()
    evaluations = []

    # Initialize training state
    states, _ = env.reset()
    total_timesteps = 0
    episode_count = 0

    print("Starting vectorized training...")

    while total_timesteps < args.max_timesteps:
        # Print progress every 10k timesteps during early training
        if total_timesteps % 10000 == 0 and total_timesteps > 0:
            elapsed = time.time() - start_time
            sps = total_timesteps / elapsed
            print(
                f"Training progress: {total_timesteps}/{args.max_timesteps} timesteps, SPS: {sps:.1f}"
            )

        # Action selection: exploration vs exploitation
        if total_timesteps < args.start_timesteps:
            actions = env.action_space.sample()
        else:
            # Use batch action selection for efficiency
            if obs_normalizer is not None:
                normalized_states = obs_normalizer.normalize(states)
                actions = policy.select_actions_batch(normalized_states, add_noise=True)
            else:
                actions = policy.select_actions_batch(states, add_noise=True)

        # Environment step
        next_states, rewards, terminations, truncations, _ = env.step(actions)
        dones = np.logical_or(terminations, truncations)

        # Update observation normalizer
        if obs_normalizer is not None and total_timesteps >= args.start_timesteps:
            obs_normalizer.update(states)

        # Store transitions in replay buffer
        replay_buffer.add_batch(
            states, actions, next_states, rewards, terminations.astype(np.float32)
        )

        states = next_states
        total_timesteps += args.num_envs

        # Training updates
        if total_timesteps >= args.start_timesteps:
            for _ in range(args.num_updates):
                # Sample batch and apply observation normalization
                state, action, next_state, reward, not_done = replay_buffer.sample(args.batch_size)
                if obs_normalizer is not None:
                    state = obs_normalizer.normalize(state)
                    next_state = obs_normalizer.normalize(next_state)

                # Perform training step
                batch = (state, action, next_state, reward, not_done)
                policy._train_step(batch)

        # Episode completion logging
        if dones.any():
            episode_count += np.sum(dones)
            if episode_count % 100 == 0:
                elapsed = time.time() - start_time
                sps = total_timesteps / elapsed
                print(f"Episodes: {episode_count}, Timesteps: {total_timesteps}, SPS: {sps:.1f}")

        # Periodic evaluation
        if total_timesteps % args.eval_freq == 0:
            eval_env_to_use = eval_env if args.num_envs > 10 else env
            eval_reward = evaluate_policy(
                policy, eval_env_to_use, eval_episodes=10, obs_normalizer=obs_normalizer
            )
            evaluations.append(eval_reward)

            elapsed = time.time() - start_time
            sps = total_timesteps / elapsed
            print(
                f"\n[Eval] Timesteps: {total_timesteps}, Time: {int(elapsed)}s, "
                f"Reward: {eval_reward:.3f}, SPS: {sps:.1f}\n"
            )

            # Save results and model
            np.save(f"./results/{args.policy}_{args.env}_{args.seed}", evaluations)
            if args.save_model:
                policy.save(f"./models/{args.policy}_{args.env}_{args.seed}")

    return evaluations


def main():
    """Main training function."""
    args = parse_arguments()

    if args.num_updates == 1:
        args.num_updates = min(64, max(1, args.num_envs // 2))

    # Print configuration
    print("=" * 50)
    print(f"FastTD3 Training Configuration")
    print(f"Environment: {args.env}")
    print(f"Seed: {args.seed}")
    print(f"Parallel Environments: {args.num_envs}")
    print(f"Training Updates per Step: {args.num_updates}")
    print(f"Distributional Critic: {args.distributional}")
    print(f"Observation Normalization: {args.normalize_obs}")
    print("=" * 50)

    # Setup directories and random seed
    setup_directories(args.save_model)
    np.random.seed(args.seed)

    # Setup environments
    env, eval_env, state_dim, action_dim, max_action = setup_environment(args)

    # Initialize components
    obs_normalizer = (
        utils.EmpiricalNormalization(shape=(state_dim,)) if args.normalize_obs else None
    )
    policy = create_policy(args, state_dim, action_dim, max_action)

    # Load pretrained model if specified
    if args.load_model:
        model_path = (
            f"{args.policy}_{args.env}_{args.seed}"
            if args.load_model == "default"
            else args.load_model
        )
        policy.load(f"./models/{model_path}")
        print(f"Loaded model from ./models/{model_path}")

    # Initialize replay buffer
    replay_buffer = (
        utils.VectorizedReplayBuffer(args.num_envs, state_dim, action_dim)
        if args.num_envs > 1
        else utils.ReplayBuffer(state_dim, action_dim)
    )

    # Run training
    start_time = time.time()
    if args.num_envs > 1:
        evaluations = train_vectorized(args, env, eval_env, policy, replay_buffer, obs_normalizer)
        env.close()
        if args.num_envs > 10:
            eval_env.close()
    else:
        # Single environment fallback (simplified)
        evaluations = train_single_env(args, env, eval_env, policy, replay_buffer, obs_normalizer)
        env.close()
        eval_env.close()

    # Training summary
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.1f}s")
    if evaluations:
        print(f"Final evaluation reward: {evaluations[-1]:.3f}")
        print(f"Best evaluation reward: {max(evaluations):.3f}")


def train_single_env(args, env, eval_env, policy, replay_buffer, obs_normalizer):
    """Simplified single environment training for compatibility."""
    evaluations = []
    state, _ = env.reset()
    episode_reward = 0
    episode_count = 0

    for t in range(int(args.max_timesteps)):
        # Action selection
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            if obs_normalizer is not None:
                normalized_state = obs_normalizer.normalize(state.reshape(1, -1)).squeeze()
                action = policy.select_action(normalized_state, add_noise=True)
            else:
                action = policy.select_action(state, add_noise=True)

        # Environment step
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store transition and update normalizer
        if obs_normalizer is not None:
            obs_normalizer.update(state.reshape(1, -1))
        replay_buffer.add(state, action, next_state, reward, float(terminated))

        state = next_state
        episode_reward += reward

        # Training
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        # Episode completion
        if done:
            if episode_count % 100 == 0:
                print(f"Episode {episode_count + 1}: Reward = {episode_reward:.3f}")
            state, _ = env.reset()
            episode_reward = 0
            episode_count += 1

        # Evaluation
        if (t + 1) % args.eval_freq == 0:
            eval_reward = evaluate_single_episode(policy, eval_env, obs_normalizer)
            evaluations.append(eval_reward)
            print(f"Step {t + 1}: Evaluation = {eval_reward:.3f}")

    return evaluations


def evaluate_single_episode(policy, eval_env, obs_normalizer):
    """Evaluate policy on single episode."""
    state, _ = eval_env.reset()
    episode_reward = 0
    done = False

    while not done:
        if obs_normalizer is not None:
            normalized_state = obs_normalizer.normalize(state.reshape(1, -1)).squeeze()
            action = policy.select_action(normalized_state)
        else:
            action = policy.select_action(state)

        state, reward, terminated, truncated, _ = eval_env.step(action)
        episode_reward += reward
        done = terminated or truncated

    return episode_reward


if __name__ == "__main__":
    main()
