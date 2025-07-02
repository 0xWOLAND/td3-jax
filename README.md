# FastTD3 JAX Implementation

A clean and efficient implementation of FastTD3 (Distributional Twin Delayed Deep Deterministic Policy Gradient) in JAX/Flax.

## Features

- **Distributional Critic**: Uses categorical distribution (C51) for improved value estimation
- **Observation Normalization**: Empirical normalization for stable training
- **Learning Rate Scheduling**: Linear and cosine decay schedules
- **Vectorized Environment Support**: Uses Gymnasium's built-in `gym.make_vec` for efficient parallel data collection
- **JAX/Flax Implementation**: Fast, JIT-compiled training
- **Clean, minimal codebase**

## Installation

```bash
pip install -r requirements.txt
```

Or using uv:
```bash
uv sync
```

## Usage

### Basic Training

```bash
python main.py --env HalfCheetah-v5 --seed 0
```

### FastTD3 Training with Vectorized Environments

```bash
python main.py --env HalfCheetah-v5 --num_envs 16 --num_updates 4 --batch_size 512
```

### Key Arguments

**Basic Arguments:**
- `--env`: Environment name (default: HalfCheetah-v5)
- `--seed`: Random seed (default: 0)
- `--max_timesteps`: Total training timesteps (default: 1e6)
- `--start_timesteps`: Random exploration timesteps (default: 25e3)
- `--batch_size`: Batch size for training (default: 256)
- `--num_envs`: Number of parallel environments (default: 1)
- `--num_updates`: Gradient updates per env step (default: 1)

**FastTD3 Specific:**
- `--distributional`: Use distributional critic (default: True)
- `--num_atoms`: Number of atoms for categorical distribution (default: 101)
- `--v_min`, `--v_max`: Value range for distribution (default: -250, 250)
- `--normalize_obs`: Normalize observations (default: True)
- `--lr_schedule`: Learning rate schedule: 'linear', 'cosine' or None
- `--hidden_dim`: Hidden layer dimension (default: 256)

**Training Parameters:**
- `--discount`: Discount factor (default: 0.99)
- `--tau`: Target network update rate (default: 0.005)
- `--policy_noise`: Noise for target policy smoothing (default: 0.2)
- `--noise_clip`: Noise clipping range (default: 0.5)
- `--policy_freq`: Delayed policy updates (default: 2)
- `--actor_lr`, `--critic_lr`: Learning rates (default: 3e-4)

### Example Commands

**Standard TD3 (without distributional critic):**
```bash
python main.py --env HalfCheetah-v5 --no-distributional --no-normalize_obs
```

**FastTD3 with all features:**
```bash
python main.py --env HalfCheetah-v5 --num_envs 16 --num_updates 4 \
    --batch_size 512 --hidden_dim 512 --lr_schedule linear
```

## Files

- `TD3.py`: Core FastTD3 algorithm with distributional critic
- `main.py`: Training script with observation normalization and vectorized env support
- `utils.py`: Replay buffer, empirical normalization, and utilities
- `plot.py`: Plotting utilities

## Key FastTD3 Features Implemented

1. **Distributional Critic (C51)**:
   - Categorical distribution over fixed support
   - Configurable number of atoms and value range
   - Proper distributional Bellman update with projection

2. **Observation Normalization**:
   - Running mean/variance estimation
   - Numerically stable updates using Welford's algorithm
   - Optional centering and scaling

3. **Enhanced Training**:
   - Learning rate scheduling (linear/cosine decay)
   - Vectorized environment support for efficient data collection
   - Multiple gradient updates per environment step

## Results

Results are saved to `./results/` and models to `./models/`.