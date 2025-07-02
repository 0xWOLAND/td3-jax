# Train FastTD3 on HalfCheetah with vectorized environments
train:
    python main.py --env HalfCheetah-v5 --seed 0 --num_envs 16

# Train with single environment
train-single:
    python main.py --env HalfCheetah-v5 --seed 0 --num_envs 1

# Train without distributional critic (standard TD3)
train-standard:
    python main.py --env HalfCheetah-v5 --seed 0 --distributional false --normalize_obs false

# Train with custom parameters
train-custom env="HalfCheetah-v5" seed="0" num_envs="16":
    python main.py --env {{env}} --seed {{seed}} --num_envs {{num_envs}}

# Evaluate saved model
eval env="HalfCheetah-v5" seed="0":
    python main.py --env {{env}} --seed {{seed}} --load_model default --max_timesteps 0

# Plot results
plot env="HalfCheetah-v5" seed="0":
    python plot.py --file results/TD3_{{env}}_{{seed}}.npy

# Quick test run
test:
    python main.py --env CartPole-v1 --max_timesteps 5000 --eval_freq 1000 --batch_size 64