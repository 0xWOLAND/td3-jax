# default recipe
default: train

# train td3 with optimized defaults
train env="HalfCheetah-v5":
    python scripts/train.py --env {{env}} --max_timesteps 100000 --eval_freq 20000

# quick training for testing  
train-quick env="HalfCheetah-v5":
    python scripts/train.py --env {{env}} --max_timesteps 50000 --eval_freq 10000

# run tests
test:
    uv run pytest

# plot results
plot env="HalfCheetah-v5" seed="0":
    python -m fasttd3.plotting results/TD3_{{env}}_{{seed}}.npy

# clean build artifacts
clean:
    rm -rf __pycache__ src/fasttd3/__pycache__ tests/__pycache__
    rm -rf .pytest_cache .coverage htmlcov
    rm -rf *.egg-info build dist