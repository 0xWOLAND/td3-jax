# default recipe
default: train

# train td3
train env="HalfCheetah-v5":
    python scripts/train.py --env {{env}}

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