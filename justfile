
# Install dependencies
install:
    uv pip install -e ".[dev]"

# Run tests
test:
    pytest

# Run linting
lint:
    ruff check .

# Run all experiments (reproduce results)
experiments:
    @for seed in 0 1 2 3 4; do \
        for env in Pendulum-v1 MountainCarContinuous-v0; do \
            uv run python main.py --policy TD3 --env $$env --seed $$seed --save_model; \
        done; \
    done