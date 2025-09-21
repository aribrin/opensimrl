# OpenSimRL 🚀

A scalable reinforcement learning simulation framework built with PyTorch, designed for research and production.

## Attribution

This project includes adapted implementations of Proximal Policy Optimization (PPO) based on OpenAI Spinning Up in Deep RL.

- Website: https://spinningup.openai.com
- Repository: https://github.com/openai/spinningup

Files adapted (so far):
- opensimrl/core/ppo_core.py
- opensimrl/algorithms/ppo.py

See THIRD_PARTY_NOTICES.md for license details and the full MIT license text from OpenAI Spinning Up.

## Features

- 🧠 **Modern RL Algorithms**: PPO, SAC, Dreamer (planned)
- 🌐 **Distributed Training**: Ray integration for scaling
- ☁️ **Cloud Native**: Kubernetes & Argo Workflows support
- 📊 **Experiment Tracking**: W&B and optional MLflow integration
- 🔧 **Modular Design**: Easy to extend and customize
- ⚙️ **Config-Driven**: Hydra-based configuration, CLI overrides, and multirun

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Default training run (CartPole via Hydra configs)
python -m opensimrl.train

# Override hyperparameters (examples)
python -m opensimrl.train algorithm.epochs=5 algorithm.steps_per_epoch=1000

# Switch environment to GridWorld (size configurable)
python -m opensimrl.train env=gridworld env.size=5

# Enable MLflow-configured run (if MLflow is installed)
python -m opensimrl.train logger=mlflow logger.experiment_name=cartpole

# Multirun across seeds (runs 3 jobs locally)
python -m opensimrl.train -m seed=0,1,2

# Run tests
pytest tests/
```

## Documentation

- Website: See docs/ with MkDocs configuration (mkdocs.yml)
- Tutorials: docs/tutorials.md
- API Reference: docs/api_reference.md
- References & Credits: docs/references.md

## Notes

- Hydra composes configs from `configs/` and creates a unique working directory per run. Model checkpoints and artifacts will be saved under that run directory by default.
- The PPO training loop supports both discrete and continuous action spaces and includes optional MLflow logging (safe to ignore if MLflow is not installed).
