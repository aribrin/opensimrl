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
- 📊 **Experiment Tracking**: W&B integration
- 🔧 **Modular Design**: Easy to extend and customize

## Quick Start
```bash
# Install
pip install -r requirements.txt

# Train PPO on GridWorld
python -m opensimrl.train

# Run tests
pytest tests/
