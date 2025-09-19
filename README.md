# OpenSimRL 🚀

A scalable reinforcement learning simulation framework built with PyTorch, designed for research and production.

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