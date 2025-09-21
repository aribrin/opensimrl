# OpenSimRL Documentation

Welcome to the OpenSimRL documentation! This framework provides tools for running large-scale, distributed reinforcement learning simulations.

## Features
- Modular architecture for pluggable RL environments & algorithms
- Support for cutting-edge algorithms (PPO, SAC, Dreamer, etc.)
- Distributed rollout collection and training
- Experiment logging and visualization
- Kubernetes/Argo/Ray orchestration

## Getting Started
Install OpenSimRL (or install dependencies locally):
```bash
pip install opensimrl
# or, from source:
pip install -r requirements.txt
```

Run PPO training with Hydra:
```bash
python -m opensimrl.train
# override hyperparameters
python -m opensimrl.train algorithm.epochs=5 algorithm.steps_per_epoch=1000
# switch environment
python -m opensimrl.train env=gridworld env.size=5
# multirun sweep over seeds
python -m opensimrl.train -m seed=0,1,2
```

Check out our [tutorials](tutorials.md) for more examples.
