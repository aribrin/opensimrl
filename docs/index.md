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

## Select a logging backend (MLflow or W&B)

By default, runs print to the console and save checkpoints locally. You can switch the experiment logger at runtime:

- MLflow
```bash
python -m opensimrl.train logger=mlflow logger.experiment_name=cartpole
# Optional: configure a tracking server in configs/logger/mlflow.yaml (tracking_uri)
```

- Weights & Biases
```bash
wandb login
python -m opensimrl.train logger=wandb logger.project_name=opensimrl logger.run_name=ppo
```

You can also select the backend when using the module CLIs:
```bash
# PPO (module CLI)
python -m opensimrl.algorithms.ppo --logger wandb --env CartPole-v1 --epochs 1 --steps 200

# SAC (module CLI)
python -m opensimrl.algorithms.sac --logger mlflow --env Pendulum-v1 --epochs 1
```

Notes:
- Metrics use TotalEnvInteracts as the logging step for consistent x-axes across backends.
- If MLflow or W&B are unavailable, the run safely falls back to console logging.

Check out our [tutorials](tutorials.md) for more examples.
