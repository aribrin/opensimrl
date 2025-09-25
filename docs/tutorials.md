# OpenSimRL Tutorials

## 1) Basic Training (Hydra CLI)
Run the default PPO training on CartPole using Hydra configs:
```bash
python -m opensimrl.train
```

Override hyperparameters from the CLI:
```bash
python -m opensimrl.train algorithm.epochs=5 algorithm.steps_per_epoch=1000
```

## 2) Switch Environments
Train on the SimpleGridWorld environment:
```bash
python -m opensimrl.train env=gridworld env.size=5
```

Train on any Gymnasium environment by ID (default is CartPole-v1):
```bash
python -m opensimrl.train env.name=CartPole-v1
```

## 3) Select a Logging Backend (Console, MLflow, or W&B)
By default, runs log to the console and save checkpoints locally. You can switch the experiment logger at runtime:

- MLflow:
```bash
python -m opensimrl.train logger=mlflow logger.experiment_name=cartpole
# Optional: set configs/logger/mlflow.yaml: tracking_uri: http://localhost:5000
```

- Weights & Biases:
```bash
wandb login
python -m opensimrl.train logger=wandb logger.project_name=opensimrl logger.run_name=ppo
```

- SAC via unified entrypoint:
```bash
python -m opensimrl.train algorithm=sac env.name=Pendulum-v1
```

## 4) Multirun Sweeps (Hydra)
Run multiple seeds locally:
```bash
python -m opensimrl.train -m seed=0,1,2
```

Sweep a hyperparameter:
```bash
python -m opensimrl.train -m algorithm.pi_lr=3e-4,1e-3
```

## 5) Notebook Example (Local)
You can still run training in notebooks without Hydra by calling a helper script.
For example, the repository includes a CartPole training utility:

```python
from opensimrl.train_ppo import train_ppo_cartpole

# Console logging (default)
history = train_ppo_cartpole(epochs=5, steps_per_epoch=1000, seed=0, logger_kind="console")

# Or use W&B
# history = train_ppo_cartpole(epochs=5, steps_per_epoch=1000, seed=0, logger_kind="wandb")
```

Plot the training curve:
```python
import matplotlib.pyplot as plt

plt.plot(history)
plt.title("PPO Training on CartPole (Mean Return per Epoch)")
plt.xlabel("Epoch")
plt.ylabel("Mean Episode Return")
plt.show()
```

## 6) Running Tests
```bash
pytest tests/
```

Notes:
- Hydra composes configurations from the `configs/` directory and creates a unique working directory per run. Checkpoints and artifacts will be stored under each run directory by default.
- PPO supports both discrete and continuous action spaces.
- Logging backends are selectable: console (default), MLflow, or W&B. If a backend is unavailable, the run safely falls back to console logging. Metrics use TotalEnvInteracts as the logging step.
