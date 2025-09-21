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

## 3) Enable MLflow (optional)
If MLflow is installed, you can select the MLflow logger config:
```bash
python -m opensimrl.train logger=mlflow logger.experiment_name=cartpole
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
from opensimrl.train_cartpole import train_cartpole

history = train_cartpole(epochs=5, steps_per_epoch=1000, seed=0)
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
- The PPO implementation supports both discrete and continuous action spaces and includes optional MLflow logging.
