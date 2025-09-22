# OpenSimRL API Reference

## Environments

### SimpleGridWorld
```python
class SimpleGridWorld(size: int = 5)
```
- reset() -> (obs, info)
- step(action) -> (obs, reward, terminated, truncated, info)
- render() -> None

## Algorithms

### PPO (function)
```python
from opensimrl.algorithms.ppo import ppo

def ppo(
    env_fn,
    actor_critic=core.MLPActorCritic,
    ac_kwargs: dict = {},
    seed: int = 0,
    steps_per_epoch: int = 4000,
    epochs: int = 50,
    gamma: float = 0.99,
    clip_ratio: float = 0.2,
    pi_lr: float = 3e-4,
    vf_lr: float = 1e-3,
    train_pi_iters: int = 80,
    train_v_iters: int = 80,
    lam: float = 0.97,
    max_ep_len: int = 1000,
    target_kl: float = 0.01,
    logger_kind: str = "console",        # "console" | "mlflow" | "wandb"
    logger_kwargs: dict = {},            # backend-specific kwargs
    save_freq: int = 10,
    return_history: bool = False,
)
```
- env_fn: Callable[[], gymnasium.Env]
- actor_critic: module implementing pi(obs, act)/v(obs)
- ac_kwargs: e.g., hidden_sizes=[64,64]
- logger_kind: select logging backend
- logger_kwargs:
  - console: {experiment_name, run_name}
  - mlflow: {experiment_name, run_name, tracking_uri?}
  - wandb: {project_name, run_name}
- return: if return_history=True, returns list[float] of per-epoch mean returns

CLI (module):
```bash
python -m opensimrl.algorithms.ppo --logger console|mlflow|wandb --env CartPole-v1 --epochs 1 --steps 200
```

### SAC (function)
```python
from opensimrl.algorithms.sac import sac

def sac(
    env_fn,
    actor_critic=core.MLPActorCritic,
    ac_kwargs: dict = {},
    seed: int = 0,
    steps_per_epoch: int = 4000,
    epochs: int = 100,
    replay_size: int = int(1e6),
    gamma: float = 0.99,
    polyak: float = 0.995,
    lr: float = 1e-3,
    alpha: float = 0.2,
    batch_size: int = 100,
    start_steps: int = 10000,
    update_after: int = 1000,
    update_every: int = 50,
    num_test_episodes: int = 10,
    max_ep_len: int = 1000,
    logger_kind: str = "console",        # "console" | "mlflow" | "wandb"
    logger_kwargs: dict = {},            # backend-specific kwargs
    save_freq: int = 1,
)
```
- Supports continuous (Box) action spaces
- Uses deterministic test episodes per epoch for evaluation

CLI (module):
```bash
python -m opensimrl.algorithms.sac --logger console|mlflow|wandb --env Pendulum-v1 --epochs 1
```

## Training Entry Points

### Hydra entrypoint (recommended)
```python
python -m opensimrl.train
```
- Config root: `configs/`
- Defaults (in configs/config.yaml):
  - algorithm: ppo
  - env: cartpole
  - logger: console
- Logger selection at runtime:
  - MLflow: `logger=mlflow logger.experiment_name=<name> [logger.tracking_uri=<uri>]`
  - W&B:   `logger=wandb logger.project_name=<project> logger.run_name=<run>`

### Helper scripts
- `opensimrl/train_ppo.py::train_ppo_cartpole(..., logger_kind="console")`
- `opensimrl/train_sac.py::train_sac_pendulum(..., logger_kind="console")`
- Both pass `logger_kind` and `logger_kwargs` through to the algorithms.

## Core Utilities

### RunLogger (unified logging)
```python
from opensimrl.core.run_logger import RunLogger, create_run_logger
```
Backends:
- ConsoleRunLogger (default): console prints, safe fallback
- MLflowRunLogger: experiment/run, params, metrics (step=TotalEnvInteracts), artifacts
- WandbRunLogger: project/run, params via config.update, metrics, simple file save

Factory:
```python
logger = create_run_logger(
    kind,                # "console" | "mlflow" | "wandb"
    experiment_name=..., # console/mlflow
    run_name=...,        # console/mlflow/wandb
    tracking_uri=...,    # mlflow only
    project_name=...,    # wandb only
)
```
Common methods:
- start(experiment_name, run_name, extras=None)
- log_params(dict)
- log_metrics(dict, step=None)
- log_artifact(path, artifact_path=None)
- end()

### ExperimentLogger (legacy)
```python
from opensimrl.core.logger import ExperimentLogger
```
A W&B-centric utility retained for backward compatibility and notebook use. The primary training loops use the unified RunLogger API.

## Notes
- Metrics are logged with `TotalEnvInteracts` as the step for consistent x-axes across backends.
- If a backend is unavailable or misconfigured, logging safely falls back to console behavior while training continues.
