# OpenSimRL API Reference

## Environments
### `SimpleGridWorld`
```python
class SimpleGridWorld(size=5)
```
- `reset()`: Reset environment to initial state
- `step(action)`: Take action in environment
- `render()`: Render current state

## Algorithms
### `SimplePPO`
```python
class SimplePPO(observation_dim, action_dim)
```
- `get_action(observation)`: Select action based on observation
- `update(observations, actions, rewards, old_probabilities)`: Update policy

## Core Utilities
### `ExperimentLogger`
```python
class ExperimentLogger(project_name, experiment_name, config)
```
- `log_metrics(metrics, step)`: Log metrics to W&B and console
- `log_plot(figure, name)`: Log matplotlib figure
- `finish()`: Finalize W&B run
