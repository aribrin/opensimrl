import wandb
import matplotlib.pyplot as plt
import numpy as np


class ExperimentLogger:
    """Handles logging to W&B and local files."""

    def __init__(self, project_name="opensimrl", experiment_name=None, config=None):
        self.use_wandb = True
        try:
            wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                reinit=True
            )
        except Exception as e:
            print(f"W&B initialization failed: {e}")
            self.use_wandb = False
    
    def log_metrics(self, metrics, step=None):
        """Logs metrics to W&B and print to console."""
        if self.use_wandb:
            wandb.log(metrics, step=step)

        # Always print to console
        step_str = f"Step {step}: " if step is not None else ""
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"{step_str}{metrics_str}")

    def log_plot(self, figure, name):
        """Logs a matplotlib figure to W&B."""
        if self.use_wandb:
            wandb.log({name: wandb.Image(figure)})
        plt.savefig(f"{name}.png")

    def finish(self):
        """Finalize the W&B run."""
        if self.use_wandb:
            wandb.finish()