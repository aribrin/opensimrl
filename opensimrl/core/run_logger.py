from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


class RunLogger:
    """Unified logging interface across MLflow, Weights & Biases, or console-only."""

    def start(self, experiment_name: str, run_name: str, extras: Optional[Dict[str, Any]] = None) -> None:
        raise NotImplementedError

    def log_params(self, params: Dict[str, Any]) -> None:
        raise NotImplementedError

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        raise NotImplementedError

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        raise NotImplementedError

    def end(self) -> None:
        raise NotImplementedError


@dataclass
class ConsoleRunLogger(RunLogger):
    """No-op logger with console printing (safe default)."""

    experiment_name: str = "opensimrl"
    run_name: str = "run"

    def start(self, experiment_name: str, run_name: str, extras: Optional[Dict[str, Any]] = None) -> None:
        self.experiment_name = experiment_name or self.experiment_name
        self.run_name = run_name or self.run_name
        print(f"[logger:console] start: experiment={self.experiment_name} run={self.run_name}")

    def log_params(self, params: Dict[str, Any]) -> None:
        # Keep concise to avoid spam
        print(f"[logger:console] params: {list(params.keys())}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        # Algorithms already print a rich log_line; we keep a compact echo here
        step_str = f" step={step}" if step is not None else ""
        log_line = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"[logger:console] metrics{step_str}: {log_line}")

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        apath = f"{artifact_path}/" if artifact_path else ""
        print(f"[logger:console] artifact: {apath}{path}")

    def end(self) -> None:
        print(f"[logger:console] end: experiment={self.experiment_name} run={self.run_name}")


@dataclass
class MLflowRunLogger(RunLogger):
    experiment_name: str = "opensimrl"
    run_name: str = "run"
    tracking_uri: Optional[str] = None
    _mlflow: Any = None
    _active: bool = False

    def __post_init__(self) -> None:
        try:
            import mlflow  # type: ignore
            self._mlflow = mlflow
        except Exception as e:
            print(f"[logger:mlflow] MLflow not available: {e}")
            self._mlflow = None

    def start(self, experiment_name: str, run_name: str, extras: Optional[Dict[str, Any]] = None) -> None:
        if self._mlflow is None:
            print("[logger:mlflow] falling back to console; mlflow unavailable")
            return
        self.experiment_name = experiment_name or self.experiment_name
        self.run_name = run_name or self.run_name
        if self.tracking_uri:
            try:
                self._mlflow.set_tracking_uri(self.tracking_uri)
            except Exception as e:
                print(f"[logger:mlflow] failed to set tracking_uri: {e}")
        try:
            self._mlflow.set_experiment(self.experiment_name)
            self._mlflow.start_run(run_name=self.run_name)
            self._active = True
        except Exception as e:
            print(f"[logger:mlflow] start failed: {e}")

    def log_params(self, params: Dict[str, Any]) -> None:
        if self._mlflow is None or not self._active:
            return
        try:
            self._mlflow.log_params(params)
        except Exception as e:
            print(f"[logger:mlflow] log_params failed: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self._mlflow is None or not self._active:
            return
        try:
            if step is not None:
                self._mlflow.log_metrics(metrics, step=step)
            else:
                self._mlflow.log_metrics(metrics)
        except Exception as e:
            print(f"[logger:mlflow] log_metrics failed: {e}")

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        if self._mlflow is None or not self._active:
            return
        try:
            if artifact_path:
                self._mlflow.log_artifact(path, artifact_path=artifact_path)
            else:
                self._mlflow.log_artifact(path)
        except Exception as e:
            print(f"[logger:mlflow] log_artifact failed: {e}")

    def end(self) -> None:
        if self._mlflow is None or not self._active:
            return
        try:
            self._mlflow.end_run()
        except Exception as e:
            print(f"[logger:mlflow] end failed: {e}")
        finally:
            self._active = False


@dataclass
class WandbRunLogger(RunLogger):
    project_name: str = "opensimrl"
    run_name: str = "run"
    _wandb: Any = None
    _active: bool = False

    def __post_init__(self) -> None:
        try:
            import wandb  # type: ignore
            self._wandb = wandb
        except Exception as e:
            print(f"[logger:wandb] W&B not available: {e}")
            self._wandb = None

    def start(self, experiment_name: str, run_name: str, extras: Optional[Dict[str, Any]] = None) -> None:
        if self._wandb is None:
            print("[logger:wandb] falling back to console; wandb unavailable")
            return
        self.run_name = run_name or self.run_name
        try:
            self._wandb.init(
                project=self.project_name,
                name=self.run_name,
                config=extras or {},
                reinit=True,
            )
            self._active = True
        except Exception as e:
            print(f"[logger:wandb] start failed: {e}")

    def log_params(self, params: Dict[str, Any]) -> None:
        if self._wandb is None or not self._active:
            return
        try:
            self._wandb.config.update(params, allow_val_change=True)
        except Exception as e:
            print(f"[logger:wandb] log_params failed: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self._wandb is None or not self._active:
            return
        try:
            if step is not None:
                self._wandb.log(metrics, step=step)
            else:
                self._wandb.log(metrics)
        except Exception as e:
            print(f"[logger:wandb] log_metrics failed: {e}")

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        # For simplicity, log file as an image/artifact-like file under the key if provided
        # Users can use W&B Artifacts explicitly in higher-level scripts if needed.
        if self._wandb is None or not self._active:
            return
        try:
            # W&B "artifacts" API is more involved; a simple file upload via save is adequate here
            self._wandb.save(path, policy="now")
        except Exception as e:
            print(f"[logger:wandb] log_artifact failed: {e}")

    def end(self) -> None:
        if self._wandb is None or not self._active:
            return
        try:
            self._wandb.finish()
        except Exception as e:
            print(f"[logger:wandb] end failed: {e}")
        finally:
            self._active = False


def create_run_logger(kind: str, **kwargs: Any) -> RunLogger:
    """Factory for RunLogger.

    kind: 'mlflow' | 'wandb' | 'console'
    kwargs are backend-specific:
      - mlflow: experiment_name, run_name, tracking_uri
      - wandb: project_name, run_name
      - console: experiment_name, run_name
    """
    k = (kind or "console").lower()
    if k == "mlflow":
        return MLflowRunLogger(
            experiment_name=kwargs.get("experiment_name", "opensimrl"),
            run_name=kwargs.get("run_name", "run"),
            tracking_uri=kwargs.get("tracking_uri"),
        )
    if k == "wandb":
        return WandbRunLogger(
            project_name=kwargs.get("project_name", "opensimrl"),
            run_name=kwargs.get("run_name", "run"),
        )
    # default
    return ConsoleRunLogger(
        experiment_name=kwargs.get("experiment_name", "opensimrl"),
        run_name=kwargs.get("run_name", "run"),
    )
