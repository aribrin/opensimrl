import subprocess
import sys
import os
import shlex


def run_cmd(cmd: str, timeout: int = 120):
    """
    Run a shell command and return (returncode, stdout, stderr).
    Uses the same Python interpreter to ensure module resolution.
    """
    # Replace 'python' with current interpreter for reliability
    if cmd.startswith("python "):
        cmd = cmd.replace("python", shlex.quote(sys.executable), 1)
    elif cmd.startswith("python -m "):
        cmd = cmd.replace("python", shlex.quote(sys.executable), 1)

    env = os.environ.copy()
    # Helpful during CI/debugging of Hydra errors
    env.setdefault("HYDRA_FULL_ERROR", "1")

    proc = subprocess.run(
        cmd,
        shell=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        universal_newlines=True,
        cwd=os.getcwd(),
    )
    return proc.returncode, proc.stdout


def test_unified_ppo_smoke():
    """
    Smoke test PPO via the unified Hydra entrypoint.
    Keeps settings very small to be CPU-friendly.
    """
    cmd = "python -m opensimrl.train algorithm.epochs=1 algorithm.steps_per_epoch=200"
    code, out = run_cmd(cmd, timeout=180)
    # For debugging on failure
    if code != 0:
        print(out)
    assert code == 0, f"PPO unified entrypoint failed with exit code {code}"
    # Expect PPO-style metrics in output
    assert "Epoch" in out and "LossV" in out and "logger:console" in out


def test_unified_sac_smoke():
    """
    Smoke test SAC via the unified Hydra entrypoint.
    Uses Pendulum-v1 and tiny settings to keep runtime low.
    """
    cmd = (
        "python -m opensimrl.train "
        "algorithm=sac env.name=Pendulum-v1 "
        "algorithm.epochs=1 algorithm.steps_per_epoch=200 "
        "algorithm.start_steps=0 algorithm.update_after=0 algorithm.update_every=1"
    )
    code, out = run_cmd(cmd, timeout=240)
    # For debugging on failure
    if code != 0:
        print(out)
    assert code == 0, f"SAC unified entrypoint failed with exit code {code}"
    # Expect SAC-style metrics in output
    # 'LossQ' and 'LossPi' are printed in consolidated epoch output
    assert "Epoch" in out and "LossQ" in out and "LossPi" in out and "logger:console" in out
