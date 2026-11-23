"""Run before the session:  python smoke_test.py   -> should end with ALL GOOD"""
import sys

print("Python", sys.version.split()[0])
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10 or newer is needed. See the setup guide.")

try:
    import numpy
    import gymnasium as gym
    import pygame  # noqa: F401  (draws the game window)
except ImportError as e:
    raise SystemExit(f"Missing package: {e.name}. Activate your environment, then run:\n"
                     f"    python -m pip install -r requirements.txt") from None
print("gymnasium", gym.__version__, "| numpy", numpy.__version__)

try:
    import gridlab  # noqa: F401  (needed by missions.py)
except ImportError:
    print("Note: gridlab.py isn't in this folder. Put it next to missions.py before the missions.")

cliff = max((k for k in gym.registry if k.split("-v")[0] == "CliffWalking"), key=lambda k: int(k.split("-v")[1]))
try:
    env = gym.make(cliff, render_mode="human")
    env.reset(seed=0)
    for _ in range(20):
        env.step(env.action_space.sample())
    env.close()
except Exception as e:  # no display: WSL, a remote machine, Colab
    print(f"Couldn't open a window ({str(e)[:60] or type(e).__name__}). That's fine: set WATCH = False in the scripts.")
    print("ALL GOOD (without a window)")
else:
    print("A window with a grid should have flashed up. ALL GOOD")
