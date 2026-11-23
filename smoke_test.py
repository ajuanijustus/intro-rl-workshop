"""Run before the session:  python smoke_test.py   -> should end with ALL GOOD"""
import sys
print("Python", sys.version.split()[0])
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10 or newer is needed. See the setup guide.")
import numpy, gymnasium as gym, pygame  # noqa: F401
print("gymnasium", gym.__version__, "| numpy", numpy.__version__)
cliff = max((k for k in gym.registry if k.split("-v")[0] == "CliffWalking"), key=lambda k: int(k.split("-v")[1]))
env = gym.make(cliff, render_mode="human")
env.reset(seed=0)
for _ in range(20):
    env.step(env.action_space.sample())
env.close()
print("A window with a grid should have flashed up. ALL GOOD")
