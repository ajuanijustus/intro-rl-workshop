"""Q-learning on the Cliff Walker grid from the talk. Edit the TWEAK block, then run:  python q_learn.py"""
import time
import numpy as np
import gymnasium as gym

# ====================== TWEAK THESE ======================
# My map for FrozenLake: S=start, F=frozen, H=hole, G=goal. Rows must be equal length.
MY_MAP = ["SFFF",
          "FHFH",
          "FFFH",
          "HFFG"]
ENV = "CliffWalking"   # "CliffWalking", "FrozenLake" or "Taxi"
ENV_KWARGS = {}        # e.g. FrozenLake: {"is_slippery": False, "desc": MY_MAP}
EPISODES = 500         # how many games the agent practises
ALPHA = 0.1            # learning rate: how much each new experience overwrites the old estimate
GAMMA = 0.99           # discount: how much future reward matters (0 = only now, ~1 = long-term)
EPSILON = 0.1          # exploration: chance of a random move instead of the best-known move
STEP_BONUS = 0.0       # added to EVERY step's reward
WATCH = True           # open a window and watch the trained agent (set False on Colab/WSL)
SEED = 0
# =========================================================


def latest_id(name):
    ids = [k for k in gym.registry if k.split("-v")[0] == name]
    if not ids:
        raise SystemExit(f"No environment called {name!r}. Check the spelling.")
    return max(ids, key=lambda k: int(k.split("-v")[1]))


class StepBonus(gym.Wrapper):
    def step(self, action):
        s, r, terminated, truncated, info = self.env.step(action)
        return s, r + STEP_BONUS, terminated, truncated, info


env_id = latest_id(ENV)
env = StepBonus(gym.make(env_id, **ENV_KWARGS))
rng = np.random.default_rng(SEED)
n_states, n_actions = env.observation_space.n, env.action_space.n
Q = np.zeros((n_states, n_actions))


def choose(s, eps):
    if rng.random() < eps:
        return int(rng.integers(n_actions))           # explore
    best = np.flatnonzero(Q[s] == Q[s].max())
    return int(rng.choice(best))                      # exploit (break ties randomly)


if ENV == "CliffWalking":
    print("Cliff Walker scoring here: -1 for every step, -100 for falling off (and you go back to the start).\n"
          "There's no +100 at the goal in this version, so the best possible score is -13 (13 steps).\n")
print(f"Practising on {env_id} for {EPISODES} games...")
returns, block = [], max(1, EPISODES // 10)
for ep in range(EPISODES):
    s, _ = env.reset(seed=SEED + ep)
    total, done, steps = 0.0, False, 0
    while not done and steps < 1000:
        a = choose(s, EPSILON)
        s2, r, terminated, truncated, _ = env.step(a)
        target = r if terminated else r + GAMMA * Q[s2].max()   # Bellman target
        Q[s, a] += ALPHA * (target - Q[s, a])                    # Q-learning update
        s, total, done, steps = s2, total + r, terminated or truncated, steps + 1
    returns.append(total)
    if (ep + 1) % block == 0:
        print(f"  game {ep + 1:5d} | average score of the last {block} practice games: {np.mean(returns[-block:]):9.1f}")
print("  (Practice scores include the random exploring moves, so they stay noisy.)")

# Print the learned policy as arrows (works without a window)
ARROWS = {"CliffWalking": "^>v<", "FrozenLake": "<v>^"}
if ENV in ARROWS:
    u = env.unwrapped
    shape = tuple(getattr(u, "shape", ())) or u.desc.shape
    cells = [ARROWS[ENV][int(np.argmax(Q[i]))] for i in range(n_states)]
    if ENV == "CliffWalking":                         # mark the cliff and goal instead of meaningless arrows
        cols = shape[1]
        for c in range(1, cols - 1):
            cells[(shape[0] - 1) * cols + c] = "C"
        cells[shape[0] * cols - 1] = "G"
    else:
        for i, ch in enumerate(u.desc.flatten()):
            if ch in (b"H", b"G"):
                cells[i] = ch.decode()
    print("\nThe agent's plan (best move from each square; C = cliff, G = goal, H = hole):")
    print("\n".join(" ".join(row) for row in np.array(cells).reshape(shape)))

# Test the plan with NO random moves: this is the agent's real skill
s, _ = env.reset(seed=SEED)
score, steps, terminated, truncated = 0.0, 0, False, False
while not (terminated or truncated) and steps < 100:
    s, r, terminated, truncated, _ = env.step(int(np.argmax(Q[s])))
    score, steps = score + r, steps + 1
if not terminated:
    verdict = "got stuck and never reached the goal"
elif ENV == "FrozenLake" and env.unwrapped.desc.flatten()[s] == b"H":
    verdict = f"fell in a hole after {steps} steps"
else:
    verdict = f"finished in {steps} steps"
print(f"\nTEST RUN (no random moves): the agent {verdict}. Score: {score:g}")
env.close()

# Watch the greedy agent
if WATCH:
    watch = gym.make(env_id, render_mode="human", **ENV_KWARGS)
    s, _ = watch.reset(seed=SEED)
    for _ in range(30):                                # a stuck agent would otherwise loop for ages
        s, r, terminated, truncated, _ = watch.step(int(np.argmax(Q[s])))
        if terminated or truncated:
            break
    time.sleep(1)
    watch.close()
