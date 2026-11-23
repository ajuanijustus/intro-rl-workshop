"""GridLab: a tiny multi-goal grid world for reward-design missions.

You don't need to edit this file. Edit missions.py instead.
"""
import time
import numpy as np

try:  # Gymnasium is the standard RL environment API; we follow it here.
    import gymnasium as gym
    from gymnasium import spaces
    _Base = gym.Env
except ImportError:  # still works without it, just without the official base class
    gym, spaces = None, None
    _Base = object

# --------------------------------------------------------------------------
# The map. Edges are walls.
#   S start   A B C goals   L lava   # wall   * breadcrumb   . floor
# --------------------------------------------------------------------------
MAP = [
    "A#.....B",
    ".#.LL...",
    "S...*...",
    "..#LL...",
    ".......C",
]
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
ARROWS = "^>v<"
DEFAULT_REWARDS = {"A": 1.0, "B": 1.0, "C": 1.0, "lava": -1.0, "breadcrumb": 0.0, "step": 0.0}


class _Discrete:  # stand-in for gymnasium.spaces.Discrete if gymnasium is missing
    def __init__(self, n): self.n = n


class GridLab(_Base):
    """Goals A, B, C end the episode (unless a goal is made a checkpoint).
    Lava ends the episode. A breadcrumb pays out every time you step on it."""
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, rewards=None, checkpoint=None, remember_checkpoint=False, render_mode=None):
        self.grid = np.array([list(r) for r in MAP])
        self.rows, self.cols = self.grid.shape
        self.rewards = {**DEFAULT_REWARDS, **(rewards or {})}
        self.checkpoint = checkpoint                  # e.g. "A": pays once, does not end the episode
        self.remember = remember_checkpoint           # adds "have I visited the checkpoint?" to the state
        self.start = tuple(np.argwhere(self.grid == "S")[0])
        n_cells = self.rows * self.cols
        S = _Discrete if spaces is None else spaces.Discrete
        self.observation_space = S(n_cells * (2 if self.remember else 1))
        self.action_space = S(4)
        self.render_mode = render_mode
        self._window = None

    # ----- Gymnasium API ---------------------------------------------------
    def reset(self, seed=None, options=None):
        self.pos, self.visited = self.start, False
        return self._obs(), {}

    def step(self, action):
        r, c = self.pos
        dr, dc = ACTIONS[int(action)]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr, nc] != "#":
            self.pos = (nr, nc)
        cell = self.grid[self.pos]
        reward, terminated = self.rewards["step"], False
        if cell == "L":
            reward += self.rewards["lava"]; terminated = True
        elif cell == "*":
            reward += self.rewards["breadcrumb"]
        elif cell in "ABC":
            if cell == self.checkpoint:
                if not self.visited:
                    reward += self.rewards[cell]
                self.visited = True
            else:
                reward += self.rewards[cell]; terminated = True
        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, terminated, False, {"cell": cell}

    def _obs(self):
        idx = self.pos[0] * self.cols + self.pos[1]
        return idx + (self.rows * self.cols if (self.remember and self.visited) else 0)

    # ----- Drawing ---------------------------------------------------------
    def text(self, policy=None):
        out = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                ch = self.grid[r, c]
                if (r, c) == getattr(self, "pos", None) and policy is None:
                    ch = "@"
                elif policy is not None and ch in ".S*":
                    ch = ARROWS[policy[r * self.cols + c]]
                row.append(ch)
            out.append(" ".join(row))
        return "\n".join(out)

    def render(self):
        if self.render_mode == "ansi":
            return self.text()
        if self.render_mode != "human":
            return None
        import pygame
        cell = 80
        if self._window is None:
            pygame.init()
            self._window = pygame.display.set_mode((self.cols * cell, self.rows * cell))
            pygame.display.set_caption("GridLab")
            self._font = pygame.font.SysFont(None, 48)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close(); return None
        colours = {"#": (60, 60, 60), "L": (230, 90, 40), "A": (90, 160, 240), "B": (120, 200, 120),
                   "C": (240, 200, 70), "*": (250, 240, 200), ".": (245, 245, 245), "S": (245, 245, 245)}
        self._window.fill((255, 255, 255))
        for r in range(self.rows):
            for c in range(self.cols):
                ch = self.grid[r, c]
                rect = pygame.Rect(c * cell, r * cell, cell, cell)
                pygame.draw.rect(self._window, colours[ch], rect)
                pygame.draw.rect(self._window, (200, 200, 200), rect, 1)
                if ch in "ABCS*":
                    label = self._font.render(ch, True, (30, 30, 30))
                    self._window.blit(label, label.get_rect(center=rect.center))
        r, c = self.pos
        pygame.draw.circle(self._window, (30, 30, 30), (c * cell + cell // 2, r * cell + cell // 2), cell // 4)
        pygame.display.flip()
        time.sleep(1 / self.metadata["render_fps"])

    def close(self):
        if self._window is not None:
            import pygame
            pygame.display.quit(); pygame.quit()
            self._window = None


# --------------------------------------------------------------------------
# Tabular Q-learning (the same algorithm as in the talk)
# --------------------------------------------------------------------------
def train(env, gamma, seed, episodes=2000, alpha=0.5, epsilon=0.3, max_steps=200):
    """Each practice episode starts on a random floor square ("exploring starts"),
    so the agent gets to see the whole map, not just the area near S."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    floor = [tuple(p) for p in np.argwhere(np.isin(env.grid, list(".S*")))]
    for _ in range(episodes):
        env.reset()
        env.pos = floor[rng.integers(len(floor))]
        env.visited = bool(env.remember and rng.random() < 0.5)
        s = env._obs()
        for _ in range(max_steps):
            if rng.random() < epsilon:
                a = int(rng.integers(4))                                   # explore
            else:
                a = int(rng.choice(np.flatnonzero(Q[s] == Q[s].max())))    # exploit
            s2, r, terminated, truncated, _ = env.step(a)
            target = r if terminated else r + gamma * Q[s2].max()          # Bellman target
            Q[s, a] += alpha * (target - Q[s, a])                          # Q-learning update
            s = s2
            if terminated or truncated:
                break
    return Q


def greedy_run(env, Q, max_steps=60):
    """Follow the best-known action from the start. Returns (path, outcome)."""
    s, _ = env.reset()
    path, seen = [env.pos], {s}
    for _ in range(max_steps):
        s, r, terminated, _, info = env.step(int(np.argmax(Q[s])))
        path.append(env.pos)
        if terminated:
            return path, ("lava" if info["cell"] == "L" else info["cell"])
        if s in seen:
            return path, "loop"
        seen.add(s)
    return path, "loop"
