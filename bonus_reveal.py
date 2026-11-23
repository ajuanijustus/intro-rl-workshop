"""The answer to mission 6 (BONUS: Grand tour). Have a go at the mission first!  Run:  python bonus_reveal.py

Why mission 6 can't be solved: the agent's state is only WHERE it is. A is a dead end, so the agent
must walk into A and then back out through the same square. In that square it has to move towards A
the first time and away from A the second time, but a policy that only sees "where am I" must pick the
same move both times. No reward can fix that.

The fix is to give the agent memory: add "have I visited A yet?" to its state. Same rewards, same
algorithm, twice as many states, and it now solves it.
"""
from gridlab import GridLab, train, greedy_run

REWARDS = {"A": 1, "B": 1, "C": 0, "step": -0.01}
GAMMA = 0.95

for remember in (False, True):
    print(f"\nState = position{' + visited-A flag' if remember else ' only'}:")
    for seed in range(3):
        env = GridLab(REWARDS, checkpoint="A", remember_checkpoint=remember)
        Q = train(env, GAMMA, seed)
        path, outcome = greedy_run(env, Q)
        visited = any(env.grid[p] == "A" for p in path)
        print(f"   Agent {seed + 1}: ended '{outcome}' after {len(path) - 1} steps, "
              f"visited A first: {'yes' if visited else 'no'}")
