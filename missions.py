"""
REWARD-DESIGN MISSIONS
======================
The agent learns with Q-learning, exactly as in the talk. You can't touch the algorithm.
You can only change what it is REWARDED for. Can you make it do what you want?

How to play
  1. Set MISSION below (start at 1).
  2. Read the brief it prints, then put your changes in CHANGES.
  3. Run:  python missions.py
  The agent trains 3 times (3 different random seeds). The mission only counts if all 3 succeed.

The map (edges are walls):
    A # . . . . . B        S  start            A B C  goals (stepping on one ends the game)
    . # . L L . . .        L  lava (ends the game)
    S . . . * . . .        #  wall             *      breadcrumb (pays out EVERY time you step on it)
    . . # L L . . .
    . . . . . . . C
"""

MISSION = 1

CHANGES = {
    # Put your changes here, e.g.
    # "B": 2.0,
}

WATCH = True   # show the trained agent in a window (set to False if no window appears)

# ===================== You don't need to edit below this line =====================
import sys
import time
from gridlab import GridLab, train, greedy_run, DEFAULT_REWARDS

MISSIONS = {
    1: dict(title="Change of plans",
            brief="All three goals are worth +1, so the agent heads for the nearest one (A).\n"
                  "Make it go to B instead.",
            settings={"A": 1, "B": 1, "C": 1, "GAMMA": 0.9},
            locked=["GAMMA"], target="B",
            hint="Hint: a reward k steps away is worth reward x GAMMA^(k-1) to the agent. A is 2 steps away; B and C are 9."),
    2: dict(title="Budget cuts",
            brief="A is now worth +5 and you can't change it. You may only change B's reward.\n"
                  "Make the agent go to B. Can you work out the smallest reward that works, before you test it?",
            settings={"A": 5, "B": 1, "C": 1, "GAMMA": 0.9},
            locked=["A", "C", "lava", "step", "breadcrumb", "GAMMA"], target="B",
            hint="Hint: A (+5, 2 steps away) is worth 5 x 0.9 = 4.5 right now. B is 9 steps away, so it is worth B x 0.9^8. Solve for B."),
    3: dict(title="No rush",
            brief="GAMMA is now 1.0: the agent values future reward exactly as much as reward now.\n"
                  "Only B pays out. Make the agent reach B in 9 steps (the shortest route).\n"
                  "You may change the step and lava rewards only.",
            settings={"A": 0, "B": 1, "C": 0, "GAMMA": 1.0},
            locked=["A", "B", "C", "breadcrumb", "GAMMA"], target="B", max_steps=9,
            hint="Hint: with GAMMA = 1 and nothing to lose per step, wandering forever is as good as finishing. Make dawdling cost something, but only a little."),
    4: dict(title="Think long-term   [GAMMA unlocked]",
            brief="A is worth +10, B is worth +12, and every reward is locked.\n"
                  "The only thing you can change is GAMMA. Make the agent go to B.",
            settings={"A": 10, "B": 12, "C": 0, "GAMMA": 0.9},
            locked=["A", "B", "C", "lava", "step", "breadcrumb"], target="B",
            hint="Hint: B must be worth more than A *from the start*: 12 x GAMMA^8 > 10 x GAMMA. Try a calculator before you try the code."),
    5: dict(title="A helpful hint?",
            brief="We want the agent at C. To help it, we put a +0.5 breadcrumb (*) on the map.\n"
                  "Run it as it is first and watch what happens. Then fix it. Everything is unlocked.",
            settings={"A": 0, "B": 0, "C": 1, "breadcrumb": 0.5, "GAMMA": 0.95},
            locked=[], target="C",
            hint="Hint: count what the agent earns by stepping on and off the breadcrumb forever, then compare it with C."),
    6: dict(title="BONUS: Grand tour   [this one has a twist]",
            brief="A is now a checkpoint: it pays out once and doesn't end the game.\n"
                  "Make the agent visit A first, then finish at B. Everything is unlocked.\n"
                  "If you can't crack it, that might be the point. We'll explain at the end.",
            settings={"A": 1, "B": 1, "C": 0, "step": -0.01, "GAMMA": 0.95},
            locked=[], target="B", checkpoint="A",
            hint="Hint: watch the square just below A. What move does the agent need there on the way in, and on the way out?"),
}


def main():
    if MISSION not in MISSIONS:
        sys.exit(f"MISSION must be one of {sorted(MISSIONS)}.")
    m = MISSIONS[MISSION]
    settings = {**DEFAULT_REWARDS, **m["settings"]}
    for key, value in CHANGES.items():
        if key not in settings:
            sys.exit(f"Unknown setting {key!r}. Choose from: {', '.join(settings)}")
        if key in m["locked"]:
            sys.exit(f"{key!r} is locked in this mission. Remove it from CHANGES.")
        settings[key] = float(value)
    gamma = settings.pop("GAMMA")
    if not 0 <= gamma <= 1:
        sys.exit("GAMMA must be between 0 and 1.")

    print(f"\nMISSION {MISSION}: {m['title']}\n{m['brief']}\n")
    print("Settings (locked ones marked *):")
    for k, v in {**settings, "GAMMA": gamma}.items():
        print(f"   {k:<11}{v:>7}{'  *' if k in m['locked'] else ''}")
    print("\nTraining 3 agents...")

    passes, first = 0, None
    for seed in range(3):
        env = GridLab(settings, checkpoint=m.get("checkpoint"))
        Q = train(env, gamma, seed)
        path, outcome = greedy_run(env, Q)
        steps = len(path) - 1
        visited_cp = m.get("checkpoint") is not None and any(env.grid[p] == m["checkpoint"] for p in path)
        ok = outcome == m["target"] and steps <= m.get("max_steps", 999) and (visited_cp or "checkpoint" not in m)
        passes += ok
        desc = {"loop": "went round in circles and never finished", "lava": "fell in the lava"}.get(
            outcome, f"finished at {outcome} in {steps} steps")
        if m.get("checkpoint"):
            desc += f" ({'did' if visited_cp else 'did NOT'} visit {m['checkpoint']} first)"
        print(f"   Agent {seed + 1}: {desc}  {'PASS' if ok else 'FAIL'}")
        if first is None:
            first = (env, Q, path)

    env, Q, path = first
    print("\nAgent 1's plan (best move from each square):")
    print(env.text(policy=Q[: env.rows * env.cols].argmax(1)))
    if passes == 3:
        print("\nMISSION COMPLETE! Try the next one.")
    else:
        print(f"\nNot yet: {passes}/3 agents succeeded. Change CHANGES and try again.")
        if passes == 2:
            print("So close! Borderline values pass on some seeds and fail on others. Push it a bit further.")
        print(m["hint"])

    if WATCH:
        try:
            show = GridLab(settings, checkpoint=m.get("checkpoint"), render_mode="human")
            show.reset()
            for pos in path[:30]:
                show.pos = pos
                show.render()
            time.sleep(1.5)
            show.close()
        except Exception as e:  # no display (WSL, remote machine) or pygame missing
            print(f"(Couldn't open a window: {e}. Set WATCH = False to hide this message.)")


if __name__ == "__main__":
    main()
