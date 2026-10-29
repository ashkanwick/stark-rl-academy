# 🧠 EXPLAINER — understand this project in 10 minutes

This repo is a **teaching demo**: three AIs learn to fly Iron Man suits, and
every file here exists so you can *see exactly how they learned*.
Start here. No math background needed beyond arithmetic.

---

## 1. The problem (the story)

Iron Man's suit is hovering in the lab. It can tilt on one line:

```
💥 ---- 🎯 ---- 💥
 -4   ... 0  ... +4
```

- `0` = perfect hover 🎯
- `-4` / `+4` = the suit smashed into a wall 💥

Wind keeps shoving the suit around. **Nobody tells JARVIS (the AI) what to do.**
JARVIS only ever feels a **reward**:

| event | reward |
|---|---|
| lands on `0` | **+20** |
| any normal step | **−2** (fuel/time costs a little) |
| hits a wall | **−100**, episode over |

Learning = figuring out, purely from these numbers, which action to take in
each situation. That is *all* reinforcement learning is.

---

## 2. The entire algorithm (it fits in 4 lines)

Repeat thousands of times:

1. **LOOK** — where am I? (the tilt → this is the **state**)
2. **ACT** — pick one of 3 actions: thrust right ▶ / hold • / thrust left ◀
3. **FEEL** — the world answers with a **new tilt** and a **reward**
4. **LEARN** — update one number in a table

The "table" is the **Q-table**: 9 tilts × 3 actions = 27 numbers. Each cell
means *"if I do this action while tilted here, how much total future reward
do I expect?"* It starts as all zeros — total ignorance. Experience fills it.

---

## 3. The one formula

```
Q(s,a)  <-  Q(s,a) + α · ( reward + γ · max Q(s',·) − Q(s,a) )
```

| symbol | meaning | here |
|---|---|---|
| `Q(s,a)` | current belief: "how good is action `a` in state `s`" | one of the 27 cells |
| `reward` | what the world just gave me | +20 / −2 / −100 |
| `max Q(s',·)` | best I can hope for in the **new** state | looks ahead one step |
| `γ = 0.95` | discount — future rewards matter slightly less than now | hyperparameter |
| `α = 0.1` | learning rate — nudge the belief, don't jump | hyperparameter |

In words: **"nudge your old belief toward what I just got, plus the best of
what comes next."** If the episode ended in a crash there is no "next", so
the target is just the reward. That's it. Every RL method is a variation on
this sentence.

---

## 4. See it with your own eyes (5 minutes)

```bash
pip install numpy
python stark_rl_simple.py
```

That file is ~150 lines and contains **the entire algorithm** — read it top to
bottom. Things to notice in its output:

- The learned policy: thrust toward the center, **hold** when on `0`.
- The demo flight: from tilt −3 it corrects in 3 steps, then micro-wobbles
  around the target (that wobble is normal — holding at 0 and stepping off 0
  have nearly equal long-run values).
- One beautiful detail: `Q(−3, LEFT) = −100.0`. That cell is JARVIS's
  **memory of a crash**. Nobody coded "walls are bad" — the −100 reward
  wrote that memory by itself.

---

## 5. Exploration — the ε trick

A table of zeros is useless if you only ever take the "best known" action —
you'd never discover better ones. So early on JARVIS acts **randomly** with
probability ε (epsilon), and uses the table otherwise. ε decays from 1.0 to
0.01 over training: **explore a lot first, exploit later.**

This is the same strategy you'd use in a new city: try streets at random the
first day; by the last day, walk your favorite route.

---

## 6. Multi-agent (House Party Protocol) in 5 sentences

Part 2 puts **three suits** on a 6×6 grid. Each must reach its own reactor at
the bottom row, avoiding fire cells (−50) and *each other* (collision = −80
to **both** suits). Each suit has **its own separate Q-table** and can only
see its own cell plus a tiny "radar" of its two teammates. There is **no
coordinator** — yet they end up weaving around each other, because
collisions hurt everyone's reward, and reward is the only thing that talks.
That's the magic the seminar wants to show.

---

## 7. Where everything lives

| you want to... | go to |
|---|---|
| read the whole algorithm in one page | `stark_rl_simple.py` |
| learn step-by-step with live animations | `stark_rl_academy.ipynb` |
| play a game with the trained brains | open `stark_rl_game.html` (tab **🎓 ACADEMY** teaches interactively) |
| see the full seminar script (animations, dashboard) | `stark_rl_seminar.py` |
| rebuild the academy notebook | `python _build_academy_nb.py` |

| concept | where to find it in the code |
|---|---|
| environment (`step` function) | `stark_rl_simple.py` §1 · `StarkFlightBalancerEnv` |
| Q-table + Bellman update | `stark_rl_simple.py` §3 · `train_flight_stabilizer` |
| ε-greedy | `epsilon_schedule_preview()` · `choose_action()` |
| multi-agent + radar state | `HousePartyGridEnv` · `teach_house_radar_example()` |

---

## 8. Things people often mix up

- **Reward vs. punishment.** The −100 wall hit is *punishment* (it decreases
  the chance of repeating what led there). "Negative reinforcement" is a
  different thing — it means *removing* something bad. Don't mix them on a slide.
- **Q-table vs. policy.** The Q-table is the *knowledge*; the policy
  (`argmax` of each row) is the *behavior* it produces.
- **Q-Learning is off-policy.** It learns the value of the *greedy* future
  even while exploring randomly — that's why `max Q(s',·)` appears in the formula.
- **The multi-agent suits don't "predict" each other.** They're independent
  Q-learners (IQL); coordination *emerges* from the shared team penalty.

---

*«Sometimes you gotta run before you can walk.» — Tony Stark* 🦾
