# -*- coding: utf-8 -*-
"""
==============================================================================
  STARK INDUSTRIES  --  REINFORCEMENT LEARNING SEMINAR DEMO  (Iron Man theme)
  سمینار یادگیری تقویتی -- شبیه‌سازی آزمایشگاه تونی استارک
==============================================================================

  ENGLISH                                              فارسی
  --------                                              -----
  PART 1: Single-agent tabular Q-Learning              بخش ۱: یادگیری تک‌عامله
          "Iron Man Flight Stabilizer"                       با Q-Learning جدولی
          (the first flight scene)                           (صحنه‌ی اولین پرواز)
  PART 2: Multi-Agent Independent Q-Learning           بخش ۲: یادگیری چندعامله
          "House Party Protocol" (3 suits)                   مستقل (پروتکل مهمانی
                                                             خانگی با ۳ زره)
  PART 3: Stark HUD matplotlib dashboard               بخش ۳: داشبورد تله‌متری
                                                             گرافیکی به سبک استارک
  LIVE  : after training, frame-by-frame animations    انیمیشن زنده‌ی سیاستِ
          of the LEARNED policy (Jupyter clear_output  آموخته‌شده پس از آموزش
          or terminal cls):                                (فریم‌به‌فریم)
          animate_flight_stabilizer(Q1)  /  animate_house_party(Qs2)

  Requirements : numpy + matplotlib ONLY               وابستگی: فقط numpy و matplotlib
  Terminal run : python stark_rl_seminar.py            اجرا در ترمینال
  Jupyter      : %run stark_rl_seminar.py              اجرا در جوپیتر

  NOTES / نکات:
   * SEED is fixed (42): the whole demo is deterministic -> safe to re-run
     live on stage.   (seed ثابت، پس خروجی هر بار دقیقاً یکسان است)
   * stdout is forced to UTF-8 so emojis never crash on Windows consoles.
     (خروجی UTF-8 می‌شود تا ایموجی‌ها در کنسول ویندوز خطا ندهند)
==============================================================================
"""

import os
import random
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

# --- Windows fix: force UTF-8 console so emojis (🦾 💥 ✅ 🔥 ⚡) never crash
#     رفع مشکل ویندوز: اجبار خروجی به UTF-8 برای چاپ ایموجی‌ها
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# --- reproducibility for the live seminar / تکرارپذیری برای ارائه‌ی زنده
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# folder of this script (works in Jupyter too) / پوشه‌ی اسکریپت (در جوپیتر هم کار می‌کند)
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()


# ============================================================================
#  Shared helpers / توابع کمکی مشترک
# ============================================================================

def epsilon_at(ep, episodes, eps_start=1.0, eps_end=0.01):
    """Linear epsilon decay: 1.0 -> 0.01 over training.
    کاهش خطی اپسیلون: از اکتشاف کامل به بهره‌برداری."""
    return max(eps_end, eps_start - (eps_start - eps_end) * ep / max(1, episodes - 1))


def choose_action(q_row, epsilon):
    """epsilon-greedy action selection (with random tie-breaking).
    انتخاب اکشن به روش اپسیلون-حریصانه:
      با احتمال eps  -> اکتشاف (اکشن تصادفی)  Exploration
      در غیر این صورت -> بهره‌برداری از بهترین Q  Exploitation"""
    if random.random() < epsilon:
        return random.randrange(len(q_row))
    return int(np.random.choice(np.flatnonzero(q_row == q_row.max())))


def banner(text):
    print("\n" + "=" * 78)
    print("  " + text)
    print("=" * 78)


def moving_average(x, w=50):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        w = len(x)
    if w == 0:
        return np.array([])
    return np.convolve(x, np.ones(w) / w, mode="valid")


# ============================================================================
# ============================================================================
#   PART 1 -- SINGLE-AGENT Q-LEARNING :  "IRON MAN FLIGHT STABILIZER"
#   بخش ۱ -- یادگیری تقویتی تک‌عامله : پایدارساز پرواز آیرون‌من
# ============================================================================
# ============================================================================

class StarkFlightBalancerEnv:
    """1-D discrete tilt axis of the suit (the "first flight" scene).
    محیط پرواز: محور کج‌شدگی گسسته‌ی زره در فضای یک‌بعدی.

    States  / وضعیت‌ها  : -4 .. +4  (tilt angle; 0 = perfect hover)
                          -4 and +4 are CRASH states (lab wall) -> terminal
    Actions / اکشن‌ها    : 0 fire LEFT  thruster -> tilt +1
                           1 HOLD (gravity/turbulence randomly pushes ±1)
                           2 fire RIGHT thruster -> tilt -1
    Rewards / پاداش‌ها   : land on 0 -> +20 (stable hover)
                          land on ±4 -> -100 (crash, episode ends)
                          any other state -> -2 (time/fuel penalty)
    Episode ends on crash or after MAX_STEPS (continuing task truncated).
    """

    MIN_TILT, MAX_TILT = -4, 4
    MAX_STEPS = 30
    N_STATES = 9          # -4..+4  => index = tilt + 4
    N_ACTIONS = 3

    ACTION_NAMES = {
        0: "FIRE LEFT THRUSTER 🚀",
        1: "HOLD (turbulence) 🌪",
        2: "FIRE RIGHT THRUSTER🚀",
    }

    def reset(self, start_tilt=None):
        # suit is dropped at a random tilt / رها شدن زره با زاویه‌ی تصادفی
        self.tilt = start_tilt if start_tilt is not None else random.randint(-3, 3)
        self.steps = 0
        return self.tilt

    def step(self, action):
        """Apply action -> (next_tilt, reward, done).
        NOTE: the reward is a function of the state you LAND IN
        (this is exactly R_{t+1} in the Bellman equation below!).
        پاداش تابعِ وضعیتی است که عامل به آن «می‌رسد» — همان R_{t+1}
        در معادله‌ی بلمن."""
        self.steps += 1
        if action == 0:                       # fire left  -> pushed right
            nxt = self.tilt + 1
        elif action == 2:                     # fire right -> pushed left
            nxt = self.tilt - 1
        else:                                 # do nothing -> turbulence
            nxt = self.tilt + random.choice((-1, 1))
        nxt = max(self.MIN_TILT, min(self.MAX_TILT, nxt))

        if nxt in (self.MIN_TILT, self.MAX_TILT):
            reward, done = -100, True         # 💥 crash into the lab wall
        elif nxt == 0:
            reward, done = +20, False         # ✅ stable hover moment
        else:
            reward, done = -2, False          # ⛽ time / fuel penalty
        self.tilt = nxt
        return nxt, reward, done


def tilt_bar(state):
    """One-line console gauge:  💥 walls, ✅ balance point, 🦾 the suit."""
    cells = []
    for s in range(-4, 5):
        if s == state:
            cells.append("🦾")
        elif s in (-4, 4):
            cells.append("💥")
        elif s == 0:
            cells.append("✅")
        else:
            cells.append("·")
    return " ".join(cells)


def train_flight_stabilizer(env, episodes=800, alpha=0.1, gamma=0.95,
                            eps_start=1.0, eps_end=0.01, progress_every=200,
                            on_episode=None):
    """Tabular Q-Learning from scratch.
    آموزش جدولی Q-Learning از صفر — بدون هیچ کتابخانه‌ی RL.
    on_episode(ep, Q) is called after every episode (used by the ACADEMY
    to snapshot the Q-table while it learns) — بعد از هر اپیزود صدا زده می‌شود."""
    Q = np.zeros((env.N_STATES, env.N_ACTIONS))
    rewards_hist, crash_hist = [], []

    for ep in range(episodes):
        eps = epsilon_at(ep, episodes, eps_start, eps_end)
        s = env.reset()
        total, crashed = 0, False

        for _ in range(env.MAX_STEPS):
            a = choose_action(Q[s + 4], eps)
            s2, r, done = env.step(a)

            # =================================================================
            #  ★ THE BELLMAN / Q-LEARNING UPDATE  ★
            #  ★ به‌روزرسانی معادله‌ی بلمن  ★
            #
            #     Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
            #
            #  English : new estimate = old estimate + learning-rate * (received
            #            reward + discounted best future value - current belief)
            #  فارسی   : مقدار جدید = مقدار قبلی + نرخ یادگیری ×
            #            (پاداش دریافتی + تخفیف × بهترین ارزش آینده − برآورد فعلی)
            #
            #  alpha=0.1  learning rate / نرخ یادگیری (چقدر اطلاعات جدید جایگزین شود)
            #  gamma=0.95 discount      / ضریب تخفیف (اهمیت پاداش‌های آینده)
            #  If the episode ended (crash) there IS no future -> target = r
            #  اگر اپیزود تمام شده باشد آینده‌ای وجود ندارد، پس هدف فقط r است
            # =================================================================
            best_next = 0.0 if done else np.max(Q[s2 + 4])
            Q[s + 4, a] += alpha * (r + gamma * best_next - Q[s + 4, a])

            s, total = s2, total + r
            if done:
                crashed = True
                break

        rewards_hist.append(total)
        crash_hist.append(crashed)
        if on_episode:
            on_episode(ep, Q)

        if progress_every and (ep + 1) % progress_every == 0:
            w = moving_average(rewards_hist, 50)
            print(f"   [training] ep {ep + 1:4d} | avg reward (MA50) = {w[-1]:+8.1f} | "
                  f"crash rate (last 50) = {100 * np.mean(crash_hist[-50:]):5.1f}%")

    return Q, rewards_hist, crash_hist


def run_flight_episode(env, Q, epsilon, start_tilt):
    """Run one full episode silently; returns (log_lines, total, crashed, steps).
    اجرای بی‌صدا یک اپیزود و برگرداندن لاگ خط‌به‌خط آن."""
    if Q is None:
        Q = np.zeros((env.N_STATES, env.N_ACTIONS))
    s = env.reset(start_tilt)
    lines = [
        f"  Suit dropped at tilt {s:+d}   (gauge:  💥=wall  ✅=balance  🦾=suit)",
        f"  {'':>8}{tilt_bar(s)}   <- start",
    ]
    total, steps, crashed = 0, 0, False
    for t in range(env.MAX_STEPS):
        a = choose_action(Q[s + 4], epsilon)
        s2, r, done = env.step(a)
        steps, total = steps + 1, total + r
        lines.append(f"  step {t + 1:02d} | tilt {s:+d} -> {s2:+d} | "
                     f"{env.ACTION_NAMES[a]:<21} | R {r:+4d} | {tilt_bar(s2)}")
        s = s2
        if done:
            crashed = True
            lines.append("  💥 CRASH! The suit slammed into the Stark lab wall. "
                         "Episode terminated.")
            break
    if not crashed:
        lines.append("  ✅ Stable hover held for the whole episode "
                     "(only micro-corrections).")
    outcome = "CRASH 💥" if crashed else "STABLE HOVER ✅"
    lines.append(f"  ➤ RESULT: total reward = {total:+4d} | steps = {steps:2d} "
                 f"| outcome = {outcome}")
    return lines, total, crashed, steps


def print_flight_demo(env, Q, epsilon, start_tilt, title):
    """Run one episode and pretty-print it / اجرا و چاپ یک اپیزود کامل."""
    print("\n" + "-" * 78)
    print(f"  🎬 {title}")
    print("-" * 78)
    lines, total, crashed, _steps = run_flight_episode(env, Q, epsilon, start_tilt)
    print("\n".join(lines))
    return total, crashed


def print_learned_policy(env, Q):
    """The final policy table — perfect for a slide / جدول سیاست نهایی برای اسلاید."""
    print("\n  📋 LEARNED POLICY (argmax of the Q-table)  |  سیاست آموخته‌شده:")
    for tilt in range(-3, 4):
        a = int(np.argmax(Q[tilt + 4]))
        print(f"     tilt {tilt:+d}  ->  {env.ACTION_NAMES[a]}")
    print("     tilt ±4  ->  💥 terminal (crash — never reached by the trained suit)")


# ============================================================================
# ============================================================================
#   PART 2 -- MULTI-AGENT RL :  "HOUSE PARTY PROTOCOL"
#   بخش ۲ -- یادگیری چندعامله : پروتکل مهمانی خانگی
# ============================================================================
# ============================================================================

class HousePartyGridEnv:
    """6x6 cooperative airspace for 3 independent suits (Independent Q-Learning).

    شبکه‌ی هوایی ۶×۶ برای ۳ زره‌ی مستقل — یادگیری Q مستقل (IQL).

    Suits / زره‌ها    : Mark 42 🤖 (0,0)  Silver Centurion 🦾 (0,2)  Hulkbuster 🛡 (0,4)
    Targets / هدف‌ها  : each suit must dock on ITS OWN Arc Reactor at row 5
    Fires  / موانع    : static fire/laser cells 🔥 in the middle rows

    REWARD DESIGN / طراحی پاداش:
        -1   per move (time/fuel)                     +50  reach assigned target
        -50  enter a fire cell                        -80  TEAM PENALTY on collision
                                                        (both suits are punished)

    COORDINATION MECHANICS / مکانیزم هماهنگی:
        * Movement is TURN-BASED in a fixed order (Mark42 -> Centurion -> Hulkbuster):
          حرکت‌ها نوبتی هستند؛ به این ترتیب «برخورد تعویضی» هم به‌طور طبیعی
          به‌صورت ورود به خانه‌ی اشغال‌شده شناسایی می‌شود.
        * Each suit's STATE = its own cell + a compact "radar" reading of the
          other two suits (see radar()).  State هر زره = خانه‌ی خودش + «رادار»
          هم‌تیمی‌ها؛ بدون این رادار، یادگیری پرهیز از برخورد غیرممکن است چون
          محیط از دید هر عامل غیرایستا (non-stationary) می‌شود.
        * A landed suit is REMOVED from the airspace (docked at its reactor):
          زره‌ای که به هدفش رسید فرود می‌آید و از شبکه حذف می‌شود.
    """

    SIZE = 6
    MAX_ROUNDS = 25
    N_AGENTS = 3
    N_ACTIONS = 5

    ACTION_NAMES = {0: "UP ⬆", 1: "DOWN ⬇", 2: "LEFT ⬅", 3: "RIGHT ➡", 4: "STAY ✋"}
    DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}

    SUIT_NAMES = ("Mark 42", "Silver Centurion", "Hulkbuster")
    SUIT_EMOJI = ("🤖", "🦾", "🛡")
    SUIT_INITIAL = ("M", "S", "H")

    STARTS = ((0, 0), (0, 2), (0, 4))
    TARGETS = ((5, 0), (5, 2), (5, 4))
    TARGET_EMOJI = ("🔵", "🟡", "🟢")
    FIRES = frozenset({(2, 1), (2, 3), (3, 2), (3, 4)})

    def __init__(self):
        # state = own cell (36) x radar1 (6) x radar2 (6) = 1296 states
        # فضای وضعیت: ۳۶ خانه × دو قرائت رادار ۶حالته = ۱۲۹۶ وضعیت
        self.n_states = self.SIZE * self.SIZE * 6 * 6
        self.reset()

    def reset(self):
        self.pos = list(self.STARTS)      # pos[i] = None once landed / پس از فرود None
        self.landed = [False] * self.N_AGENTS
        self.round_no = 0
        return self.state_index(0)

    # ------------------------------------------------------------------ state
    def radar(self, i, j):
        """How suit 'i' senses suit 'j' on its radar / رادار زره i نسبت به زره j:
        0 = off-grid / far (Chebyshev distance > 1),  1=N, 2=S, 3=W, 4=E, 5=diagonal"""
        pj = self.pos[j]
        if pj is None:
            return 0
        dr, dc = pj[0] - self.pos[i][0], pj[1] - self.pos[i][1]
        if max(abs(dr), abs(dc)) > 1:
            return 0
        if (dr, dc) == (-1, 0):
            return 1
        if (dr, dc) == (1, 0):
            return 2
        if (dr, dc) == (0, -1):
            return 3
        if (dr, dc) == (0, 1):
            return 4
        return 5

    def state_index(self, i):
        others = [j for j in range(self.N_AGENTS) if j != i]
        r, c = self.pos[i]
        return ((r * self.SIZE + c) * 6 + self.radar(i, others[0])) * 6 + self.radar(i, others[1])

    def _occupied(self, i):
        """Cells currently occupied by other ACTIVE suits / خانه‌های اشغال‌شده توسط بقیه."""
        return {self.pos[j] for j in range(self.N_AGENTS) if j != i and self.pos[j] is not None}

    # ------------------------------------------------------------------- step
    def step(self, i, action):
        """One suit moves. Returns (s2, reward, landed?, event_text, info).
        حرکت یک زره در نوبت خودش."""
        r, c = self.pos[i]
        dr, dc = self.DELTA[action]
        nr, nc = r + dr, c + dc
        reward = -1                                   # ⛽ time/fuel cost
        event = "holds position"
        info = {"collision": False, "teammate_penalty": 0}

        if action == 4:
            event = "holds position (scanning)"
        elif not (0 <= nr < self.SIZE and 0 <= nc < self.SIZE):
            nr, nc = r, c
            event = "bumps the arena wall"
        elif (nr, nc) in self.FIRES:
            nr, nc = r, c                             # move rejected, suit bounces back
            reward -= 50
            event = "HITS FIRE/LASER 🔥 (-50)"
        elif (nr, nc) in self._occupied(i):
            nr, nc = r, c                             # move rejected
            reward -= 80
            info["collision"] = True
            info["teammate_penalty"] = -80            # TEAM PENALTY hits BOTH suits
            event = "COLLISION 💥 (-80 team penalty, both suits)"
        elif (nr, nc) == self.TARGETS[i]:
            reward += 50
            self.landed[i] = True                     # docked & removed from airspace
            event = "DOCKS ON ARC REACTOR ⚡ (+50)"
        else:
            event = f"clears -> ({nr},{nc})  ✈"

        self.pos[i] = None if self.landed[i] else (nr, nc)
        done_i = self.landed[i]
        s2 = self.state_index(i) if not done_i else -1
        return s2, reward, done_i, event, info

    # ----------------------------------------------------------------- render
    def render(self):
        lines = ["        0     1     2     3     4     5"]
        for r in range(self.SIZE):
            cells = []
            for c in range(self.SIZE):
                ch = "·"
                if (r, c) in self.FIRES:
                    ch = "🔥"
                for t in range(self.N_AGENTS):
                    if self.TARGETS[t] == (r, c):
                        ch = self.TARGET_EMOJI[t]
                for a in range(self.N_AGENTS):        # suits drawn on top
                    if self.pos[a] == (r, c):
                        ch = self.SUIT_EMOJI[a]
                cells.append(ch)
            lines.append(f"     {r}  " + "     ".join(cells))
        return "\n".join(lines)


def train_house_party(env, episodes=1500, alpha=0.1, gamma=0.95,
                      eps_start=1.0, eps_end=0.01, progress_every=250,
                      on_episode=None):
    """Train 3 INDEPENDENT Q-tables simultaneously.
    آموزش هم‌زمان سه جدول Q مستقل — هر زره فقط از دید خودش یاد می‌گیرد.

    IQL simplification (worth mentioning on stage): each suit's Q-update uses
    the state right after ITS OWN move — teammates' same-round moves are not
    unfolded. ساده‌سازی IQL: به‌روزرسانی با وضعیت بلافاصله بعد از حرکت خودِ عامل."""
    Qs = [np.zeros((env.n_states, env.N_ACTIONS)) for _ in range(env.N_AGENTS)]
    team_hist, coll_hist, succ_hist = [], [], []

    for ep in range(episodes):
        eps = epsilon_at(ep, episodes, eps_start, eps_end)
        env.reset()
        team, collisions = 0, 0

        for _ in range(env.MAX_ROUNDS):
            if all(env.landed):
                break
            for i in range(env.N_AGENTS):
                if env.landed[i]:
                    continue
                s = env.state_index(i)
                a = choose_action(Qs[i][s], eps)
                s2, r, done_i, _event, info = env.step(i, a)

                team += r + info["teammate_penalty"]
                if info["collision"]:
                    collisions += 1

                # same Bellman update, one table per suit / همان معادله‌ی بلمن، برای هر زره
                target = r if done_i else r + gamma * np.max(Qs[i][s2])
                Qs[i][s, a] += alpha * (target - Qs[i][s, a])

                if all(env.landed):
                    break

        team_hist.append(team)
        coll_hist.append(collisions)
        succ_hist.append(all(env.landed))
        if on_episode:
            on_episode(ep, Qs)

        if progress_every and (ep + 1) % progress_every == 0:
            print(f"   [training] ep {ep + 1:4d} | team score (MA50) = {moving_average(team_hist, 50)[-1]:+7.1f} | "
                  f"collisions (MA50) = {moving_average(coll_hist, 50)[-1]:5.2f} | "
                  f"success (last 50) = {100 * np.mean(succ_hist[-50:]):5.1f}%")

    return Qs, team_hist, coll_hist, succ_hist


def print_house_party_demo(env, Qs):
    """Greedy (epsilon=0) synchronized mission, printed round by round.
    ماموریت نهایی با سیاستِ آموخته‌شده (حریصانه) — چاپ مرحله‌به‌مرحله."""
    print("\n  🎬 FINAL MISSION — trained suits, greedy policy (epsilon = 0)")
    print("     legend: 🤖 Mark 42   🦾 Silver Centurion   🛡 Hulkbuster   "
          "🔥 fire   🔵🟡🟢 arc reactors")
    env.reset()
    trajectories = [[env.STARTS[i]] for i in range(env.N_AGENTS)]
    collisions = 0
    rounds_used = env.MAX_ROUNDS

    for rnd in range(1, env.MAX_ROUNDS + 1):
        print(f"\n  ⏱ ROUND {rnd}")
        events = []
        for i in range(env.N_AGENTS):
            if env.landed[i]:
                continue
            s = env.state_index(i)
            a = int(np.argmax(Qs[i][s]))              # pure exploitation
            _s2, _r, done_i, event, info = env.step(i, a)
            if info["collision"]:
                collisions += 1
            trajectories[i].append(env.TARGETS[i] if done_i else env.pos[i])
            events.append(f"     {env.SUIT_NAMES[i]:<17}{env.SUIT_EMOJI[i]}  "
                          f"{env.ACTION_NAMES[a]:<9} {event}")
        print(env.render())
        for e in events:
            print(e)
        if all(env.landed):
            rounds_used = rnd
            print(f"\n  ✅ ALL 3 SUITS DOCKED — synchronized, collision-free mission "
                  f"in {rounds_used} rounds!  (House Party Protocol complete)")
            break
    else:
        print("\n  ⚠ Mission timed out before all suits docked.")

    # per-suit trajectories / مسیر هر زره
    print("\n  🛰 Trajectories / مسیر پرواز هر زره:")
    for i in range(env.N_AGENTS):
        path = " -> ".join(f"({r},{c})" for (r, c) in trajectories[i])
        print(f"     {env.SUIT_EMOJI[i]} {env.SUIT_NAMES[i]:<17}: {path}")

    # composite path overlay / نقشه‌ی ترکیبی مسیرها
    print("\n  🗺 COMPOSITE FLIGHT MAP (M/S/H = suits passed through the cell):")
    lines = ["        0    1    2    3    4    5"]
    for r in range(env.SIZE):
        cells = []
        for c in range(env.SIZE):
            if (r, c) in env.FIRES:
                txt = "🔥"
            else:
                txt = "".join(env.SUIT_INITIAL[i]
                              for i in range(env.N_AGENTS) if (r, c) in set(trajectories[i]))
                for t in range(env.N_AGENTS):
                    if env.TARGETS[t] == (r, c):
                        txt = "⚡" + txt
                if not txt:
                    txt = "·"
            cells.append(f"{txt:<3}")
        lines.append(f"     {r}  " + " ".join(cells))
    print("\n".join(lines))

    return collisions, rounds_used


# ============================================================================
# ============================================================================
#   LIVE ANIMATIONS -- frame-by-frame rendering (terminal + Jupyter)
#   انیمیشن زنده -- نمایش فریم‌به‌فریم در ترمینال و ژوپیتر نوت‌بوک
# ============================================================================
#   After the Q-tables are trained, these two functions replay the LEARNED
#   policy live (the agents really act with argmax(Q) -- nothing is scripted):
#   پس از آموزش، این دو تابع سیاستِ آموخته‌شده را زنده اجرا می‌کنند:
#       animate_flight_stabilizer(Q)   -> 1-D HUD gauge, wind gusts included
#       animate_house_party(Qs)        -> 6x6 multi-agent radar, 1 frame/round
#   Inside Jupyter each frame replaces the previous one via
#   IPython.display.clear_output(wait=True); in a plain terminal cls/clear
#   is used instead.  در جوپیتر فریم‌ها جای‌هم را می‌گیرند؛ در ترمینال cls.
# ============================================================================

LIVE_ANIMATIONS = True   # set False to skip the animations on terminal runs

# glyphs used only by the animations / ایموجی‌های مخصوص انیمیشن
ANIM_SUIT, ANIM_WALL, ANIM_CENTER = "🦾", "💥", "🎯"
ANIM_FLAME, ANIM_WIND = "🔥", "🌪"
ANIM_HOUSE_SUITS = ("🔴", "🔵", "🟡")               # Mark 42 / Centurion / Hulkbuster
ANIM_HOUSE_FIRE, ANIM_HOUSE_TARGET = "💥", "🔋"
ANIM_HOUSE_EMPTY = " ⬜ "


def _make_frame_clearer():
    """Return a 'clear previous frame' function for the animation loop.
    تابع پاک‌سازی فریم قبلی: در جوپیتر clear_output(wait=True)، در ترمینال
    تعاملی cls/clear — و اگر خروجی pipe/redirect شده باشد فقط یک خط جداکننده
    (چون cls بدون کنسول واقعی ممکن است بسیار کند یا حتی معلق شود)."""
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            from IPython.display import clear_output
            return lambda: clear_output(wait=True)
    except ImportError:
        pass
    if sys.stdout.isatty():
        cmd = "cls" if os.name == "nt" else "clear"
        return lambda: os.system(cmd)
    return lambda: print("─" * 64)   # piped/background: keep a frame separator


# --------------------------------------------------------------- PART 1 anim
def _flight_gauge(tilt, action=None):
    """One console gauge row: 💥 walls, 🎯 balance point, 🦾 suit (+🔥 flame).
    یک خط گیج: دیوارها، نقطه‌ی تعادل، زره — و شعله‌ی رانشگر در لحظه‌ی شلیک."""
    cells = [" . "] * 9
    cells[0], cells[8] = f" {ANIM_WALL} ", f" {ANIM_WALL} "
    cells[4] = f" {ANIM_CENTER} "
    here = tilt + 4
    if action is not None:                # show the thruster that is firing now
        flame = here - 1 if action == 0 else here + 1 if action == 2 else None
        if flame is not None and cells[flame] == " . ":
            cells[flame] = f" {ANIM_FLAME} "
    cells[here] = f" {ANIM_SUIT} "
    return "[ " + "".join(cells) + " ]"


def animate_flight_stabilizer(q_table, env=None, start_tilt=-3, steps=15,
                              frame_dt=0.6, seed=7, hold_last=1.5):
    """LIVE HUD animation: the suit stabilises itself on 0° with its LEARNED
    policy while random wind gusts knock it off course.
    انیمیشن زنده‌ی HUD: زره با سیاستِ آموخته‌شده خودش را روی 0 قفل می‌کند،
    حتی وقتی تلاطم باد ناگهانی آن را از مسیر خارج می‌کند."""
    if env is None:
        env = StarkFlightBalancerEnv()
    rng = random.Random(seed)
    clear_frame = _make_frame_clearer()

    # compact policy strip from the Q-table: ▶ = pushed right, • = hold, ◀ = pushed left
    # نوار سیاستِ آموخته‌شده: برای هر شیب کدام رانشگر شلیک می‌شود
    effect = {0: "▶", 1: "•", 2: "◀"}
    policy_strip = "  ".join(
        "💥" if t in (-4, 4) else effect[int(np.argmax(q_table[t + 4]))]
        for t in range(-4, 5))

    gust_at = set(rng.sample(range(3, steps), min(3, max(0, steps - 3))))
    tilt = env.reset(start_tilt)
    total, was_gust = 0, False

    for step in range(1, steps + 1):
        action = int(np.argmax(q_table[tilt + 4]))      # pure learned policy
        angle = tilt * 15
        status = ("✅ STABLE HOVER — تعادل کامل" if tilt == 0
                  else "⚠️ CORRECTING TILT — در حال اصلاح شیب")
        q_row = q_table[tilt + 4]

        clear_frame()
        print("=" * 64)
        print("   🤖 J.A.R.V.I.S  FLIGHT STABILIZER HUD   (LIVE FEED) 🤖")
        print("=" * 64)
        print(f" ⏱ گام زمانی: {step:2d}/{steps}  [{'▰' * step}{'▱' * (steps - step)}]")
        print(f" 📐 زاویه انحراف: {angle:+3d}°   |   پاداش تجمعی: {total:+4d}   |   {status}\n")
        print("      چپ  " + _flight_gauge(tilt, action) + "  راست")
        print("            ▲  🎯 = نقطه‌ی تعادل (0°)   |   💥 = دیوار آزمایشگاه\n")
        print(f" ⚙️ فرمان جارویس: {env.ACTION_NAMES[action]}")
        print(f" 🧠 Q(شیب {tilt:+d}) =   ▶ {q_row[0]:+7.1f}    • {q_row[1]:+7.1f}    ◀ {q_row[2]:+7.1f}")
        if was_gust:
            print(f" {ANIM_WIND} GUST! باد شدید زره را از مسیر خارج کرد — جارویس در حال واکنش...")
        print(f" 📋 سیاست آموخته‌شده:  {policy_strip}")
        print("=" * 64)

        time.sleep(frame_dt)

        # --- apply the commanded action with the REAL env dynamics ---
        tilt, r, done = env.step(action)
        total += r
        was_gust = step in gust_at
        if was_gust:                        # surprise turbulence 🌪 pushes the suit
            push = rng.choice((-2, -1, 1, 2))
            tilt = max(-3, min(3, tilt + push))
            env.tilt = tilt
        if done:                            # safety net: never crash in the demo
            tilt = max(-3, min(3, tilt))
            env.tilt = tilt

    # final steady frame / فریم پایانی
    clear_frame()
    print("=" * 64)
    print("   🤖 J.A.R.V.I.S  FLIGHT STABILIZER HUD   (LIVE FEED) 🤖")
    print("=" * 64)
    if tilt == 0:
        print(f" ✅ مأموریت کامل: {steps} گام پرواز بدون سقوط — تعادل کامل تثبیت شد")
    else:
        print(f" ✅ مأموریت کامل: {steps} گام پرواز بدون سقوط — پرواز پایدار "
              f"(لرزش میکرو حول 🎯 طبیعی است)")
    print(f"    زاویه‌ی نهایی: {tilt * 15:+3d}°   |   پاداش کل: {total:+4d}\n")
    print("      چپ  " + _flight_gauge(tilt) + "  راست")
    print("\n    \"I am Iron Man.\" 🦾")
    print("=" * 64)
    time.sleep(hold_last)
    return total, tilt


# --------------------------------------------------------------- PART 2 anim
def _house_grid(env, shown):
    """The 6x6 radar grid; 'shown' = last position of every suit (docked suits
    stay visible on their reactor).  شبکه‌ی رادار؛ زره‌ی فرودآمده روی رآکتور می‌ماند."""
    lines = ["      " + "".join(f"{c:^3}" for c in range(env.SIZE))]
    for r in range(env.SIZE):
        cells = []
        for c in range(env.SIZE):
            glyph = ANIM_HOUSE_EMPTY
            if (r, c) in env.FIRES:
                glyph = f" {ANIM_HOUSE_FIRE} "
            for t in range(env.N_AGENTS):
                if env.TARGETS[t] == (r, c) and shown[t] != (r, c):
                    glyph = f" {ANIM_HOUSE_TARGET} "
            for i in range(env.N_AGENTS):   # suits are drawn on top
                if shown[i] == (r, c):
                    glyph = f" {ANIM_HOUSE_SUITS[i]} "
            cells.append(glyph)
        lines.append(f"   {r}  " + "".join(cells))
    return "\n".join(lines)


def animate_house_party(q_tables, env=None, frame_dt=0.5, hold_last=2.0):
    """LIVE radar animation: the 3 suits fly from row 0 to their arc reactors
    at row 5, each moving with its OWN learned Q-table (greedy), one frame per
    round, turn-based exactly like in training.
    انیمیشن زنده‌ی رادار: ۳ زره با سیاست‌های آموخته‌شده‌ی خودشان و بدون برخورد
    از ردیف ۰ به رآکتورهایشان در ردیف ۵ می‌روند — یک فریم در هر راند."""
    if env is None:
        env = HousePartyGridEnv()
    clear_frame = _make_frame_clearer()
    env.reset()
    shown = list(env.STARTS)                # last seen position of each suit
    docked_round = [None] * env.N_AGENTS
    collisions = 0
    rounds_used = env.MAX_ROUNDS

    def status_lines(rnd, moves):
        move_map = {i: (before, after, a) for i, before, after, a in moves}
        out = []
        for i in range(env.N_AGENTS):
            icon, name = ANIM_HOUSE_SUITS[i], env.SUIT_NAMES[i]
            if docked_round[i] is not None:
                out.append(f"   {icon} {name:<16}: ✅ روی رآکتورش فرود آمد (راند {docked_round[i]})")
            elif i in move_map:
                before, after, a = move_map[i]
                out.append(f"   {icon} {name:<16}: {before} → {after}   [{env.ACTION_NAMES[a]}]")
            else:
                out.append(f"   {icon} {name:<16}: آماده‌ی پرتاب ✈")
        return out

    def draw(rnd, moves=()):
        n_docked = sum(d is not None for d in docked_round)
        clear_frame()
        print("=" * 64)
        print("  🚀 HOUSE PARTY PROTOCOL — MULTI-AGENT RADAR (LIVE)")
        print("=" * 64)
        print(f"  📡 راند: {rnd:2d}/{env.MAX_ROUNDS}   |   فرود: {n_docked}/3   |   "
              f"برخورد تیمی: {collisions} {'✅ ایمن' if collisions == 0 else '💥 خطر!'}\n")
        print(_house_grid(env, shown))
        print()
        for line in status_lines(rnd, moves):
            print(line)
        print("\n  🔴 Mark 42   🔵 Silver Centurion   🟡 Hulkbuster   🔋 رآکتور آرک   💥 آتش")
        print("=" * 64)

    draw(0)                                  # opening frame: the launch pad
    time.sleep(max(1.0, frame_dt))           # let the viewer see the layout

    for rnd in range(1, env.MAX_ROUNDS + 1):
        moves = []
        for i in range(env.N_AGENTS):        # fixed turn order, like in training
            if env.landed[i]:
                continue
            s = env.state_index(i)
            a = int(np.argmax(q_tables[i][s]))           # pure learned policy
            before = env.pos[i]
            _s2, _r, done_i, _ev, info = env.step(i, a)
            collisions += int(info["collision"])
            after = env.TARGETS[i] if done_i else env.pos[i]
            if done_i:
                docked_round[i] = rnd
            shown[i] = after
            moves.append((i, before, after, a))
        draw(rnd, moves)
        time.sleep(frame_dt)
        if all(env.landed):
            rounds_used = rnd
            break

    # completion frame / فریم پایانی مأموریت
    clear_frame()
    print("=" * 64)
    print("  🚀 HOUSE PARTY PROTOCOL — MULTI-AGENT RADAR (LIVE)")
    print("=" * 64)
    if all(env.landed):
        print(f"  ✅ هر ۳ زره روی رآکتور خود فرود آمدند — {rounds_used} راند، "
              f"{collisions} برخورد!\n")
    else:
        print("  ⚠ مهلت مأموریت تمام شد؛ همه‌ی زره‌ها فرود نیامدند.\n")
    print(_house_grid(env, shown))
    print("\n  🔴 Mark 42   🔵 Silver Centurion   🟡 Hulkbuster   🔋 رآکتور آرک   💥 آتش")
    print("=" * 64)
    time.sleep(hold_last)
    return collisions, rounds_used


# ============================================================================
# ============================================================================
#   TEACHING / ACADEMY  --  learn the RL logic step by step, operation by
#   operation (used by stark_rl_academy.ipynb and the ACADEMY tab of the game)
#   بخش آموزشی آکادمی -- یادگیری گام‌به‌گامِ منطق RL، هر عملیات را جدا می‌بینیم
# ============================================================================
# ============================================================================

RL_LOOP_DIAGRAM = r"""
        ┌───────────────────────┐                 ┌───────────────────────┐
        │      🤖 AGENT         │                 │    🌍 ENVIRONMENT      │
        │  (J.A.R.V.I.S / Q)    │                 │  (Suit + Lab + Wind)   │
        └───────────┬───────────┘                 └───────────┬───────────┘
                    │                                         │
                    │   action  aₜ   (fire thruster / hold)    │
                    │  ────────────────────────────────────────▶
                    │                                         │
                    │   state   sₜ₊₁  (new tilt)               │
                    │   reward  rₜ₊₁  (+20 / −2 / −100)        │
                    ◀──────────────────────────────────────── │
                    │                                         │
        ┌───────────┴───────────────────────────────────────────┐
        │  🧠 LEARN:  Q(sₜ,aₜ) ← Q(sₜ,aₜ) + α[rₜ₊₁ + γ·max Q(sₜ₊₁,a)] │
        └─────────────────────────────────────────────────────────┘
"""


def print_rl_loop_diagram():
    """The agent-environment loop -- THE picture of reinforcement learning.
    حلقه‌ی عامل-محیط — همان «یک عکس» که کل یادگیری تقویتی را توضیح می‌دهد."""
    print(RL_LOOP_DIAGRAM)
    print("  🔁 این حلقه در هر گام تکرار می‌شود: انتخاب اکشن → دیدن پاداش و وضعیت جدید → یادگیری")
    print("  🔁 نه دستور مستقیم داده می‌شود، نه مثال برچسب‌دار — فقط «پاداش». این همان فرق RL با بقیه است.")


def pretty_q_table(Q, title="Q-TABLE", highlight=None):
    """Console rendering of the 9x3 Q-table (optionally highlight (s,a)).
    نمایش جدول ۹×۳ — اختیاراً یک خانه (s,a) را هایلایت می‌کند."""
    print(f"\n  🧠 {title}   (هر خانه = «چقدر خوب بود این اکشن در این وضعیت؟»)")
    print("      tilt  │   ▶ RIGHT   │    • HOLD   │   ◀ LEFT")
    print("     " + "────┼" + "─────────────┼" + "─────────────┼" + "─────────────")
    for t in range(-4, 5):
        row = Q[t + 4]
        cells = []
        for a in range(3):
            txt = f"{row[a]:+13.1f}"
            if highlight == (t + 4, a):
                txt = f"  ⟨{row[a]:+9.1f}⟩"    # ← the cell being updated
            cells.append(txt)
        wall = "  💥" if t in (-4, 4) else ("  🎯" if t == 0 else "   ")
        print(f"     {t:+d}   │{cells[0]}│{cells[1]}│{cells[2]}│{wall}")
    print("     " + "────┴" + "─────────────┴" + "─────────────┴" + "─────────────")


def demonstrate_env_steps(env=None, actions=(0, 1, 2, 1), start_tilt=0):
    """Feel the environment with your own hands: each call = one transition
    (s, a) -> (s', r).  محیط را دستی تجربه کن: هر فرمان یک انتقال واقعی است."""
    if env is None:
        env = StarkFlightBalancerEnv()
    s = env.reset(start_tilt)
    print("\n  🕹 محیط را امتحان کنیم — هر خط یک «انتقال» واقعی محیط است:")
    print(f"     شروع: tilt = {s:+d}     (پاداش‌ها: 🎯 0° ← +20 | هر گام دیگر ← −2 | 💥 دیوار ← −100)")
    for t, a in enumerate(actions, 1):
        s2, r, done = env.step(a)
        print(f"     گام {t}: tilt {s:+d} --{env.ACTION_NAMES[a]:<24}--> tilt {s2:+d} | "
              f"پاداش r = {r:+4d} | {tilt_bar(s2)}")
        s = s2
        if done:
            print("     💥 اپیزود تمام شد — برخورد با دیوار!")
            break
    return s


def demo_bellman_by_hand(env=None, Q=None, s=-2, a=0, alpha=0.1, gamma=0.95):
    """ONE full Q-learning update, every number spelled out.
    یک به‌روزرسانی کامل بلمن با همه‌ی عددها — همان فرمول، این بار با عدد واقعی."""
    if env is None:
        env = StarkFlightBalancerEnv()
    if Q is None:
        Q = np.zeros((env.N_STATES, env.N_ACTIONS))
    env.reset(s)
    old = Q[s + 4, a]
    s2, r, done = env.step(a)
    best_next = 0.0 if done else np.max(Q[s2 + 4])
    target = r + gamma * best_next
    delta = target - old
    new = old + alpha * delta

    print("\n  ✍️ یک قدم یادگیری — با دست، با عدد:")
    print(f"     وضعیت s = {s:+d} (شیب زره)، اکشن a = {env.ACTION_NAMES[a]}")
    print(f"     ① محیط جواب می‌دهد: s' = {s2:+d}، پاداش r = {r:+d}"
          + ("  (اپیزود تمام شد → آینده‌ای نیست)" if done else "") + "")
    print(f"     ② بهترین ارزش آینده: max Q(s',·) = {best_next:+.1f}")
    print(f"     ③ هدف یادگیری:  r + γ·max Q(s',·) = {r:+d} + {gamma}×{best_next:+.1f} = {target:+.2f}")
    print(f"     ④ خطای فعلی (TD error): هدف − باور فعلی = {target:+.2f} − {old:+.1f} = {delta:+.2f}")
    print(f"     ⑤ باور جدید: Q ← {old:+.1f} + {alpha}×({delta:+.2f}) = {new:+.2f}")
    print("        یعنی: «کمی به سمت حقیقتِ تازه حرکت کن — نه پرش کامل» (این کارِ α است)")

    Q[s + 4, a] = new                        # actually apply it / واقعاً اعمال کن
    pretty_q_table(Q, title="Q-TABLE بعد از فقط یک به‌روزرسانی", highlight=(s + 4, a))
    return Q


def record_first_training_steps(env=None, n_steps=10, alpha=0.1, gamma=0.95, seed=123):
    """Replay the VERY FIRST training steps of an empty Q-table and record the
    full Bellman arithmetic of each (used by the notebook + the game's ACADEMY).
    اولین قدم‌های آموزشِ جدولِ خالی را ضبط می‌کند — با همه‌ی جزئیات معادله‌ی بلمن."""
    if env is None:
        env = StarkFlightBalancerEnv()
    st = random.getstate()
    random.seed(seed)
    try:
        Q = np.zeros((env.N_STATES, env.N_ACTIONS))
        s = env.reset(-2)                    # start two steps left of the target
        steps = []
        for t in range(1, n_steps + 1):
            a = random.randrange(3)          # ε=1.0 at the start: pure exploration
            old = float(Q[s + 4, a])
            s2, r, done = env.step(a)
            best_next = 0.0 if done else float(np.max(Q[s2 + 4]))
            target = r + gamma * best_next
            new = old + alpha * (target - old)
            Q[s + 4, a] = new
            steps.append({
                "t": t, "s": int(s), "a": int(a), "s2": int(s2), "r": int(r),
                "done": bool(done), "gamma": gamma, "alpha": alpha,
                "old": round(old, 2), "best_next": round(best_next, 2),
                "target": round(target, 2), "td": round(target - old, 2),
                "new": round(new, 2), "q_after": [round(float(v), 2) for v in Q[s + 4]],
                "q_all": [[round(float(v), 2) for v in row] for row in Q],
            })
            s = s2
            if done:
                s = env.reset(-2)
        return steps
    finally:
        random.setstate(st)


def train_flight_with_snapshots(env=None, episodes=800, snapshot_at=(1, 50, 100, 200, 400, 800),
                                alpha=0.1, gamma=0.95):
    """train_flight_stabilizer + Q-table snapshots at chosen episodes.
    آموزش + عکس‌های لحظه‌ای از جدول Q در اپیزودهای انتخابی (برای انیمیشن شکل‌گیری مغز)."""
    if env is None:
        env = StarkFlightBalancerEnv()
    snapset = set(snapshot_at)
    snaps = []

    def cb(ep, Q):
        if ep + 1 in snapset:
            snaps.append({"ep": ep + 1,
                          "epsilon": round(epsilon_at(ep, episodes), 3),
                          "Q": np.array(Q, copy=True)})

    Q, rewards, crashes = train_flight_stabilizer(
        env, episodes=episodes, alpha=alpha, gamma=gamma, progress_every=0, on_episode=cb)
    # enrich snapshots with the learning curve up to that episode
    # افزودن منحنی یادگیری تا آن اپیزود به هر اسنپ‌شات
    for sn in snaps:
        upto = rewards[:sn["ep"]]
        sn["avg_reward"] = round(float(np.mean(upto[-50:])), 1)
        sn["crash_rate"] = round(100 * float(np.mean(crashes[:sn["ep"]][-50:])), 1)
    return Q, rewards, crashes, snaps


def train_house_with_snapshots(env=None, episodes=1500,
                               snapshot_at=(1, 100, 300, 600, 1000, 1500),
                               alpha=0.1, gamma=0.95):
    """train_house_party + per-suit policy maps (arrows) at chosen episodes.
    آموزش چندعامله + نقشه‌ی سیاستِ هر زره (فلش‌ها) در اپیزودهای انتخابی."""
    if env is None:
        env = HousePartyGridEnv()
    arrow = {0: "↑", 1: "↓", 2: "←", 3: "→", 4: "•"}

    def policy_maps(Qs):
        maps = []
        for i in range(env.N_AGENTS):
            grid = []
            for r in range(env.SIZE):
                row = ""
                for c in range(env.SIZE):
                    if (r, c) in env.FIRES:
                        row += "🔥 "
                    elif (r, c) == env.TARGETS[i]:
                        row += "⚡ "
                    else:  # policy assuming no teammate nearby (radar = 0,0)
                        s = ((r * env.SIZE + c) * 6 + 0) * 6 + 0
                        row += arrow[int(np.argmax(Qs[i][s]))] + " "
                grid.append(row)
            maps.append(grid)
        return maps

    snapset = set(snapshot_at)
    snaps = []

    def cb(ep, Qs):
        if ep + 1 in snapset:
            snaps.append({"ep": ep + 1,
                          "epsilon": round(epsilon_at(ep, episodes), 3),
                          "policies": policy_maps(Qs)})

    Qs, team, coll, succ = train_house_party(
        env, episodes=episodes, alpha=alpha, gamma=gamma, progress_every=0, on_episode=cb)
    for sn in snaps:
        upto = team[:sn["ep"]]
        sn["team_avg"] = round(float(np.mean(upto[-50:])), 1)
        sn["coll_avg"] = round(float(np.mean(coll[:sn["ep"]][-50:])), 2)
    return Qs, team, coll, succ, snaps


def epsilon_schedule_preview(episodes=800, marks=(1, 100, 200, 400, 600, 800)):
    """Show how ε decays: from total explorer to total exploiter.
    نشان‌دادن کاهش ε: از کاوشگر کامل به بهره‌بردار کامل."""
    print("\n  🎲 ε (اپسیلون) در طول آموزش — احتمال «اکشن تصادفی» به‌جای «بهترینِ جدول»:")
    for m in marks:
        e = epsilon_at(m - 1, episodes)
        explore = int(e * 20)
        print(f"     اپیزود {m:4d}: ε = {e:.2f}   [{'🎲' * explore}{'·' * (20 - explore)}]"
              f"   {(1 - e) * 100:4.0f}% از جدول استفاده می‌کند")
    print("     اول همه‌چیز را امتحان می‌کنیم (کاوش)، آخر فقط بهترین را اجرا می‌کنیم (بهره‌برداری).")


# ------------------------------------------------- animate the Q-table learning
def _q_row_bars(vals, scale, width=8):
    """Tiny bar per value on a SHARED scale (heights comparable across rows).
    میله‌ی کوچک برای هر مقدار Q با مقیاس مشترک تا ارتفاع‌ها قابل مقایسه باشند."""
    out = []
    for v in vals:
        n = int(round(abs(v) / scale * width)) if scale else 0
        out.append(("█" * n).ljust(width))
    return out


def animate_q_table_learning(snapshots, frame_dt=1.1, hold_last=2.0):
    """LIVE animation: watch the 9x3 Q-table LEARN itself, snapshot by snapshot
    (episode counter, ε, crash rate and the moving-average curve update too).
    انیمیشن زنده‌ی شکل‌گیری جدول Q — اسنپ‌شات به اسنپ‌شات، با ε و نرخ سقوط و منحنی یادگیری."""
    if not snapshots:
        print("  (no snapshots to animate)")
        return
    clear_frame = _make_frame_clearer()
    effect = {0: "▶", 1: "•", 2: "◀"}
    mx_all = max(abs(float(v)) for sn in snapshots for v in np.asarray(sn["Q"]).ravel()) or 1.0

    for i, sn in enumerate(snapshots):
        Q = np.asarray(sn["Q"])
        clear_frame()
        print("=" * 64)
        print("   🧠 J.A.R.V.I.S BRAIN SCAN — جدول Q در حال یادگیری است")
        print("=" * 64)
        print(f" ⏱ اپیزود {sn['ep']:4d}   |   ε = {sn['epsilon']:.2f}   |   "
              f"میانگین پاداش (۵۰ اپیزود اخیر) = {sn['avg_reward']:+8.1f}   |   "
              f"نرخ سقوط = {sn['crash_rate']:5.1f}%")
        print(f" 📈 پیشرفت: [{'▰' * (i + 1)}{'▱' * (len(snapshots) - i - 1)}]"
              f"  {100 * (i + 1) // len(snapshots):3d}%\n")
        print("      tilt │  ▶ RIGHT        • HOLD         ◀ LEFT       سیاست")
        print("     " + "────┼" + "──────────────────────────────────" + "┼───────")
        for t in range(-4, 5):
            row = Q[t + 4]
            bars = _q_row_bars(list(row), mx_all)
            pol = "💥" if t in (-4, 4) else effect[int(np.argmax(row))]
            wall = "💥" if t in (-4, 4) else ("🎯" if t == 0 else " ")
            print(f"     {t:+d}   │ {row[0]:+7.1f} {bars[0]} │ {row[1]:+7.1f} {bars[1]}"
                  f" │ {row[2]:+7.1f} {bars[2]}│   {pol}  {wall}")
        print()
        best = np.max(np.abs(Q[4])) or 1.0
        print(f" 💡 نکته: مقدار Q(tilt=0, HOLD) الان {Q[4, 1]:+7.1f} است — "
              f"جدول دارد می‌فهمد «روی 🎯 بمان» یعنی پاداش.")
        if i == 0:
            print("    (اولین اسنپ‌شات: تقریباً همه‌چیز صفر — جارویس هنوز چیزی ندیده است!)")
        elif i == len(snapshots) - 1:
            print("    (آخرین اسنپ‌شات: جدول پر شده — سیاست نهایی همین argmax ها است!)")
        print("=" * 64)
        time.sleep(frame_dt)

    time.sleep(hold_last)
    return snapshots[-1]["ep"]


# ------------------------------------------------- animate multi-agent policies
def animate_policy_arrows(snapshots, env=None, suit=0, frame_dt=1.1, hold_last=2.0):
    """LIVE animation: one suit's brain organising itself on the 6x6 grid --
    each frame = one training snapshot; every cell shows the learned action
    (↑↓←→•) assuming no teammate nearby.  شکل‌گیری مغزِ یک زره روی شبکه:
    فلش هر خانه = اکشن آموخته‌شده؛ از آشوب به مسیر منظم به سمت رآکتور."""
    if env is None:
        env = HousePartyGridEnv()
    if not snapshots:
        print("  (no snapshots to animate)")
        return
    clear_frame = _make_frame_clearer()

    for i, sn in enumerate(snapshots):
        grid = sn["policies"][suit]
        clear_frame()
        print("=" * 64)
        print(f"   🧠 BRAIN SCAN — {env.SUIT_NAMES[suit]} {env.SUIT_EMOJI[suit]}"
              f"   (نقشه‌ی سیاست، بدون هم‌تیمی در رادار)")
        print("=" * 64)
        print(f" ⏱ اپیزود {sn['ep']:5d} | ε = {sn['epsilon']:.2f} | "
              f"امتیاز تیمی (MA50) = {sn['team_avg']:+7.1f} | برخورد = {sn['coll_avg']:.2f}")
        print(f" 📈 [{'▰' * (i + 1)}{'▱' * (len(snapshots) - i - 1)}]"
              f"  {100 * (i + 1) // len(snapshots):3d}%\n")
        print("        0    1    2    3    4    5")
        for r in range(env.SIZE):
            # each policy row stores 2 chars per cell / هر خانه دو کاراکتر است
            cells = "  ".join(grid[r][c * 2:c * 2 + 2].strip().ljust(2) for c in range(6))
            print(f"     {r}  {cells}")
        print("\n    ↑↓←→ اکشن آموخته‌شده در هر خانه   🔥 آتش (F)   ⚡/T رآکتورِ خودش   • بمان")
        if i == 0:
            print("    (اپیزود ۱: فلش‌ها عملاً تصادفی‌اند — زره هیچ چیزی نفهمیده!)")
        else:
            print(f"    (اپیزود {sn['ep']}: فلش‌ها دارند به سمت ⚡ پایین شبکه منظم می‌شوند…)")
        print("=" * 64)
        time.sleep(frame_dt)

    time.sleep(hold_last)
    return snapshots[-1]["ep"]


def teach_house_radar_example(env=None):
    """How one suit's state is built: own cell + radar of both teammates.
    وضعیتِ یک زره چگونه ساخته می‌شود: خانه‌ی خودش + رادار دو هم‌تیمی."""
    if env is None:
        env = HousePartyGridEnv()
    env.reset()
    # put the teammates right next to suit 0 to make the radar interesting
    # هم‌تیمی‌ها را کنار زره ۰ می‌گذاریم تا رادار جالب شود
    env.pos[1], env.pos[2] = (1, 0), (0, 2)
    i = 0
    r, c = env.pos[i]
    others = [j for j in range(env.N_AGENTS) if j != i]
    rad1, rad2 = env.radar(i, others[0]), env.radar(i, others[1])
    s = env.state_index(i)

    print("\n  📡 وضعیت (state) زره Mark 42 چطور ساخته می‌شود؟")
    print(env.render())
    names = {0: "خارج از رادار", 1: "شمال ⬆", 2: "جنوب ⬇", 3: "غرب ⬅", 4: "شرق ➡", 5: "مورب ◤"}
    print(f"     ① خانه‌ی خودش: ({r},{c})   → عدد {(r * env.SIZE + c):2d}")
    print(f"     ② رادارِ {env.SUIT_NAMES[others[0]]}: {names[rad1]} → عدد {rad1}")
    print(f"     ③ رادارِ {env.SUIT_NAMES[others[1]]}: {names[rad2]} → عدد {rad2}")
    print(f"     ⟹ شاخص وضعیت = (({r}×6+{c})×6+{rad1})×6+{rad2} = {s}"
          f"   (از {env.n_states} وضعیت ممکن)")
    print("     بدون رادار، دو زره که به هم نزدیک می‌شوند از نظر هم نامرئی بودند → برخورد!")
    return s




def plot_stark_dashboard(rewards1, crashes1, team2, coll2, window=50):
    """One dark 'Stark HUD' figure with 2 telemetry subplots + saves a PNG.
    یک تصویر تیره‌ی «HUD استارک» با دو نمودار + ذخیره‌ی خروجی PNG برای اسلایدها."""
    plt.rcParams.update({
        "figure.facecolor": "#0b0f14",
        "axes.facecolor": "#10161d",
        "axes.edgecolor": "#2b3648",
        "axes.labelcolor": "#c9d6e3",
        "text.color": "#c9d6e3",
        "xtick.color": "#8fa3b8",
        "ytick.color": "#8fa3b8",
        "grid.color": "#223041",
        "font.size": 10,
    })
    gold, cyan, red = "#ffc83d", "#3fb6ff", "#ff4d4d"
    box_style = dict(boxstyle="round,pad=0.45", facecolor="#141b24",
                     edgecolor=cyan, alpha=0.92)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.8))
    fig.suptitle("STARK INDUSTRIES // J.A.R.V.I.S  --  R.L. TELEMETRY HUD",
                 color=gold, fontsize=15, fontweight="bold")

    # ---------------- Subplot 1 : flight stabilizer learning curve
    ep1 = np.arange(1, len(rewards1) + 1)
    ax1.plot(ep1, rewards1, color=cyan, alpha=0.30, lw=0.9, label="reward per episode")
    ma1 = moving_average(rewards1, window)
    start1 = len(rewards1) - len(ma1) + 1
    ax1.plot(np.arange(start1, start1 + len(ma1)), ma1, color=gold, lw=2.4,
             label=f"moving average ({window})")
    crash_eps = np.where(crashes1)[0] + 1
    if len(crash_eps):
        ax1.scatter(crash_eps, np.full(len(crash_eps), min(rewards1) - 30),
                    marker="x", s=26, color=red, linewidths=1.4, label="crash")
    ax1.set_ylim(min(rewards1) - 60, max(rewards1) + 40)
    ax1.set_title("PART 1 - Flight Stabilizer (Single-Agent Q-Learning)", color=gold)
    ax1.set_xlabel("episode")
    ax1.set_ylabel("total reward")
    ax1.grid(True, alpha=0.5)
    ax1.legend(loc="lower right", facecolor="#141b24", edgecolor="#2b3648")
    stats1 = (f"avg reward  ep 1-100   : {np.mean(rewards1[:100]):+8.1f}\n"
              f"avg reward  ep 701-800 : {np.mean(rewards1[-100:]):+8.1f}\n"
              f"crash rate  ep 1-100   : {100 * np.mean(crashes1[:100]):7.1f}%\n"
              f"crash rate  ep 701-800 : {100 * np.mean(crashes1[-100:]):7.1f}%")
    ax1.text(0.03, 0.97, stats1, transform=ax1.transAxes, va="top", ha="left",
             fontsize=8.5, family="monospace", bbox=box_style)

    # ---------------- Subplot 2 : house party coordination
    ep2 = np.arange(1, len(team2) + 1)
    ma_t = moving_average(team2, window)
    start2 = len(team2) - len(ma_t) + 1
    ax2.plot(np.arange(start2, start2 + len(ma_t)), ma_t, color=gold, lw=2.4,
             label="team coordination score (MA 50)")
    ax2.set_title("PART 2 - House Party Protocol (Multi-Agent RL)", color=gold)
    ax2.set_xlabel("episode")
    ax2.set_ylabel("team total reward")
    ax2.grid(True, alpha=0.5)

    ax2t = ax2.twinx()
    ma_c = moving_average(coll2, window)
    ax2t.plot(np.arange(start2, start2 + len(ma_c)), ma_c, color=red, lw=1.9,
              ls="--", label="collisions / episode (MA 50)")
    ax2t.set_ylabel("collisions per episode", color=red)
    ax2t.tick_params(axis="y", colors=red)
    ax2t.spines["right"].set_color(red)
    ax2t.set_ylim(0, max(1.0, max(ma_c) * 1.25))

    stats2 = (f"team score  ep 1-100     : {np.mean(team2[:100]):+8.1f}\n"
              f"team score  ep 1401-1500 : {np.mean(team2[-100:]):+8.1f}")
    ax2.text(0.03, 0.97, stats2, transform=ax2.transAxes, va="top", ha="left",
             fontsize=8.5, family="monospace", bbox=box_style)

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2t.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="lower right", facecolor="#141b24",
               edgecolor="#2b3648")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = os.path.join(BASE_DIR, "stark_hud_dashboard.png")
    fig.savefig(png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n  💾 Dashboard saved to: {png}   (use it directly on a slide)")
    plt.show()


# ============================================================================
# ============================================================================
#   MAIN
# ============================================================================
# ============================================================================

def main():
    t0 = time.time()
    banner("⚡ S.T.A.R.K. INDUSTRIES — R.L. LIVE DEMO ⚡")
    print("   \"Sometimes you gotta run before you can walk.\"  — Tony Stark")
    print("   SEED =", SEED, "(fixed => deterministic output, safe for the live demo)")

    # ---------------------------------------------------------------- PART 1
    banner("PART 1 — SINGLE-AGENT Q-LEARNING : IRON MAN FLIGHT STABILIZER")
    env1 = StarkFlightBalancerEnv()

    # The untrained suit crashes in ~2/3 of early episodes. To guarantee the
    # dramatic "untrained vs trained" contrast on stage, we replay the FIRST
    # crashing attempt (deterministic under the fixed SEED).
    # زره‌ی آموزش‌ندیده در حدود دوسوم اپیزودهای اول سقوط می‌کند؛ برای اینکه
    # مقایسه‌ی دراماتیک روی صحنه همیشه دیده شود، اولین تلاشِ سقوط‌دار را
    # نمایش می‌دهیم (با seed ثابت، خروجی هر بار یکسان است).
    attempt, lines = 0, None
    for k in range(1, 11):
        lines, _tot, crashed, _st = run_flight_episode(env1, None, 1.0, -3)
        attempt = k
        if crashed:
            break
    print("\n" + "-" * 78)
    print(f"  🎬 EPISODE {attempt} (EARLY TRAINING) — UNTRAINED SUIT "
          f"(pure random actions, ε = 1.0)")
    print("-" * 78)
    print("\n".join(lines))

    print("\n  🧠 TRAINING J.A.R.V.I.S — 800 episodes of trial & error ...")
    print("     update rule / معادله‌ی به‌روزرسانی:")
    print("        Q(s,a) <- Q(s,a) + α [ r + γ · max_a' Q(s',a') − Q(s,a) ]")
    print("        α = 0.1 (learning rate / نرخ یادگیری)   γ = 0.95 (discount / ضریب تخفیف)")
    Q1, rewards1, crashes1 = train_flight_stabilizer(env1, episodes=800)
    print_learned_policy(env1, Q1)

    print_flight_demo(env1, Q=Q1, epsilon=0.0, start_tilt=-3,
                      title="EPISODE 800 — TRAINED J.A.R.V.I.S (greedy policy, ε = 0)")

    if LIVE_ANIMATIONS:
        print("\n  🎬 LIVE ANIMATION — HUD تعادل پرواز، فریم‌به‌فریم "
              "(سیاست آموخته‌شده + تلاطم باد) ...")
        animate_flight_stabilizer(Q1, env1)

    # ---------------------------------------------------------------- PART 2
    banner("PART 2 — MULTI-AGENT RL : HOUSE PARTY PROTOCOL (3 SUITS, COOPERATIVE)")
    env2 = HousePartyGridEnv()
    print("  Initial airspace / فضای هوایی اولیه:")
    env2.reset()
    print(env2.render())
    print("\n  🧠 TRAINING 3 INDEPENDENT Q-TABLES — 1500 episodes ...")
    Qs2, team2, coll2, succ2 = train_house_party(env2, episodes=1500)

    print("\n  📈 TRAINING SUMMARY / خلاصه‌ی آموزش:")
    print(f"     team score    : first 100 eps = {np.mean(team2[:100]):+8.1f}   ->   "
          f"last 100 eps = {np.mean(team2[-100:]):+8.1f}")
    print(f"     collisions    : first 100 eps = {np.mean(coll2[:100]):8.2f}   ->   "
          f"last 100 eps = {np.mean(coll2[-100:]):8.2f}  (per episode)")
    print(f"     success rate  : first 100 eps = {100 * np.mean(succ2[:100]):7.1f}%   ->   "
          f"last 100 eps = {100 * np.mean(succ2[-100:]):7.1f}%")

    demo_collisions, rounds_used = print_house_party_demo(env2, Qs2)
    if demo_collisions == 0:
        print("\n  ✅ Final mission was 100% collision-free — coordination learned!")
    else:
        print(f"\n  ⚠ Final mission had {demo_collisions} collision(s) — "
              f"mention that IQL is a simple baseline.")

    if LIVE_ANIMATIONS:
        print("\n  🎬 LIVE ANIMATION — رادار چندعامله، یک فریم در هر راند "
              "(هر ۳ زره با سیاست آموخته‌شده‌ی خودشان) ...")
        animate_house_party(Qs2, env2)

    # ---------------------------------------------------------------- PART 3
    banner("PART 3 — STARK HUD TELEMETRY DASHBOARD (matplotlib)")
    plot_stark_dashboard(rewards1, crashes1, team2, coll2)

    banner("DONE — SEMINAR READY ✅")
    print(f"   total runtime: {time.time() - t0:.1f}s   |   "
          f"\"I am Iron Man.\" 🦾")


if __name__ == "__main__":
    main()
