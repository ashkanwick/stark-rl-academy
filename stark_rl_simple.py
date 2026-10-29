# -*- coding: utf-8 -*-
"""
==============================================================================
  STARK RL — the WHOLE idea of Q-learning in ONE small file
  کل ایده‌ی Q-Learning در یک فایل کوچک — بدون بازی، بدون انیمیشن، فقط منطق
==============================================================================

  THE STORY / داستان:
    Iron Man's suit hovers on a 1-D line. Wind keeps pushing it around.
    JARVIS must learn — by TRIAL AND ERROR ONLY — to keep it centered.
    Nobody ever tells JARVIS what to do. It only feels REWARDS.
    زره روی یک خط می‌لرزد؛ جارویس فقط با «پاداش» و آزمون‌وخطا یاد می‌گیرد
    تعادل را نگه دارد.

  READ THIS FILE TOP TO BOTTOM — it really is the whole algorithm.
  RUN:  python stark_rl_simple.py        (needs only numpy)
==============================================================================
"""

import random

import numpy as np

random.seed(42)                     # same results every run / خروجی تکرارپذیر
np.random.seed(42)


# ============================================================================
# 1) THE WORLD  —  محیط (فیزیک ساده‌ی پرواز)
# ============================================================================
# The suit's tilt can be -4, -3, ..., +4      (0 = perfect hover)
# tilt = -4 or +4  -> CRASH into the lab wall (episode ends immediately)
#
# Actions:  0 = thrust RIGHT (tilt +1)
#           1 = hold still  (turbulence pushes randomly -1 or +1)
#           2 = thrust LEFT  (tilt -1)

def step(tilt, action):
    """One moment of physics: (tilt, action) -> (new_tilt, reward, done).

    This function is the ENVIRONMENT. Notice what it does NOT do:
    it knows nothing about learning, Q-tables, or JARVIS. It just answers.
    """
    if action == 0:
        new = tilt + 1
    elif action == 2:
        new = tilt - 1
    else:
        new = tilt + random.choice((-1, 1))      # 🌪 turbulence

    new = max(-4, min(4, new))
    if new in (-4, 4):
        return new, -100, True       # 💥 crash  -> episode over
    if new == 0:
        return new, +20, False       # ✅ perfect hover
    return new, -2, False            # ⛽ flying costs a little (time/fuel)


# ============================================================================
# 2) THE BRAIN  —  مغز: یک جدول ۹×۳ از عددها
# ============================================================================
# Q[tilt + 4][action]  means:
#   "If I am tilted at `tilt` and I do `action`,
#    how much TOTAL future reward do I expect from now on?"
#
# It starts at ZERO — JARVIS knows literally nothing. Experience fills it.
#   ۹ شیب × ۳ اکشن = ۲۷ عدد؛ روز اول همه صفر — هیچ چیزی نمی‌داند.

Q = np.zeros((9, 3))

ALPHA = 0.1     # learning rate: how much ONE experience may move a belief
GAMMA = 0.95    # discount: how much future reward matters compared to now
EPISODES = 800  # trial-and-error attempts / تعداد دفعات آزمون‌وخطا


# ============================================================================
# 3) THE LEARNING LOOP  —  حلقه‌ی یادگیری (همین سه خط، قلب کل RL است)
# ============================================================================
print("🧠 Training JARVIS for", EPISODES, "episodes ...")

for episode in range(1, EPISODES + 1):

    # exploration decays: explore a lot early, trust the table later
    # اول کاوشِ زیاد 🎲، آخر فقط بهترینِ جدول ▶
    epsilon = max(0.01, 1.0 - episode / EPISODES)

    tilt = random.randint(-3, 3)          # drop the suit at a random tilt

    for _ in range(30):                   # one episode = up to 30 moments

        # --- 1. CHOOSE an action: usually the best known, sometimes random
        if random.random() < epsilon:
            action = random.choice((0, 1, 2))            # 🎲 explore
        else:
            action = int(np.argmax(Q[tilt + 4]))         # ▶ exploit

        # --- 2. ASK the world: it answers with a new tilt and a reward
        new_tilt, reward, crashed = step(tilt, action)

        # --- 3. LEARN — the ONE formula of all reinforcement learning:
        #
        #   Q(s,a)  <-  Q(s,a) + alpha * ( reward + gamma * max Q(s',.) - Q(s,a) )
        #
        #   in words:  "nudge your old belief a little toward
        #               (what I just got) + (discounted best of what comes next)"
        #   فارسی: باور قدیمی را کمی به سمت «پاداش گرفتی + بهترین آینده» جابه‌جا کن
        #
        future = 0.0 if crashed else np.max(Q[new_tilt + 4])   # no future after 💥
        Q[tilt + 4][action] += ALPHA * (reward + GAMMA * future - Q[tilt + 4][action])

        tilt = new_tilt
        if crashed:
            break


# ============================================================================
# 4) THE RESULT  —  before vs after
# ============================================================================

def gauge(tilt):
    """One-line picture of the world: 💥 walls, 🎯 target, 🦾 suit."""
    cells = []
    for s in range(-4, 5):
        cells.append("🦾" if s == tilt else ("💥" if abs(s) == 4 else ("🎯" if s == 0 else "·")))
    return " ".join(cells)


print("\n📋 LEARNED POLICY — what the filled table tells JARVIS to do:")
names = {0: "thrust RIGHT ▶", 1: "hold •", 2: "thrust LEFT ◀"}
for tilt in range(-3, 4):
    best = int(np.argmax(Q[tilt + 4]))
    print(f"   tilt {tilt:+d}  ->  {names[best]}      Q-row: "
          f"[{Q[tilt + 4][0]:+7.1f} {Q[tilt + 4][1]:+7.1f} {Q[tilt + 4][2]:+7.1f}]")

print("\n🎬 DEMO FLIGHT — JARVIS flies with ONLY this table (no randomness):")
tilt, total = -3, 0
for t in range(1, 16):
    action = int(np.argmax(Q[tilt + 4]))
    tilt, reward, crashed = step(tilt, action)
    total += reward
    print(f"   step {t:2d} | {names[action]:<13} | reward {reward:+4d} | {gauge(tilt)}")
    if crashed:
        break
print(f"   ➤ total reward: {total:+d}   "
      f"(before training, a random pilot scores about −100 or worse 💥)")

print("\n🧠 FINAL Q-TABLE (each cell = expected future reward):")
print("      tilt |   ▶ RIGHT |    • HOLD |   ◀ LEFT")
for tilt in range(-4, 5):
    row = Q[tilt + 4]
    tag = " 💥" if abs(tilt) == 4 else (" 🎯" if tilt == 0 else "")
    print(f"     {tilt:+d}    |  {row[0]:+8.1f} | {row[1]:+8.1f} | {row[2]:+8.1f}{tag}")

print('\n   "Sometimes you gotta run before you can walk."  — Tony Stark 🦾')
