# -*- coding: utf-8 -*-
"""
==============================================================================
  STARK RL ARCADE  --  game-style HTML HUD for the RL seminar  (Iron Man theme)
  حالت بازی برای سمینار یادگیری تقویتی — HUD گرافیکی HTML به سبک بازی
==============================================================================

  WHAT IT DOES / چه کاری می‌کند:
    * Trains (or reuses) the two Q-learning parts, records the learned policy
      as a replay of frames, then bakes EVERYTHING into ONE self-contained
      HTML file:  stark_rl_game.html   (no server, no internet, no CDN).
      Q-table ها و سیاست آموخته‌شده داخل «یک» فایل HTML مستقل پخت می‌شود.
    * Open it in any browser (or embed it in Jupyter) and you get a GAME:
        - MISSION 1  Flight Stabilizer : 👀 WATCH J.A.R.V.I.S  or  🎮 YOU FLY
          (thruster buttons / arrow keys, live Q-bars, wind gusts, scoring)
        - MISSION 2  House Party       : animated 6x6 radar, play/pause/step
    * Every AI move comes from argmax(Q) recorded here — nothing scripted.
      هر حرکت هوش مصنوعی از همان جدول Q واقعی ضبط شده است.

  RUN / اجرا:
      python stark_rl_game.py                     -> writes stark_rl_game.html
      python stark_rl_game.py <fragment_path>     -> also writes a bare HTML
                                                     fragment (inline preview)
  In Jupyter:   from stark_rl_game import build_game_html
                build_game_html(Q1, Qs2)
==============================================================================
"""

import json
import os
import random
import sys

import numpy as np

from stark_rl_seminar import (
    StarkFlightBalancerEnv,
    HousePartyGridEnv,
    train_flight_stabilizer,
    train_house_party,
    record_first_training_steps,
    train_flight_with_snapshots,
    train_house_with_snapshots,
)

GAME_FILENAME = "stark_rl_game.html"


# ============================================================================
#  1) RECORD the learned policies as replayable frames
#     ضبط سیاست‌های آموخته‌شده به‌صورت فریم‌های قابل پخش
# ============================================================================

def record_flight_frames(q_table, env=None, steps=15, start_tilt=-3,
                         gust_schedule=((5, 2), (9, -2), (12, 1)), seed=90210):
    """Greedy replay of the trained flight policy with fixed dramatic gusts.
    پخش حریصانه‌ی سیاست پرواز با تلاطم‌های ثابت و دراماتیک (خروجی قطعی)."""
    if env is None:
        env = StarkFlightBalancerEnv()
    gust_at = dict(gust_schedule)
    # freeze global RNG so the replay is identical whenever it is recorded
    # قطع‌کردن تصادفیت سراسری تا پخش همیشه یکسان بماند
    st = random.getstate()
    random.seed(seed)
    try:
        tilt = env.reset(start_tilt)
        frames = []
        for step in range(1, steps + 1):
            action = int(__import__("numpy").argmax(q_table[tilt + 4]))
            q_row = [round(float(v), 1) for v in q_table[tilt + 4]]
            tilt, reward, _done = env.step(action)
            gust = gust_at.get(step, 0)
            if gust:                                    # 🌪 turbulence, clamped
                tilt = max(-3, min(3, tilt + gust))
                env.tilt = tilt
            frames.append({
                "step": step, "tilt": tilt, "action": action, "q": q_row,
                "tilt_after": int(tilt), "reward": int(reward), "gust": gust,
            })
        ai_score = sum(f["reward"] for f in frames)
        return frames, ai_score
    finally:
        random.setstate(st)


def record_house_rounds(q_tables, env=None):
    """Greedy synchronized mission, recorded move by move.
    مأموریت حریصانه‌ی هم‌زمان، حرکت‌به‌حرکت ضبط می‌شود."""
    if env is None:
        env = HousePartyGridEnv()
    env.reset()
    rounds = []
    for rnd in range(1, env.MAX_ROUNDS + 1):
        moves = []
        for i in range(env.N_AGENTS):
            if env.landed[i]:
                continue
            s = env.state_index(i)
            a = int(__import__("numpy").argmax(q_tables[i][s]))
            q_val = round(float(q_tables[i][s][a]), 1)   # confidence of the choice
            before = list(env.pos[i])
            _s2, r, done_i, _ev, info = env.step(i, a)
            after = list(env.TARGETS[i]) if done_i else list(env.pos[i])
            moves.append({
                "i": i, "frm": before, "to": after, "action": a,
                "reward": int(r), "q": q_val,
                "landed": bool(done_i),
                "collision": bool(info["collision"]),
            })
        rounds.append({"n": rnd, "moves": moves})
        if all(env.landed):
            break
    return rounds


# ============================================================================
#  2) BAKE everything into one self-contained HTML game
#     پخت همه‌چیز داخل یک فایل HTML مستقل
# ============================================================================

_INNER_TEMPLATE = r'''<style>
#stark-arcade{
  --gold:#ffc83d; --cyan:#3fb6ff; --red:#ff5a5a; --green:#3ddc84;
  --bg:#05070b; --panel:#0d141d; --panel2:#121b26; --line:#223041;
  --txt:#c9d6e3; --dim:#8fa3b8;
  --cw:clamp(30px,8vw,46px); --gc:clamp(36px,7vw,54px); --gg:6px;
  font-family:"Segoe UI",Tahoma,sans-serif; color:var(--txt); font-size:14px;
  background:radial-gradient(1100px 480px at 50% -120px,#0e1a26 0%,var(--bg) 62%);
  border:1px solid var(--line); border-radius:14px;
  padding:14px 16px 18px; position:relative; overflow:hidden;
}
#stark-arcade *{box-sizing:border-box}
#stark-arcade [hidden]{display:none!important}
.sa-scan{pointer-events:none;position:absolute;inset:0;z-index:40;
  background:repeating-linear-gradient(0deg,rgba(63,182,255,.03) 0 1px,transparent 1px 3px)}
.sa-top{display:flex;align-items:center;gap:12px;border-bottom:1px solid var(--line);
  padding-bottom:10px;flex-wrap:wrap}
.sa-reactor{width:34px;height:34px;border-radius:50%;flex:none;position:relative;
  background:radial-gradient(circle,#dff4ff 0 18%,var(--cyan) 22% 34%,#0a2530 36% 62%,var(--cyan) 66% 74%,transparent 78%);
  box-shadow:0 0 12px rgba(63,182,255,.8)}
.sa-reactor::after{content:"";position:absolute;inset:-5px;border-radius:50%;
  border:1px dashed rgba(63,182,255,.5);animation:sa-spin 7s linear infinite}
.sa-title{font-weight:700;letter-spacing:.16em;color:var(--gold);font-size:15px}
.sa-sub{color:var(--dim);font-size:12px}
.sa-tabs{display:flex;gap:8px;margin-left:auto;flex-wrap:wrap}
.sa-tab{background:var(--panel2);border:1px solid var(--line);color:var(--dim);
  padding:6px 12px;border-radius:8px;cursor:pointer;font-size:12px;letter-spacing:.05em}
.sa-tab.on{color:var(--gold);border-color:var(--gold);box-shadow:0 0 8px rgba(255,200,61,.25)}
.sa-screen{margin-top:14px}
.num{font-family:Consolas,Monaco,monospace}
.gold{color:var(--gold)} .good{color:var(--green)} .bad{color:var(--red)} .dim{color:var(--dim)}
.warn{color:var(--red)}

/* ---- boot ---- */
#sa-bootlog{white-space:pre;font-family:Consolas,Monaco,monospace;font-size:13px;
  color:var(--green);background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:14px;max-width:600px;min-height:118px;margin:0 auto;text-align:left;direction:ltr}
.sa-missions{display:flex;gap:14px;flex-wrap:wrap;justify-content:center;margin-top:16px}
.sa-mission{flex:1 1 250px;max-width:350px;background:var(--panel);border:1px solid var(--line);
  border-radius:14px;padding:16px;text-align:left;cursor:pointer;color:var(--txt);
  transition:border-color .2s,box-shadow .2s}
.sa-mission:hover{border-color:var(--cyan);box-shadow:0 0 16px rgba(63,182,255,.25)}
.sa-m-title{color:var(--gold);font-weight:700;letter-spacing:.05em;margin-bottom:6px}
.sa-m-desc{font-size:13px;color:var(--dim);line-height:1.8}

/* ---- HUD stats ---- */
.sa-hud{display:flex;gap:10px;flex-wrap:wrap;justify-content:center}
.sa-stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:6px 14px;min-width:88px;text-align:center}
.sa-stat .sa-l{display:block;font-size:11px;letter-spacing:.12em;color:var(--dim)}
.sa-stat b{font-size:18px;font-family:Consolas,Monaco,monospace;color:var(--cyan)}
.sa-stat b.gold{color:var(--gold)}

/* ---- mission 1 corridor ---- */
#m1-corridor{margin:16px auto 8px;width:max-content;max-width:100%;
  background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 14px;overflow-x:auto}
#m1-strip{position:relative;display:flex;gap:6px;width:max-content;margin:0 auto;
  height:calc(var(--cw) + 14px);align-items:center}
.sa-c1{width:var(--cw);height:var(--cw);flex:none;display:flex;align-items:center;justify-content:center;
  background:var(--panel2);border:1px solid var(--line);border-radius:8px;color:var(--dim);
  font-size:calc(var(--cw)*.42)}
.sa-c1.wall{border-color:rgba(255,90,90,.55);animation:sa-pulse 1.6s ease-in-out infinite}
.sa-c1.center{border-color:rgba(255,200,61,.5)}
#m1-suit,#m1-flame,#m1-wind{position:absolute;top:50%;margin-top:-4px;pointer-events:none;
  left:calc(var(--i,0) * (var(--cw) + 6px) + var(--cw)/2);transform:translate(-50%,-50%)}
#m1-suit{width:calc(var(--cw) + 8px);height:calc(var(--cw) + 8px);display:flex;align-items:center;
  justify-content:center;font-size:calc(var(--cw)*.62);transition:left .28s ease;border-radius:10px;
  background:rgba(63,182,255,.12);border:1px solid rgba(63,182,255,.55);
  box-shadow:0 0 14px rgba(63,182,255,.35)}
#m1-suit.ok{border-color:rgba(61,220,132,.8);background:rgba(61,220,132,.13);
  box-shadow:0 0 16px rgba(61,220,132,.5)}
#m1-suit.boom{animation:sa-boom .5s ease}
#m1-flame{font-size:calc(var(--cw)*.5);transition:left .28s ease;animation:sa-flick .45s infinite alternate;
  filter:drop-shadow(0 0 6px rgba(255,150,0,.85))}
#m1-wind{font-size:calc(var(--cw)*.72);animation:sa-flick .4s infinite alternate;transition:left .2s ease}
#m1-corridor.shake{animation:sa-shake .42s ease}

/* ---- Q bars ---- */
.sa-cols{display:flex;gap:14px;flex-wrap:wrap;justify-content:center;align-items:stretch;margin-top:10px}
.sa-qwrap{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:10px 14px;text-align:center}
.sa-qhead{font-size:12px;letter-spacing:.1em;color:var(--dim);margin-bottom:8px}
.sa-qrow{display:flex;gap:10px;justify-content:center}
.sa-qcol{width:54px;display:flex;flex-direction:column;align-items:center;gap:4px;
  border:1px solid var(--line);border-radius:10px;padding:8px 6px;background:var(--panel2);
  transition:border-color .2s,box-shadow .2s}
.sa-qcol.chosen{border-color:var(--gold);box-shadow:0 0 10px rgba(255,200,61,.28)}
.sa-qbar{height:62px;width:22px;background:rgba(63,182,255,.12);border-radius:5px;
  display:flex;align-items:flex-end;overflow:hidden}
.sa-qfill{width:100%;background:linear-gradient(180deg,var(--cyan),#1f6fa8);
  border-radius:5px 5px 0 0;transition:height .3s ease}
.sa-qcol.chosen .sa-qfill{background:linear-gradient(180deg,var(--gold),#a97a12)}
.sa-qval{font-family:Consolas,Monaco,monospace;font-size:11px;color:var(--dim)}
.sa-qlab{font-size:14px}
#m1-jarvis{font-size:12.5px;color:var(--cyan);margin-top:8px}
.sa-eventbox{flex:1 1 240px;max-width:330px;background:var(--panel);border:1px solid var(--line);
  border-radius:12px;padding:12px 14px;display:flex;align-items:center}
#m1-action{font-size:13.5px;line-height:2}

/* ---- controls ---- */
.sa-controls{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin-top:12px}
.sa-ctrlgrp{display:flex;gap:8px;flex-wrap:wrap;justify-content:center}
.sa-btn{background:var(--panel2);border:1px solid var(--line);color:var(--txt);border-radius:9px;
  padding:7px 14px;font-size:12.5px;cursor:pointer;letter-spacing:.04em;transition:border-color .15s,box-shadow .15s}
.sa-btn:hover{border-color:var(--cyan)}
.sa-btn.on{color:var(--gold);border-color:var(--gold);box-shadow:0 0 8px rgba(255,200,61,.2)}
.sa-btn.gold{border-color:var(--gold);color:var(--gold)}
.sa-btn.sa-big{padding:10px 16px;font-size:13.5px}
.sa-btn .kbd{opacity:.55;font-size:11px}
.sa-hint{color:var(--dim);font-size:12.5px;text-align:center;margin-top:12px;line-height:2}

/* ---- mission 2 ---- */
.sa-m2wrap{width:max-content;max-width:100%;margin:0 auto;background:var(--panel);
  border:1px solid var(--line);border-radius:12px;padding:12px;overflow-x:auto}
#m2-arena{position:relative;width:max-content}
#m2-arena.flash{animation:sa-flash .35s ease}
#m2-grid{display:grid;grid-template-columns:repeat(6,var(--gc));gap:var(--gg)}
.sa-gcell{width:var(--gc);height:var(--gc);display:flex;align-items:center;justify-content:center;
  background:var(--panel2);border:1px solid var(--line);border-radius:8px;font-size:calc(var(--gc)*.5)}
.sa-gcell.fire{animation:sa-flick 1.1s infinite alternate}
.sa-gcell.target{border-color:rgba(61,220,132,.45)}
.sa-agent{position:absolute;left:0;top:0;width:calc(var(--gc) - 8px);height:calc(var(--gc) - 8px);
  display:flex;align-items:center;justify-content:center;font-size:calc(var(--gc)*.55);
  transition:left .3s ease,top .3s ease;border-radius:50%;
  filter:drop-shadow(0 0 7px currentColor);pointer-events:none}
.sa-agent.landed{animation:sa-ring 1.5s ease-out infinite}
.sa-sidebox{flex:1 1 250px;max-width:330px}
.sa-legend{font-size:12px;color:var(--dim);margin-bottom:8px;line-height:1.9;text-align:center}
#m2-log{background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:8px 12px;min-height:150px;font-size:12.5px}
.sa-logline{padding:4px 0;border-bottom:1px dashed rgba(34,48,65,.6);
  font-family:Consolas,Monaco,monospace;direction:ltr;text-align:left}
.s0{color:#ff8a8a}.s1{color:#7cc9ff}.s2{color:#ffd76b}

/* ---- modal ---- */
#sa-modal{position:absolute;inset:0;background:rgba(4,6,10,.87);display:flex;align-items:center;
  justify-content:center;z-index:60;border-radius:14px}
.sa-modal-card{background:var(--panel);border:1px solid var(--gold);border-radius:14px;
  padding:20px 24px;max-width:440px;width:calc(100% - 40px);
  box-shadow:0 0 30px rgba(255,200,61,.22);text-align:center}
#sa-modal-title{color:var(--gold);font-size:18px;font-weight:700;letter-spacing:.05em;margin-bottom:10px}
#sa-modal-lines{font-size:13.5px;line-height:2}
#sa-modal-btns{display:flex;gap:10px;justify-content:center;margin-top:14px;flex-wrap:wrap}

/* ---- academy ---- */
.sa-slidehead{display:flex;align-items:center;gap:10px;margin-top:14px;flex-wrap:wrap}
.sa-slidehead .sa-lesson{color:var(--gold);font-weight:700;letter-spacing:.05em;font-size:14px}
.sa-dots{display:flex;gap:6px;margin-left:auto;align-items:center}
.sa-dot{width:9px;height:9px;border-radius:50%;background:var(--panel2);border:1px solid var(--line);cursor:pointer}
.sa-dot.on{background:var(--gold);border-color:var(--gold);box-shadow:0 0 6px rgba(255,200,61,.5)}
#edu-body{margin-top:12px;min-height:390px;position:relative}
.edu-card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px 18px;line-height:2.1;font-size:13.5px}
.edu-card h4{color:var(--gold);margin:0 0 8px;font-size:14px;letter-spacing:.04em}
.edu-note{color:var(--dim);font-size:12.5px}
.edu-big{font-size:15px}
.sa-loop{display:flex;align-items:center;justify-content:center;gap:10px;flex-wrap:wrap;margin:10px 0}
.sa-loopbox{background:var(--panel2);border:1px solid var(--cyan);border-radius:12px;padding:10px 14px;
  text-align:center;min-width:130px;box-shadow:0 0 10px rgba(63,182,255,.15)}
.sa-loopbox b{color:var(--cyan);display:block;font-size:13px}
.sa-loopbox span{font-size:11.5px;color:var(--dim)}
.sa-looparrow{color:var(--gold);font-size:16px;text-align:center;line-height:1.3;font-size:12.5px}
.sa-looparrow b{color:var(--gold);display:block;font-size:14px}
.edu-qgrid{display:grid;grid-template-columns:56px repeat(3,minmax(64px,86px));gap:4px;
  width:max-content;margin:8px auto;font-family:Consolas,Monaco,monospace}
.edu-qhead{color:var(--dim);font-size:11.5px;text-align:center;background:var(--panel2);
  border:1px solid var(--line);border-radius:6px;padding:3px 4px;font-family:"Segoe UI",Tahoma,sans-serif}
.edu-qcell{position:relative;border:1px solid var(--line);border-radius:6px;padding:5px 4px 3px;
  text-align:center;font-size:12.5px;transition:background .5s,color .5s,border-color .3s}
.edu-qcell .bar{height:4px;border-radius:2px;background:var(--cyan);margin-top:3px;transition:width .5s,background .5s}
.edu-qcell.hot{border-color:var(--gold);box-shadow:0 0 9px rgba(255,200,61,.4)}
.edu-qcell.tilt{color:var(--dim);font-size:12px;background:transparent;border-color:transparent}
.edu-qrowlab{color:var(--dim);font-size:11.5px;text-align:right;padding-right:6px;align-self:center}
.edu-formula{font-family:Consolas,Monaco,monospace;font-size:13px;background:var(--panel2);
  border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin:8px 0;text-align:center;direction:ltr}
.edu-formula .dim2{opacity:.4}
.edu-arrowgrid{display:grid;grid-template-columns:repeat(6,30px);gap:3px;width:max-content;margin:4px auto}
.edu-acell{width:30px;height:30px;display:flex;align-items:center;justify-content:center;font-size:15px;
  background:var(--panel2);border:1px solid var(--line);border-radius:6px;transition:background .3s,color .3s}
.edu-acell.fire{border-color:rgba(255,90,90,.5)}
.edu-acell.tgt{border-color:rgba(61,220,132,.5)}
.edu-statline{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin:8px 0;font-size:12.5px}
.edu-chip{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:3px 10px}
.edu-chip b{color:var(--cyan);font-family:Consolas,Monaco,monospace}
.edu-mini{font-size:12px;color:var(--dim)}
.edu-2col{display:flex;gap:16px;flex-wrap:wrap;justify-content:center;align-items:flex-start}
.edu-2col>div{flex:1 1 300px;max-width:430px}
.edu-reveal{opacity:0;transform:translateY(4px);transition:opacity .4s,transform .4s}
.edu-reveal.show{opacity:1;transform:none}
.edu-flash{animation:sa-flash2 .8s ease}
@keyframes sa-flash2{0%{background:rgba(255,200,61,.18)}100%{background:transparent}}
.edu-reward{font-weight:700;font-size:15px}

@keyframes sa-spin{to{transform:rotate(360deg)}}
@keyframes sa-pulse{0%,100%{box-shadow:0 0 0 rgba(255,90,90,0)}50%{box-shadow:0 0 12px rgba(255,90,90,.5)}}
@keyframes sa-flick{from{opacity:.55}to{opacity:1}}
@keyframes sa-shake{0%,100%{transform:translateX(0)}25%{transform:translateX(-4px)}75%{transform:translateX(4px)}}
@keyframes sa-boom{0%{transform:translate(-50%,-50%) scale(1)}40%{transform:translate(-50%,-50%) scale(1.35)}100%{transform:translate(-50%,-50%) scale(1)}}
@keyframes sa-flash{0%{box-shadow:inset 0 0 0 3px rgba(255,90,90,.9)}100%{box-shadow:inset 0 0 0 3px rgba(255,90,90,0)}}
@keyframes sa-ring{0%{box-shadow:0 0 0 0 rgba(61,220,132,.7)}100%{box-shadow:0 0 0 12px rgba(61,220,132,0)}}
@media (prefers-reduced-motion: reduce){
  #stark-arcade *{animation:none!important;transition:none!important}
}
</style>

<div id="stark-arcade" dir="ltr">
  <div class="sa-scan" aria-hidden="true"></div>

  <div class="sa-top">
    <div class="sa-reactor" aria-hidden="true"></div>
    <div>
      <div class="sa-title">STARK RL ARCADE</div>
      <div class="sa-sub">J.A.R.V.I.S · یادگیری تقویتی زنده · Q-Learning</div>
    </div>
    <div class="sa-tabs">
      <button class="sa-tab" id="tab-edu" type="button">🎓 ACADEMY</button>
      <button class="sa-tab on" id="tab-boot" type="button">🏠 MISSIONS</button>
      <button class="sa-tab" id="tab-m1" type="button">🛩 M1 STABILIZER</button>
      <button class="sa-tab" id="tab-m2" type="button">🚀 M2 HOUSE PARTY</button>
    </div>
  </div>

  <!-- ============ BOOT ============ -->
  <div class="sa-screen" id="scr-boot">
    <div id="sa-bootlog">STARK OS BOOTING...</div>
    <div class="sa-missions">
      <button class="sa-mission" id="go-edu" type="button">
        <div class="sa-m-title">🎓 STARK ACADEMY — از صفر یاد بگیر</div>
        <div class="sa-m-desc">۸ درس تعاملی: حلقه‌ی عامل-محیط، پاداش، جدول Q، معادله‌ی بلمن با عدد واقعی، تماشای شکل‌گیری مغز، و هماهنگی چندعامله — بعدش برو سراغ مأموریت‌ها!</div>
      </button>
      <button class="sa-mission" id="go-m1" type="button">
        <div class="sa-m-title">🛩 MISSION 1 — FLIGHT STABILIZER</div>
        <div class="sa-m-desc">Q-Learning تک‌عامله · جارویس را تماشا کن که با جدول Q پرواز می‌کند، یا خودت خلبان شو و امتیازت را با هوش مصنوعی مقایسه کن!</div>
      </button>
      <button class="sa-mission" id="go-m2" type="button">
        <div class="sa-m-title">🚀 MISSION 2 — HOUSE PARTY PROTOCOL</div>
        <div class="sa-m-desc">RL چندعامله · ۳ زره‌ی مستقل هم‌زمان به رآکتورهایشان می‌رسند — بدون برخورد و بدون هیچ برنامه‌ریز مرکزی.</div>
      </button>
    </div>
  </div>

  <!-- ============ ACADEMY ============ -->
  <div class="sa-screen" id="scr-edu" hidden>
    <div class="sa-slidehead">
      <span class="sa-lesson" id="edu-title">ACADEMY</span>
      <span class="sa-sub" id="edu-sub"></span>
      <div class="sa-dots" id="edu-dots"></div>
    </div>
    <div id="edu-body"></div>
    <div class="sa-controls">
      <div class="sa-ctrlgrp">
        <button class="sa-btn" id="edu-prev" type="button">◀ درس قبلی</button>
        <button class="sa-btn gold sa-big" id="edu-next" type="button">درس بعدی ▶</button>
      </div>
    </div>
  </div>

  <!-- ============ MISSION 1 ============ -->
  <div class="sa-screen" id="scr-m1" hidden>
    <div class="sa-hud">
      <div class="sa-stat"><span class="sa-l">TILT</span><b id="m1-tilt" class="num">-45°</b></div>
      <div class="sa-stat"><span class="sa-l">SCORE</span><b id="m1-score" class="gold num">+0</b></div>
      <div class="sa-stat"><span class="sa-l">STREAK</span><b id="m1-streak" class="num">—</b></div>
      <div class="sa-stat"><span class="sa-l">STEP</span><b id="m1-step" class="num">0/15</b></div>
      <div class="sa-stat"><span class="sa-l">BEST</span><b id="m1-best" class="num">+0</b></div>
    </div>

    <div id="m1-corridor"><div id="m1-strip">
      <div id="m1-wind" hidden>🌪</div>
      <div id="m1-flame" hidden>🔥</div>
      <div id="m1-suit" style="--i:1">🦾</div>
    </div></div>

    <div class="sa-cols">
      <div class="sa-qwrap">
        <div class="sa-qhead">🧠 Q(s,a) — LIVE</div>
        <div class="sa-qrow">
          <div class="sa-qcol" id="qcol0"><div class="sa-qbar"><div class="sa-qfill" id="qfill0" style="height:30%"></div></div><div class="sa-qval num" id="qval0">—</div><div class="sa-qlab">▶</div></div>
          <div class="sa-qcol" id="qcol1"><div class="sa-qbar"><div class="sa-qfill" id="qfill1" style="height:30%"></div></div><div class="sa-qval num" id="qval1">—</div><div class="sa-qlab">✋</div></div>
          <div class="sa-qcol" id="qcol2"><div class="sa-qbar"><div class="sa-qfill" id="qfill2" style="height:30%"></div></div><div class="sa-qval num" id="qval2">—</div><div class="sa-qlab">◀</div></div>
        </div>
        <div id="m1-jarvis">J.A.R.V.I.S → —</div>
      </div>
      <div class="sa-eventbox"><div id="m1-action" aria-live="polite">—</div></div>
    </div>

    <div class="sa-controls">
      <div class="sa-ctrlgrp">
        <button class="sa-btn on" id="m1-mode-watch" type="button">👀 WATCH J.A.R.V.I.S</button>
        <button class="sa-btn" id="m1-mode-manual" type="button">🎮 YOU FLY IT</button>
      </div>
      <div class="sa-ctrlgrp" id="m1-watch-ctrls">
        <button class="sa-btn" id="m1-play" type="button">⏸ PAUSE</button>
        <button class="sa-btn" id="m1-restart" type="button">🔁 RESTART</button>
        <button class="sa-btn" id="m1-speed" type="button">1×</button>
      </div>
      <div class="sa-ctrlgrp" id="m1-manual-ctrls" hidden>
        <button class="sa-btn sa-big" id="m1-btn0" type="button">▶ PUSH RIGHT <span class="kbd">→</span></button>
        <button class="sa-btn sa-big" id="m1-btn1" type="button">✋ HOLD <span class="kbd">↓</span></button>
        <button class="sa-btn sa-big" id="m1-btn2" type="button">◀ PUSH LEFT <span class="kbd">←</span></button>
      </div>
    </div>
    <div class="sa-hint" dir="rtl">WATCH: هر حرکت مستقیم از argmax جدول Q می‌آید. &nbsp;YOU FLY: با دکمه‌ها یا کلیدهای جهت‌دار زره را روی 🎯 نگه دار و با امتیاز جارویس (<span class="num" id="m1-airef">—</span>) مقایسه کن!</div>
  </div>

  <!-- ============ MISSION 2 ============ -->
  <div class="sa-screen" id="scr-m2" hidden>
    <div class="sa-hud">
      <div class="sa-stat"><span class="sa-l">ROUND</span><b id="m2-round" class="num">0</b></div>
      <div class="sa-stat"><span class="sa-l">DOCKED</span><b id="m2-docked" class="num">0/3</b></div>
      <div class="sa-stat"><span class="sa-l">COLLISIONS</span><b id="m2-coll" class="num">0</b></div>
      <div class="sa-stat"><span class="sa-l">TEAM SCORE</span><b id="m2-team" class="gold num">+0</b></div>
    </div>

    <div class="sa-cols">
      <div class="sa-m2wrap"><div id="m2-arena"><div id="m2-grid"></div></div></div>
      <div class="sa-sidebox">
        <div class="sa-legend">🔴 Mark 42 · 🔵 Silver Centurion · 🟡 Hulkbuster<br>🔋 Arc Reactor · 💥 Fire / Laser</div>
        <div id="m2-log" aria-live="polite"></div>
      </div>
    </div>

    <div class="sa-controls"><div class="sa-ctrlgrp">
      <button class="sa-btn" id="m2-play" type="button">⏸ PAUSE</button>
      <button class="sa-btn" id="m2-step" type="button">⏭ STEP</button>
      <button class="sa-btn" id="m2-speed" type="button">1×</button>
      <button class="sa-btn" id="m2-restart" type="button">🔁 RESTART</button>
    </div></div>
    <div class="sa-hint" dir="rtl">هر زره فقط جدول Q خودش را دارد (وضعیتش + رادار هم‌تیمی‌ها) — هماهنگی صرفاً از پاداش تیمی بیرون آمده است.</div>
  </div>

  <div id="sa-modal" hidden>
    <div class="sa-modal-card">
      <div id="sa-modal-title"></div>
      <div id="sa-modal-lines"></div>
      <div id="sa-modal-btns"></div>
    </div>
  </div>
</div>

<script>
(() => {
const D = __DATA__;
const $ = id => document.getElementById(id);
const fmt = n => (n > 0 ? '+' : '') + n;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const argmax = row => { let b = 0; for (let k = 1; k < row.length; k++) if (row[k] > row[b]) b = k; return b; };

/* ---------- static build ---------- */
const strip = $('m1-strip'), corr = $('m1-corridor');
for (let i = 0; i < 9; i++) {
  const d = document.createElement('div');
  d.className = 'sa-c1' + (i === 0 || i === 8 ? ' wall' : '') + (i === 4 ? ' center' : '');
  d.textContent = (i === 0 || i === 8) ? '💥' : (i === 4 ? '🎯' : '·');
  strip.insertBefore(d, $('m1-wind'));
}
D.p2.suits.forEach((s, i) => {
  const el = document.createElement('div');
  el.className = 'sa-agent'; el.id = 'ag' + i; el.textContent = s.icon;
  el.style.color = s.color; el.style.transform = 'translate(4px,4px)';
  $('m2-arena').appendChild(el);
});
for (let r = 0; r < 6; r++) for (let c = 0; c < 6; c++) {
  const d = document.createElement('div'); d.className = 'sa-gcell';
  if (D.p2.fires.some(f => f[0] === r && f[1] === c)) { d.classList.add('fire'); d.textContent = '💥'; }
  const ti = D.p2.targets.findIndex(t => t[0] === r && t[1] === c);
  if (ti >= 0) { d.classList.add('target'); d.textContent = '🔋'; }
  $('m2-grid').appendChild(d);
}
$('m1-airef').textContent = fmt(D.p1.aiScore);

/* ---------- screens & tabs ---------- */
function show(id) {
  ['scr-boot', 'scr-m1', 'scr-m2', 'scr-edu'].forEach(s => { $(s).hidden = (s !== id); });
  $('tab-boot').classList.toggle('on', id === 'scr-boot');
  $('tab-m1').classList.toggle('on', id === 'scr-m1');
  $('tab-m2').classList.toggle('on', id === 'scr-m2');
  $('tab-edu').classList.toggle('on', id === 'scr-edu');
  hideModal();                       // never carry a modal across screens
}

/* ---------- modal ---------- */
function showModal(title, lines, btns) {
  $('sa-modal-title').textContent = title;
  $('sa-modal-lines').innerHTML = lines.map(l => '<div>' + l + '</div>').join('');
  const box = $('sa-modal-btns'); box.innerHTML = '';
  btns.forEach(b => {
    const el = document.createElement('button');
    el.className = 'sa-btn' + (b.primary ? ' gold' : '');
    el.textContent = b.label; el.type = 'button';
    el.addEventListener('click', () => { hideModal(); b.fn && b.fn(); });
    box.appendChild(el);
  });
  $('sa-modal').hidden = false;
}
const hideModal = () => { $('sa-modal').hidden = true; };

/* =================  MISSION 1  ================= */
const M1 = { mode: 'watch', idx: 0, total: 0, paused: true, speed: 1, timer: null,
             mTilt: -3, mScore: 0, mSteps: 0, mStreak: 0, mBest: 0, mBusy: false };
const ACT = ['▶ PUSH RIGHT · رانشگر چپ 🔥', '✋ HOLD · رها در تلاطم', '◀ PUSH LEFT · رانشگر راست 🔥'];

function placeSuit(t) {
  const s = $('m1-suit');
  s.style.setProperty('--i', t + 4);
  s.classList.toggle('ok', t === 0);
  $('m1-tilt').textContent = (t * 15 > 0 ? '+' : '') + t * 15 + '°';
}
function placeFlame(t, a) {
  const f = $('m1-flame');
  if (a === null || a === undefined || a === 1) { f.hidden = true; return; }
  f.hidden = false;
  f.style.setProperty('--i', t + 4 + (a === 2 ? 1 : -1));
}
function qBars(row, chosen) {
  const mn = Math.min(...row), mx = Math.max(...row), sp = (mx - mn) || 1;
  for (let k = 0; k < 3; k++) {
    $('qfill' + k).style.height = (10 + ((row[k] - mn) / sp) * 86) + '%';
    $('qval' + k).textContent = row[k].toFixed(1);
    $('qcol' + k).classList.toggle('chosen', chosen === k);
  }
}
const hint = a => { $('m1-jarvis').textContent = 'J.A.R.V.I.S → ' + D.p1.actNames[a]; };
function m1Hud() {
  $('m1-score').textContent = fmt(M1.mode === 'watch' ? M1.total : M1.mScore);
  $('m1-streak').textContent = M1.mode === 'watch' ? '—' : M1.mStreak;
  $('m1-best').textContent = fmt(M1.mBest);
}
function windFx(atIdx) {
  const w = $('m1-wind');
  w.hidden = false; w.style.setProperty('--i', atIdx);
  corr.classList.add('shake');
  setTimeout(() => corr.classList.remove('shake'), 430);
  setTimeout(() => { w.hidden = true; }, 640);
}

function eduComment(f) {
  const t = f.tilt, a = f.action;
  if (a === 1 && t === 0) return 'روی 🎯 است و HOLD می‌کند — جدول می‌گوید همین‌جا بمانی بهترین است (Q(HOLD) بزرگ‌ترین).';
  if (a === 1) return 'هیچ‌کاری نمی‌کند و منتظر باد می‌ماند — ریسکِ باد را پذیرفته؛ تجربه‌ی جدول همین را گفته.';
  const right = a === 0;
  const needed = right ? 'چپ' : 'راست';
  const correct = (t < 0 && right) || (t > 0 && !right);
  return correct
    ? `شیب به ${needed} افتاده → رانشگر سمت ${needed} شلیک می‌کند 🔥 تا زره به سمت 🎯 هل بخورد (argmaxِ همان ردیف).`
    : `رانشگر سمت ${right ? 'چپ' : 'راست'} — اصلاحِ ریز برای برگشت به 🎯.`;
}
function renderWatchFrame(f) {
  placeSuit(f.tilt); placeFlame(f.tilt, f.action);
  qBars(f.q, f.action); hint(f.action);
  $('m1-action').innerHTML = '🧠 فرمان جارویس: <b>' + D.p1.actNames[f.action] + '</b>'
    + (f.gust ? '   <span class="warn">🌪 تلاطم در راه است…</span>' : '')
    + '<div class="edu-mini" style="margin-top:4px">🧑‍🏫 ' + eduComment(f) + '</div>';
}
function m1WatchReset() {
  M1.mode = 'watch'; M1.idx = 0; M1.total = 0; M1.paused = true;
  $('m1-step').textContent = '0/' + D.p1.frames.length;
  $('m1-play').textContent = '▶ PLAY';
  renderWatchFrame(D.p1.frames[0]); m1Hud();
}
function m1Tick() {
  if ($('scr-m1').hidden || M1.mode !== 'watch') { M1.timer = setTimeout(m1Tick, 220); return; }
  if (!$('sa-modal').hidden || M1.paused) { M1.timer = setTimeout(m1Tick, 140); return; }
  if (M1.idx >= D.p1.frames.length) { m1WatchEnd(); return; }
  const f = D.p1.frames[M1.idx];
  renderWatchFrame(f);
  if (f.gust) windFx(f.tilt_after + 4);
  M1.timer = setTimeout(() => {
    placeSuit(f.tilt_after); placeFlame(f.tilt_after, null);
    M1.total += f.reward; m1Hud();
    M1.idx++; $('m1-step').textContent = M1.idx + '/' + D.p1.frames.length;
    M1.timer = setTimeout(m1Tick, 420 / M1.speed);
  }, 560 / M1.speed);
}
function m1WatchEnd() {
  const last = D.p1.frames[D.p1.frames.length - 1];
  placeSuit(last.tilt_after); placeFlame(last.tilt_after, null);
  $('m1-action').textContent = '✅ HOVER LOCKED — پرواز کامل بدون سقوط';
  showModal('MISSION COMPLETE ✅',
    ['J.A.R.V.I.S SCORE: <b class="gold">' + fmt(M1.total) + '</b>',
     '۱۵ گام، ۳ تلاطم باد، صفر سقوط — این نتیجه‌ی همان جدول ۹×۳ است!'],
    [{ label: '👀 WATCH AGAIN', primary: true, fn: () => { m1WatchReset(); M1.paused = false; $('m1-play').textContent = '⏸ PAUSE'; } },
     { label: '🎮 YOU FLY IT', fn: () => setMode('manual') },
     { label: '🏠 MENU', fn: () => show('scr-boot') }]);
}

function suggest() {
  const row = D.p1.Q[M1.mTilt + 4];
  qBars(row, -1); hint(argmax(row));
}
function m1ManualReset() {
  M1.mode = 'manual'; M1.mTilt = -3; M1.mScore = 0; M1.mSteps = 0; M1.mStreak = 0; M1.mBusy = false;
  placeSuit(-3); placeFlame(-3, null); m1Hud();
  $('m1-step').textContent = '0/30';
  $('m1-action').textContent = '🎮 فرمان با توست — زره را روی 🎯 نگه دار!';
  suggest();
}
function m1Step(a) {
  if (M1.mode !== 'manual' || M1.mBusy) return;
  M1.mBusy = true;
  const row = D.p1.Q[M1.mTilt + 4];
  qBars(row, a); hint(argmax(row));
  const d = a === 0 ? 1 : a === 2 ? -1 : (Math.random() < 0.5 ? -1 : 1);
  const nt = clamp(M1.mTilt + d, -4, 4);
  let r, crashed = false;
  if (nt === -4 || nt === 4) { r = -100; crashed = true; }
  else if (nt === 0) { r = 20; M1.mStreak++; }
  else { r = -2; M1.mStreak = 0; }
  M1.mScore += r; M1.mSteps++;
  placeFlame(M1.mTilt, a);
  if (!crashed) placeSuit(nt);
  $('m1-action').innerHTML = '🎮 شما: <b>' + ACT[a] + '</b> → پاداش <b class="' + (r > 0 ? 'good' : 'bad') + '">' + fmt(r) + '</b>';
  $('m1-step').textContent = M1.mSteps + '/30';
  m1Hud();
  setTimeout(() => {
    placeFlame(M1.mTilt, null);
    if (crashed) {
      $('m1-suit').classList.add('boom');
      setTimeout(() => { $('m1-suit').classList.remove('boom'); m1Crashed(); }, 480);
      return;
    }
    if (Math.random() < 0.22) {                       // 🌪 sudden turbulence
      const g = [-2, -1, 1, 2][Math.floor(Math.random() * 4)];
      const nt2 = clamp(nt + g, -3, 3);
      windFx(nt2 + 4); placeSuit(nt2); M1.mTilt = nt2;
    } else {
      M1.mTilt = nt;
    }
    suggest(); M1.mBusy = false;
    if (M1.mSteps >= 30) m1ManualEnd();
  }, 330);
}
function saveBest() { if (M1.mScore > M1.mBest) M1.mBest = M1.mScore; m1Hud(); }
function m1Crashed() {
  saveBest();
  showModal('SUIT DOWN 💥',
    ['SCORE: <b class="bad">' + fmt(M1.mScore) + '</b>',
     'برخورد با دیوار آزمایشگاه! جارویس این را هرگز تجربه نمی‌کند — چون پاداش ‎−۱۰۰ را در جدولش ثبت کرده است.'],
    [{ label: '🔁 RETRY', primary: true, fn: m1ManualReset },
     { label: '👀 WATCH JARVIS', fn: () => setMode('watch') },
     { label: '🏠 MENU', fn: () => show('scr-boot') }]);
}
function m1ManualEnd() {
  saveBest();
  const s = M1.mScore, g = s >= 140 ? 'S' : s >= 80 ? 'A' : s >= 20 ? 'B' : 'C';
  showModal('MISSION COMPLETE ✅',
    ['YOUR SCORE: <b class="gold">' + fmt(s) + '</b>  |  J.A.R.V.I.S: <b>' + fmt(D.p1.aiScore) + '</b>',
     'GRADE: <b class="gold">' + g + '</b>',
     g === 'S' ? 'درجه S! تو هم در سطح خلبان‌های استارکی.' : 'جارویس با همان جدول Q این کار را می‌کند — تفاوت سیاستِ آموخته با شهود انسانی!'],
    [{ label: '🔁 AGAIN', primary: true, fn: m1ManualReset },
     { label: '🏠 MENU', fn: () => show('scr-boot') }]);
}
function setMode(m) {
  M1.mode = m;
  $('m1-mode-watch').classList.toggle('on', m === 'watch');
  $('m1-mode-manual').classList.toggle('on', m === 'manual');
  $('m1-watch-ctrls').hidden = m !== 'watch';
  $('m1-manual-ctrls').hidden = m !== 'manual';
  if (m === 'watch') m1WatchReset(); else m1ManualReset();
}

$('m1-mode-watch').addEventListener('click', () => setMode('watch'));
$('m1-mode-manual').addEventListener('click', () => setMode('manual'));
$('m1-play').addEventListener('click', () => {
  M1.paused = !M1.paused;
  $('m1-play').textContent = M1.paused ? '▶ PLAY' : '⏸ PAUSE';
});
$('m1-restart').addEventListener('click', () => { m1WatchReset(); M1.paused = false; $('m1-play').textContent = '⏸ PAUSE'; });
$('m1-speed').addEventListener('click', () => {
  M1.speed = M1.speed === 1 ? 2 : M1.speed === 2 ? 4 : 1;
  $('m1-speed').textContent = M1.speed + '×';
});
$('m1-btn0').addEventListener('click', () => m1Step(0));
$('m1-btn1').addEventListener('click', () => m1Step(1));
$('m1-btn2').addEventListener('click', () => m1Step(2));
document.addEventListener('keydown', e => {
  if ($('scr-m1').hidden || M1.mode !== 'manual' || !$('sa-modal').hidden) return;
  if (e.key === 'ArrowRight') { m1Step(0); e.preventDefault(); }
  else if (e.key === 'ArrowLeft') { m1Step(2); e.preventDefault(); }
  else if (e.key === 'ArrowDown') { m1Step(1); e.preventDefault(); }
});

/* =================  MISSION 2  ================= */
const P2 = { ri: 0, k: 0, paused: true, speed: 1, timer: null, team: 0, coll: 0, docked: [null, null, null] };

function placeAgent(i, r, c) {
  const el = $('ag' + i);
  el.style.left = 'calc(' + c + ' * (var(--gc) + var(--gg)))';
  el.style.top = 'calc(' + r + ' * (var(--gc) + var(--gg)))';
  el.classList.toggle('landed', P2.docked[i] !== null);
}
function p2Reset() {
  P2.ri = 0; P2.k = 0; P2.team = 0; P2.coll = 0; P2.docked = [null, null, null];
  $('m2-log').innerHTML = '';
  $('m2-round').textContent = '0'; $('m2-docked').textContent = '0/3';
  $('m2-coll').textContent = '0'; $('m2-team').textContent = '+0';
  D.p2.starts.forEach((s, i) => placeAgent(i, s[0], s[1]));
}
function logLine(html) {
  const el = $('m2-log');
  el.insertAdjacentHTML('afterbegin', '<div class="sa-logline">' + html + '</div>');
  while (el.children.length > 8) el.removeChild(el.lastChild);
}
function p2Apply(mv, rnd) {
  P2.team += mv.reward;
  if (mv.collision) P2.coll++;
  if (mv.landed) P2.docked[mv.i] = rnd;
  const s = D.p2.suits[mv.i];
  logLine('<b class="s' + mv.i + '">' + s.icon + ' ' + s.name + '</b> (' + mv.frm[0] + ',' + mv.frm[1] + ')→(' + mv.to[0] + ',' + mv.to[1] + ') ' + D.p2.actNames[mv.action]
    + (mv.landed ? ' ⚡DOCKED +50' : ' ' + fmt(mv.reward)) + ' <span class="dim">Q=' + mv.q + '</span>');
  placeAgent(mv.i, mv.to[0], mv.to[1]);
  $('m2-team').textContent = fmt(P2.team);
  $('m2-coll').textContent = P2.coll;
  $('m2-docked').textContent = P2.docked.filter(x => x !== null).length + '/3';
  if (mv.collision) {
    const a = $('m2-arena');
    a.classList.add('flash'); setTimeout(() => a.classList.remove('flash'), 360);
  }
}
function p2Advance() {
  if (P2.ri >= D.p2.rounds.length) return true;
  const rd = D.p2.rounds[P2.ri];
  if (P2.k === 0) $('m2-round').textContent = rd.n;
  if (P2.k < rd.moves.length) { p2Apply(rd.moves[P2.k], rd.n); P2.k++; }
  if (P2.k >= rd.moves.length) { P2.k = 0; P2.ri++; return false; }
  return false;
}
function p2Tick() {
  if ($('scr-m2').hidden) { P2.timer = setTimeout(p2Tick, 220); return; }
  if (!$('sa-modal').hidden || P2.paused) { P2.timer = setTimeout(p2Tick, 140); return; }
  if (P2.ri >= D.p2.rounds.length) { p2End(); return; }
  const boundary = !p2Advance();
  P2.timer = setTimeout(p2Tick, (boundary ? 780 : 560) / P2.speed);
}
function p2End() {
  const done = P2.docked.filter(x => x !== null).length;
  const g = (done === 3 && P2.coll === 0) ? 'S' : (done === 3 ? 'A' : 'B');
  showModal('HOUSE PARTY COMPLETE 🎉',
    ['ROUNDS: <b>' + D.p2.rounds.length + '</b> · DOCKED: <b>' + done + '/3</b> · COLLISIONS: <b class="' + (P2.coll ? 'bad' : 'good') + '">' + P2.coll + '</b>',
     'TEAM SCORE: <b class="gold">' + fmt(P2.team) + '</b> · GRADE: <b class="gold">' + g + '</b>',
     'سه عاملِ کاملاً مستقل، بدون برنامه‌ریز مرکزی — هماهنگی فقط از پاداش تیمی بیرون آمد!'],
    [{ label: '🔁 REPLAY', primary: true, fn: () => { p2Reset(); P2.paused = false; $('m2-play').textContent = '⏸ PAUSE'; } },
     { label: '🏠 MENU', fn: () => show('scr-boot') }]);
}
$('m2-play').addEventListener('click', () => {
  P2.paused = !P2.paused;
  $('m2-play').textContent = P2.paused ? '▶ PLAY' : '⏸ PAUSE';
});
$('m2-step').addEventListener('click', () => {
  P2.paused = true; $('m2-play').textContent = '▶ PLAY'; p2Advance();
});
$('m2-speed').addEventListener('click', () => {
  P2.speed = P2.speed === 1 ? 2 : P2.speed === 2 ? 4 : 1;
  $('m2-speed').textContent = P2.speed + '×';
});
$('m2-restart').addEventListener('click', () => { p2Reset(); P2.paused = false; $('m2-play').textContent = '⏸ PAUSE'; });

/* =================  ACADEMY (teaching slides)  ================= */
const EDU = {
  i: 0, stop: null,
  bell: { step: 0, reveal: 0 },
  brain: { idx: 0, timer: null, paused: true, speed: 1 },
  team: { idx: 0, timer: null, paused: true, speed: 1 },
  sand: { tilt: 0, score: 0, steps: 0 },
};
const E = D.edu;
const EDUMX = Math.max(1, ...E.qSnaps.map(sn => Math.max(...sn.Q.flat().map(Math.abs))));
const AR = ['▶', '•', '◀'];

function heatBg(v) {
  const f = v / EDUMX;
  return f >= 0 ? `rgba(63,182,255,${(f * .78).toFixed(2)})`
                : `rgba(255,90,90,${(-f * .78).toFixed(2)})`;
}
function qGrid(Q, hot, small) {
  let h = '<div class="edu-qgrid">';
  h += '<div class="edu-qrowlab">tilt</div>' + AR.map(a => `<div class="edu-qhead">${a}</div>`).join('');
  for (let t = -4; t <= 4; t++) {
    const row = Q[t + 4];
    h += `<div class="edu-qrowlab">${t > 0 ? '+' : ''}${t}${t === 0 ? ' 🎯' : (Math.abs(t) === 4 ? ' 💥' : '')}</div>`;
    for (let a = 0; a < 3; a++) {
      const v = row[a], f = Math.abs(v) / EDUMX;
      const isHot = hot && hot[0] === t + 4 && hot[1] === a;
      h += `<div class="edu-qcell${isHot ? ' hot' : ''}" style="background:${heatBg(v)}">
        <span class="${small ? 'edu-mini' : ''}">${v.toFixed(1)}</span>
        <div class="bar" style="width:${Math.round(f * 100)}%;background:${v >= 0 ? 'var(--cyan)' : 'var(--red)'}"></div></div>`;
    }
  }
  return h + '</div>';
}
function corridorHTML(tilt, cls) {
  let h = '<div style="display:flex;gap:5px;justify-content:center;margin:10px 0">';
  for (let i = -4; i <= 4; i++) {
    const ch = Math.abs(i) === 4 ? '💥' : (i === 0 ? '🎯' : (i === tilt ? '🦾' : '·'));
    h += `<div class="${cls}" style="width:38px;height:38px;display:flex;align-items:center;justify-content:center;
      border:1px solid ${i === tilt ? 'var(--gold)' : 'var(--line)'};border-radius:8px;font-size:18px;
      background:${i === tilt ? 'rgba(255,200,61,.12)' : 'var(--panel2)'}">${ch}</div>`;
  }
  return h + '</div>';
}
function curveSVG(vals, marker01) {
  const W = 300, H = 64, mn = Math.min(...vals), mx = Math.max(...vals), sp = (mx - mn) || 1;
  const pts = vals.map((v, i) =>
    `${(i / (vals.length - 1) * W).toFixed(1)},${(H - 6 - (v - mn) / sp * (H - 12)).toFixed(1)}`).join(' ');
  const mx0 = Math.round(marker01 * W);
  return `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" style="max-width:100%">
    <polyline points="${pts}" fill="none" stroke="#3fb6ff" stroke-width="2"/>
    <line x1="${mx0}" y1="2" x2="${mx0}" y2="${H - 2}" stroke="#ffc83d" stroke-width="2" stroke-dasharray="4 3"/>
  </svg>`;
}

/* ---- slide renderers ---- */
function slStory() {
  $('edu-body').innerHTML = `<div class="edu-card">
    <h4>🎬 درس ۰ — داستان: چرا «یادگیری تقویتی»؟</h4>
    <div>تونی استارک نمی‌تواند برای هر لحظه‌ی پرواز دستور بنویسد — باد 🌪 هر بار از سمتی دیگر می‌آید.
    پس به‌جای برنامه‌نویسی مستقیم، به جارویس فقط <b class="gold">یک عدد</b> می‌دهد: <b>پاداش</b>. بقیه‌اش را خودش،
    با آزمون‌وخطا، یاد می‌گیرد. تمام یادگیری تقویتی همین یک حلقه است:</div>
    <div class="sa-loop" dir="ltr">
      <div class="sa-loopbox"><b>🤖 AGENT — عامل</b><span>جارویس + جدول Q (مغزش)</span></div>
      <div class="sa-looparrow"><b>→ action aₜ</b> شلیک رانشگر یا هیچ‌کاره<b style="margin-top:10px">← state sₜ₊₁ + reward rₜ₊₁</b> شیب جدید + پاداش</div>
      <div class="sa-loopbox"><b>🌍 ENV — محیط</b><span>زره + آزمایشگاه + باد</span></div>
    </div>
    <div class="edu-formula">Q(s,a) ← Q(s,a) + α · [ r + γ · max Q(s′,·) − Q(s,a) ]
      <div class="edu-mini">بعد از هر حرکت، مغز کمی اصلاح می‌شود — این تنها فرمولِ کل سمینار است</div></div>
    <div>💡 در این بازی هر عدد و هر حرکت از <b>آموزش واقعی</b> ضبط شده — هیچ چیز اسکریپت نیست. در درس‌های بعدی همین حلقه را تکه‌تکه می‌بینیم.</div>
  </div>`;
}
function slStates() {
  $('edu-body').innerHTML = `<div class="edu-card">
    <h4>🎯 درس ۱ — وضعیت (State): کل دنیای زره یک عدد است</h4>
    <div>محیط پرواز فقط یک عدد نگه می‌دارد: <b class="gold">شیب زره (tilt)</b> از ‎−۴‎ تا ‎+۴‎.</div>
    ${corridorHTML(null, 'sa-c1')}
    <div class="edu-statline">
      <span class="edu-chip">🎯 tilt = 0 → تعادل کامل</span>
      <span class="edu-chip">💥 tilt = ±4 → برخورد با دیوار، پایان اپیزود</span>
      <span class="edu-chip">۹ وضعیت ممکن</span>
    </div>
    <div>عامل این عددها را «انتخاب» نمی‌کند — محیط به او می‌گوید کجاست. کار عامل فقط انتخاب <b>اکشن</b> در هر وضعیت است.</div>
  </div>`;
}
function slActions() {
  const S = EDU.sand;
  $('edu-body').innerHTML = `<div class="edu-card">
    <h4>🕹 درس ۲ — اکشن و پاداش: خودت عاملِ آموزش‌ندیده باش!</h4>
    <div>سه اکشن داری؛ بعد از هر کلیک محیط پاسخ می‌دهد. سعی کن امتیاز بگیری — دقیقاً همان کاری که جارویس باید یاد بگیرد:</div>
    ${corridorHTML(S.tilt, 'sa-c1')}
    <div class="edu-statline">
      <span class="edu-chip">امتیاز <b class="gold">${fmt(S.score)}</b></span>
      <span class="edu-chip">گام‌ها <b>${S.steps}</b></span>
      <span class="edu-chip">قانون: 🎯 +۲۰ · هر گام دیگر −۲ · 💥 −۱۰۰</span>
    </div>
    <div class="sa-controls"><div class="sa-ctrlgrp">
      <button class="sa-btn sa-big" data-act="sand0" type="button">▶ رانشگر چپ 🔥</button>
      <button class="sa-btn sa-big" data-act="sand1" type="button">✋ هیچ‌کاری نکن 🌪</button>
      <button class="sa-btn sa-big" data-act="sand2" type="button">◀ رانشگر راست 🔥</button>
      <button class="sa-btn" data-act="sandreset" type="button">🔁 از اول</button>
    </div></div>
    <div id="sand-msg" class="edu-big" style="text-align:center;min-height:34px">—</div>
    <div class="edu-note">چطور بود؟ نگه‌داشتن تعادل با «گام‌به‌گام فیدبک» سخت است — حالا تصور کن ۸۰۰ اپیزود این کار را بکنی؛ همان کاری که چند خط کد انجام می‌دهد.</div>
  </div>`;
}
function sandMsg(html) { const m = $('sand-msg'); if (m) m.innerHTML = html; }
function sandStep(a) {
  const S = EDU.sand;
  const d = a === 0 ? 1 : a === 2 ? -1 : (Math.random() < .5 ? -1 : 1);
  const nt = clamp(S.tilt + d, -4, 4);
  let r, msg;
  if (Math.abs(nt) === 4) { r = -100; msg = '💥<b class="bad"> برخورد با دیوار! ‎−۱۰۰</b> — اپیزود تمام شد (از اول)'; S.tilt = 0; S.steps = 0; S.score = 0; slActions(); sandMsg(msg); return; }
  if (nt === 0) { r = 20; msg = '✅<b class="good"> تعادل! ‎+۲۰</b> — این همان چیزی است که باید یاد بگیری'; }
  else { r = -2; msg = `⚠️ <span class="dim">شیب ${nt > 0 ? '+' : ''}${nt} → ‎−۲</span> (هزینه‌ی سوخت/زمان)`; }
  S.tilt = nt; S.score += r; S.steps++;
  slActions(); sandMsg(msg);
}
function slQTable() {
  $('edu-body').innerHTML = `<div class="edu-card">
    <h4>🧠 درس ۳ — جدول Q: مغز جارویس = ۲۷ عدد</h4>
    <div>برای هر وضعیت (۹ شیب) و هر اکشن (۳ فرمان)، یک عدد نگه می‌داریم: <b class="gold">«اگر اینجا این کار را بکنم، چقدر پاداش در انتظار من است؟»</b></div>
    ${qGrid([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], null, true)}
    <div style="text-align:center">روز اول: همه صفر — جارویس <b>هیچ چیزی نمی‌داند</b>. سیاست = argmax هر ردیف؛ فعلاً بی‌معنی.</div>
    <div class="edu-note">سؤال کلیدی: این ۲۷ عدد را از کجا بیاوریم؟ → با معادله‌ی بلمن، یک قدم به‌قدم. درس بعد ⬅</div>
  </div>`;
}
function slBellman() {
  const B = EDU.bell, st = E.steps[B.step];
  if (!st) { slBellDone(); return; }
  const show = B.reveal;                      // how many blocks revealed
  const blocks = [];
  blocks.push(`<div class="edu-reveal${show > 0 ? ' show' : ''}">① محیط پاسخ داد: از شیب <b>${st.s > 0 ? '+' : ''}${st.s}</b> با اکشن <b>${AR[st.a]}</b> به شیب
    <b>${st.s2 > 0 ? '+' : ''}${st.s2}</b> رسیدیم و پاداش گرفتیم <b class="${st.r >= 0 ? 'good' : 'bad'}">${fmt(st.r)}</b>
    ${st.r === 20 ? '(🎯 تعادل)' : st.r === -100 ? '(💥 سقوط!)' : st.a === 1 ? '(🌪 نتیجه‌ی باد)' : '(⛽ هزینه)'}</div>`);
  blocks.push(`<div class="edu-reveal${show > 1 ? ' show' : ''}">② بهترین ارزشِ آینده: max Q(s′,·) = <b>${st.best_next.toFixed(1)}</b>
    → هدف: r + γ·max = ${st.r} + ${st.gamma}×${st.best_next.toFixed(1)} = <b class="gold">${st.target.toFixed(2)}</b></div>`);
  blocks.push(`<div class="edu-reveal${show > 2 ? ' show' : ''}">③ یادگیری (کمی به سمت حقیقت): Q ← ${st.old.toFixed(1)} + ${st.alpha}×(${st.target.toFixed(2)} − ${st.old.toFixed(1)})
    = <b class="gold">${st.new.toFixed(2)}</b>${st.r === -100 ? ' — سقوط را برای همیشه در این خانه ثبت کرد!' : ''}</div>`);
  const doneAll = show > 3;
  $('edu-body').innerHTML = `<div class="edu-card">
    <h4>✍️ درس ۴ — معادله‌ی بلمن با عدد واقعی (قدم ${st.t} از ${E.steps.length})</h4>
    <div class="edu-mini">این‌ها قدم‌های اولِ اپیزود ۱ هستند — چون ε=1 هنوز همه‌چیز تصادفی است؛ جدول دارد اولین تجربه‌هایش را جمع می‌کند.</div>
    ${blocks.join('')}
    <div class="edu-reveal${doneAll ? ' show' : ''}">${qGrid(st.q_all, [st.s + 4, st.a], true)}
      <div style="text-align:center" class="edu-mini">خانه‌ی طلایی = همان خانه‌ای که همین الان به‌روز شد (شیب ${st.s > 0 ? '+' : ''}${st.s}، اکشن ${AR[st.a]})</div></div>
    <div class="sa-controls"><div class="sa-ctrlgrp">
      <button class="sa-btn gold sa-big" data-act="bellnext" type="button">${show >= 4 ? 'قدم بعدی ▶' : 'ادامه ▶'}</button>
      <button class="sa-btn" data-act="bellskip" type="button">⏭ تا آخر</button>
    </div></div>
  </div>`;
}
function slBellDone() {
  const last = E.steps[E.steps.length - 1];
  $('edu-body').innerHTML = `<div class="edu-card">
    <h4>✅ درس ۴ تمام — حالا بلمن را دیده‌ای با چشم خودت</h4>
    ${qGrid(last.q_all, null, true)}
    <div style="text-align:center">بعد از ${E.steps.length} قدم هنوز تقریباً همه‌چیز صفر است — یادگیری <b>تدریجی</b> است.
    اگر همین را ۸۰۰ اپیزود ادامه بدی… درس بعد تماشایش کن ⬅</div>
  </div>`;
}
function brainFrame() {
  const B = EDU.brain, sn = E.qSnaps[B.idx];
  const Q = sn.Q;
  let pol = '<div style="display:grid;gap:4px;margin-left:8px;justify-items:center">';
  for (let t = -4; t <= 4; t++)
    pol += `<div style="font-size:15px">${Math.abs(t) === 4 ? '💥' : AR[[Q[t + 4][0], Q[t + 4][1], Q[t + 4][2]].indexOf(Math.max(...Q[t + 4]))]}</div>`;
  pol += '</div>';
  $('edu-body').innerHTML = `<div class="edu-card">
    <h4>📈 درس ۵ — تماشای شکل‌گیری مغز (جدول Q در طول آموزش واقعی)</h4>
    <div class="edu-statline">
      <span class="edu-chip">اپیزود <b>${sn.ep}</b></span>
      <span class="edu-chip">ε = <b>${sn.eps.toFixed(2)}</b> (${Math.round((1 - sn.eps) * 100)}٪ از جدول)</span>
      <span class="edu-chip">میانگین پاداش <b>${fmt(Math.round(sn.avg))}</b></span>
      <span class="edu-chip">نرخ سقوط <b class="${sn.crash > 30 ? 'bad' : 'good'}">${Math.round(sn.crash)}٪</b></span>
    </div>
    <div class="edu-2col">
      <div>
        <div style="display:flex;justify-content:center">${qGrid(Q, null, true)}${pol}</div>
        <div class="edu-mini" style="text-align:center">ستون آخر = سیاست (argmax) — از بی‌معنی به «همیشه به سمت 🎯»</div>
      </div>
      <div style="text-align:center">
        ${curveSVG(E.curve1, B.idx / (E.qSnaps.length - 1))}
        <div class="edu-mini">منحنی یادگیری (میانگین پاداش هر ۱۰ اپیزود) — خط طلایی: الان کجاییم</div>
        <div class="sa-controls"><div class="sa-ctrlgrp">
          <button class="sa-btn" data-act="brainplay" type="button">${B.paused ? '▶ PLAY' : '⏸ PAUSE'}</button>
          <button class="sa-btn" data-act="brainspeed" type="button">${B.speed}×</button>
        </div></div>
        <div class="edu-mini" style="margin-top:6px">آبی روشن = باور مثبت، قرمز = تجربه‌ی بد (مثلاً راهِ سقوط)</div>
      </div>
    </div>
  </div>`;
}
function brainTick() {
  const B = EDU.brain;
  B.timer = setTimeout(() => {
    if ($('scr-edu').hidden || EDU.i !== 5) return;
    if (!B.paused) { B.idx = (B.idx + 1) % E.qSnaps.length; brainFrame(); }
    brainTick();
  }, (B.paused ? 300 : 1400) / B.speed);
}
function teamFrame() {
  const T = EDU.team, sn = E.policySnaps[T.idx];
  const grids = D.p2.suits.map((su, i) => {
    let h = `<div style="text-align:center"><div class="edu-mini" style="color:${su.color}">${su.icon} ${su.name}</div><div class="edu-arrowgrid">`;
    for (let r = 0; r < 6; r++) for (let c = 0; c < 6; c++) {
      const ch = sn.suits[i][r][c];
      const cls = ch === '🔥' ? ' fire' : (ch === '⚡' ? ' tgt' : '');
      h += `<div class="edu-acell${cls}">${ch}</div>`;
    }
    return h + '</div></div>';
  }).join('');
  $('edu-body').innerHTML = `<div class="edu-card">
    <h4>🚀 درس ۶ — چندعامله: هر زره یک مغز جدا + یک رادار</h4>
    <div class="edu-statline">
      <span class="edu-chip">اپیزود <b>${sn.ep}</b></span>
      <span class="edu-chip">ε = <b>${sn.eps.toFixed(2)}</b></span>
      <span class="edu-chip">امتیاز تیمی <b>${fmt(Math.round(sn.team))}</b></span>
      <span class="edu-chip">برخورد <b class="${sn.coll > 1 ? 'bad' : 'good'}">${sn.coll}</b></span>
    </div>
    <div style="display:flex;gap:18px;justify-content:center;flex-wrap:wrap">${grids}</div>
    <div class="sa-controls"><div class="sa-ctrlgrp">
      <button class="sa-btn" data-act="teamplay" type="button">${T.paused ? '▶ PLAY' : '⏸ PAUSE'}</button>
      <button class="sa-btn" data-act="teamspeed" type="button">${T.speed}×</button>
    </div></div>
    <div class="edu-mini" style="margin-top:6px">وضعیت هر زره = خانه‌ی خودش + رادارِ دو هم‌تیمی (۶ حالت هرکدام). برخورد؟ ‎−۸۰‎ برای <b>هر دو</b> —
    هماهنگی بدون هیچ فرماندهی مرکزی، فقط از پاداش تیمی بیرون می‌آید. فلش‌ها از آشوب به مسیرِ منظم به سمت ⚡ می‌روند.</div>
  </div>`;
}
function teamTick() {
  const T = EDU.team;
  T.timer = setTimeout(() => {
    if ($('scr-edu').hidden || EDU.i !== 6) return;
    if (!T.paused) { T.idx = (T.idx + 1) % E.policySnaps.length; teamFrame(); }
    teamTick();
  }, (T.paused ? 300 : 1400) / T.speed);
}
function slGrad() {
  $('edu-body').innerHTML = `<div class="edu-card" style="text-align:center">
    <h4>🎓 درس ۷ — فارغ‌التحصیل! حالا همه‌ی قطعات را دیده‌ای</h4>
    <div style="text-align:right;line-height:2.4">
      ✅ حلقه‌ی عامل-محیط: اکشن → پاداش و وضعیت جدید<br>
      ✅ وضعیت = شیب زره (۹ حالت) · اکشن = ۳ فرمان · پاداش = ‎+۲۰/−۲/−۱۰۰<br>
      ✅ جدول Q = ۲۷ حافظه‌ی «چقدر خوب بود؟»<br>
      ✅ معادله‌ی بلمن = باور جدید = باور قدیم + α×(هدف − باور)<br>
      ✅ ε: اول کاوش (تصادفی)، بعد بهره‌برداری (argmax)<br>
      ✅ چندعامله: مغزهای جدا + رادار + پاداش تیمی
    </div>
    <div class="edu-big" style="margin:10px 0">حالا برو سراغ مأموریت‌ها — می‌فهمی چرا جارویس این‌طور پرواز می‌کند!</div>
    <div class="sa-controls"><div class="sa-ctrlgrp">
      <button class="sa-btn gold sa-big" data-act="go-m1w" type="button">👀 M1 — تماشای جارویس</button>
      <button class="sa-btn sa-big" data-act="go-m1f" type="button">🎮 M1 — خودت پرواز کن</button>
      <button class="sa-btn sa-big" data-act="go-m2" type="button">🚀 M2 — House Party</button>
    </div></div>
  </div>`;
}

const SLIDES = [
  { t: 'درس ۰ — داستان', s: 'WHY REINFORCEMENT LEARNING?', r: slStory },
  { t: 'درس ۱ — وضعیت', s: 'STATE', r: slStates },
  { t: 'درس ۲ — اکشن و پاداش', s: 'ACTIONS & REWARDS — TRY IT!', r: () => { slActions(); sandMsg('یک دکمه بزن!'); } },
  { t: 'درس ۳ — جدول Q', s: 'THE BRAIN = 27 NUMBERS', r: slQTable },
  { t: 'درس ۴ — بلمن با عدد', s: 'ONE UPDATE, BY HAND', r: () => { EDU.bell = { step: 0, reveal: 0 }; slBellman(); } },
  { t: 'درس ۵ — شکل‌گیری مغز', s: 'WATCH THE Q-TABLE LEARN', r: () => { EDU.brain.paused = false; brainFrame(); brainTick(); } },
  { t: 'درس ۶ — تیم چندعامله', s: '3 BRAINS + RADAR + TEAM REWARD', r: () => { EDU.team.paused = false; teamFrame(); teamTick(); } },
  { t: 'درس ۷ — فارغ‌التحصیل', s: 'GRADUATION', r: slGrad },
];

function eduGo(n) {
  n = Math.max(0, Math.min(SLIDES.length - 1, n));
  if (EDU.stop) { EDU.stop(); EDU.stop = null; }
  clearTimeout(EDU.brain.timer); clearTimeout(EDU.team.timer);
  EDU.i = n;
  const dots = SLIDES.map((_, k) => `<div class="sa-dot${k === n ? ' on' : ''}" data-slide="${k}"></div>`).join('');
  $('edu-dots').innerHTML = dots;
  $('edu-title').textContent = SLIDES[n].t;
  $('edu-sub').textContent = SLIDES[n].s;
  $('edu-prev').disabled = n === 0;
  $('edu-next').disabled = n === SLIDES.length - 1;
  SLIDES[n].r();
}
$('edu-prev').addEventListener('click', () => eduGo(EDU.i - 1));
$('edu-next').addEventListener('click', () => eduGo(EDU.i + 1));
$('edu-dots').addEventListener('click', e => {
  const d = e.target.closest('.sa-dot'); if (d) eduGo(+d.dataset.slide);
});
$('edu-body').addEventListener('click', e => {
  const b = e.target.closest('[data-act]'); if (!b) return;
  const act = b.dataset.act;
  if (act.startsWith('sand') && act !== 'sandreset') sandStep(+act.slice(4));
  else if (act === 'sandreset') { EDU.sand = { tilt: 0, score: 0, steps: 0 }; slActions(); sandMsg('از اول!'); }
  else if (act === 'bellnext') {
    const B = EDU.bell;
    if (B.reveal < 4) B.reveal++;
    else { B.step++; B.reveal = 1; }
    slBellman();
  } else if (act === 'bellskip') { EDU.bell = { step: E.steps.length, reveal: 0 }; slBellDone(); }
  else if (act === 'brainplay') { EDU.brain.paused = !EDU.brain.paused; brainFrame(); }
  else if (act === 'brainspeed') { EDU.brain.speed = EDU.brain.speed === 1 ? 2 : EDU.brain.speed === 2 ? 4 : 1; brainFrame(); }
  else if (act === 'teamplay') { EDU.team.paused = !EDU.team.paused; teamFrame(); }
  else if (act === 'teamspeed') { EDU.team.speed = EDU.team.speed === 1 ? 2 : EDU.team.speed === 2 ? 4 : 1; teamFrame(); }
  else if (act === 'go-m1w') { show('scr-m1'); setMode('watch'); M1.paused = false; $('m1-play').textContent = '⏸ PAUSE'; }
  else if (act === 'go-m1f') { show('scr-m1'); setMode('manual'); }
  else if (act === 'go-m2') { show('scr-m2'); p2Reset(); P2.paused = false; $('m2-play').textContent = '⏸ PAUSE'; }
});
document.addEventListener('keydown', e => {
  if ($('scr-edu').hidden || !$('sa-modal').hidden) return;
  if (e.key === 'ArrowRight') { eduGo(EDU.i + 1); e.preventDefault(); }
  else if (e.key === 'ArrowLeft') { eduGo(EDU.i - 1); e.preventDefault(); }
});

/* ---------- tabs & missions ---------- */
$('tab-boot').addEventListener('click', () => show('scr-boot'));
$('tab-edu').addEventListener('click', () => { show('scr-edu'); eduGo(EDU.i || 0); });
$('tab-m1').addEventListener('click', () => { show('scr-m1'); setMode(M1.mode || 'watch'); });
$('tab-m2').addEventListener('click', () => { show('scr-m2'); p2Reset(); P2.paused = false; $('m2-play').textContent = '⏸ PAUSE'; });
$('go-edu').addEventListener('click', () => { show('scr-edu'); eduGo(0); });
$('go-m1').addEventListener('click', () => { show('scr-m1'); setMode('watch'); M1.paused = false; $('m1-play').textContent = '⏸ PAUSE'; });
$('go-m2').addEventListener('click', () => { show('scr-m2'); p2Reset(); P2.paused = false; $('m2-play').textContent = '⏸ PAUSE'; });

/* ---------- boot log typewriter ---------- */
const BOOT = [
  'STARK INDUSTRIES // J.A.R.V.I.S. OS v9.2',
  'LOADING Q-TABLES ......... OK',
  'ACADEMY MODULE .......... READY',
  'CALIBRATING THRUSTERS .... OK',
  'ARC REACTOR OUTPUT ....... 3.6 GJ/s',
  'READY. "Sometimes you gotta run before you can walk."'
];
(function typeBoot() {
  let li = 0, ci = 0;
  const el = $('sa-bootlog');
  const t = setInterval(() => {
    if (li >= BOOT.length) { clearInterval(t); return; }
    ci++;
    el.textContent = BOOT.slice(0, li).join('\n') + (li ? '\n' : '') + BOOT[li].slice(0, ci) + '▌';
    if (ci >= BOOT[li].length) { li++; ci = 0; }
  }, 16);
})();

/* ---------- init ---------- */
m1ManualReset(); m1WatchReset(); p2Reset(); eduGo(0);
m1Tick(); p2Tick();
show('scr-boot');

/* deep links (testing / slides):  ?m=edu  ?m=1  ?m=2  &speed=4 */
const qp = new URLSearchParams(location.search);
if (qp.get('m') === 'edu') {
  show('scr-edu'); eduGo(0);
}
if (qp.get('m') === '1') {
  show('scr-m1'); setMode('watch'); M1.paused = false; $('m1-play').textContent = '⏸ PAUSE';
  if (qp.get('speed')) { M1.speed = +qp.get('speed') || 1; $('m1-speed').textContent = M1.speed + '×'; }
}
if (qp.get('m') === '2') {
  show('scr-m2'); p2Reset(); P2.paused = false; $('m2-play').textContent = '⏸ PAUSE';
  if (qp.get('speed')) { P2.speed = +qp.get('speed') || 1; $('m2-speed').textContent = P2.speed + '×'; }
}
})();
</script>
'''


def _bucket(values, n_buckets):
    """Average 'values' into n_buckets buckets (compact learning curve for JS).
    میانگین‌گیری سطلی برای منحنی یادگیری فشرده."""
    v = np.asarray(values, dtype=float)
    k = max(1, len(v) // n_buckets)
    return [round(float(x), 1) for x in v[:k * n_buckets].reshape(n_buckets, k).mean(1)]


def _edu_data(env1=None, env2=None):
    """Everything the ACADEMY tab teaches from — all REAL training artifacts.
    Returns (data, Q1, Qs2) so callers can reuse the very same tables.
    همه‌ی داده‌های تب آکادمی — همه از آموزش واقعی ضبط شده‌اند؛ خود جدول‌ها را هم
    برمی‌گرداند تا همین جدول‌ها در مأموریت‌ها پرواز کنند."""
    if env1 is None:
        env1 = StarkFlightBalancerEnv()
    if env2 is None:
        env2 = HousePartyGridEnv()

    steps = record_first_training_steps(env1, n_steps=12)

    Q1, rewards1, crashes1, snaps1 = train_flight_with_snapshots(env1, episodes=800)
    q_snaps = [{
        "ep": sn["ep"], "eps": sn["epsilon"],
        "avg": sn["avg_reward"], "crash": sn["crash_rate"],
        "Q": [[round(float(v), 1) for v in row] for row in np.asarray(sn["Q"])],
    } for sn in snaps1]

    Qs2, team2, coll2, _succ, snaps2 = train_house_with_snapshots(env2, episodes=1500)
    policy_snaps = []
    for sn in snaps2:
        suits = []
        for grid in sn["policies"]:               # each grid = 6 rows x (2 chars/cell)
            suits.append([[grid[r][c * 2:c * 2 + 2].strip() for c in range(6)]
                          for r in range(6)])
        policy_snaps.append({
            "ep": sn["ep"], "eps": sn["epsilon"],
            "team": sn["team_avg"], "coll": sn["coll_avg"],
            "suits": suits,
        })

    data = {
        "steps": steps,
        "qSnaps": q_snaps,
        "curve1": _bucket(rewards1, 80),
        "crash1": _bucket([100 * float(c) for c in crashes1], 80),
        "policySnaps": policy_snaps,
        "curve2": _bucket(team2, 80),
        "fires": sorted(list(f) for f in HousePartyGridEnv.FIRES),
        "targets": [list(t) for t in HousePartyGridEnv.TARGETS],
    }
    return data, Q1, Qs2


def _game_data(q1, qs2, edu=None):
    """Collect everything the game needs / همه‌ی داده‌هایی که بازی لازم دارد."""
    frames, ai_score = record_flight_frames(q1)
    rounds = record_house_rounds(qs2)
    return {
        "p1": {
            "Q": [[round(float(v), 1) for v in row] for row in np.asarray(q1)],
            "frames": frames,
            "aiScore": ai_score,
            "actNames": ["▶ PUSH RIGHT", "✋ HOLD", "◀ PUSH LEFT"],
        },
        "p2": {
            "fires": sorted(list(f) for f in HousePartyGridEnv.FIRES),
            "targets": [list(t) for t in HousePartyGridEnv.TARGETS],
            "starts": [list(s) for s in HousePartyGridEnv.STARTS],
            "suits": [
                {"name": "Mark 42", "icon": "🔴", "color": "#ff8a8a"},
                {"name": "Silver Centurion", "icon": "🔵", "color": "#7cc9ff"},
                {"name": "Hulkbuster", "icon": "🟡", "color": "#ffd76b"},
            ],
            "actNames": ["UP ⬆", "DOWN ⬇", "LEFT ⬅", "RIGHT ➡", "STAY ✋"],
            "rounds": rounds,
        },
        "edu": edu if edu is not None else _edu_data()[0],
    }


def build_game_fragment(q1, qs2, edu=None):
    """Bare HTML fragment (style + markup + script), for inline previews.
    فریممنت خام HTML برای نمایش درون‌خطی."""
    return _INNER_TEMPLATE.replace("__DATA__",
                                   json.dumps(_game_data(q1, qs2, edu), ensure_ascii=False))


def build_game_html(q1, qs2, edu=None, out_dir=None, open_browser=False):
    """Write the standalone game file and return its path.
    فایل مستقل بازی را می‌نویسد و مسیرش را برمی‌گرداند.
    edu = academy data (pass _edu_data() output to keep the ACADEMY tables
    identical to the flying tables) — داده‌های تب آکادمی."""
    inner = build_game_fragment(q1, qs2, edu)
    standalone = ("<!doctype html>\n<html lang=\"fa\">\n<head>\n<meta charset=\"utf-8\">\n"
                  "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
                  "<title>STARK RL ARCADE — Iron Man RL Game HUD</title>\n"
                  "<style>html,body{margin:0;padding:24px 12px;background:#05070b}</style>\n"
                  "</head>\n<body>\n" + inner + "\n</body>\n</html>\n")
    path = os.path.join(out_dir or os.getcwd(), GAME_FILENAME)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(standalone)
    print(f"🎮 STARK RL ARCADE written to: {path}")
    if open_browser and hasattr(os, "startfile"):
        os.startfile(path)                      # Windows: open in default browser
    return path


def main(fragment_path=None):
    """Train from scratch (deterministic, ~3s) and bake the game.
    آموزش از صفر (قطعی و حدود ۳ ثانیه) و پخت بازی."""
    from stark_rl_seminar import banner
    banner("🎮 STARK RL ARCADE — baking the game HUD (+ 🎓 ACADEMY)")
    env1 = StarkFlightBalancerEnv()
    env2 = HousePartyGridEnv()
    # one training pass: the ACADEMY data and the flying tables come from the
    # SAME run — یک بار آموزش: داده‌ی آکادمی و جدول‌های پرواز از یک اجرا
    edu, q1, qs2 = _edu_data(env1, env2)
    build_game_html(q1, qs2, edu)
    if fragment_path:
        with open(fragment_path, "w", encoding="utf-8") as fh:
            fh.write(build_game_fragment(q1, qs2, edu))
        print(f"🧩 fragment written to: {fragment_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
