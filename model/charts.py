"""Charts for the Clasico forecast. Run after clasico_model_v2.py."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "outputs"
CH = OUT / "charts"
CH.mkdir(parents=True, exist_ok=True)

BAR_C, RMA_C, DRAW_C = "#2a78d6", "#eb6834", "#8a8984"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlesize": 11,
})
R = json.loads((OUT / "results.json").read_text())
matches = pd.read_csv(DATA / "matches_2026_27.csv")
scen = pd.read_csv(OUT / "in_play_scenarios.csv")
BIN_LABELS = ["1-15", "16-30", "31-45+", "46-60", "61-75", "76-90+"]

def save(fig, name):
    fig.tight_layout()
    fig.savefig(CH / name, dpi=160, facecolor=SURF)
    plt.close(fig)

# 1. Form: goals for / against per match ---------------------------------------
ABBR = {"Paris Saint-Germain": "PSG", "Racing Santander": "Racing", "Athletic Club": "Athletic",
        "Rayo Vallecano": "Rayo", "Real Sociedad": "R.Sociedad", "Atletico Madrid": "Atletico",
        "Galatasaray": "Galata.", "RB Leipzig": "Leipzig", "Real Betis": "Betis"}
fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharey=True)
for ax, team, col in [(axes[0], "Barcelona", BAR_C), (axes[1], "Real Madrid", RMA_C)]:
    d = matches[matches.team == team].reset_index(drop=True)
    x = np.arange(len(d))
    ax.bar(x - 0.2, d.gf, 0.38, color=col, label="Scored", edgecolor=SURF, linewidth=1)
    ax.bar(x + 0.2, d.ga, 0.38, color=DRAW_C, label="Conceded", edgecolor=SURF, linewidth=1)
    labels = [f"{ABBR.get(o, o)}\n{v}{'*' if c == 'UCL' else ''}" for o, v, c in zip(d.opponent, d.venue, d.competition)]
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_title(f"{team}: every match this season (LaLiga + UCL*), oldest to newest", loc="left")
    ax.set_ylabel("Goals")
    ax.legend(frameon=False, loc="upper right", ncol=2)
    ax.grid(axis="x", visible=False)
save(fig, "01_form_by_match.png")

# 2. Goal timing --------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
x = np.arange(6)
for ax, team, col in [(axes[0], "Barcelona", BAR_C), (axes[1], "Real Madrid", RMA_C)]:
    t = R["timing"][team]
    ax.bar(x - 0.2, t["for_counts"], 0.38, color=col, label="Scored", edgecolor=SURF)
    ax.bar(x + 0.2, t["against_counts"], 0.38, color=DRAW_C, label="Conceded", edgecolor=SURF)
    for i, v in enumerate(t["for_counts"]):
        ax.text(i - 0.2, v + 0.2, str(v), ha="center", fontsize=8, color=INK2)
    for i, v in enumerate(t["against_counts"]):
        ax.text(i + 0.2, v + 0.2, str(v), ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, BIN_LABELS)
    ax.set_xlabel("Minute")
    ax.set_title(f"{team}: when goals happen (12 matches)", loc="left")
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Goals")
axes[1].legend(frameon=False)
save(fig, "02_goal_timing.png")

# 3. Simulated scoring intensity, minute by minute ----------------------------
haz = np.load(OUT / "hazards.npy") * 90  # goals per 90 at each minute
fig, ax = plt.subplots(figsize=(10, 3.8))
m = np.arange(1, haz.shape[1] + 1)
ax.axvspan(58, 70, color=GRID, alpha=0.6, lw=0)
ax.axvspan(78, 90, color=GRID, alpha=0.6, lw=0)
ax.text(64, haz.max() * 1.02, "1st sub wave", ha="center", fontsize=8, color=INK2)
ax.text(84, haz.max() * 1.02, "Late subs / Espi window", ha="center", fontsize=8, color=INK2)
ax.plot(m, haz[0], color=BAR_C, lw=2, label="Barcelona scoring rate")
ax.plot(m, haz[1], color=RMA_C, lw=2, label="Real Madrid scoring rate")
ax.axvline(47, color=INK2, lw=0.8, ls=":")
ax.text(47.5, haz.min() * 0.9, "HT", fontsize=8, color=INK2)
ax.set_xlim(1, haz.shape[1])
ax.set_ylim(0, haz.max() * 1.1)
ax.set_xlabel("Match minute (incl. stoppage)")
ax.set_ylabel("Expected goals per 90 at that minute")
ax.set_title("Simulated Clasico: each side's scoring threat across the 90 minutes", loc="left")
ax.legend(frameon=False, loc="upper left")
save(fig, "03_minute_intensity.png")

# 4. Comeback windows ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 3.8))
s10 = scen[scen.score == "1-0"].sort_values("minute")
s01 = scen[scen.score == "0-1"].sort_values("minute")
s00 = scen[scen.score == "0-0"].sort_values("minute")
ax.plot(s10.minute, (s10.p_draw + s10.p_rma) * 100, color=RMA_C, lw=2, marker="o", ms=4,
        label="Madrid avoid defeat if 0-1 down at this minute")
ax.plot(s01.minute, (s01.p_draw + s01.p_bar) * 100, color=BAR_C, lw=2, marker="o", ms=4,
        label="Barca avoid defeat if 0-1 down at this minute")
ax.plot(s00.minute, s00.p_bar * 100, color=BAR_C, lw=1.5, ls="--", label="Barca win if 0-0 at this minute")
ax.set_xlabel("Minute")
ax.set_ylabel("Probability (%)")
ax.set_ylim(0, 100)
ax.set_title("How long each side can wait: probability from a given game state", loc="left")
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
save(fig, "04_comeback_windows.png")

# 5. Score grid ---------------------------------------------------------------
grid = np.load(OUT / "score_grid.npy")[:6, :6] * 100
fig, ax = plt.subplots(figsize=(5.6, 4.8))
ax.grid(False)
im = ax.imshow(grid, cmap="Blues", origin="upper")
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=8,
                color="white" if grid[i, j] > grid.max() * 0.6 else INK)
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xlabel("Real Madrid goals"); ax.set_ylabel("Barcelona goals")
ax.set_title("Scoreline probabilities (%)", loc="left")
save(fig, "05_score_grid.png")

# 6. Ensemble -----------------------------------------------------------------
sources = [("Form model\n(minute sim)", R["model"]["p"]), ("Market\nconsensus", R["market"]["p"]),
           ("Pundits", R["pundits"]["p"])]
for o in R["other_models"]:
    if o["source"].startswith("Opta"):
        sources.append(("Opta\nsupercomputer", [o["p_bar"], o["p_draw"], o["p_rma"]]))
sources.append(("FINAL\nensemble", R["final"]))
fig, ax = plt.subplots(figsize=(10, 3.8))
x = np.arange(len(sources))
for k, (lab, col) in enumerate([("Barca win", BAR_C), ("Draw", DRAW_C), ("Madrid win", RMA_C)]):
    vals = [s[1][k] * 100 for s in sources]
    ax.bar(x + (k - 1) * 0.26, vals, 0.25, color=col, label=lab, edgecolor=SURF)
    for xi, v in zip(x, vals):
        ax.text(xi + (k - 1) * 0.26, v + 1, f"{v:.0f}", ha="center", fontsize=8, color=INK2)
ax.set_xticks(x, [s[0] for s in sources])
ax.set_ylabel("Probability (%)")
ax.set_ylim(0, 85)
ax.grid(axis="x", visible=False)
ax.legend(frameon=False, ncol=3, loc="upper right")
ax.set_title("Who wins? Each input and the final blend", loc="left")
save(fig, "06_ensemble.png")
print("charts written to", CH)
