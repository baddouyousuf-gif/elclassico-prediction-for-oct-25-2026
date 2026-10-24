"""
El Clasico forecast engine v2 - Barcelona vs Real Madrid, LaLiga MD10, 25 Oct 2026, Camp Nou
=============================================================================================

Pipeline (credit-risk analogy in brackets):
  1. Form ratings      opponent-adjusted, time-decayed attack/defence from every 2026-27
                       LaLiga + UCL match                                       [obligor scoring]
  2. xG regression     finishing luck pulled back toward expected goals          [noise filter]
  3. Prior test        2025-26 seasons enter ONLY if leave-one-out log-likelihood
                       improves (Flick's Barca; Madrid's prior down-weighted for coach change)
                                                                                 [out-of-sample validation]
  4. Availability      injuries/suspensions as multiplicative shocks            [stress overlays]
  5. Minute engine     minute-by-minute Monte Carlo with team-specific timing profiles,
                       game-state effects (leading/trailing), momentum, red cards [path-dependent MC]
  6. Ensemble          model + de-vigged market consensus + pundit sentiment     [model vs market PD]

Run:  python clasico_model_v2.py      (reads ../data, writes ../outputs)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(20261025)

# =============================================================== ASSUMPTIONS
BINS = [(1, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 99)]
BIN_LABELS = ["1-15", "16-30", "31-45+", "46-60", "61-75", "76-90+"]
# Typical top-flight goal-timing shares (approximate reference distribution; goals rise late)
REF_PROFILE = np.array([0.135, 0.155, 0.180, 0.160, 0.170, 0.200])
PROFILE_PSEUDO_GOALS = 30          # shrinkage of team timing profiles toward the reference
HOME_ADV, AWAY_ADJ = 1.15, 0.87    # goal multipliers for home / away side
HALF_LIFE = 8                      # matches; recency weight halves every 8 matches
XG_ALPHA = 0.6                     # weight on xG vs goals when correcting finishing luck
OPP_SHRINK = 0.5                   # shrink opponent ratings halfway to average (9-game tables are noisy)
# UCL opponents, rated relative to an average LaLiga side (judgement from club strength/current form)
UCL_OPP = {
    "Inter":               dict(att=1.35, dfn=0.70),
    "Paris Saint-Germain": dict(att=1.55, dfn=0.70),
    "Roma":                dict(att=1.10, dfn=0.85),
    "RB Leipzig":          dict(att=1.15, dfn=1.00),
    "Feyenoord":           dict(att=1.05, dfn=1.15),
    "Galatasaray":         dict(att=1.15, dfn=1.05),
}
# Availability shocks (multiply the goal rate they affect)
SHOCKS = {
    "BAR_attack":  0.99 * 0.99,  # Olmo out (Fermin in form); small Yamal groin doubt
    "BAR_defence": 1.06,         # Kounde, E. Garcia, Christensen out -> Madrid scores more
    "RMA_attack":  0.97,         # Vinicius suspended (Diomande in form offsets most of it)
    "RMA_defence": 1.01,         # Mendy, Asencio out
}
# In-match parameters
P_RED = {"BAR": 0.07, "RMA": 0.12}   # chance of a red card per match (Madrid: 2 reds in 12; Mourinho Clasico history)
RED_EFFECT_SELF, RED_EFFECT_OPP = 0.75, 1.35
STATE_PSEUDO_MIN = 450               # shrinkage (minutes) for game-state effects toward 1.0
LIT_STATE = {"trailing": 1.12, "level": 1.0, "leading": 0.92}  # literature-style prior for state effects
MOMENTUM_WINDOW = 10                 # minutes after a goal
N_SIMS = 100_000
MATCH_LEN = 96                       # simulated minutes incl. stoppage (1H ~47, 2H ~49)

# =============================================================== LOAD
matches = pd.read_csv(DATA / "matches_2026_27.csv")
goals = pd.read_csv(DATA / "goal_events.csv")
subs = pd.read_csv(DATA / "substitutions.csv")
table = pd.read_csv(DATA / "league_table_md9.csv").set_index("team")
mk = pd.read_csv(DATA / "market_and_model_probabilities.csv")
experts = pd.read_csv(DATA / "expert_predictions.csv")

LEAGUE_AVG = table.gf.sum() / table.played.sum()          # goals per team-game, 2026-27
TEAM_KEY = {"Barcelona": "BAR", "Real Madrid": "RMA"}

# =============================================================== 1. OPPONENT FACTORS
OPP_NAME = {"Athletic Club": "Athletic Club", "Malaga": "Malaga", "Atletico Madrid": "Atletico Madrid"}

def opp_factors(row):
    opp = row.opponent
    if opp in UCL_OPP:
        return UCL_OPP[opp]["att"], UCL_OPP[opp]["dfn"]
    t = table.loc[OPP_NAME.get(opp, opp)]
    # remove this match from the opponent's record so the team isn't rated against itself
    p = t.played - 1
    att = ((t.gf - row.ga) / p) / LEAGUE_AVG
    dfn = ((t.ga - row.gf) / p) / LEAGUE_AVG
    return 1 + OPP_SHRINK * (att - 1), 1 + OPP_SHRINK * (dfn - 1)

matches[["opp_att", "opp_def"]] = matches.apply(lambda r: pd.Series(opp_factors(r)), axis=1)
venue_for = np.where(matches.venue == "H", HOME_ADV, AWAY_ADJ)
venue_against = np.where(matches.venue == "H", AWAY_ADJ, HOME_ADV)
matches["base_for"] = LEAGUE_AVG * matches.opp_def * venue_for
matches["base_against"] = LEAGUE_AVG * matches.opp_att * venue_against
matches["order"] = matches.groupby("team").cumcount()

# xG luck factors (LaLiga totals; applied to whole-season ratings)
xg_adj = {}
for team in TEAM_KEY:
    t = table.loc[team]
    xg_adj[team] = dict(att=(t.xg / t.gf) ** XG_ALPHA, dfn=(t.xga / t.ga) ** XG_ALPHA)

# Last-season priors, as requested: Flick's 2025-26 Barcelona and Mourinho's 2025-26 Benfica.
# (Real Madrid's own 2025-26 numbers are NOT used - different coach, and the brief was this season only.)
# Strengths are relative to each league's average goals per team-game.
LALIGA_25_AVG = 2.69 / 2           # 1,024 goals / 380 matches
PRIMEIRA_25_AVG = 1.30             # assumption: ~2.6 goals per Primeira Liga match
PRIORS = {"Barcelona": dict(att=95 / 38 / LALIGA_25_AVG, dfn=36 / 38 / LALIGA_25_AVG),
          "Real Madrid": dict(att=74 / 34 / PRIMEIRA_25_AVG, dfn=25 / 34 / PRIMEIRA_25_AVG)}  # Mourinho's Benfica

def ratings(df, team, k_prior, half_life=HALF_LIFE, exclude=None):
    d = df[df.team == team]
    if exclude is not None:
        d = d[d.match_id != exclude]
    n = df[df.team == team].order.max()
    w = 0.5 ** ((n - d.order) / half_life) if half_life else np.ones(len(d))
    avg_base_f, avg_base_a = d.base_for.mean(), d.base_against.mean()
    pr = PRIORS[team]
    att = (np.sum(w * d.gf) + k_prior * pr["att"] * avg_base_f) / (np.sum(w * d.base_for) + k_prior * avg_base_f)
    dfn = (np.sum(w * d.ga) + k_prior * pr["dfn"] * avg_base_a) / (np.sum(w * d.base_against) + k_prior * avg_base_a)
    return att * xg_adj[team]["att"], dfn * xg_adj[team]["dfn"]

# =============================================================== 2-3. LEAVE-ONE-OUT TESTS
def loo_loglik(team, k_prior, half_life):
    d = matches[matches.team == team]
    ll = 0.0
    for _, r in d.iterrows():
        a, f = ratings(matches, team, k_prior, half_life, exclude=r.match_id)
        ll += poisson.logpmf(r.gf, a * r.base_for) + poisson.logpmf(r.ga, f * r.base_against)
    return ll

K_GRID = [0, 2, 4, 8, 12, 20, 38]
HL_GRID = [4, 8, 0]  # 0 = no decay
loo = []
for team in TEAM_KEY:
    for hl in HL_GRID:
        for k in K_GRID:
            loo.append(dict(team=team, half_life=hl if hl else "none", k_prior_games=k,
                            loo_loglik=round(loo_loglik(team, k, hl), 3)))
loo = pd.DataFrame(loo)
no_prior = loo[(loo.k_prior_games == 0)].groupby("team").loo_loglik.max()
PARSIMONY_TOL = 0.05   # take the smallest prior weight within 0.05 log-lik of the best (avoid over-trusting a prior)
CHOICE = {}
for team in TEAM_KEY:
    d = loo[loo.team == team]
    top = d.loo_loglik.max()
    hl = d.loc[d.loo_loglik.idxmax(), "half_life"]
    d_hl = d[d.half_life == hl]
    k = int(d_hl[d_hl.loo_loglik >= top - PARSIMONY_TOL].k_prior_games.min())
    chosen_ll = float(d_hl[d_hl.k_prior_games == k].loo_loglik.iloc[0])
    CHOICE[team] = dict(k_prior=k, half_life=0 if hl == "none" else int(hl),
                        best_k_in_grid=int(d.loc[d.loo_loglik.idxmax(), "k_prior_games"]),
                        gain_vs_no_prior=round(chosen_ll - float(no_prior[team]), 3),
                        prior_used=k > 0)

att_b, def_b = ratings(matches, "Barcelona", CHOICE["Barcelona"]["k_prior"], CHOICE["Barcelona"]["half_life"])
att_m, def_m = ratings(matches, "Real Madrid", CHOICE["Real Madrid"]["k_prior"], CHOICE["Real Madrid"]["half_life"])

# Rating-model skill check: LOO W/D/L Brier vs naive base rates
def wdl(lh, la, n=10):
    g = np.arange(n + 1)
    M = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    return np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()

brier_model, brier_naive, rows = [], [], []
for team in TEAM_KEY:
    for _, r in matches[matches.team == team].iterrows():
        a, f = ratings(matches, team, CHOICE[team]["k_prior"], CHOICE[team]["half_life"], exclude=r.match_id)
        p = np.array(wdl(a * r.base_for, f * r.base_against))
        y = np.array([r.gf > r.ga, r.gf == r.ga, r.gf < r.ga], float)
        brier_model.append(((p - y) ** 2).sum())
        brier_naive.append(((np.array([0.45, 0.27, 0.28]) - y) ** 2).sum())
        rows.append(dict(match=r.match_id, opp=r.opponent, p_win=round(p[0], 3), result=f"{r.gf}-{r.ga}"))
backtest = dict(brier_model=round(float(np.mean(brier_model)), 3),
                brier_naive=round(float(np.mean(brier_naive)), 3), detail=rows)

# =============================================================== 4. MATCH GOAL RATES
lam_b = LEAGUE_AVG * att_b * def_m * HOME_ADV * SHOCKS["BAR_attack"] * SHOCKS["RMA_defence"]
lam_m = LEAGUE_AVG * att_m * def_b * AWAY_ADJ * SHOCKS["RMA_attack"] * SHOCKS["BAR_defence"]

# =============================================================== 5a. MINUTE-LEVEL ANALYTICS
def bin_of(m):
    for i, (a, b) in enumerate(BINS):
        if a <= m <= b:
            return i
    return len(BINS) - 1

goals["bin"] = goals.minute.apply(bin_of)
timing = {}
for team in TEAM_KEY:
    g = goals[goals.team == team]
    f = np.bincount(g[g.side == "for"].bin, minlength=6)
    a = np.bincount(g[g.side == "against"].bin, minlength=6)
    timing[team] = dict(for_counts=f.tolist(), against_counts=a.tolist())

def shrunk_profile(counts):
    counts = np.asarray(counts, float)
    n = counts.sum()
    share = counts / n if n else REF_PROFILE
    w = n / (n + PROFILE_PSEUDO_GOALS)
    p = w * share + (1 - w) * REF_PROFILE
    return p / p.sum()

prof = {t: dict(score=shrunk_profile(timing[t]["for_counts"]),
                concede=shrunk_profile(timing[t]["against_counts"])) for t in TEAM_KEY}

def match_timeline(mid):
    """Return per-goal (minute, +1 for / -1 against) and final length."""
    g = goals[goals.match_id == mid].sort_values("minute")
    return [(int(m), 1 if s == "for" else -1) for m, s in zip(g.minute, g.side)]

def state_minutes(team):
    """Minutes and goals for/against while leading, level, trailing."""
    out = {s: dict(minutes=0.0, gf=0, ga=0) for s in ["leading", "level", "trailing"]}
    for mid in matches[matches.team == team].match_id:
        ev = match_timeline(mid)
        diff, t_prev = 0, 0
        for m, s in ev + [(MATCH_LEN, 0)]:
            st = "leading" if diff > 0 else "trailing" if diff < 0 else "level"
            out[st]["minutes"] += max(m - t_prev, 0)
            if s == 1:
                out[st]["gf"] += 1
            elif s == -1:
                out[st]["ga"] += 1
            diff += s
            t_prev = m
    for s in out:
        mins = out[s]["minutes"]
        out[s]["gf_per90"] = round(out[s]["gf"] / mins * 90, 2) if mins else None
        out[s]["ga_per90"] = round(out[s]["ga"] / mins * 90, 2) if mins else None
        out[s]["minutes"] = round(mins)
    return out

states = {t: state_minutes(t) for t in TEAM_KEY}

def state_multipliers(team):
    st = states[team]
    tot_gf = sum(st[s]["gf"] for s in st)
    tot_min = sum(st[s]["minutes"] for s in st)
    base = tot_gf / tot_min
    mult = {}
    for s in st:
        mins, gf = st[s]["minutes"], st[s]["gf"]
        # Bayesian shrink: pseudo-minutes at the literature multiplier
        mult[s] = (gf + STATE_PSEUDO_MIN * base * LIT_STATE[s]) / ((mins + STATE_PSEUDO_MIN) * base)
    return mult

state_mult = {t: state_multipliers(t) for t in TEAM_KEY}

def response_stats(team):
    """After conceding: minutes until the team's next goal (censored at full time)."""
    times, answered15 = [], 0
    n = 0
    for mid in matches[matches.team == team].match_id:
        ev = match_timeline(mid)
        for i, (m, s) in enumerate(ev):
            if s == -1:
                n += 1
                nxt = next((m2 for m2, s2 in ev[i + 1:] if s2 == 1), None)
                if nxt is not None:
                    times.append(nxt - m)
                    answered15 += (nxt - m) <= 15
    return dict(goals_conceded=n, answered=len(times), answered_within_15=answered15,
                median_minutes_to_reply=float(np.median(times)) if times else None)

responses = {t: response_stats(t) for t in TEAM_KEY}

def momentum_stats():
    """Pairs of consecutive goals within MOMENTUM_WINDOW minutes: same side again vs other side."""
    same = other = 0
    for mid in matches.match_id:
        ev = match_timeline(mid)
        for (m1, s1), (m2, s2) in zip(ev, ev[1:]):
            if m2 - m1 <= MOMENTUM_WINDOW:
                same += s1 == s2
                other += s1 != s2
    # expected 'same' share without momentum ~ goal share of scoring side; both teams outscore opponents ~2.7:1
    return dict(pairs=same + other, same_side=same, other_side=other)

momentum = momentum_stats()
# Momentum multiplier: shrunk toward 1 (literature finds small effects). Use 1 + 0.10 if clustering observed.
total_for = (goals.side == "for").sum()
exp_same = (total_for / len(goals)) ** 2 + (1 - total_for / len(goals)) ** 2
obs_same = momentum["same_side"] / max(momentum["pairs"], 1)
MOMENTUM_MULT = 1 + float(np.clip((obs_same - exp_same) * 0.5, 0, 0.15))
momentum.update(expected_same_share=round(exp_same, 3), observed_same_share=round(obs_same, 3),
                momentum_multiplier=round(MOMENTUM_MULT, 3))

def bench_stats(team):
    g = goals[(goals.team == team) & (goals.side == "for")]
    bench = int((g.scorer_off_bench == 1).sum())
    late = int((g.minute >= 76).sum())
    early = int((g.minute <= 60).sum())
    mid_late = int((g.minute > 60).sum())
    n = matches[matches.team == team].shape[0]
    s = subs[subs.team == team]
    first_wave = s[s.reason.str.startswith("tactical")].groupby("match_id").minute.min()
    return dict(goals=len(g), bench_goals=bench, bench_share=round(bench / len(g), 3),
                goals_76_plus=late, share_76_plus=round(late / len(g), 3),
                per90_min1_60=round(early / (n * 60) * 90, 2), per90_min61_end=round(mid_late / (n * 36) * 90, 2),
                median_first_tactical_sub=float(first_wave.median()) if len(first_wave) else None)

bench = {t: bench_stats(t) for t in TEAM_KEY}

# =============================================================== 5b. MINUTE-BY-MINUTE SIMULATION
def minute_profile(score_prof, concede_prof):
    """Per-minute relative intensity (mean 1) combining attacker's scoring and defender's conceding timing."""
    comb = np.sqrt(score_prof * concede_prof)
    comb = comb / comb.sum()
    per_min = np.zeros(MATCH_LEN)
    for i, (a, b) in enumerate(BINS):
        lo, hi = a - 1, min(b, MATCH_LEN)
        if i == 2:
            hi = 47           # first half incl. ~2 min stoppage
        if i == 3:
            lo = 47
        per_min[lo:hi] = comb[i] / (hi - lo)
    return per_min / per_min.mean()

PROF_B = minute_profile(prof["Barcelona"]["score"], prof["Real Madrid"]["concede"])
PROF_M = minute_profile(prof["Real Madrid"]["score"], prof["Barcelona"]["concede"])

def simulate(lb, lm, n=N_SIMS, start_min=0, start_score=(0, 0), seed=1, record=False):
    r = np.random.default_rng(seed)
    gb = np.full(n, start_score[0]); gm = np.full(n, start_score[1])
    red_b_min = np.where(r.random(n) < P_RED["BAR"], r.integers(25, MATCH_LEN, n), 999)
    red_m_min = np.where(r.random(n) < P_RED["RMA"], r.integers(25, MATCH_LEN, n), 999)
    last_goal_b = np.full(n, -99); last_goal_m = np.full(n, -99)
    first_goal_min = np.full(n, -1); first_goal_side = np.zeros(n, int)
    ht = None
    hb = np.zeros(MATCH_LEN); hm = np.zeros(MATCH_LEN)
    sb = {s: lb / MATCH_LEN for s in ["leading", "level", "trailing"]}
    for t in range(start_min, MATCH_LEN):
        d = gb - gm
        st_b = np.where(d > 0, state_mult["Barcelona"]["leading"],
                        np.where(d < 0, state_mult["Barcelona"]["trailing"], state_mult["Barcelona"]["level"]))
        st_m = np.where(d < 0, state_mult["Real Madrid"]["leading"],
                        np.where(d > 0, state_mult["Real Madrid"]["trailing"], state_mult["Real Madrid"]["level"]))
        ib = lb / MATCH_LEN * PROF_B[t] * st_b
        im = lm / MATCH_LEN * PROF_M[t] * st_m
        ib = ib * np.where(t - last_goal_b <= MOMENTUM_WINDOW, MOMENTUM_MULT, 1.0)
        im = im * np.where(t - last_goal_m <= MOMENTUM_WINDOW, MOMENTUM_MULT, 1.0)
        rb, rm = t >= red_b_min, t >= red_m_min
        ib = ib * np.where(rb, RED_EFFECT_SELF, 1) * np.where(rm, RED_EFFECT_OPP, 1)
        im = im * np.where(rm, RED_EFFECT_SELF, 1) * np.where(rb, RED_EFFECT_OPP, 1)
        hb[t], hm[t] = ib.mean(), im.mean()
        sb_ = r.random(n) < ib
        sm_ = r.random(n) < im
        new_first = (first_goal_min < 0) & (sb_ | sm_)
        first_goal_min = np.where(new_first, t + 1, first_goal_min)
        first_goal_side = np.where(new_first, np.where(sb_ & ~sm_, 1, np.where(sm_ & ~sb_, -1, 1)), first_goal_side)
        gb = gb + sb_; gm = gm + sm_
        last_goal_b = np.where(sb_, t, last_goal_b); last_goal_m = np.where(sm_, t, last_goal_m)
        if t == 46:
            ht = (gb.copy(), gm.copy())
    res = dict(gb=gb, gm=gm, first_min=first_goal_min, first_side=first_goal_side,
               red_b=red_b_min < 999, red_m=red_m_min < 999, haz_b=hb, haz_m=hm)
    if ht is not None:
        res["ht"] = ht
    return res

# Calibrate base rates so simulated mean goals equal the structural lambdas
cal_b, cal_m = lam_b, lam_m
for _ in range(3):
    s = simulate(cal_b, cal_m, n=40_000, seed=7)
    cal_b *= lam_b / s["gb"].mean()
    cal_m *= lam_m / s["gm"].mean()

sim = simulate(cal_b, cal_m, seed=11)
gb, gm = sim["gb"], sim["gm"]
p_model = np.array([(gb > gm).mean(), (gb == gm).mean(), (gb < gm).mean()])

score_counts = pd.Series(list(zip(gb, gm))).value_counts(normalize=True)
top_scores = [(f"{a}-{b}", round(float(p), 4)) for (a, b), p in score_counts.head(10).items()]
grid = np.zeros((7, 7))
for (a, b), p in score_counts.items():
    if a < 7 and b < 7:
        grid[a, b] = p

ht_b, ht_m = sim["ht"]
ht_ft = pd.crosstab(np.sign(ht_b - ht_m), np.sign(gb - gm), normalize=True)

# Goals per 15-minute bin in the simulated Clasico
def bin_expect(h):
    return [float(h[a - 1:min(b, MATCH_LEN)].sum()) for a, b in BINS]

exp_bins = dict(Barcelona=bin_expect(sim["haz_b"]), Real_Madrid=bin_expect(sim["haz_m"]))

# Conditional / in-play scenarios: how long can each side wait?
scen = []
# Only the goal difference drives the simulated outcome, so 1-0 and 2-1 are equivalent states
for minute in range(0, 90, 5):
    for sc in [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2)]:
        r_ = simulate(cal_b, cal_m, n=20_000, start_min=minute, start_score=sc, seed=100 + minute)
        b_, m_ = r_["gb"], r_["gm"]
        scen.append(dict(minute=minute, score=f"{sc[0]}-{sc[1]}",
                         p_bar=round(float((b_ > m_).mean()), 3), p_draw=round(float((b_ == m_).mean()), 3),
                         p_rma=round(float((b_ < m_).mean()), 3)))
scen = pd.DataFrame(scen)

first_goal = dict(
    p_bar_first=round(float((sim["first_side"] == 1).mean()), 3),
    p_rma_first=round(float((sim["first_side"] == -1).mean()), 3),
    p_no_goal=round(float((sim["first_min"] < 0).mean()), 3),
    median_first_goal_minute=float(np.median(sim["first_min"][sim["first_min"] > 0])),
)
cond = {}
for side, name in [(1, "if_bar_score_first"), (-1, "if_rma_score_first")]:
    msk = sim["first_side"] == side
    cond[name] = [round(float(x), 3) for x in [(gb[msk] > gm[msk]).mean(), (gb[msk] == gm[msk]).mean(),
                                               (gb[msk] < gm[msk]).mean()]]

# =============================================================== 6. ENSEMBLE
def american_to_prob(o):
    o = float(o)
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)

books = []
for _, r in mk[mk.type == "sportsbook"].iterrows():
    odds = [x.strip() for x in r.raw.split("/")]
    p = np.array([american_to_prob(x) for x in odds])
    books.append(p / p.sum())
markets = [np.array([r.p_bar, r.p_draw, r.p_rma]) / (r.p_bar + r.p_draw + r.p_rma)
           for _, r in mk[mk.type == "prediction market"].iterrows()]
p_market = np.mean(books + markets, axis=0)
other_models = mk[mk.type == "model"][["source", "p_bar", "p_draw", "p_rma"]]

pick_map = {"BAR": np.array([0.6, 0.2, 0.2]), "BAR (lean)": np.array([0.5, 0.25, 0.25]),
            "DRAW": np.array([0.3, 0.4, 0.3]), "RMA": np.array([0.2, 0.2, 0.6])}
pk = experts[experts.pick.isin(pick_map)]
p_pundit = np.mean([pick_map[p] for p in pk.pick], axis=0)
pundit_tally = experts.pick.value_counts().to_dict()
pundit_scores = experts.dropna(subset=["score_home"])
pundit_total_goals = float((pundit_scores.score_home + pundit_scores.score_away).mean())

# Markets are the most efficient single input, so they get the largest weight;
# the form model is the main "independent view"; pundits are a small sentiment term.
W = dict(model=0.4, market=0.5, pundits=0.1)
p_final = W["model"] * p_model + W["market"] * p_market + W["pundits"] * p_pundit

# Market-implied expected goals: fit Poisson rates to the market's H/D/A and Over-4.5 price (Kalshi 47%)
from scipy.optimize import minimize

def _poisson_grid(lb, lm, n=15):
    g = np.arange(n)
    return np.outer(poisson.pmf(g, lb), poisson.pmf(g, lm))

def _market_loss(x, target=p_market, o45=0.47):
    M = _poisson_grid(*np.exp(x))
    p = np.array([np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()])
    idx = np.add.outer(np.arange(15), np.arange(15))
    return ((p - target) ** 2).sum() + (M[idx > 4].sum() - o45) ** 2

lam_market = np.exp(minimize(_market_loss, np.log([2.5, 1.5]), method="Nelder-Mead").x)

# Sensitivity of the form model (analytic Poisson, fast)
def _wdl(lb, lm):
    M = _poisson_grid(lb, lm)
    return [round(float(x), 3) for x in (np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum())]

att_b12, def_b12 = ratings(matches, "Barcelona", 12, CHOICE["Barcelona"]["half_life"])
att_m12, def_m12 = ratings(matches, "Real Madrid", 12, CHOICE["Real Madrid"]["half_life"])
sensitivity = [
    dict(case="Base form model", lb=lam_b, lm=lam_m),
    dict(case="Force 12-game last-season priors on both", lb=LEAGUE_AVG * att_b12 * def_m12 * HOME_ADV * SHOCKS["BAR_attack"] * SHOCKS["RMA_defence"],
         lm=LEAGUE_AVG * att_m12 * def_b12 * AWAY_ADJ * SHOCKS["RMA_attack"] * SHOCKS["BAR_defence"]),
    dict(case="Elite-matchup dampener 0.85 on both", lb=lam_b * 0.85, lm=lam_m * 0.85),
    dict(case="Barca defence crisis worse (x1.15)", lb=lam_b, lm=lam_m / SHOCKS["BAR_defence"] * 1.15),
    dict(case="No home advantage (neutral)", lb=lam_b / HOME_ADV, lm=lam_m / AWAY_ADJ),
    dict(case="Raw goals only (no xG correction)", lb=lam_b / xg_adj["Barcelona"]["att"] / xg_adj["Real Madrid"]["dfn"],
         lm=lam_m / xg_adj["Real Madrid"]["att"] / xg_adj["Barcelona"]["dfn"]),
    dict(case="Market-implied rates", lb=lam_market[0], lm=lam_market[1]),
]
for s_ in sensitivity:
    s_["p"] = _wdl(s_["lb"], s_["lm"])
    s_["lb"], s_["lm"] = round(float(s_["lb"]), 2), round(float(s_["lm"]), 2)

# =============================================================== SAVE
results = dict(
    league_avg_goals_per_team_game=round(LEAGUE_AVG, 3),
    prior_test=dict(choice=CHOICE, loo_table=loo.to_dict(orient="records")),
    rating_backtest=backtest,
    ratings=dict(Barcelona=dict(att=round(att_b, 3), dfn=round(def_b, 3)),
                 Real_Madrid=dict(att=round(att_m, 3), dfn=round(def_m, 3))),
    xg_adjustment=xg_adj,
    lambdas=dict(Barcelona=round(lam_b, 3), Real_Madrid=round(lam_m, 3), total=round(lam_b + lam_m, 3)),
    timing=timing, profiles={t: {k: np.round(v, 3).tolist() for k, v in prof[t].items()} for t in prof},
    game_states=states, state_multipliers={t: {k: round(v, 3) for k, v in state_mult[t].items()} for t in state_mult},
    responses=responses, momentum=momentum, bench=bench,
    model=dict(p=np.round(p_model, 4).tolist(), mean_goals=[round(float(gb.mean()), 3), round(float(gm.mean()), 3)],
               over_2_5=round(float(((gb + gm) > 2.5).mean()), 3), over_3_5=round(float(((gb + gm) > 3.5).mean()), 3),
               over_4_5=round(float(((gb + gm) > 4.5).mean()), 3), btts=round(float(((gb > 0) & (gm > 0)).mean()), 3),
               top_scores=top_scores, first_goal=first_goal, conditional_on_first_goal=cond,
               p_red_any=round(float((sim["red_b"] | sim["red_m"]).mean()), 3),
               ht_ft=ht_ft.round(4).to_dict(), expected_goals_by_bin=exp_bins),
    market=dict(p=np.round(p_market, 4).tolist(), n_sources=len(books) + len(markets)),
    other_models=other_models.to_dict(orient="records"),
    pundits=dict(p=np.round(p_pundit, 4).tolist(), tally=pundit_tally, avg_total_goals_predicted=round(pundit_total_goals, 2)),
    weights=W, final=np.round(p_final, 4).tolist(),
    market_implied_lambdas=np.round(lam_market, 3).tolist(),
    sensitivity=sensitivity,
)
(OUT / "results.json").write_text(json.dumps(results, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
scen.to_csv(OUT / "in_play_scenarios.csv", index=False)
np.save(OUT / "score_grid.npy", grid)
np.save(OUT / "hazards.npy", np.vstack([sim["haz_b"], sim["haz_m"]]))

if __name__ == "__main__":
    print(f"League avg goals/team-game: {LEAGUE_AVG:.3f}")
    print("Prior test choice:", CHOICE)
    print("Backtest Brier model vs naive:", backtest["brier_model"], backtest["brier_naive"])
    print(f"Ratings  BAR att {att_b:.2f} def {def_b:.2f} | RMA att {att_m:.2f} def {def_m:.2f}")
    print(f"Lambdas  BAR {lam_b:.2f}  RMA {lam_m:.2f}")
    print("Model  H/D/A", np.round(p_model, 3), "goals", gb.mean().round(2), gm.mean().round(2))
    print("Market H/D/A", np.round(p_market, 3))
    print("Pundit H/D/A", np.round(p_pundit, 3), pundit_tally)
    print("FINAL  H/D/A", np.round(p_final, 3))
    print("Market-implied lambdas", np.round(lam_market, 2))
    for s_ in sensitivity:
        print(f"  {s_['case']:40s} {s_['lb']:.2f}-{s_['lm']:.2f}  {s_['p']}")
    print(scen.to_string())
    print("Top scores", top_scores[:6])
    print("First goal", first_goal, cond)
    print("States", json.dumps(states, indent=1))
    print("State mult", state_mult)
    print("Responses", responses)
    print("Momentum", momentum)
    print("Bench", bench)
    print("Timing", timing)
