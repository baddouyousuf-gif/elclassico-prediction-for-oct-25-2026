"""
El Clásico forecast engine — Barcelona vs Real Madrid, La Liga MD10, 25 Oct 2026 (Camp Nou)

Same architecture as a credit-risk engine:
  1. Structural model   -> Poisson goal intensities (lambda) from attack/defence ratings
                           (analogue of Merton: latent "strength" drives the outcome)
  2. Bayesian shrinkage -> 9-game sample is thin, so blend with priors (last season + H2H)
  3. Scenario overlays   -> absences / fatigue as multiplicative shocks on lambda
  4. Dependence fix      -> Dixon-Coles rho corrects low-score correlation (like a copula tweak)
  5. Monte Carlo         -> simulate parameter uncertainty + match outcomes (like MC VaR)
  6. Benchmark           -> compare to market-implied probabilities (like model PD vs CDS-implied PD)

Inputs current as of 24 Oct 2026. Edit the INPUTS block to re-run.
"""
import numpy as np
from scipy.stats import poisson

rng = np.random.default_rng(2026)

# ----------------------------------------------------------------- INPUTS
GAMES = 9
TEAMS = {
    #            goals for, xG for, goals against   (2026-27 La Liga, 9 games)
    "Barcelona":   dict(gf=39, xg=33.6, ga=9),
    "Real Madrid": dict(gf=23, xg=20.9, ga=9),
}
# Priors: per-game rates from last season (approx., 2025-26 La Liga)
PRIOR = {
    "Barcelona":   dict(att=2.6, dfn=1.0),
    "Real Madrid": dict(att=2.1, dfn=0.95),
}
PRIOR_WEIGHT_GAMES = 9          # prior counts as 9 games of evidence
XG_WEIGHT = 0.7                 # attack signal = 70% xG, 30% actual goals

LEAGUE_HOME_GPG = 1.45          # La Liga avg goals/game, home side
LEAGUE_AWAY_GPG = 1.15          # away side
LEAGUE_AVG = (LEAGUE_HOME_GPG + LEAGUE_AWAY_GPG) / 2

# Head-to-head: last 10 Clásicos (all comps) — Barça 25 goals, Madrid 19
H2H_BARCA_GPG, H2H_MADRID_GPG = 2.5, 1.9
H2H_WEIGHT = 0.25

# Elite-opponent dampener: season rates are inflated by weaker opponents
ELITE_DAMPENER = 0.82

# Scenario shocks (multiplicative on lambda)
SHOCKS = {
    "barca_attack":  0.97,   # Olmo, Bardghji out
    "barca_defence": 1.08,   # Koundé, Christensen, Eric García out -> Madrid scores more
    "madrid_attack": 0.88,   # Vinícius suspended, Rodrygo out
    "madrid_defence":1.03,   # Mendy, Asencio out -> Barça scores more
}
RHO = -0.08                     # Dixon-Coles low-score dependence
LAMBDA_CV = 0.15                # parameter uncertainty (coef. of variation on lambda)
N_SIMS = 200_000
MAX_GOALS = 10

# Benchmarks (home / draw / away)
MARKETS = {
    "Opta supercomputer": (0.544, 0.195, 0.261),
    "Polymarket (18 Oct)": (0.66, 0.16, 0.20),
}

# ----------------------------------------------------------------- 1-2. RATINGS
def rates(team):
    d, p = TEAMS[team], PRIOR[team]
    att_obs = (XG_WEIGHT * d["xg"] + (1 - XG_WEIGHT) * d["gf"]) / GAMES
    dfn_obs = d["ga"] / GAMES
    w = GAMES / (GAMES + PRIOR_WEIGHT_GAMES)
    att = w * att_obs + (1 - w) * p["att"]
    dfn = w * dfn_obs + (1 - w) * p["dfn"]
    return att / LEAGUE_AVG, dfn / LEAGUE_AVG     # strengths relative to league

b_att, b_def = rates("Barcelona")
m_att, m_def = rates("Real Madrid")

lam_b = LEAGUE_HOME_GPG * b_att * m_def * ELITE_DAMPENER
lam_m = LEAGUE_AWAY_GPG * m_att * b_def * ELITE_DAMPENER

lam_b = (1 - H2H_WEIGHT) * lam_b + H2H_WEIGHT * H2H_BARCA_GPG
lam_m = (1 - H2H_WEIGHT) * lam_m + H2H_WEIGHT * H2H_MADRID_GPG

# ----------------------------------------------------------------- 3. SHOCKS
lam_b *= SHOCKS["barca_attack"] * SHOCKS["madrid_defence"]
lam_m *= SHOCKS["madrid_attack"] * SHOCKS["barca_defence"]

# ----------------------------------------------------------------- 4. DIXON-COLES MATRIX
def dc_matrix(lh, la, rho=RHO, n=MAX_GOALS):
    g = np.arange(n + 1)
    M = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    M[0, 0] *= 1 - lh * la * rho
    M[0, 1] *= 1 + lh * rho
    M[1, 0] *= 1 + la * rho
    M[1, 1] *= 1 - rho
    return M / M.sum()

def outcome(M):
    return np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()

M = dc_matrix(lam_b, lam_m)
analytic = outcome(M)

# ----------------------------------------------------------------- 5. MONTE CARLO
# Gamma draws keep lambda positive with mean = point estimate, CV = LAMBDA_CV
k = 1 / LAMBDA_CV**2
lb = rng.gamma(k, lam_b / k, N_SIMS)
lm = rng.gamma(k, lam_m / k, N_SIMS)
gb = rng.poisson(lb)
gm = rng.poisson(lm)
# Dixon-Coles adjustment via importance weights on low scores
w = np.ones(N_SIMS)
w[(gb == 0) & (gm == 0)] *= 1 - lb[(gb == 0) & (gm == 0)] * lm[(gb == 0) & (gm == 0)] * RHO
w[(gb == 0) & (gm == 1)] *= 1 + lb[(gb == 0) & (gm == 1)] * RHO
w[(gb == 1) & (gm == 0)] *= 1 + lm[(gb == 1) & (gm == 0)] * RHO
w[(gb == 1) & (gm == 1)] *= 1 - RHO
w /= w.sum()

mc = (w[gb > gm].sum(), w[gb == gm].sum(), w[gb < gm].sum())
over25 = w[(gb + gm) > 2.5].sum()
btts = w[(gb > 0) & (gm > 0)].sum()

# Bootstrap-style CI on the win prob from parameter uncertainty
def p_home_given(lh, la):
    return outcome(dc_matrix(lh, la))[0]
idx = rng.choice(N_SIMS, 2000, replace=False)
pw = np.array([p_home_given(lb[i], lm[i]) for i in idx])
ci = np.percentile(pw, [5, 95])

# Top scorelines
flat = [((i, j), M[i, j]) for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1)]
top = sorted(flat, key=lambda x: -x[1])[:6]

# ----------------------------------------------------------------- OUTPUT
if __name__ == "__main__":
    print(f"Expected goals   Barcelona {lam_b:.2f}  |  Real Madrid {lam_m:.2f}")
    print(f"Analytic (DC)    H {analytic[0]:.1%}  D {analytic[1]:.1%}  A {analytic[2]:.1%}")
    print(f"Monte Carlo      H {mc[0]:.1%}  D {mc[1]:.1%}  A {mc[2]:.1%}   (n={N_SIMS:,})")
    print(f"Barça win 90% band from parameter risk: {ci[0]:.1%} – {ci[1]:.1%}")
    print(f"Over 2.5 goals {over25:.1%}   BTTS {btts:.1%}")
    print("Most likely scores:", ", ".join(f"{a}-{b} ({p:.1%})" for (a, b), p in top))
    print("\nVs benchmarks (H/D/A):")
    for k_, v in MARKETS.items():
        edge = mc[0] - v[0]
        print(f"  {k_:22s} {v[0]:.1%} / {v[1]:.1%} / {v[2]:.1%}   model-minus-market on Barça: {edge:+.1%}")
