"""Build the self-contained HTML report (charts embedded). Run after clasico_model_v2.py and charts.py."""
import base64
import json
from html import escape
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "outputs"
R = json.loads((OUT / "results.json").read_text())
scen = pd.read_csv(OUT / "in_play_scenarios.csv")
squads = pd.read_csv(DATA / "squads_availability.csv")
experts = pd.read_csv(DATA / "expert_predictions.csv")
mk = pd.read_csv(DATA / "market_and_model_probabilities.csv")

def img(name):
    b = base64.b64encode((OUT / "charts" / name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b}" alt="{name}">'

def pct(x):
    return f"{x * 100:.0f}%"

def table(df, cls=""):
    return df.to_html(index=False, border=0, classes=f"tbl {cls}", escape=True, na_rep="")

fin, mod, mar, pun = R["final"], R["model"]["p"], R["market"]["p"], R["pundits"]["p"]
lam = R["lambdas"]
fg = R["model"]["first_goal"]
cond = R["model"]["conditional_on_first_goal"]
pc = R["prior_test"]["choice"]
bench = R["bench"]
resp = R["responses"]
st = R["game_states"]
tm = R["timing"]
bins = ["1-15", "16-30", "31-45+", "46-60", "61-75", "76-90+"]

timing_df = pd.DataFrame({
    "Minutes": bins,
    "Barca scored": tm["Barcelona"]["for_counts"], "Barca conceded": tm["Barcelona"]["against_counts"],
    "Madrid scored": tm["Real Madrid"]["for_counts"], "Madrid conceded": tm["Real Madrid"]["against_counts"],
    "Clasico xG: Barca": [round(x, 2) for x in R["model"]["expected_goals_by_bin"]["Barcelona"]],
    "Clasico xG: Madrid": [round(x, 2) for x in R["model"]["expected_goals_by_bin"]["Real_Madrid"]],
})
state_rows = []
for team in ["Barcelona", "Real Madrid"]:
    for s in ["leading", "level", "trailing"]:
        d = st[team][s]
        state_rows.append({"Team": team, "State": s, "Minutes": d["minutes"], "Scored": d["gf"], "Conceded": d["ga"],
                           "Scored /90": d["gf_per90"], "Conceded /90": d["ga_per90"],
                           "Model multiplier (shrunk)": R["state_multipliers"][team][s]})
state_df = pd.DataFrame(state_rows)

key_min = scen[scen.minute.isin([0, 30, 45, 60, 70, 75, 80, 85])].copy()
key_min = key_min.pivot_table(index="minute", columns="score", values=["p_bar", "p_draw", "p_rma"])
inplay = pd.DataFrame({"Minute": key_min.index})
for sc, lab in [("0-0", "0-0"), ("1-0", "Barca +1"), ("0-1", "Madrid +1")]:
    inplay[f"{lab}: B / D / M"] = [f"{pct(key_min.loc[m, ('p_bar', sc)])} / {pct(key_min.loc[m, ('p_draw', sc)])} / {pct(key_min.loc[m, ('p_rma', sc)])}"
                                  for m in key_min.index]

loo = pd.DataFrame(R["prior_test"]["loo_table"])
loo_piv = loo.pivot_table(index=["team", "half_life"], columns="k_prior_games", values="loo_loglik").reset_index()
loo_piv.columns = [str(c) if not isinstance(c, str) else c for c in loo_piv.columns]
loo_piv = loo_piv.rename(columns={"team": "Team", "half_life": "Recency half-life"})

sens = pd.DataFrame([{"Scenario": s["case"], "Barca xG": s["lb"], "Madrid xG": s["lm"],
                      "Barca": pct(s["p"][0]), "Draw": pct(s["p"][1]), "Madrid": pct(s["p"][2])} for s in R["sensitivity"]])

def xi(team):
    d = squads[(squads.team == team) & squads.projected_role.fillna("").str.startswith("Starter")]
    return ", ".join(f"{p} ({pos})" for p, pos in zip(d.player, d.position))

out_df = squads[squads.status.str.upper().str.startswith(("OUT", "SUSP"))][["team", "player", "status", "detail"]]
out_df.columns = ["Team", "Player", "Status", "Detail"]
squad_df = squads.fillna("")[["team", "player", "position", "status", "projected_role", "detail"]]
squad_df.columns = ["Team", "Player", "Pos", "Status", "Projected role", "Notes"]

exp_df = experts.fillna("")
exp_df["Score"] = [f"{int(h)}-{int(a)}" if h != "" else "" for h, a in zip(exp_df.score_home, exp_df.score_away)]
exp_df = exp_df[["source", "person", "type", "pick", "Score", "date"]]
exp_df.columns = ["Outlet", "Who", "Type", "Pick", "Score", "Date"]

mk_df = mk.fillna("")[["source", "type", "p_bar", "p_draw", "p_rma", "raw", "as_of"]]
mk_df.columns = ["Source", "Type", "Barca", "Draw", "Madrid", "Raw price", "As of"]

top = ", ".join(f"{s} ({p * 100:.1f}%)" for s, p in R["model"]["top_scores"][:6])

CSS = """
:root{--bg:#f7f6f3;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e6e5e1;--bar:#2a78d6;--rma:#eb6834;--draw:#8a8984}
@media (prefers-color-scheme: dark){:root{--bg:#121211;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#33332f;--bar:#3987e5;--rma:#d95926}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,Segoe UI,Inter,Roboto,sans-serif}
main{max-width:1060px;margin:0 auto;padding:28px 16px 80px}
h1{font-size:30px;margin:0 0 4px}h2{font-size:21px;margin:42px 0 10px;padding-top:10px;border-top:1px solid var(--line)}
h3{font-size:16px;margin:22px 0 6px}.sub{color:var(--ink2);margin:0 0 18px}
.hero{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.tile .n{font-size:34px;font-weight:700}.tile .l{color:var(--ink2);font-size:13px}
.b .n{color:var(--bar)}.m .n{color:var(--rma)}.d .n{color:var(--draw)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:12px 0;overflow-x:auto}
img{max-width:100%;height:auto;border-radius:8px;background:#fcfcfb}
.tbl{border-collapse:collapse;width:100%;font-size:13px}.tbl th,.tbl td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
.tbl th{color:var(--ink2);font-weight:600}ul{padding-left:20px}li{margin:4px 0}
.note{color:var(--ink2);font-size:13px}.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
.kpis .tile .n{font-size:22px}
@media (max-width:640px){.hero{grid-template-columns:1fr}h1{font-size:24px}}
"""

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>El Clasico Forecast</title><style>{CSS}</style></head><body><main>
<h1>El Clásico forecast: Barcelona vs Real Madrid</h1>
<p class="sub">LaLiga matchday 10 · Spotify Camp Nou · Sun 25 Oct 2026, 21:00 Spain / 4:00 p.m. Ottawa · built 24 Oct 2026 from this season's LaLiga + Champions League data</p>

<div class="hero">
 <div class="tile b"><div class="n">{pct(fin[0])}</div><div class="l">Barcelona win</div></div>
 <div class="tile d"><div class="n">{pct(fin[1])}</div><div class="l">Draw</div></div>
 <div class="tile m"><div class="n">{pct(fin[2])}</div><div class="l">Real Madrid win</div></div>
</div>
<div class="kpis">
 <div class="tile"><div class="n">{lam['Barcelona']:.1f} – {lam['Real_Madrid']:.1f}</div><div class="l">Expected goals (model)</div></div>
 <div class="tile"><div class="n">{R['model']['top_scores'][0][0]}</div><div class="l">Single most likely score ({R['model']['top_scores'][0][1] * 100:.1f}%)</div></div>
 <div class="tile"><div class="n">{pct(R['model']['over_2_5'])}</div><div class="l">Over 2.5 goals</div></div>
 <div class="tile"><div class="n">{pct(R['model']['btts'])}</div><div class="l">Both teams score</div></div>
 <div class="tile"><div class="n">{pct(fg['p_bar_first'])}</div><div class="l">Barça score first (median first goal ~{fg['median_first_goal_minute']:.0f}')</div></div>
</div>

<h2>Bottom line</h2>
<ul>
<li><b>Barça are clear favourites but not runaway ones.</b> The final blend is the form-and-minute model ({pct(mod[0])} / {pct(mod[1])} / {pct(mod[2])}), the betting markets ({pct(mar[0])} / {pct(mar[1])} / {pct(mar[2])}) and the pundits ({pct(pun[0])} / {pct(pun[1])} / {pct(pun[2])}). The model sits slightly more pro-Barça than the market.</li>
<li><b>Expect goals at both ends.</b> Barça have scored in every match, and Madrid haven't kept a clean sheet in their last 12 Clásicos. The most likely scores are {top}.</li>
<li><b>The first goal is huge.</b> If Barça score first, they win {pct(cond['if_bar_score_first'][0])} of simulations. If Madrid score first, it's close to a coin flip: Barça {pct(cond['if_rma_score_first'][0])}, draw {pct(cond['if_rma_score_first'][1])}, Madrid {pct(cond['if_rma_score_first'][2])}.</li>
<li><b>Madrid's path to a result:</b> stay in the game until about 75 minutes, then use their late bench. {bench['Real Madrid']['share_76_plus'] * 100:.0f}% of their goals come after the 75th minute, and {bench['Real Madrid']['bench_goals']} of {bench['Real Madrid']['goals']} came from substitutes (Espí 3, Endrick, Güler, Diomande). They need Barça's 46–60 burst to be survivable and Courtois at his best.</li>
<li><b>Barça's path:</b> score early and keep scoring. Barça lead in {st['Barcelona']['leading']['minutes'] / (st['Barcelona']['leading']['minutes'] + st['Barcelona']['level']['minutes'] + st['Barcelona']['trailing']['minutes']) * 100:.0f}% of their minutes this season, and when they concede they reply fast: median {resp['Barcelona']['median_minutes_to_reply']:.0f} minutes, and {resp['Barcelona']['answered_within_15']} of {resp['Barcelona']['goals_conceded']} goals conceded were answered within 15 minutes.</li>
</ul>

<h2>Who wins: every input and the blend</h2>
<div class="card">{img('06_ensemble.png')}</div>
<p class="note">Weights: market {R['weights']['market']:.0%}, form model {R['weights']['model']:.0%}, pundits {R['weights']['pundits']:.0%}. Markets get the most weight because they are the most efficient single input. The model is the independent view, the same idea as comparing a model PD with a CDS-implied PD. Market-implied expected goals are Barça {R['market_implied_lambdas'][0]:.2f}, Madrid {R['market_implied_lambdas'][1]:.2f}. The model's figures are Barça {lam['Barcelona']:.2f}, Madrid {lam['Real_Madrid']:.2f}, so the disagreement is almost entirely about how many Barça score.</p>

<h2>How each team is playing right now</h2>
<h3>Barcelona (Flick): 9/9 in LaLiga, 50 goals in 12 games, first loss at PSG on Tuesday</h3>
<ul>
<li>Barça have 39 LaLiga goals, the second-best start ever, from 30.3 xG, while conceding 9 from 8.0 xGA. They average about 70% possession and play a very high line. Raphinha leads LaLiga with 12 league goals (14 in all competitions). Yamal has 8 league goals and 6 assists, and 14 goal involvements since 31 August, the most in Europe's top five leagues.</li>
<li><b>The weakness is the back four.</b> Barça have used 10 different back lines in 12 games. Koundé, Eric García and Christensen are out, which leaves Cubarsí and Gerard Martín as the only senior centre-backs, with Cancelo and Espart at full-back. Goals have come from headers off crosses and set pieces (3), long shots (2, both at PSG) and individual errors (2). PSG scored at 81' and 86' to win 4-2, and Flick said: "We have to defend better than we did in Paris."</li>
<li>Rodri was kept on the bench at PSG with this game in mind. Olmo picked up an adductor injury in Paris (out about a month), so Fermín (5 goals) comes in. Yamal has reported some groin discomfort but is expected to play.</li>
<li>Barça have won 23 straight LaLiga home games and 6 of Flick's 7 Clásicos.</li>
</ul>
<h3>Real Madrid (Mourinho): 4 straight wins, 12 goals in the last 3</h3>
<ul>
<li>Madrid are 7W-2L in LaLiga (both defeats away, at Betis and Atlético) and 3/3 in the Champions League, beating Inter 2-1, Roma 4-3 from 3-1 down and Leipzig 4-1. Mbappé has 9 league goals (12 in all competitions) and has scored 12 in 10 games against Barça.</li>
<li><b>Shape:</b> 4-2-3-1. Valverde has been used at right-back, with Tchouaméni and Bernardo (or Camavinga) as the pivot, Güler drifting in from the right, Bellingham at 10, Diomande wide left and Mbappé up front. The plan is to counter: unpredictable runs against Barça's high line, balls over the top for Mbappé, and Diomande attacking Espart's side.</li>
<li><b>Set pieces are a real weapon.</b> At Roma, Madrid scored three times from corner sequences between 83' and 89'. Rüdiger has two late set-piece headers, and Güler takes most corners. Flick called Madrid "one of the best teams on set pieces."</li>
<li><b>Defensive concerns:</b> no away clean sheet this season, and 5 of the 14 goals they've conceded came early in the second half (46–60'). They've also been reckless: Huijsen was sent off at Atlético and Vinícius against Villarreal.</li>
</ul>
<h3>The Vinícius situation</h3>
<ul>
<li>After the derby loss at Atlético (20 Sep), Mourinho benched Vinícius against Villarreal on 10 Oct. Mourinho said the aim was to "protect" him and "bring him on when the game called for a player like Vini." Vinícius came on at half-time and had a goal disallowed by VAR for pulling Nathan Saliba's hair, then got a straight red (about 73').</li>
<li>He got a 4-match LaLiga ban, 2 games for the incident plus 2 for protests, upheld on appeal. This is game 2 of the ban, so he's out until around 21 Nov (Celta). His season so far: 1 goal and 3 assists in 11 games, with fewer successful dribbles than usual and reports of a confidence dip after the World Cup.</li>
<li><b>Effect on the model:</b> small (Madrid attack ×0.97). Diomande has 3 goals in his last 4 games, and Madrid beat Sevilla 4-1 with him starting on the left in Vinícius's place. Pundits like Troy Deeney see a selection headache coming when Vinícius returns.</li>
</ul>

<h2>Rosters, availability and predicted line-ups</h2>
<div class="card"><p><b>Barcelona (4-2-3-1):</b> {escape(xi('Barcelona'))}</p>
<p><b>Real Madrid (4-2-3-1):</b> {escape(xi('Real Madrid'))}</p>
<p class="note">These are the consensus predicted XIs from Sports Mole, Rotowire, SI, Opta and the Yahoo/Standard previews. Valverde at right-back versus Dumfries, and Endrick versus Güler on the right, are Madrid's main open selection questions.</p></div>
<h3>Out / suspended</h3>
<div class="card">{table(out_df)}</div>
<details><summary><b>Full squad table</b></summary><div class="card">{table(squad_df)}</div></details>

<h2>Recent form: every match this season</h2>
<div class="card">{img('01_form_by_match.png')}</div>

<h2>Minute-by-minute analysis</h2>
<h3>When the goals come</h3>
<div class="card">{img('02_goal_timing.png')}</div>
<div class="card">{table(timing_df)}</div>
<ul>
<li><b>Madrid are a late team.</b> They've scored 12 of 33 goals after the 75th minute ({bench['Real Madrid']['share_76_plus'] * 100:.0f}%, compared with about 20% in a typical league). Their 61st-to-end scoring rate ({bench['Real Madrid']['per90_min61_end']}/90) is higher than their 1st-60th rate ({bench['Real Madrid']['per90_min1_60']}/90). Mourinho's first tactical substitution comes at a median of {bench['Real Madrid']['median_first_tactical_sub']:.0f}', and a second wave around 80–88' brings on Espí and Endrick for the final push.</li>
<li><b>Madrid concede early in the second half.</b> 5 of their 14 goals conceded came in minutes 46–60, which is when the model gives Barça their highest scoring rate in the Clásico.</li>
<li><b>Barça score throughout the game</b> (7 / 8 / 6 / 9 / 8 / 12 by 15-minute block). Their goals conceded cluster at 16–30 and 76–90+ (late leaks at Levante and PSG). Their bench adds goals rather than relying on them: {bench['Barcelona']['bench_goals']} goals from substitutes (Adeyemi ×2, Eric García, Olmo).</li>
</ul>
<h3>Game state: how each team plays when leading, level or trailing</h3>
<div class="card">{table(state_df)}</div>
<p class="note">The multipliers are each team's scoring rate in that state relative to their overall rate. They are shrunk toward typical league effects (trailing ×1.12, leading ×0.92), because some of these samples are only 60–100 minutes long.</p>
<h3>Momentum: how long does a swing last?</h3>
<ul>
<li><b>Goals don't snowball, they get answered.</b> Across all 24 matches, when two goals came within 10 minutes of each other, the second went to the same team {R['momentum']['observed_same_share'] * 100:.0f}% of the time, below the {R['momentum']['expected_same_share'] * 100:.0f}% you'd expect from scoring rates alone. So the model applies no "hot streak" bonus (momentum multiplier = {R['momentum']['momentum_multiplier']}).</li>
<li><b>Barça bounce back fast:</b> {resp['Barcelona']['answered']} of {resp['Barcelona']['goals_conceded']} goals conceded were answered, {resp['Barcelona']['answered_within_15']} of them within 15 minutes (median {resp['Barcelona']['median_minutes_to_reply']:.0f}').</li>
<li><b>Madrid bounce back slowly but late:</b> {resp['Real Madrid']['answered']} of {resp['Real Madrid']['goals_conceded']} goals conceded were answered, but only {resp['Real Madrid']['answered_within_15']} within 15 minutes (median {resp['Real Madrid']['median_minutes_to_reply']:.1f}'). Their replies arrive when the substitutes come on.</li>
</ul>
<h3>The simulated Clásico, minute by minute</h3>
<div class="card">{img('03_minute_intensity.png')}</div>
<p class="note">Each line combines the attacking team's scoring profile with the defending team's conceding profile, then runs 100,000 simulated matches with game-state effects, red cards (P(any red) = {pct(R['model']['p_red_any'])}) and stoppage time.</p>
<h3>How long can each side wait? In-play probabilities</h3>
<div class="card">{img('04_comeback_windows.png')}</div>
<div class="card">{table(inplay)}</div>
<ul>
<li><b>If it's 0-0 at half-time,</b> Barça's win probability drops to {pct(key_min.loc[45, ('p_bar', '0-0')])}. At 0-0 after 75', a draw becomes the favourite ({pct(key_min.loc[75, ('p_draw', '0-0')])}), which is roughly when the game state starts to suit Madrid.</li>
<li><b>Madrid 1-0 up:</b> at 60' they win {pct(key_min.loc[60, ('p_rma', '0-1')])}, they pass 50% around 70' ({pct(key_min.loc[70, ('p_rma', '0-1')])}), and reach {pct(key_min.loc[75, ('p_rma', '0-1')])} at 75'. That's the length of time Mourinho's side would need to hold out.</li>
<li><b>Barça 1-0 up</b> is close to decisive at any minute (about {pct(key_min.loc[60, ('p_bar', '1-0')])} to win), because Madrid's realistic comeback window is only the last 15 minutes.</li>
</ul>

<h2>Did last season improve the model?</h2>
<p>Each option was tested with leave-one-out cross-validation: each of the 24 matches this season is predicted using only the other 23, and the Poisson log-likelihood of the actual goals is scored (higher is better). A prior is kept only if it improves that score.</p>
<ul>
<li><b>Flick's 2025-26 Barcelona (95 scored, 36 conceded in 38): rejected.</b> Every amount of prior weight made predictions worse, because this season's Barça are better than last season's. Barça are rated on this season only.</li>
<li><b>Mourinho's 2025-26 Benfica (74 scored, 25 conceded in 34, unbeaten league season, compact 5-3-2 block against strong sides): accepted</b> (+{pc['Real Madrid']['gain_vs_no_prior']:.2f} log-likelihood). Mourinho's defensive identity is predictive of how this Madrid side performs. The test's best weight was {pc['Real Madrid']['best_k_in_grid']} games, but I used the most conservative weight within 0.05 of the best ({pc['Real Madrid']['k_prior']} games), because it comes from a weaker league.</li>
<li>The test also found that Madrid's most recent games are more predictive (recency half-life {pc['Real Madrid']['half_life']} matches), while Barça's form is steady (half-life {pc['Barcelona']['half_life']}).</li>
</ul>
<details><summary><b>Leave-one-out table</b> (columns = prior weight in games)</summary><div class="card">{table(loo_piv)}</div></details>

<h2>Pundits, TV analysts and markets</h2>
<h3>Pundit picks: {R['pundits']['tally'].get('BAR', 0) + R['pundits']['tally'].get('BAR (lean)', 0)} Barça, {R['pundits']['tally'].get('DRAW', 0)} draw, 0 Madrid. Average predicted total: {R['pundits']['avg_total_goals_predicted']} goals</h3>
<div class="card">{table(exp_df)}</div>
<ul>
<li><b>CBS Sports Golazo:</b> Troy Deeney said "It is impossible for them to beat Barcelona... a team full of individuals against the best team," and expects chaos and possibly a red card. Jamie Carragher (via AS) said Madrid "could well lose to Barcelona and then lose the La Liga title."</li>
<li><b>ESPN</b> (Marsden 3-1, Kirkland 4-2) both back Barça. ESPN notes Mourinho's first Madrid spell against Barça was 2W 4D 5L, although Opta has him unbeaten in his last five against them.</li>
<li><b>Luis Figo</b> predicts 2-2 (Raphinha and Yamal; Bellingham and Mbappé) because "Madrid will surely have chances on the counter." Sports Mole and Hooligan Soccer also go 2-2.</li>
<li><b>Fox Sports</b> (Echegaray) refused to predict. Jürgen Klopp said he'll be supporting Barça as "the better team" (a preference, not a forecast).</li>
<li><b>TSN:</b> it covered the build-up (Gordon v Getafe, Diomande v Villarreal) but I couldn't find a published TSN pick.</li>
<li><b>TikTok / Instagram creators (Ilyes, Faysal, Skillz, etc.):</b> their posts aren't indexed by web search, and reading them needs the in-app browser, which was declined this session. They aren't included, so the pundit layer is mainstream media only.</li>
</ul>
<h3>Markets and other models</h3>
<div class="card">{table(mk_df)}</div>
<p class="note">Sportsbook prices are de-vigged by normalising implied probabilities. Market drift: Polymarket had Barça at 66% on 18 Oct and 59.5% on 24 Oct ($517K volume), after Barça lost at PSG and Madrid beat Leipzig. Kalshi prices Over 4.5 goals at about 47%.</p>

<h2>Model method</h2>
<ol>
<li><b>Form ratings:</b> opponent-adjusted attack and defence from all 24 LaLiga and UCL matches. LaLiga opponents are rated from the current table (with the match itself removed and the ratings shrunk 50%), and UCL opponents are rated by judgment. Home ×1.15, away ×0.87, with time-decay weighting.</li>
<li><b>xG correction:</b> goals are pulled toward xG (60% weight). Barça are scoring above their xG (39 vs 30.3), and Madrid have conceded below theirs (9 vs 10.6 xGA).</li>
<li><b>Last-season priors:</b> only where cross-validation shows an improvement (see above).</li>
<li><b>Availability shocks:</b> Barça defence ×1.06 (three centre-backs out), Barça attack ×0.98 (Olmo, Yamal doubt), Madrid attack ×0.97 (Vinícius), Madrid defence ×1.01 (Mendy, Asencio).</li>
<li><b>Minute engine:</b> team-specific 15-minute scoring and conceding profiles (shrunk toward a league-typical profile), game-state multipliers, momentum test, red-card risk and stoppage time, simulated 100,000 times.</li>
<li><b>Ensemble</b> with the de-vigged market consensus and pundit sentiment.</li>
</ol>
<p class="note">Backtest: the rating model's leave-one-out Brier score is {R['rating_backtest']['brier_model']}, against {R['rating_backtest']['brier_naive']} for a naive base-rate forecast. This mainly shows it separates strong from weak opponents. Calibration in elite-vs-elite games is less certain, which is why the market gets the largest weight.</p>
<h3>Sensitivity: what moves the answer</h3>
<div class="card">{table(sens)}</div>
<div class="card">{img('05_score_grid.png')}</div>

<h2>Caveats</h2>
<ul>
<li>The sample is 12 matches per team. Some goal minutes (Racing, Sevilla, Getafe) and some substitution times are approximate and flagged in the data files.</li>
<li>UCL opponent strengths, home advantage and the availability shocks are judgment calls. The sensitivity table shows how much each one matters.</li>
<li>Team news can change on the day. Re-run with the confirmed XIs: edit <code>SHOCKS</code> in <code>model/clasico_model_v2.py</code>, then run the model, <code>charts.py</code> and <code>build_report.py</code>.</li>
<li>This is a probabilistic forecast, not betting advice.</li>
</ul>
<p class="note">Main sources: UEFA, LaLiga, Sky Sports table, OddAlerts xG, VAVEL / Flashscore / Fox / Football Web Pages match reports, Al Jazeera, ESPN, The Analyst (Opta), Goal, Sports Mole, Rotowire, SI, Yahoo/Standard, Covers, VSiN, CryptoSlate/Polymarket, Managing Madrid, FC Barcelona. The full URL list is in <code>data/*.csv</code> and <code>README.md</code>.</p>
</main></body></html>"""

(ROOT / "El_Clasico_Forecast_Report.html").write_text(html, encoding="utf-8")
print("report written:", ROOT / "El_Clasico_Forecast_Report.html", f"{len(html) / 1e6:.2f} MB")
