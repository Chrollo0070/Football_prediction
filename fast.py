# fast.py — Enhanced FastAPI app with beautiful, modern UI

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


# ========= File paths (adjust if needed) =========
FIFA_CSV = "fifa2020-2024.csv"
RESULTS_CSV = "result1.csv"
SCORERS_CSV = "goalscorers.csv"


# ========= Load data =========
try:
    df_fifa = pd.read_csv(FIFA_CSV)
    df_results = pd.read_csv(RESULTS_CSV)
    goals_df = pd.read_csv(SCORERS_CSV)
except FileNotFoundError as e:
    raise RuntimeError(f"Required CSV missing: {e.filename}. Place CSVs next to fast.py or update paths.") from e


# ========= Preprocess & merge =========
df_fifa['date'] = pd.to_datetime(df_fifa['date'], errors='coerce')
df_results['date'] = pd.to_datetime(df_results['date'], errors='coerce')
goals_df['date'] = pd.to_datetime(goals_df['date'], errors='coerce')


df_fifa = df_fifa.sort_values('date').reset_index(drop=True)
df_results = df_results.sort_values('date').reset_index(drop=True)


if 'total.points' in df_fifa.columns:
    fifa_points_col = 'total.points'
elif 'total_points' in df_fifa.columns:
    fifa_points_col = 'total_points'
else:
    raise RuntimeError("FIFA CSV must have 'total.points' or 'total_points' column.")


goals_df['own_goal'] = goals_df['own_goal'].astype(str).str.upper().isin(['TRUE', '1', 'YES'])
goals_df['penalty']  = goals_df['penalty'].astype(str).str.upper().isin(['TRUE', '1', 'YES'])
if 'minute' in goals_df.columns:
    goals_df['minute'] = pd.to_numeric(goals_df['minute'], errors='coerce')


fifa_home = df_fifa.rename(columns={'rank': 'home_rank', fifa_points_col: 'home_points'})[['date', 'team', 'home_rank', 'home_points']]
fifa_away = df_fifa.rename(columns={'rank': 'away_rank', fifa_points_col: 'away_points'})[['date', 'team', 'away_rank', 'away_points']]


df_merged = pd.merge_asof(
    df_results, fifa_home,
    left_on='date', right_on='date',
    left_by='home_team', right_by='team',
    direction='backward'
).rename(columns={'team': 'home_team_fifa_name'})


df_merged = pd.merge_asof(
    df_merged, fifa_away,
    left_on='date', right_on='date',
    left_by='away_team', right_by='team',
    direction='backward'
).rename(columns={'team': 'away_team_fifa_name'})


required_fifa_cols = ['home_rank', 'home_points', 'away_rank', 'away_points']
df_merged = df_merged.dropna(subset=required_fifa_cols)


df_merged['rank_difference'] = df_merged['home_rank'] - df_merged['away_rank']
df_merged['points_difference'] = df_merged['home_points'] - df_merged['away_points']
df_merged['home_win'] = (df_merged['home_score'] > df_merged['away_score']).astype(int)
df_merged['score_difference'] = df_merged['home_score'] - df_merged['away_score']


if 'tournament' in df_merged.columns:
    df_merged = pd.get_dummies(df_merged, columns=['tournament'], prefix='tournament')
else:
    df_merged['tournament_missing'] = 1


if 'neutral' not in df_merged.columns:
    df_merged['neutral'] = 0
df_merged['neutral'] = df_merged['neutral'].astype(int)


engineered_features = ['rank_difference', 'points_difference', 'neutral']
tournament_features = [c for c in df_merged.columns if c.startswith('tournament_')]
features = [f for f in engineered_features + tournament_features if f in df_merged.columns]


X = df_merged[features].copy()
y_clf = df_merged['home_win'].copy()
y_reg = df_merged['score_difference'].copy()


X_train, X_test, y_train_clf, y_test_clf, y_train_reg, y_test_reg = train_test_split(
    X, y_clf, y_reg, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
X_train_scaled = X_train_scaled.fillna(X_train_scaled.mean())
X_test_scaled  = X_test_scaled.fillna(X_test_scaled.mean())

# Load pretrained models and scaler if available; otherwise train and save artifacts.
from joblib import load
import os
import json

MODEL_DIR = os.environ.get('MODEL_DIR', 'models')

def _load_models_or_train():
  global model_clf, model_reg, scaler, feature_columns
  clf_path = os.path.join(MODEL_DIR, 'model_clf.joblib')
  reg_path = os.path.join(MODEL_DIR, 'model_reg.joblib')
  scaler_path = os.path.join(MODEL_DIR, 'scaler.joblib')
  features_path = os.path.join(MODEL_DIR, 'features.json')

  if os.path.exists(clf_path) and os.path.exists(reg_path) and os.path.exists(scaler_path) and os.path.exists(features_path):
    model_clf = load(clf_path)
    model_reg = load(reg_path)
    scaler = load(scaler_path)
    with open(features_path, 'r', encoding='utf-8') as f:
      feature_columns = json.load(f)
  else:
    # Train and save using helper script
    try:
      from train_models import train_and_save
      train_and_save(FIFA_CSV, RESULTS_CSV, SCORERS_CSV, MODEL_DIR)
      model_clf = load(clf_path)
      model_reg = load(reg_path)
      scaler = load(scaler_path)
      with open(features_path, 'r', encoding='utf-8') as f:
        feature_columns = json.load(f)
    except Exception as e:
      raise RuntimeError(f"Failed to train or load models: {e}") from e


_load_models_or_train()


# ========= Helpers =========
def _last_n_matches(df: pd.DataFrame, team: str, n: int = 5) -> pd.DataFrame:
    m = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
    if m.empty:
        return m
    m = m.sort_values('date', ascending=False).head(n).copy()
    def opponent(row): return row['away_team'] if row['home_team'] == team else row['home_team']
    def venue(row):
        if row.get('neutral', 0) == 1: return 'Neutral'
        return 'Home' if row['home_team'] == team else 'Away'
    def result(row):
        hs, as_ = row['home_score'], row['away_score']
        if row['home_team'] == team:
            return 'W' if hs > as_ else ('D' if hs == as_ else 'L')
        return 'W' if as_ > hs else ('D' if as_ == hs else 'L')
    def scoreline(row):
        if row['home_team'] == team: return f"{row['home_score']}-{row['away_score']}"
        return f"{row['away_score']}-{row['home_score']}"
    m['Opponent'], m['Venue'], m['Result'], m['Scoreline'] = (
        m.apply(opponent, axis=1), m.apply(venue, axis=1), m.apply(result, axis=1), m.apply(scoreline, axis=1)
    )
    cols = ['date', 'Opponent', 'Venue', 'tournament', 'Result', 'Scoreline', 'home_team', 'away_team', 'home_score', 'away_score']
    cols = [c for c in cols if c in m.columns]
    return m[cols].rename(columns={'date': 'Date', 'tournament': 'Tournament'})


def _current_fifa_snapshot(team: str):
    tdf = df_fifa[df_fifa['team'] == team]
    if tdf.empty: return None
    latest = tdf.sort_values('date', ascending=False).iloc[0]
    pts_col = 'total.points' if 'total.points' in latest.index else ('total_points' if 'total_points' in latest.index else None)
    points = float(latest[pts_col]) if pts_col else None
    return {'rank': int(latest['rank']), 'points': points, 'date': latest['date']}


def _head_to_head(df: pd.DataFrame, team_a: str, team_b: str, n: int = 5) -> Tuple[pd.DataFrame, Dict[str, int]]:
    h2h = df[((df['home_team'] == team_a) & (df['away_team'] == team_b)) | ((df['home_team'] == team_b) & (df['away_team'] == team_a))].copy()
    if h2h.empty:
        return h2h, {'A_wins': 0, 'B_wins': 0, 'Draws': 0}
    h2h = h2h.sort_values('date', ascending=False)
    def res_for(team, row):
        hs, as_ = row['home_score'], row['away_score']
        if row['home_team'] == team:
            return 'W' if hs > as_ else ('D' if hs == as_ else 'L')
        return 'W' if as_ > hs else ('D' if as_ == hs else 'L')
    A_wins = sum(res_for(team_a, r) == 'W' for _, r in h2h.iterrows())
    B_wins = sum(res_for(team_b, r) == 'W' for _, r in h2h.iterrows())
    Draws  = sum(res_for(team_a, r) == 'D' for _, r in h2h.iterrows())
    show = h2h.head(n).copy()
    def opp(row, team): return row['away_team'] if row['home_team'] == team else row['home_team']
    def venue_for(team, row):
        if row.get('neutral', 0) == 1: return 'Neutral'
        return 'Home' if row['home_team'] == team else 'Away'
    def scoreline_from_a(row):
        if row['home_team'] == team_a: return f"{row['home_score']}-{row['away_score']}"
        return f"{row['away_score']}-{row['home_score']}"
    show['Opponent'] = show.apply(lambda r: opp(r, team_a), axis=1)
    show['Venue(A)'] = show.apply(lambda r: venue_for(team_a, r), axis=1)
    show['Result(A)'] = show.apply(lambda r: res_for(team_a, r), axis=1)
    show['Scoreline(A)'] = show.apply(scoreline_from_a, axis=1)
    keep = ['date', 'Opponent', 'Venue(A)', 'tournament', 'Result(A)', 'Scoreline(A)']
    keep = [c for c in keep if c in show.columns]
    return show[keep].rename(columns={'date': 'Date', 'tournament': 'Tournament'}), {'A_wins': int(A_wins), 'B_wins': int(B_wins), 'Draws': int(Draws)}


def team_scorer_weights(team, as_of_date=None, half_life_days=365, include_penalties=True, penalty_weight=0.7):
    tdf = goals_df[(goals_df['team'] == team) & (~goals_df['own_goal'])].copy()
    if tdf.empty:
        return pd.DataFrame(columns=['scorer', 'weight', 'goals', 'recent_goals'])
    if as_of_date is None:
        as_of_date = df_results['date'].max() if 'date' in df_results.columns else pd.Timestamp.today()
    age_days = (as_of_date - tdf['date']).dt.days.clip(lower=0).fillna(0)
    decay = np.exp(-np.log(2) * (age_days / float(half_life_days)))
    if include_penalties:
        pen_adj = np.where(tdf['penalty'], penalty_weight, 1.0)
        weights = decay * pen_adj
    else:
        tdf = tdf[~tdf['penalty']]
        if tdf.empty:
            return pd.DataFrame(columns=['scorer', 'weight', 'goals', 'recent_goals'])
        age_days = (as_of_date - tdf['date']).dt.days.clip(lower=0).fillna(0)
        decay = np.exp(-np.log(2) * (age_days / float(half_life_days)))
        weights = decay
    tdf = tdf.assign(weight=weights)
    agg = tdf.groupby('scorer').agg(
        weight=('weight', 'sum'),
        goals=('scorer', 'count'),
        recent_goals=('date', lambda s: (s >= (as_of_date - pd.Timedelta(days=180))).sum())
    ).reset_index()
    if agg['weight'].sum() > 0:
        agg['weight'] = agg['weight'] / agg['weight'].sum()
    return agg.sort_values(['weight', 'goals', 'recent_goals'], ascending=[False, False, False]).reset_index(drop=True)


def pick_scorers_for_team(team, goals, as_of_date=None, include_penalties=True, penalty_weight=0.7, top_k_pool=10):
    goals = int(goals)
    if goals <= 0:
        return []
    weights_df = team_scorer_weights(team, as_of_date=as_of_date, include_penalties=include_penalties, penalty_weight=penalty_weight)
    if weights_df.empty:
        return []
    pool = weights_df.head(max(top_k_pool, goals))
    names = pool['scorer'].tolist()
    picks, idx = [], 0
    for _ in range(goals):
        picks.append(names[idx % len(names)])
        idx += 1
    return picks


def predict_match_outcome(home_team, away_team, tournament, neutral, match_date=None,
                          include_penalties_for_prediction=True, penalty_weight=0.7):
    home_fifa = _current_fifa_snapshot(home_team)
    away_fifa = _current_fifa_snapshot(away_team)
    if home_fifa is None:
        raise RuntimeError(f"FIFA ranking data not found for {home_team}")
    if away_fifa is None:
        raise RuntimeError(f"FIFA ranking data not found for {away_team}")


    rank_diff = home_fifa['rank'] - away_fifa['rank']
    points_diff = (home_fifa['points'] or 0.0) - (away_fifa['points'] or 0.0)
    input_data = {'rank_difference': [rank_diff], 'points_difference': [points_diff], 'neutral': [int(bool(neutral))]}
    input_df = pd.DataFrame(input_data)


    for col in feature_columns:
        if col.startswith('tournament_') and col not in input_df.columns:
            input_df[col] = 0
    specific_tournament_col = f'tournament_{tournament}'
    if specific_tournament_col in feature_columns:
        input_df[specific_tournament_col] = 1


    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns, index=input_df.index)


    pred_clf = int(model_clf.predict(input_scaled)[0])
    proba_clf = model_clf.predict_proba(input_scaled)[0].tolist() if hasattr(model_clf, "predict_proba") else [None, None]
    pred_reg = float(model_reg.predict(input_scaled)[0])


    predicted_score_difference = int(round(pred_reg))
    winner, home_score, away_score = "", 0, 0


    if pred_clf == 1:
        winner = f"{home_team} Wins"
        score_diff = max(1, predicted_score_difference)
        if score_diff == 1:
            home_score, away_score = 2, 1
        elif score_diff == 2:
            home_score, away_score = 3, 1
        elif score_diff > 2:
            home_score, away_score = score_diff + 1, 1
        else:
            home_score, away_score = 1, 0
    else:
        if predicted_score_difference < 0:
            winner = f"{away_team} Wins"
            score_diff = max(1, abs(predicted_score_difference))
            if score_diff == 1:
                away_score, home_score = 2, 1
            elif score_diff == 2:
                away_score, home_score = 3, 1
            elif score_diff > 2:
                away_score, home_score = score_diff + 1, 1
            else:
                away_score, home_score = 1, 0
        else:
            winner = "Draw"
            home_score, away_score = 1, 1


    as_of_date = match_date if match_date is not None else (df_results['date'].max() if 'date' in df_results.columns else pd.Timestamp.today())
    home_scorers = pick_scorers_for_team(home_team, home_score, as_of_date=as_of_date,
                                         include_penalties=include_penalties_for_prediction, penalty_weight=penalty_weight)
    away_scorers = pick_scorers_for_team(away_team, away_score, as_of_date=as_of_date,
                                         include_penalties=include_penalties_for_prediction, penalty_weight=penalty_weight)


    return winner, f"{home_score}-{away_score}", {home_team: home_scorers, away_team: away_scorers}, (proba_clf[1] if proba_clf[1] is not None else 0.0)


# ========= API =========
app = FastAPI(title="Football Outcome API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    tournament: str
    neutral: bool = False
    match_date: Optional[str] = None
    include_penalties_for_prediction: bool = True
    penalty_weight: float = 0.7


class PredictResponse(BaseModel):
    winner: str
    scoreline: str
    home_scorers: List[str]
    away_scorers: List[str]
    proba_home_win: float


class InsightsResponse(BaseModel):
    last5_home: List[dict]
    last5_away: List[dict]
    h2h_last: List[dict]
    h2h_summary: Dict[str, int]


@app.get("/", response_class=HTMLResponse)
def root_ui():
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Football Match Predictor</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
  <script>
    (function () {
      var saved = localStorage.getItem('theme');
      var theme = saved === 'light' || saved === 'dark'
        ? saved
        : (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
      document.documentElement.dataset.theme = theme;
    })();
  </script>
  <style>
    :root {
      --ease: cubic-bezier(0.22, 1, 0.36, 1);
      --mono: 'IBM Plex Mono', ui-monospace, 'Cascadia Mono', monospace;
    }
    :root, :root[data-theme="dark"] {
      color-scheme: dark;
      --bg: #0b0d14;
      --card: #11141c;
      --raised: #1a1e28;
      --hairline: rgba(238, 240, 245, 0.08);
      --hairline-strong: rgba(238, 240, 245, 0.16);
      --text: #eef0f5;
      --text-2: #a3a9b8;
      --muted: #7a8296;
      --accent: #4cc38a;
      --btn-fg: #0b0d14;
      --btn-hover: #ffffff;
      --win: #4cc38a;
      --draw: #d4a72c;
      --loss: #e5484d;
      --win-bg: rgba(76, 195, 138, 0.15);
      --draw-bg: rgba(212, 167, 44, 0.15);
      --loss-bg: rgba(229, 72, 77, 0.15);
      --focus-ring: rgba(76, 195, 138, 0.18);
      --focus-ring-strong: rgba(76, 195, 138, 0.3);
      --row-hover: rgba(238, 240, 245, 0.03);
      --error-bg: rgba(229, 72, 77, 0.08);
      --error-border: rgba(229, 72, 77, 0.3);
    }
    :root[data-theme="light"] {
      color-scheme: light;
      --bg: #f5f7fa;
      --card: #ffffff;
      --raised: #eceff4;
      --hairline: rgba(17, 24, 39, 0.08);
      --hairline-strong: rgba(17, 24, 39, 0.16);
      --text: #111827;
      --text-2: #4b5565;
      --muted: #6b7487;
      --accent: #15803d;
      --btn-fg: #ffffff;
      --btn-hover: #1f2937;
      --win: #15803d;
      --draw: #a16207;
      --loss: #dc2626;
      --win-bg: rgba(21, 128, 61, 0.12);
      --draw-bg: rgba(161, 98, 7, 0.12);
      --loss-bg: rgba(220, 38, 38, 0.1);
      --focus-ring: rgba(21, 128, 61, 0.16);
      --focus-ring-strong: rgba(21, 128, 61, 0.3);
      --row-hover: rgba(17, 24, 39, 0.03);
      --error-bg: rgba(220, 38, 38, 0.06);
      --error-border: rgba(220, 38, 38, 0.25);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'IBM Plex Sans', -apple-system, 'Segoe UI', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      font-size: 14px;
      line-height: 20px;
    }

    .mono { font-family: var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap; }

    .shell { max-width: 1160px; margin: 0 auto; padding: 0 24px 48px; }

    /* Top bar */
    .topbar {
      display: flex; align-items: center; justify-content: space-between;
      gap: 16px; height: 64px; margin-bottom: 32px;
    }
    .brand { display: flex; align-items: center; gap: 10px; }
    .brand-mark { color: var(--accent); display: flex; }
    .brand-name { font-size: 16px; font-weight: 500; }
    .topbar-meta { font-size: 12px; color: var(--muted); }

    /* Panels */
    .stack { display: grid; gap: 16px; }
    .panel {
      background: var(--card);
      border: 1px solid var(--hairline);
      border-radius: 16px;
      padding: 24px;
    }
    .panel-title { font-size: 15px; font-weight: 500; margin-bottom: 4px; }
    .panel-sub { font-size: 13px; color: var(--text-2); margin-bottom: 20px; }
    .card-title {
      display: flex; align-items: center; gap: 8px;
      font-size: 14px; font-weight: 500; margin-bottom: 16px; min-height: 24px;
    }
    .team-badge {
      font-size: 12px; font-weight: 400; color: var(--text-2);
      background: var(--raised); border: 1px solid var(--hairline);
      border-radius: 6px; padding: 2px 8px;
      max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }

    /* Form */
    .form-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }
    .field { display: flex; flex-direction: column; gap: 8px; }
    .field > label { font-size: 12px; color: var(--text-2); }
    .filter-input {
      height: 36px; padding: 0 12px; font-size: 13px;
      color: var(--text); background: var(--raised);
      border: 1px solid var(--hairline); border-radius: 8px;
      font-family: inherit;
      transition: border-color 140ms var(--ease);
    }
    .filter-input::placeholder { color: var(--muted); }
    .filter-input:focus {
      outline: none; border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--focus-ring);
    }
    select {
      height: 40px; padding: 0 36px 0 12px; font-size: 13px;
      color: var(--text); background-color: var(--raised);
      border: 1px solid var(--hairline); border-radius: 8px;
      font-family: inherit; cursor: pointer; appearance: none;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%237a8296' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'/%3E%3C/svg%3E");
      background-repeat: no-repeat;
      background-position: right 12px center;
      transition: border-color 140ms var(--ease);
    }
    select:focus {
      outline: none; border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--focus-ring);
    }
    .check {
      display: flex; align-items: center; gap: 10px;
      height: 40px; padding: 0 12px;
      background: var(--raised); border: 1px solid var(--hairline); border-radius: 8px;
      cursor: pointer; user-select: none;
      transition: border-color 140ms var(--ease);
    }
    .check:hover { border-color: var(--hairline-strong); }
    input[type="checkbox"] { width: 16px; height: 16px; accent-color: var(--accent); cursor: pointer; }
    .check span { font-size: 13px; }
    .predict-btn {
      height: 44px; padding: 0 20px;
      background: var(--text); color: var(--btn-fg);
      border: none; border-radius: 8px; cursor: pointer;
      font-size: 14px; font-weight: 500; font-family: inherit;
      transition: background-color 140ms var(--ease), transform 140ms var(--ease);
    }
    .predict-btn:hover { background: var(--btn-hover); }
    .predict-btn:active { transform: scale(0.98); }
    .predict-btn:focus-visible { outline: none; box-shadow: 0 0 0 3px var(--focus-ring-strong); }

    /* Insights */
    .insights-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }

    table { width: 100%; border-collapse: collapse; }
    th {
      padding: 8px 12px; text-align: left;
      font-size: 12px; font-weight: 400; color: var(--muted);
      border-bottom: 1px solid var(--hairline);
    }
    td {
      padding: 8px 12px; height: 40px;
      font-size: 13px; color: var(--text-2);
      border-bottom: 1px solid var(--hairline);
    }
    tbody tr:last-child td { border-bottom: none; }
    tbody tr { transition: background-color 140ms var(--ease); }
    tbody tr:hover { background: var(--row-hover); }
    td.score { color: var(--text); }

    .result-badge {
      display: inline-flex; align-items: center; justify-content: center;
      width: 20px; height: 20px; border-radius: 4px;
      font-family: var(--mono); font-size: 11px; font-weight: 500;
    }
    .result-W { background: var(--win-bg); color: var(--win); }
    .result-D { background: var(--draw-bg); color: var(--draw); }
    .result-L { background: var(--loss-bg); color: var(--loss); }

    .empty-state { padding: 32px 16px; text-align: center; font-size: 13px; color: var(--muted); }

    /* H2H summary */
    .h2h-summary {
      display: flex; gap: 48px; flex-wrap: wrap;
      margin-top: 16px; padding-top: 16px;
      border-top: 1px solid var(--hairline);
    }
    .h2h-stat-label { font-size: 12px; color: var(--muted); margin-bottom: 4px; }
    .h2h-stat-value {
      font-family: var(--mono); font-variant-numeric: tabular-nums;
      font-size: 28px; line-height: 1; font-weight: 500;
    }

    /* Scoreboard moment */
    .scoreboard {
      background: var(--raised);
      border: 1px solid var(--hairline);
      border-radius: 12px;
      padding: 32px 24px;
      text-align: center;
      animation: rise 300ms var(--ease);
    }
    @keyframes rise {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: none; }
    }
    .scoreboard-label { font-size: 12px; color: var(--muted); margin-bottom: 8px; }
    .scoreboard-winner { font-size: 20px; font-weight: 500; color: var(--accent); margin-bottom: 16px; }
    .scoreline {
      font-family: var(--mono); font-variant-numeric: tabular-nums;
      font-size: 48px; line-height: 1; font-weight: 500;
      margin-bottom: 16px;
    }
    .scoreboard-prob { font-size: 13px; color: var(--text-2); }
    .scoreboard-prob b { font-family: var(--mono); font-weight: 500; color: var(--text); }
    .scorers-grid {
      display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
      margin-top: 24px; padding-top: 16px;
      border-top: 1px solid var(--hairline); text-align: left;
    }
    .scorer-label { font-size: 12px; color: var(--muted); margin-bottom: 4px; }
    .scorer-names { font-size: 13px; color: var(--text-2); }

    /* Status */
    .loading {
      display: inline-flex; align-items: center; gap: 8px;
      padding: 32px 16px; font-size: 13px; color: var(--muted);
    }
    .spinner {
      width: 14px; height: 14px; border-radius: 50%;
      border: 1.5px solid var(--hairline-strong); border-top-color: var(--accent);
      animation: spin 800ms linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .notice-error {
      padding: 16px; border: 1px solid var(--error-border);
      background: var(--error-bg); border-radius: 8px;
      font-size: 13px; color: var(--loss);
    }

    /* Theme toggle */
    .topbar-right { display: flex; align-items: center; gap: 12px; }
    .theme-toggle {
      display: flex; align-items: center; justify-content: center;
      width: 32px; height: 32px; padding: 0;
      background: transparent; border: 1px solid var(--hairline); border-radius: 8px;
      color: var(--muted); cursor: pointer;
      transition: border-color 140ms var(--ease), color 140ms var(--ease);
    }
    .theme-toggle:hover { border-color: var(--hairline-strong); color: var(--text-2); }
    .theme-toggle:focus-visible { outline: none; box-shadow: 0 0 0 3px var(--focus-ring-strong); }
    .theme-toggle svg { display: block; }
    :root[data-theme="light"] .icon-sun { display: none; }
    :root[data-theme="light"] .icon-moon { display: block; }
    :root:not([data-theme="light"]) .icon-sun { display: block; }
    :root:not([data-theme="light"]) .icon-moon { display: none; }

    @media (max-width: 900px) {
      .insights-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 768px) {
      .shell { padding: 0 16px 32px; }
      .topbar { margin-bottom: 24px; }
      .topbar-meta { display: none; }
      .panel { padding: 16px; }
      .scoreboard { padding: 24px 16px; }
      .scoreline { font-size: 36px; }
      .scorers-grid { grid-template-columns: 1fr; }
      .h2h-summary { gap: 32px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header class="topbar">
      <div class="brand">
        <span class="brand-mark">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <rect x="2" y="4" width="20" height="16" rx="2"/>
            <line x1="12" y1="4" x2="12" y2="20"/>
            <circle cx="12" cy="12" r="3"/>
          </svg>
        </span>
        <span class="brand-name">Football Match Predictor</span>
      </div>
      <div class="topbar-right">
        <span class="topbar-meta">Men's international fixtures</span>
        <button class="theme-toggle" id="themeToggle" aria-label="Toggle theme">
          <svg class="icon-sun" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="4"/>
            <path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/>
          </svg>
          <svg class="icon-moon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>
          </svg>
        </button>
      </div>
    </header>

    <div class="stack">
      <section class="panel">
        <h2 class="panel-title">Match setup</h2>
        <p class="panel-sub">Pick two teams and a tournament — you get a scoreline, win probability and likely scorers</p>

        <div class="form-grid">
          <div class="field">
            <label for="homeFilter">Home team</label>
            <input id="homeFilter" class="filter-input" placeholder="Search" autocomplete="off">
            <select id="home"></select>
          </div>

          <div class="field">
            <label for="awayFilter">Away team</label>
            <input id="awayFilter" class="filter-input" placeholder="Search" autocomplete="off">
            <select id="away"></select>
          </div>

          <div class="field">
            <label for="tourFilter">Tournament</label>
            <input id="tourFilter" class="filter-input" placeholder="Search" autocomplete="off">
            <select id="tournament"></select>
          </div>

          <div class="field">
            <label style="visibility: hidden;" for="neutral">Options</label>
            <div class="check" onclick="document.getElementById('neutral').click()">
              <input type="checkbox" id="neutral">
              <span>Neutral venue</span>
            </div>
          </div>
        </div>

        <button class="predict-btn" id="predict">Predict match outcome</button>
      </section>

      <div class="insights-grid">
        <section class="panel">
          <h3 class="card-title">Last 5 matches <span class="team-badge" id="homeBadge">Home</span></h3>
          <div id="last5_home" class="empty-state">Select a home team to see recent results</div>
        </section>

        <section class="panel">
          <h3 class="card-title">Last 5 matches <span class="team-badge" id="awayBadge">Away</span></h3>
          <div id="last5_away" class="empty-state">Select an away team to see recent results</div>
        </section>
      </div>

      <section class="panel">
        <h3 class="card-title">Head-to-head</h3>
        <div id="h2h" class="empty-state">Select both teams to see the head-to-head record</div>
        <div id="h2h_summary"></div>
      </section>

      <section class="panel">
        <h3 class="card-title">Prediction</h3>
        <div id="prediction" class="empty-state">Select teams and a tournament, then run a prediction</div>
      </section>
    </div>
  </div>

<script>
function enableFilter(inputId, selectId, allItems) {
  const inp = document.getElementById(inputId);
  const sel = document.getElementById(selectId);

  const render = (query) => {
    const q = (query || "").toLowerCase();
    const items = allItems.filter(t => t.toLowerCase().includes(q));
    sel.innerHTML = '<option value="">Select</option>' + items.map(t => `<option value="${t}">${t}</option>`).join('');
  };

  inp.addEventListener('input', () => render(inp.value));
  inp.addEventListener('focus', () => inp.select());
  render('');
}

async function loadOptions() {
  try {
    const [teams, tournaments] = await Promise.all([
      fetch('/teams').then(r => r.json()),
      fetch('/tournaments').then(r => r.json())
    ]);

    enableFilter('homeFilter', 'home', teams);
    enableFilter('awayFilter', 'away', teams);
    enableFilter('tourFilter', 'tournament', tournaments);
  } catch(e) {
    console.error('Failed to load options:', e);
  }
}

function renderTable(id, rows) {
  const container = document.getElementById(id);

  if (!rows || rows.length === 0) {
    container.innerHTML = '<div class="empty-state">No matches found</div>';
    return;
  }

  const cols = Object.keys(rows[0]);
  let html = '<table><thead><tr>';

  cols.forEach(col => {
    if (!['home_team', 'away_team', 'home_score', 'away_score'].includes(col)) {
      html += `<th>${col}</th>`;
    }
  });
  html += '</tr></thead><tbody>';

  rows.forEach(row => {
    html += '<tr>';
    cols.forEach(col => {
      if (!['home_team', 'away_team', 'home_score', 'away_score'].includes(col)) {
        let value = row[col] ?? '';
        const isMono = col === 'Date' || col === 'Scoreline' || col === 'Scoreline(A)';
        if (col === 'Result' || col === 'Result(A)') {
          value = `<span class="result-badge result-${value}">${value}</span>`;
        }
        const cls = col === 'Scoreline' || col === 'Scoreline(A)' ? 'mono score' : (isMono ? 'mono' : '');
        html += `<td${cls ? ` class="${cls}"` : ''}>${value}</td>`;
      }
    });
    html += '</tr>';
  });

  html += '</tbody></table>';
  container.innerHTML = html;
}

async function refreshInsights() {
  const home = document.getElementById('home').value;
  const away = document.getElementById('away').value;

  document.getElementById('homeBadge').textContent = home || 'Home';
  document.getElementById('awayBadge').textContent = away || 'Away';

  if (!home || !away) return;

  try {
    const qs = new URLSearchParams({ home_team: home, away_team: away }).toString();
    const res = await fetch('/insights?' + qs);
    const data = await res.json();

    if (!res.ok) {
      console.error('Insights error:', data);
      return;
    }

    renderTable('last5_home', data.last5_home);
    renderTable('last5_away', data.last5_away);
    renderTable('h2h', data.h2h_last);

    if (data.h2h_summary) {
      const summary = data.h2h_summary;
      document.getElementById('h2h_summary').innerHTML = `
        <div class="h2h-summary">
          <div class="h2h-stat">
            <div class="h2h-stat-label">${home} wins</div>
            <div class="h2h-stat-value">${summary.A_wins || 0}</div>
          </div>
          <div class="h2h-stat">
            <div class="h2h-stat-label">Draws</div>
            <div class="h2h-stat-value">${summary.Draws || 0}</div>
          </div>
          <div class="h2h-stat">
            <div class="h2h-stat-label">${away} wins</div>
            <div class="h2h-stat-value">${summary.B_wins || 0}</div>
          </div>
        </div>
      `;
    }
  } catch(e) {
    console.error('Failed to fetch insights:', e);
  }
}

async function doPredict() {
  const home = document.getElementById('home').value;
  const away = document.getElementById('away').value;
  const tournament = document.getElementById('tournament').value;
  const neutral = document.getElementById('neutral').checked;
  const out = document.getElementById('prediction');

  if (!home || !away || !tournament) {
    out.innerHTML = '<div class="notice-error">Select home team, away team and tournament</div>';
    return;
  }

  out.innerHTML = '<div class="loading"><span class="spinner"></span>Generating prediction</div>';

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ home_team: home, away_team: away, tournament, neutral })
    });

    const data = await resp.json();

    if (!resp.ok) {
      out.innerHTML = `<div class="notice-error">Prediction failed — ${JSON.stringify(data)}</div>`;
      return;
    }

    const homeScorers = (data.home_scorers || []).join(', ') || 'None';
    const awayScorers = (data.away_scorers || []).join(', ') || 'None';
    const probability = ((data.proba_home_win || 0) * 100).toFixed(1);

    out.innerHTML = `
      <div class="scoreboard">
        <div class="scoreboard-label">Predicted winner</div>
        <div class="scoreboard-winner">${data.winner || 'Unknown'}</div>
        <div class="scoreline">${data.scoreline || '0-0'}</div>
        <div class="scoreboard-prob">${home} win probability <b>${probability}%</b></div>
        <div class="scorers-grid">
          <div>
            <div class="scorer-label">${home} scorers</div>
            <div class="scorer-names">${homeScorers}</div>
          </div>
          <div>
            <div class="scorer-label">${away} scorers</div>
            <div class="scorer-names">${awayScorers}</div>
          </div>
        </div>
      </div>
    `;
  } catch(e) {
    out.innerHTML = `<div class="notice-error">Prediction failed — ${e.message}</div>`;
  }
}

document.getElementById('home').addEventListener('change', refreshInsights);
document.getElementById('away').addEventListener('change', refreshInsights);
document.getElementById('predict').addEventListener('click', doPredict);
document.getElementById('neutral').addEventListener('click', (e) => e.stopPropagation());

document.getElementById('themeToggle').addEventListener('click', () => {
  const next = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
  document.documentElement.dataset.theme = next;
  localStorage.setItem('theme', next);
});

loadOptions();
</script>
</body>
</html>
    """


@app.get("/teams", response_model=List[str])
def list_teams():
    t = set(df_fifa['team'].dropna().unique())
    t |= set(df_results['home_team'].dropna().unique())
    t |= set(df_results['away_team'].dropna().unique())
    return sorted(t)


@app.get("/tournaments", response_model=List[str])
def list_tournaments():
    if 'tournament' in df_results.columns:
        return sorted(pd.Series(df_results['tournament']).dropna().unique().tolist())
    return []


@app.get("/insights", response_model=InsightsResponse)
def insights(home_team: str = Query(...), away_team: str = Query(...)):
    try:
        last5_h = _last_n_matches(df_results, home_team, n=5)
        last5_a = _last_n_matches(df_results, away_team, n=5)
        h2h_table, h2h_sum = _head_to_head(df_results, home_team, away_team, n=5)
        return InsightsResponse(
            last5_home=(last5_h.to_dict(orient="records") if not last5_h.empty else []),
            last5_away=(last5_a.to_dict(orient="records") if not last5_a.empty else []),
            h2h_last=(h2h_table.to_dict(orient="records") if not h2h_table.empty else []),
            h2h_summary=h2h_sum
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insights failed: {e}")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        winner, scoreline, scorers, proba_home = predict_match_outcome(
            req.home_team, req.away_team, req.tournament, req.neutral,
            match_date=pd.to_datetime(req.match_date) if req.match_date else None,
            include_penalties_for_prediction=req.include_penalties_for_prediction,
            penalty_weight=req.penalty_weight
        )
        return PredictResponse(
            winner=winner or "Unknown",
            scoreline=scoreline or "0-0",
            home_scorers=scorers.get(req.home_team, []),
            away_scorers=scorers.get(req.away_team, []),
            proba_home_win=float(proba_home or 0.0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
