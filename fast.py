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
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --primary: #10b981;
      --primary-dark: #059669;
      --secondary: #3b82f6;
      --background: #f8fafc;
      --card-bg: #ffffff;
      --text: #0f172a;
      --text-light: #64748b;
      --border: #e2e8f0;
      --shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.06);
      --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
      --radius: 12px;
      --win-bg: #d1fae5;
      --draw-bg: #fef3c7;
      --loss-bg: #fee2e2;
    }
    
    * { box-sizing: border-box; margin: 0; padding: 0; }
    
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
      padding: 24px;
      color: var(--text);
    }
    
    .container {
      max-width: 1400px;
      margin: 0 auto;
    }
    
    .header {
      background: var(--card-bg);
      border-radius: var(--radius);
      padding: 32px;
      margin-bottom: 24px;
      box-shadow: var(--shadow-lg);
      text-align: center;
    }
    
    .header h1 {
      font-size: 2.5rem;
      font-weight: 700;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      margin-bottom: 8px;
    }
    
    .header p {
      color: var(--text-light);
      font-size: 1rem;
    }
    
    .prediction-panel {
      background: var(--card-bg);
      border-radius: var(--radius);
      padding: 32px;
      margin-bottom: 24px;
      box-shadow: var(--shadow-lg);
    }
    
    .prediction-title {
      font-size: 1.5rem;
      font-weight: 600;
      margin-bottom: 24px;
      color: var(--text);
    }
    
    .form-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 20px;
      margin-bottom: 24px;
    }
    
    .form-field {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    
    .form-field label {
      font-weight: 500;
      font-size: 0.875rem;
      color: var(--text);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    
    .input-wrapper {
      position: relative;
    }
    
    .filter-input {
      width: 100%;
      padding: 12px 16px;
      font-size: 0.95rem;
      border: 2px solid var(--border);
      border-radius: 8px;
      background: var(--background);
      transition: all 0.2s;
      font-family: inherit;
    }
    
    .filter-input:focus {
      outline: none;
      border-color: var(--secondary);
      background: white;
    }
    
    select {
      width: 100%;
      padding: 12px 16px;
      font-size: 0.95rem;
      border: 2px solid var(--border);
      border-radius: 8px;
      background: white;
      cursor: pointer;
      transition: all 0.2s;
      font-family: inherit;
      appearance: none;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%2364748b' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
      background-repeat: no-repeat;
      background-position: right 12px center;
      background-size: 20px;
      padding-right: 44px;
    }
    
    select:focus {
      outline: none;
      border-color: var(--secondary);
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    .checkbox-wrapper {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 16px;
      background: var(--background);
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s;
    }
    
    .checkbox-wrapper:hover {
      background: #e0e7ff;
    }
    
    input[type="checkbox"] {
      width: 20px;
      height: 20px;
      cursor: pointer;
      accent-color: var(--primary);
    }
    
    .checkbox-label {
      font-weight: 500;
      color: var(--text);
      cursor: pointer;
    }
    
    .predict-btn {
      background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
      color: white;
      padding: 14px 32px;
      font-size: 1rem;
      font-weight: 600;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.3s;
      box-shadow: var(--shadow);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    
    .predict-btn:hover {
      transform: translateY(-2px);
      box-shadow: var(--shadow-lg);
    }
    
    .predict-btn:active {
      transform: translateY(0);
    }
    
    .insights-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
      gap: 24px;
      margin-bottom: 24px;
    }
    
    .card {
      background: var(--card-bg);
      border-radius: var(--radius);
      padding: 24px;
      box-shadow: var(--shadow-lg);
      transition: all 0.3s;
    }
    
    .card:hover {
      transform: translateY(-4px);
      box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1), 0 10px 10px -5px rgba(0,0,0,0.04);
    }
    
    .card-title {
      font-size: 1.25rem;
      font-weight: 600;
      margin-bottom: 16px;
      color: var(--text);
      padding-bottom: 12px;
      border-bottom: 2px solid var(--border);
    }
    
    .team-badge {
      display: inline-block;
      padding: 4px 12px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      border-radius: 20px;
      font-size: 0.875rem;
      font-weight: 600;
    }
    
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
    }
    
    th {
      background: var(--background);
      padding: 12px;
      text-align: left;
      font-weight: 600;
      font-size: 0.875rem;
      color: var(--text);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      border-bottom: 2px solid var(--border);
    }
    
    td {
      padding: 12px;
      border-bottom: 1px solid var(--border);
      font-size: 0.9rem;
      color: var(--text);
    }
    
    tr:hover {
      background: var(--background);
    }
    
    .result-badge {
      display: inline-block;
      width: 28px;
      height: 28px;
      line-height: 28px;
      text-align: center;
      border-radius: 50%;
      font-weight: 700;
      font-size: 0.875rem;
    }
    
    .result-W { background: var(--win-bg); color: #065f46; }
    .result-D { background: var(--draw-bg); color: #92400e; }
    .result-L { background: var(--loss-bg); color: #991b1b; }
    
    .prediction-result {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 32px;
      border-radius: var(--radius);
      text-align: center;
      margin-top: 16px;
    }
    
    .prediction-result h3 {
      font-size: 2rem;
      margin-bottom: 12px;
    }
    
    .scoreline {
      font-size: 3rem;
      font-weight: 700;
      margin: 16px 0;
      text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .scorers {
      background: rgba(255,255,255,0.15);
      padding: 16px;
      border-radius: 8px;
      margin-top: 16px;
      backdrop-filter: blur(10px);
    }
    
    .scorers-title {
      font-weight: 600;
      margin-bottom: 8px;
      font-size: 1.1rem;
    }
    
    .empty-state {
      text-align: center;
      padding: 48px;
      color: var(--text-light);
      font-style: italic;
    }
    
    .h2h-summary {
      display: flex;
      justify-content: space-around;
      padding: 20px;
      background: var(--background);
      border-radius: 8px;
      margin-top: 16px;
    }
    
    .h2h-stat {
      text-align: center;
    }
    
    .h2h-stat-value {
      font-size: 2rem;
      font-weight: 700;
      color: var(--primary);
    }
    
    .h2h-stat-label {
      font-size: 0.875rem;
      color: var(--text-light);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    
    @media (max-width: 768px) {
      body { padding: 12px; }
      .header { padding: 24px 16px; }
      .header h1 { font-size: 1.75rem; }
      .prediction-panel { padding: 20px; }
      .form-grid { grid-template-columns: 1fr; }
      .insights-grid { grid-template-columns: 1fr; }
      .scoreline { font-size: 2rem; }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>⚽ Football Match Predictor</h1>
      <p>Advanced ML-powered football match predictions with comprehensive team insights</p>
    </div>
    
    <div class="prediction-panel">
      <h2 class="prediction-title">Match Setup</h2>
      
      <div class="form-grid">
        <div class="form-field">
          <label for="homeFilter">🏠 Home Team</label>
          <div class="input-wrapper">
            <input id="homeFilter" class="filter-input" placeholder="Search home team..." autocomplete="off">
            <select id="home"></select>
          </div>
        </div>
        
        <div class="form-field">
          <label for="awayFilter">✈️ Away Team</label>
          <div class="input-wrapper">
            <input id="awayFilter" class="filter-input" placeholder="Search away team..." autocomplete="off">
            <select id="away"></select>
          </div>
        </div>
        
        <div class="form-field">
          <label for="tourFilter">🏆 Tournament</label>
          <div class="input-wrapper">
            <input id="tourFilter" class="filter-input" placeholder="Search tournament..." autocomplete="off">
            <select id="tournament"></select>
          </div>
        </div>
        
        <div class="form-field">
          <label style="visibility: hidden;">Action</label>
          <div class="checkbox-wrapper" onclick="document.getElementById('neutral').click()">
            <input type="checkbox" id="neutral">
            <span class="checkbox-label">Neutral Venue</span>
          </div>
        </div>
      </div>
      
      <button class="predict-btn" id="predict">🔮 Predict Match Outcome</button>
    </div>
    
    <div class="insights-grid">
      <div class="card">
        <h3 class="card-title">
          Last 5 Matches — <span class="team-badge" id="homeBadge">Home</span>
        </h3>
        <div id="last5_home" class="empty-state">Select a home team to view recent matches</div>
      </div>
      
      <div class="card">
        <h3 class="card-title">
          Last 5 Matches — <span class="team-badge" id="awayBadge">Away</span>
        </h3>
        <div id="last5_away" class="empty-state">Select an away team to view recent matches</div>
      </div>
    </div>
    
    <div class="card">
      <h3 class="card-title">Head-to-Head Record</h3>
      <div id="h2h" class="empty-state">Select both teams to view head-to-head record</div>
      <div id="h2h_summary"></div>
    </div>
    
    <div class="card">
      <h3 class="card-title">Match Prediction</h3>
      <div id="prediction" class="empty-state">Click "Predict Match Outcome" to see the prediction</div>
    </div>
  </div>

<script>
function enableFilter(inputId, selectId, allItems) {
  const inp = document.getElementById(inputId);
  const sel = document.getElementById(selectId);
  
  const render = (query) => {
    const q = (query || "").toLowerCase();
    const items = allItems.filter(t => t.toLowerCase().includes(q));
    sel.innerHTML = '<option value="">Select a team...</option>' + items.map(t => `<option value="${t}">${t}</option>`).join('');
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
    container.innerHTML = '<div class="empty-state">No data available</div>';
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
        if (col === 'Result' || col === 'Result(A)') {
          value = `<span class="result-badge result-${value}">${value}</span>`;
        }
        html += `<td>${value}</td>`;
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
            <div class="h2h-stat-value">${summary.A_wins || 0}</div>
            <div class="h2h-stat-label">${home} Wins</div>
          </div>
          <div class="h2h-stat">
            <div class="h2h-stat-value">${summary.Draws || 0}</div>
            <div class="h2h-stat-label">Draws</div>
          </div>
          <div class="h2h-stat">
            <div class="h2h-stat-value">${summary.B_wins || 0}</div>
            <div class="h2h-stat-label">${away} Wins</div>
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
    alert('⚠️ Please select home team, away team, and tournament');
    return;
  }
  
  out.innerHTML = '<div class="empty-state">🔄 Generating prediction...</div>';
  
  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ home_team: home, away_team: away, tournament, neutral })
    });
    
    const data = await resp.json();
    
    if (!resp.ok) {
      out.innerHTML = `<div class="empty-state">❌ Prediction failed: ${JSON.stringify(data)}</div>`;
      return;
    }
    
    const homeScorers = (data.home_scorers || []).join(', ') || 'None';
    const awayScorers = (data.away_scorers || []).join(', ') || 'None';
    const probability = ((data.proba_home_win || 0) * 100).toFixed(1);
    
    out.innerHTML = `
      <div class="prediction-result">
        <h3>🎯 ${data.winner || 'Unknown'}</h3>
        <div class="scoreline">${data.scoreline || '0-0'}</div>
        <div style="font-size: 1.1rem; margin-bottom: 20px;">
          Win Probability: ${probability}%
        </div>
        <div class="scorers">
          <div class="scorers-title">🏠 ${home} Scorers:</div>
          <div>${homeScorers}</div>
        </div>
        <div class="scorers">
          <div class="scorers-title">✈️ ${away} Scorers:</div>
          <div>${awayScorers}</div>
        </div>
      </div>
    `;
  } catch(e) {
    out.innerHTML = `<div class="empty-state">❌ Network error: ${e.message}</div>`;
  }
}

document.getElementById('home').addEventListener('change', refreshInsights);
document.getElementById('away').addEventListener('change', refreshInsights);
document.getElementById('predict').addEventListener('click', doPredict);
document.getElementById('neutral').addEventListener('click', (e) => e.stopPropagation());

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
