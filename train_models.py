"""Train models from CSVs and save artifacts to a models/ directory.

This mirrors the preprocessing in `fast.py` and saves:
- models/model_clf.joblib
- models/model_reg.joblib
- models/scaler.joblib
- models/features.json
"""
import os
import json
from joblib import dump
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


def train_and_save(fifa_csv='fifa2020-2024.csv', results_csv='result1.csv', scorers_csv='goalscorers.csv', model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)

    df_fifa = pd.read_csv(fifa_csv)
    df_results = pd.read_csv(results_csv)
    goals_df = pd.read_csv(scorers_csv)

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

    model_clf = LogisticRegression(random_state=42, max_iter=1000)
    model_clf.fit(X_train_scaled, y_train_clf)

    model_reg = RandomForestRegressor(random_state=42)
    model_reg.fit(X_train_scaled, y_train_reg)

    # Save artifacts
    dump(model_clf, os.path.join(model_dir, 'model_clf.joblib'))
    dump(model_reg, os.path.join(model_dir, 'model_reg.joblib'))
    dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    with open(os.path.join(model_dir, 'features.json'), 'w', encoding='utf-8') as f:
        json.dump(features, f)

    print('Saved models to', model_dir)


if __name__ == '__main__':
    train_and_save()
