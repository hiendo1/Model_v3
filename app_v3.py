"""
Football Prediction API v3
- Uses football_model_v3.joblib with best_models structure
- Includes Dynamic Lambda Stretching, High Score Calibration, Smart Score Sync
- 29 features (24 rolling + 5 draw-specific)
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import json
from scipy.stats import poisson
import xgboost  # Required for model deserialization

app = Flask(__name__)
CORS(app)

# --- Load Model ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'football_model_v3.joblib')
print(f"Loading model from {MODEL_PATH}...")
model_package = joblib.load(MODEL_PATH)

best_models = model_package['best_models']
scaler = model_package['scaler']
feature_names = model_package['feature_names']

# --- Load Stats Lookup (Production Mode) ---
STATS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'latest_stats.json')
stats_lookup = {}
if os.path.exists(STATS_PATH):
    try:
        with open(STATS_PATH, 'r', encoding='utf-8') as f:
            stats_lookup = json.load(f)
        print(f"[OK] Stats lookup loaded: {stats_lookup.get('metadata', {}).get('total_teams', 0)} teams")
    except Exception as e:
        print(f"[ERROR] Failed to load stats lookup: {e}")

print(f"[OK] Model v3 loaded. Features: {len(feature_names)}")
print(f"  Best models: { {k: (v.get('type') if type(v) is dict else type(v).__name__) for k, v in best_models.items()} }")

# --- Configuration ---
ML_WEIGHT = 0.35  # Must match predict_gui_v3.py


@app.route('/')
def home():
    return jsonify({
        "message": "Football Prediction API v3",
        "version": "3.0",
        "endpoints": {
            "/predict-by-name": "POST - Best for FE (only needs team names)",
            "/predict-simple": "POST - Manual stats input",
            "/health": "GET - Status check",
            "/features": "GET - Required feature list"
        },
        "status": "online"
    })


@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "model_version": "v3",
        "model_loaded": True,
        "features_count": len(feature_names),
        "best_models": {k: (v.get("type") if type(v) is dict else type(v).__name__) for k, v in best_models.items()}
    })


@app.route('/features')
def get_features():
    return jsonify({
        "total_features": len(feature_names),
        "features": feature_names,
        "groups": {
            "h2h": [f for f in feature_names if 'h2h' in f],
            "home_rolling": [f for f in feature_names if f.startswith('h_')],
            "away_rolling": [f for f in feature_names if f.startswith('a_')],
            "relative": [f for f in feature_names if f.startswith('rel_')],
            "draw_specific": [f for f in feature_names if f in ['def_similarity', 'off_similarity', 'xg_convergence', 'goal_balance', 'h2h_draw_rate']]
        }
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Full prediction endpoint - requires all features.
    Input: {"features": {"h2h_home_wins": 0.5, ...}}
    """
    try:
        data = request.get_json()
        if 'features' not in data:
            return jsonify({'error': 'Missing features dict', 'hint': 'Use /features to see required fields'}), 400

        result = predict_internal(data['features'])
        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/predict-by-name', methods=['POST'])
def predict_by_name():
    """
    Production-ready endpoint for Frontend (v4).
    Input: {"home_name": "Arsenal FC", "away_name": "Liverpool FC", "match_date": "2024-05-19"}
    """
    try:
        data = request.get_json()
        h_name = data.get('home_name')
        a_name = data.get('away_name')
        match_date = data.get('match_date')
        if not h_name or not a_name:
            return jsonify({'error': 'Please provide home_name and away_name'}), 400

        if not stats_lookup:
            return jsonify({'error': 'Stats database not loaded on server.'}), 500

        team_db = stats_lookup.get('team_stats', {})
        h2h_db = stats_lookup.get('h2h_stats', {})

        if h_name not in team_db or a_name not in team_db:
            return jsonify({'error': f'One or both teams not found in database.'}), 404

        h_stats = team_db[h_name]
        a_stats = team_db[a_name]

        # 1. Start with H2H
        h2h = h2h_db.get(f"{h_name}|{a_name}")
        if not h2h:
            rev = h2h_db.get(f"{a_name}|{h_name}")
            if rev:
                h2h = {
                    'h2h_home_wins': rev['h2h_away_wins'], 'h2h_draws': rev['h2h_draws'], 'h2h_away_wins': rev['h2h_home_wins'],
                    'h2h_avg_goals_home': rev['h2h_avg_goals_away'], 'h2h_avg_goals_away': rev['h2h_avg_goals_home'],
                    'h2h_draw_rate': rev['h2h_draw_rate']
                }
            else:
                h2h = {'h2h_home_wins': 0.33, 'h2h_draws': 0.33, 'h2h_away_wins': 0.33, 'h2h_avg_goals_home': 1.4, 'h2h_avg_goals_away': 1.2, 'h2h_draw_rate': 0.33}

        # 2. Derive v4 features
        features = {}
        features.update(h2h)
        
        league = h_stats.get('league', a_stats.get('league', 'Unknown'))
        state = model_package.get("v4_state", {}).get(league, {})
        
        # Rolling Features
        metrics = ['goals_for', 'goals_against', 'xg_for', 'xg_against', 'shots_for', 'shots_against', 'goals_std', 'high_score_rate']
        for m in metrics:
            h_val = float(h_stats.get(f"roll_{m}", 0))
            a_val = float(a_stats.get(f"roll_{m}", 0))
            features[f"h_roll_{m}"] = h_val
            features[f"a_roll_{m}"] = a_val
            if m in ['goals_for', 'xg_for', 'shots_for']:
                features[f"rel_{m}"] = h_val - a_val

        # v4 state (Elo/Fatigue)
        h_elo = state.get("elo", {}).get(h_name, 1500.0)
        a_elo = state.get("elo", {}).get(a_name, 1500.0)
        features["h_elo"] = h_elo
        features["a_elo"] = a_elo
        features["rel_elo"] = h_elo - a_elo
        
        curr_date = pd.to_datetime(match_date) if match_date else pd.Timestamp.now()
        h_last = state.get("last_date", {}).get(h_name)
        a_last = state.get("last_date", {}).get(a_name)
        features["h_rest"] = min((curr_date - pd.to_datetime(h_last)).days if h_last else 15, 15)
        features["a_rest"] = min((curr_date - pd.to_datetime(a_last)).days if a_last else 15, 15)
        
        features["h_venue_roll_goals"] = h_stats.get("venue_roll_goals", h_stats.get("roll_goals_for", 1.2))
        features["a_venue_roll_goals"] = a_stats.get("venue_roll_goals", a_stats.get("roll_goals_for", 1.2))

        # Draw Specific
        features['def_similarity'] = 1.0 / (1.0 + abs(features['h_roll_goals_against'] - features['a_roll_goals_against']))
        features['off_similarity'] = 1.0 / (1.0 + abs(features['h_roll_goals_for'] - features['a_roll_goals_for']))
        features['xg_convergence'] = 1.0 / (1.0 + abs(features['h_roll_xg_for'] - features['a_roll_xg_for']))
        features['goal_balance'] = 1.0 / (1.0 + abs(features['h_roll_goals_for'] - features['h_roll_goals_against']) + 
                                          abs(features['a_roll_goals_for'] - features['a_roll_goals_against']))
        features['h2h_draw_rate'] = h2h.get('h2h_draw_rate', 0.33)

        prediction_data = predict_internal(features)
        prediction_data['match'] = {'home': h_name, 'away': a_name, 'date': str(curr_date.date())}
        return jsonify(prediction_data)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/teams')
def get_teams():
    if not stats_lookup: return jsonify([])
    return jsonify(sorted(list(stats_lookup.get('team_stats', {}).keys())))

@app.route('/teams-by-league')
def get_teams_by_league():
    if not stats_lookup: return jsonify({})
    team_db = stats_lookup.get('team_stats', {})
    
    leagues = {}
    for team, stats in team_db.items():
        league = stats.get('league', 'Unknown')
        if league not in leagues:
            leagues[league] = []
        leagues[league].append(team)
        
    # Sort teams within each league
    for l in leagues:
        leagues[l].sort()
        
    return jsonify(leagues)

@app.route('/predict-simple', methods=['POST'])
def predict_simple():
    """
    Simplified prediction endpoint.
    Input:
    {
        "home_team": {
            "name": "Arsenal FC",
            "avg_goals_scored": 2.1,
            "avg_goals_conceded": 1.0,
            "avg_xg": 1.9,
            "avg_xg_conceded": 1.2,
            "avg_shots": 15.0,
            "avg_shots_conceded": 10.0,
            "goals_std": 1.2,
            "high_score_rate": 0.3
        },
        "away_team": { ... same fields ... },
        "h2h_history": {
            "home_wins": 4, "draws": 2, "away_wins": 1,
            "avg_goals_home": 1.8, "avg_goals_away": 1.2
        }
    }
    """
    try:
        data = request.get_json()
        if 'home_team' not in data or 'away_team' not in data:
            return jsonify({'error': 'Missing home_team or away_team'}), 400

        home = data['home_team']
        away = data['away_team']
        h2h = data.get('h2h_history', {})

        features = {}

        # === H2H FEATURES ===
        total_h2h = h2h.get('home_wins', 0) + h2h.get('draws', 0) + h2h.get('away_wins', 0)
        if total_h2h > 0:
            features['h2h_home_wins'] = h2h.get('home_wins', 0) / total_h2h
            features['h2h_draws'] = h2h.get('draws', 0) / total_h2h
            features['h2h_away_wins'] = h2h.get('away_wins', 0) / total_h2h
            features['h2h_avg_goals_home'] = h2h.get('avg_goals_home', home.get('avg_goals_scored', 1.5))
            features['h2h_avg_goals_away'] = h2h.get('avg_goals_away', away.get('avg_goals_scored', 1.5))
        else:
            features['h2h_home_wins'] = 0.33
            features['h2h_draws'] = 0.33
            features['h2h_away_wins'] = 0.33
            features['h2h_avg_goals_home'] = home.get('avg_goals_scored', 1.5)
            features['h2h_avg_goals_away'] = away.get('avg_goals_scored', 1.5)

        # === RELATIVE FEATURES ===
        features['rel_goals_for'] = home.get('avg_goals_scored', 1.5) - away.get('avg_goals_scored', 1.5)
        features['rel_xg_for'] = home.get('avg_xg', 1.5) - away.get('avg_xg', 1.5)
        features['rel_shots_for'] = home.get('avg_shots', 12.0) - away.get('avg_shots', 12.0)

        # === HOME ROLLING STATS ===
        features['h_roll_goals_for'] = home.get('avg_goals_scored', 1.5)
        features['h_roll_goals_against'] = home.get('avg_goals_conceded', 1.5)
        features['h_roll_xg_for'] = home.get('avg_xg', 1.5)
        features['h_roll_xg_against'] = home.get('avg_xg_conceded', 1.5)
        features['h_roll_shots_for'] = home.get('avg_shots', 12.0)
        features['h_roll_shots_against'] = home.get('avg_shots_conceded', 12.0)
        features['h_roll_goals_std'] = home.get('goals_std', 1.0)
        features['h_roll_high_score_rate'] = home.get('high_score_rate', 0.3)

        # === AWAY ROLLING STATS ===
        features['a_roll_goals_for'] = away.get('avg_goals_scored', 1.5)
        features['a_roll_goals_against'] = away.get('avg_goals_conceded', 1.5)
        features['a_roll_xg_for'] = away.get('avg_xg', 1.5)
        features['a_roll_xg_against'] = away.get('avg_xg_conceded', 1.5)
        features['a_roll_shots_for'] = away.get('avg_shots', 12.0)
        features['a_roll_shots_against'] = away.get('avg_shots_conceded', 12.0)
        features['a_roll_goals_std'] = away.get('goals_std', 1.0)
        features['a_roll_high_score_rate'] = away.get('high_score_rate', 0.3)

        # === V4 FEATURES (Defaults) ===
        features['h_elo'] = 1500.0; features['a_elo'] = 1500.0; features['rel_elo'] = 0.0
        features['h_rest'] = 15; features['a_rest'] = 15
        features['h_venue_roll_goals'] = home.get('avg_goals_scored', 1.2)
        features['a_venue_roll_goals'] = away.get('avg_goals_scored', 1.2)

        # === DRAW-SPECIFIC FEATURES ===
        h_gc = features['h_roll_goals_against']
        a_gc = features['a_roll_goals_against']
        h_gs = features['h_roll_goals_for']
        a_gs = features['a_roll_goals_for']
        h_xg = features['h_roll_xg_for']
        a_xg = features['a_roll_xg_for']

        features['def_similarity'] = 1.0 / (1.0 + abs(h_gc - a_gc))
        features['off_similarity'] = 1.0 / (1.0 + abs(h_gs - a_gs))
        features['xg_convergence'] = 1.0 / (1.0 + abs(h_xg - a_xg))
        features['goal_balance'] = 1.0 / (1.0 + abs(h_gs - h_gc) + abs(a_gs - a_gc))
        features['h2h_draw_rate'] = features.get('h2h_draws', 0.33)

        # Predict
        result = predict_internal(features)
        result['match'] = {
            'home_team': home.get('name', 'Home'),
            'away_team': away.get('name', 'Away')
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def get_prediction(model_info, X_scaled, is_proba=False):
    """Helper to handle both single models and soft voting ensembles."""
    if type(model_info) is dict and model_info.get("type") == "ensemble":
        w_rf = model_info["weight_rf"]
        w_xgb = model_info["weight_xgb"]
        if is_proba:
            return w_rf * model_info["rf"].predict_proba(X_scaled) + w_xgb * model_info["xgb"].predict_proba(X_scaled)
        else:
            return w_rf * model_info["rf"].predict(X_scaled) + w_xgb * model_info["xgb"].predict(X_scaled)
    elif type(model_info) is dict and model_info.get("type") == "single":
        model = model_info["model"]
    else:
        model = model_info # backward compatibility
        
    return model.predict_proba(X_scaled) if is_proba else model.predict(X_scaled)

def get_classes(model_info):
    if type(model_info) is dict:
        return model_info["classes_"]
    return model_info.classes_


def predict_internal(features_dict):
    """
    Core prediction logic with Dual Hybrid Scaling.
    Mirrors predict_gui_v3.py logic exactly.
    """
    # Build feature vector
    feature_vector = []
    missing = []
    for feat in feature_names:
        if feat in features_dict:
            feature_vector.append(float(features_dict[feat]))
        else:
            missing.append(feat)
            feature_vector.append(0.0)

    X = np.array([feature_vector])
    X_scaled = scaler.transform(X)

    # --- 1X2 Prediction ---
    model_1x2 = best_models['1X2']
    probs_1x2 = get_prediction(model_1x2, X_scaled, is_proba=True)[0]
    # Map model classes to Away(0), Draw(1), Home(2)
    classes = list(get_classes(model_1x2))

    prob_away = float(probs_1x2[classes.index(0)]) if 0 in classes else 0.33
    prob_draw = float(probs_1x2[classes.index(1)]) if 1 in classes else 0.33
    prob_home = float(probs_1x2[classes.index(2)]) if 2 in classes else 0.33

    # --- O/U Prediction ---
    model_ou = best_models['O/U 2.5']
    probs_ou = get_prediction(model_ou, X_scaled, is_proba=True)[0]
    ou_classes = list(get_classes(model_ou))
    prob_under = float(probs_ou[ou_classes.index(0)]) if 0 in ou_classes else 0.5
    prob_over = float(probs_ou[ou_classes.index(1)]) if 1 in ou_classes else 0.5

    # --- BTTS Prediction ---
    model_btts = best_models.get('BTTS')
    if model_btts:
        probs_btts = get_prediction(model_btts, X_scaled, is_proba=True)[0]
        btts_classes = list(get_classes(model_btts))
        prob_btts_no = float(probs_btts[btts_classes.index(0)]) if 0 in btts_classes else 0.5
        prob_btts_yes = float(probs_btts[btts_classes.index(1)]) if 1 in btts_classes else 0.5
    else:
        # Fallback if model not yet updated
        prob_btts_yes = 0.5
        prob_btts_no = 0.5

    # --- Goal Prediction ---
    xg_h = float(get_prediction(best_models['Home Goals'], X_scaled, is_proba=False)[0])
    xg_a = float(get_prediction(best_models['Away Goals'], X_scaled, is_proba=False)[0])

    # --- DYNAMIC LAMBDA STRETCHING ---
    ou_confidence = abs(prob_over - 0.5)
    h_vol = features_dict.get('h_roll_goals_std', 1.0)
    a_vol = features_dict.get('a_roll_goals_std', 1.0)
    avg_vol = (float(h_vol) + float(a_vol)) / 2.0

    if prob_over > 0.5:
        stretch = 1.0 + (ou_confidence * 0.4) + (max(0, avg_vol - 1.0) * 0.15)
        xg_h *= stretch
        xg_a *= stretch
    else:
        shrink = 1.0 - (ou_confidence * 0.25)
        xg_h *= shrink
        xg_a *= shrink

    xg_h = max(0.3, xg_h)
    xg_a = max(0.3, xg_a)

    # --- DUAL GRID SCALING ---
    prob_matrix = np.outer(poisson.pmf(range(12), xg_h), poisson.pmf(range(12), xg_a))

    p_h_p = max(np.sum(np.tril(prob_matrix, -1)), 0.0001)
    p_d_p = max(np.sum(np.diag(prob_matrix)), 0.0001)
    p_a_p = max(np.sum(np.triu(prob_matrix, 1)), 0.0001)

    # PASS 1: Result Scaling
    target_h = (prob_home * ML_WEIGHT) + (p_h_p * (1 - ML_WEIGHT))
    target_d = (prob_draw * ML_WEIGHT) + (p_d_p * (1 - ML_WEIGHT))
    target_a = (prob_away * ML_WEIGHT) + (p_a_p * (1 - ML_WEIGHT))

    for h in range(12):
        for a in range(12):
            if h > a:
                prob_matrix[h, a] *= (target_h / p_h_p)
            elif h == a:
                prob_matrix[h, a] *= (target_d / p_d_p)
            else:
                prob_matrix[h, a] *= (target_a / p_a_p)
    prob_matrix /= np.sum(prob_matrix)

    # PASS 2: O/U Scaling
    p_o25_p = 1.0 - sum(prob_matrix[h, a] for h in range(3) for a in range(3) if h + a <= 2)
    p_u25_p = 1.0 - p_o25_p
    target_over_final = (prob_over * ML_WEIGHT) + (p_o25_p * (1 - ML_WEIGHT))
    target_under_final = 1.0 - target_over_final

    for h in range(12):
        for a in range(12):
            if h + a > 2:
                prob_matrix[h, a] *= (target_over_final / max(p_o25_p, 0.0001))
            else:
                prob_matrix[h, a] *= (target_under_final / max(p_u25_p, 0.0001))
    prob_matrix /= np.sum(prob_matrix)

    # PASS 3: BTTS Scaling
    p_btts_yes_p = 1.0 - (np.sum(prob_matrix[0, :]) + np.sum(prob_matrix[:, 0]) - prob_matrix[0, 0])
    p_btts_no_p = 1.0 - p_btts_yes_p
    target_btts_yes_final = (prob_btts_yes * ML_WEIGHT) + (p_btts_yes_p * (1 - ML_WEIGHT))
    target_btts_no_final = 1.0 - target_btts_yes_final

    for h in range(12):
        for a in range(12):
            if h > 0 and a > 0: # BTTS = Yes
                prob_matrix[h, a] *= (target_btts_yes_final / max(p_btts_yes_p, 0.0001))
            else: # BTTS = No
                prob_matrix[h, a] *= (target_btts_no_final / max(p_btts_no_p, 0.0001))
    prob_matrix /= np.sum(prob_matrix)

    # --- HIGH SCORE CALIBRATION ---
    h_hsr = float(features_dict.get('h_roll_high_score_rate', 0.3))
    a_hsr = float(features_dict.get('a_roll_high_score_rate', 0.3))
    if prob_over > 0.55 and (h_hsr > 0.35 or a_hsr > 0.35):
        boost = 1.0 + (prob_over - 0.55) * 2.0
        for h in range(12):
            for a in range(12):
                if h + a >= 4:
                    prob_matrix[h, a] *= boost
        prob_matrix /= np.sum(prob_matrix)

    # --- Derived Final Probabilities ---
    final_h = sum(prob_matrix[h, a] for h in range(12) for a in range(12) if h > a)
    final_d = sum(prob_matrix[h, a] for h in range(12) for a in range(12) if h == a)
    final_a = sum(prob_matrix[h, a] for h in range(12) for a in range(12) if h < a)
    final_o25 = 1.0 - sum(prob_matrix[h, a] for h in range(3) for a in range(3) if h + a <= 2)

    # --- Top Scores (Consensus-Aware) ---
    scores_list = []
    for h in range(7):
        for a in range(7):
            p = prob_matrix[h, a]
            
            # Outcome Alignment Weights
            # If a score aligns with the main predictions, give it a tiny boost for "visual stability"
            weight = 1.0
            if (final_o25 > 0.5 and h + a > 2) or (final_o25 <= 0.5 and h + a <= 2):
                weight *= 1.05
            if (target_btts_yes_final > 0.5 and h > 0 and a > 0) or (target_btts_yes_final <= 0.5 and (h == 0 or a == 0)):
                weight *= 1.05
            if (final_h > final_a and h > a) or (final_a > final_h and a > h) or (final_d > final_h and final_d > final_a and h == a):
                weight *= 1.05

            scores_list.append({
                'score': f"{h}-{a}",
                'probability': float(p),
                'display_p': round(float(p), 4),
                'rank_score': p * weight
            })
    
    # Sort by rank_score (probability + tiny alignment boost)
    scores_list.sort(key=lambda x: x['rank_score'], reverse=True)
    top_scores = [{'score': s['score'], 'probability': s['display_p']} for s in scores_list[:5]]

    # --- Smart Score Sync ---
    def best_score_for(cond_fn):
        best_s, max_p = (0, 0), -1.0
        for h in range(12):
            for a in range(12):
                if cond_fn(h, a) and prob_matrix[h, a] > max_p:
                    max_p = prob_matrix[h, a]
                    best_s = (h, a)
        return {'score': f"{best_s[0]}-{best_s[1]}", 'probability': round(float(max_p), 4)}

    smart_suggestions = {}
    if final_o25 > 0.5:
        smart_suggestions['primary'] = {'type': 'Over 2.5', **best_score_for(lambda h, a: h + a > 2)}
    else:
        smart_suggestions['primary'] = {'type': 'Under 2.5', **best_score_for(lambda h, a: h + a <= 2)}

    if final_h > final_a and final_h > final_d:
        smart_suggestions['trend'] = {'type': 'Home Win', **best_score_for(lambda h, a: h > a)}
    elif final_a > final_h and final_a > final_d:
        smart_suggestions['trend'] = {'type': 'Away Win', **best_score_for(lambda h, a: h < a)}
    else:
        smart_suggestions['trend'] = {'type': 'Draw', **best_score_for(lambda h, a: h == a)}

    # --- Advanced Markets (Handicap & BTTS) ---
    # BTTS logic: P(both score) = P(G1>0 and G2>0)
    # Sum of all matrix except row 0 and col 0
    p_btts_yes = 1.0 - (np.sum(prob_matrix[0, :]) + np.sum(prob_matrix[:, 0]) - prob_matrix[0, 0])
    
    # Handicap suggestion (Simple Asian Handicap estimator)
    # If final_h is 0.65 -> Handicap -0.75 or -1.0
    # Formula: AH = (Expected Goal Diff) adjusted by Win Probs
    exp_diff = xg_h - xg_a
    if final_h > 0.6:
        h_cap = f"Home -{round(abs(exp_diff)*2)/2:g}" if abs(exp_diff) > 0.4 else "Home -0.25"
    elif final_a > 0.6:
        h_cap = f"Away -{round(abs(exp_diff)*2)/2:g}" if abs(exp_diff) > 0.4 else "Away -0.25"
    else:
        h_cap = "Draw / Level (0)"

    # --- Build Response ---
    result_label = ['Away Win', 'Draw', 'Home Win'][int(np.argmax([final_a, final_d, final_h]))]

    return {
        'success': True,
        'predictions': {
            '1x2': {
                'result': result_label,
                'probabilities': {
                    'home_win': round(float(final_h), 4),
                    'draw': round(float(final_d), 4),
                    'away_win': round(float(final_a), 4)
                },
                'method': f"ML {ML_WEIGHT*100:.0f}% / Poisson {(1-ML_WEIGHT)*100:.0f}%"
            },
            'over_under': {
                'prediction': 'Over 2.5' if final_o25 > 0.5 else 'Under 2.5',
                'probabilities': {
                    'over': round(float(final_o25), 4),
                    'under': round(float(1.0 - final_o25), 4)
                }
            },
            'btts': {
                'prediction': 'Yes' if p_btts_yes > 0.5 else 'No',
                'probability': round(float(p_btts_yes), 4)
            },
            'handicap': {
                'suggestion': h_cap,
                'expected_diff': round(float(exp_diff), 2)
            },
            'score': {
                'home_expected': round(float(xg_h), 2),
                'away_expected': round(float(xg_a), 2),
                'total_expected': round(float(xg_h + xg_a), 2),
                'top_scores': top_scores
            },
            'smart_suggestions': smart_suggestions
        },
        'metadata': {
            'features_used': len(feature_names),
            'features_missing': len(missing),
            'model_version': 'v3',
            'method': 'Dual Hybrid Scaling + BTTS/Handicap Logic'
        }
    }


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(debug=False, host='0.0.0.0', port=port)
