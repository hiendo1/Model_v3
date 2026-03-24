"""
Football Prediction API (Production v4.2)
- Multi-Market Support: 1X2, BTTS, O/U 2.0, 2.5, 3.0, 3.5, Expected Goals
- 71-League Global Coverage with ELITE Disambiguation
- Robust Metadata Wrapper Handling for Production Models
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import json
import difflib
from scipy.stats import poisson
import xgboost 

app = Flask(__name__)
CORS(app)

# --- Path Resolution ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'football_model_v3.joblib')
STATS_PATH = os.path.join(BASE_DIR, 'latest_stats.json')
LEAGUE_TEAMS_PATH = os.path.join(BASE_DIR, 'league_teams.json')

# --- Global Model State ---
print(f"[*] Loading Production Model v4.2: {MODEL_PATH}")
model_pkg = joblib.load(MODEL_PATH)
best_models = model_pkg['best_models']
scaler = model_pkg['scaler']
feature_names = model_pkg['feature_names']

# --- Persistent Data Lookups ---
def load_json(path, default=None):
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[!] Error loading {path}: {e}")
    return default or {}

stats_db = load_json(STATS_PATH).get('team_stats', {})
league_teams_db = load_json(LEAGUE_TEAMS_PATH)

print(f"[+] API Ready. 8 Markets Active. Coverage: {len(league_teams_db)} leagues.")

def find_stats(query, league):
    """Fuzzy lookup for teams within a specific league."""
    # 1. Exact Match (ID|League or Name|League)
    exact_key = f"{query}|{league}"
    if exact_key in stats_db: 
        return stats_db[exact_key], query
    
    # 2. Case-Insensitive / Fuzzy Name Match within League
    league_keys = [k for k in stats_db.keys() if k.endswith(f"|{league}")]
    if not league_keys: return None, None
    
    names = [k.split('|')[0] for k in league_keys]
    
    # Case-insensitive
    query_lower = query.lower()
    for n in names:
        if n.lower() == query_lower:
            return stats_db[f"{n}|{league}"], n
            
    # Fuzzy
    matches = difflib.get_close_matches(query, names, n=1, cutoff=0.6)
    if matches:
        return stats_db[f"{matches[0]}|{league}"], matches[0]
        
    return None, None

@app.route('/')
def index():
    return jsonify({
        "engine": "Football Prediction Engine v4.2",
        "status": "online",
        "leagues": list(league_teams_db.keys())
    })

@app.route('/league-teams')
def get_league_teams():
    return jsonify(league_teams_db)

@app.route('/predict-by-name', methods=['POST'])
def predict_by_name():
    try:
        data = request.get_json()
        h_query = data.get('home_name')
        a_query = data.get('away_name')
        league = data.get('league')
        
        if not all([h_query, a_query, league]):
            return jsonify({'error': 'Missing home_name, away_name, or league'}), 400

        h_stats, h_name = find_stats(h_query, league)
        a_stats, a_name = find_stats(a_query, league)

        if not h_stats or not a_stats:
            return jsonify({
                'error': 'Team lookup failed', 
                'details': {'home': h_query, 'away': a_query, 'league': league}
            }), 404

        # Feature Mapping
        features = {}
        
        # 1. Date & Rest Days
        match_date_str = data.get('date')
        if not match_date_str:
            from datetime import datetime
            match_date_str = datetime.now().strftime('%Y-%m-%d')
        match_date = pd.to_datetime(match_date_str)

        def calc_rest(last_date_str, current_date):
            if not last_date_str: return 15
            try:
                last_date = pd.to_datetime(last_date_str)
                return min((current_date - last_date).days, 30)
            except: return 15
            
        features['h_rest'] = calc_rest(h_stats.get('last_match_date'), match_date)
        features['a_rest'] = calc_rest(a_stats.get('last_match_date'), match_date)
        features['league_tier'] = float(h_stats.get('league_tier', 4))

        # 2. Basic Stats & Elo
        metrics = ['goals_for', 'goals_against', 'xg_for', 'xg_against', 'shots_for', 'shots_against', 'avg_dist', 'danger_ratio']
        for m in metrics:
            features[f"h_roll_{m}"] = float(h_stats.get(f"roll_{m}", 0))
            features[f"a_roll_{m}"] = float(a_stats.get(f"roll_{m}", 0))

        features['h_elo'] = float(h_stats.get('elo', 1400))
        features['a_elo'] = float(a_stats.get('elo', 1400))
        features['rel_elo'] = features['h_elo'] - features['a_elo']
        
        # 3. Relative & Matchup Features
        features["rel_goals_for"] = features["h_roll_goals_for"] - features["a_roll_goals_for"]
        features["rel_xg_for"] = features["h_roll_xg_for"] - features["a_roll_xg_for"]
        features["rel_shots_for"] = features["h_roll_shots_for"] - features["a_roll_shots_for"]
        features["rel_avg_dist"] = features["h_roll_avg_dist"] - features["a_roll_avg_dist"]
        features["rel_danger_ratio"] = features["h_roll_danger_ratio"] - features["a_roll_danger_ratio"]
        
        features["h_att_v_a_def"] = features["h_roll_goals_for"] - features["a_roll_goals_against"]
        features["a_att_v_h_def"] = features["a_roll_goals_for"] - features["h_roll_goals_against"]
        features["rel_att_v_def"] = features["h_att_v_a_def"] - features["a_att_v_h_def"]
        features["xg_matchup_h"] = features["h_roll_xg_for"] - features["a_roll_xg_against"]
        features["xg_matchup_a"] = features["a_roll_xg_for"] - features["h_roll_xg_against"]
        features["goal_balance"] = (features["h_roll_goals_for"] - features["h_roll_goals_against"]) - (features["a_roll_goals_for"] - features["a_roll_goals_against"])

        # 4. Final Vector Assembly
        vector = [features.get(fn, 0.0) for fn in feature_names]
        X = scaler.transform([vector])
        
        # Market Prediction Helper (Handles Wrapper)
        def get_prob_wrapped(task, class_val):
            info = best_models[task]
            model = info['model']
            probs = model.predict_proba(X)[0]
            classes = list(info.get('classes_', []))
            if not classes and hasattr(model, 'classes_'): classes = list(model.classes_)
            return float(probs[classes.index(class_val)]) if class_val in classes else 0.5

        def get_pred_wrapped(task):
            info = best_models[task]
            return float(info['model'].predict(X)[0])

        ou_lines = ['2.0', '2.5', '3.0', '3.5']
        ou_probs = {ln: get_prob_wrapped(f'O/U {ln}', 1) for ln in ou_lines if f'O/U {ln}' in best_models}
        
        p_h, p_d, p_a = get_prob_wrapped('1X2', 2), get_prob_wrapped('1X2', 1), get_prob_wrapped('1X2', 0)
        p_btts = get_prob_wrapped('BTTS', 1)
        
        xg_h = get_pred_wrapped('Home Goals')
        xg_a = get_pred_wrapped('Away Goals')

        # Score Matrix Synthesis
        ML_W = 0.65
        m = np.outer(poisson.pmf(range(10), xg_h), poisson.pmf(range(10), xg_a))
        p_h_p, p_d_p, p_a_p = np.sum(np.tril(m, -1)), np.sum(np.diag(m)), np.sum(np.triu(m, 1))
        
        for h in range(10):
            for a in range(10):
                if h > a: m[h, a] *= (p_h * ML_W + p_h_p * (1-ML_W)) / max(p_h_p, 0.001)
                elif h == a: m[h, a] *= (p_d * ML_W + p_d_p * (1-ML_W)) / max(p_d_p, 0.001)
                else: m[h, a] *= (p_a * ML_W + p_a_p * (1-ML_W)) / max(p_a_p, 0.001)
        m /= np.sum(m)
        
        top_scores = sorted([{'score': f"{h}-{a}", 'p': float(m[h, a])} for h in range(6) for a in range(6)], key=lambda x: x['p'], reverse=True)[:5]

        return jsonify({
            "success": True,
            "match": {
                "home": h_name, 
                "away": a_name, 
                "league": league,
                "h_rest": int(features['h_rest']),
                "a_rest": int(features['a_rest']),
                "h_elo": float(features['h_elo']),
                "a_elo": float(features['a_elo'])
            },
            "predictions": {
                "1x2": {"home": round(p_h,4), "draw": round(p_d,4), "away": round(p_a,4), "suggest": "Home" if p_h>p_a and p_h>p_d else ("Away" if p_a>p_h and p_a>p_d else "Draw")},
                "over_under": {k: round(v,4) for k,v in ou_probs.items()},
                "btts": round(p_btts, 4),
                "expected_goals": {"home": round(xg_h,2), "away": round(xg_a,2)},
                "top_scores": top_scores
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=False)
