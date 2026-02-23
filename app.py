from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson

app = Flask(__name__)
CORS(app)

# Load model
print("Loading model...")
model_package = joblib.load('football_model_v2.joblib')
clf_rf = model_package['clf_rf']
clf_gb = model_package['clf_gb']
clf_ou_rf = model_package['clf_ou_rf']
clf_ou_gb = model_package['clf_ou_gb']
reg_h = model_package['reg_h']
reg_a = model_package['reg_a']
scaler = model_package['scaler']
feature_names = model_package['feature_names']

print(f"✓ Model loaded. Features required: {len(feature_names)}")

@app.route('/')
def home():
    return jsonify({
        "message": "Football Prediction API v2",
        "endpoints": {
            "/predict": "POST - Predict match result (requires all features)",
            "/predict-simple": "POST - Predict with simplified input",
            "/health": "GET - Check API health",
            "/features": "GET - Get required features list"
        },
        "status": "online"
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy", 
        "model_loaded": True,
        "features_count": len(feature_names)
    })

@app.route('/features')
def get_features():
    """Return required features for transparency"""
    return jsonify({
        "total_features": len(feature_names),
        "features": feature_names,
        "groups": {
            "h2h": [f for f in feature_names if 'h2h' in f],
            "home_stats": [f for f in feature_names if f.startswith('h_')],
            "away_stats": [f for f in feature_names if f.startswith('a_')],
            "relative": [f for f in feature_names if f.startswith('rel_')]
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Full prediction endpoint - requires all features
    Input example:
    {
        "features": {
            "h2h_home_wins": 0.5,
            "h2h_draws": 0.3,
            ...
        }
    }
    """
    try:
        data = request.get_json()
        
        # Validate input
        if 'features' not in data:
            return jsonify({
                'error': 'Missing features',
                'hint': 'Send JSON with "features" dict or use /predict-simple endpoint'
            }), 400
        
        input_features = data['features']
        
        # Create feature vector in correct order
        feature_vector = []
        missing = []
        for feat in feature_names:
            if feat in input_features:
                feature_vector.append(float(input_features[feat]))
            else:
                missing.append(feat)
                feature_vector.append(0.0)  # Default to 0
        
        if missing and len(missing) < len(feature_names) * 0.3:  # Allow some missing
            print(f"Warning: Missing {len(missing)} features: {missing[:5]}...")
        
        # Convert to numpy array and scale
        X = np.array([feature_vector])
        X_scaled = scaler.transform(X)
        
        # Ensemble predictions
        # 1x2 Result
        probs_rf = clf_rf.predict_proba(X_scaled)[0]
        probs_gb = clf_gb.predict_proba(X_scaled)[0]
        probs_1x2 = (probs_rf + probs_gb) / 2
        result_pred = int(np.argmax(probs_1x2))
        
        # Over/Under
        probs_ou_rf = clf_ou_rf.predict_proba(X_scaled)[0]
        probs_ou_gb = clf_ou_gb.predict_proba(X_scaled)[0]
        probs_ou = (probs_ou_rf + probs_ou_gb) / 2
        
        # Score prediction
        home_goals = float(reg_h.predict(X_scaled)[0])
        away_goals = float(reg_a.predict(X_scaled)[0])
        
        return jsonify({
            'success': True,
            'predictions': {
                '1x2': {
                    'result': ['Away Win', 'Draw', 'Home Win'][result_pred],
                    'probabilities': {
                        'away_win': round(float(probs_1x2[0]), 3),
                        'draw': round(float(probs_1x2[1]), 3),
                        'home_win': round(float(probs_1x2[2]), 3)
                    }
                },
                'over_under': {
                    'prediction': 'Over 2.5' if probs_ou[1] > 0.5 else 'Under 2.5',
                    'probabilities': {
                        'under': round(float(probs_ou[0]), 3),
                        'over': round(float(probs_ou[1]), 3)
                    }
                },
                'score': {
                    'home': round(home_goals, 1),
                    'away': round(away_goals, 1),
                    'total': round(home_goals + away_goals, 1)
                }
            },
            'metadata': {
                'features_used': len(feature_names),
                'features_missing': len(missing)
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict-simple', methods=['POST'])
def predict_simple():
    """
    Simplified prediction endpoint - auto-calculate features from basic team stats
    
    Input example:
    {
        "home_team": {
            "name": "Manchester City",
            "avg_goals_scored": 2.1,
            "avg_goals_conceded": 1.0,
            "avg_xg": 1.9,
            "avg_xg_conceded": 1.2,
            "avg_shots": 15.0,
            "avg_shots_conceded": 10.0,
            "avg_ppda": 8.5,
            "avg_deep": 6.0,
            "recent_form": [2, 2, 1, 2, 2],
            "clean_sheets_rate": 0.4
        },
        "away_team": {
            "name": "Liverpool",
            "avg_goals_scored": 1.8,
            "avg_goals_conceded": 1.3,
            ...
        },
        "h2h_history": {
            "home_wins": 4,
            "draws": 2,
            "away_wins": 1
        }
    }
    """
    try:
        data = request.get_json()
        
        if 'home_team' not in data or 'away_team' not in data:
            return jsonify({
                'error': 'Missing home_team or away_team data',
                'hint': 'Check API documentation for required fields'
            }), 400
        
        home = data['home_team']
        away = data['away_team']
        h2h = data.get('h2h_history', {})
        
        # Build features dict
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
            # Default H2H if no history
            features['h2h_home_wins'] = 0.33
            features['h2h_draws'] = 0.33
            features['h2h_away_wins'] = 0.33
            features['h2h_avg_goals_home'] = home.get('avg_goals_scored', 1.5)
            features['h2h_avg_goals_away'] = away.get('avg_goals_scored', 1.5)
        
        # === HOME ROLLING STATS ===
        features['h_roll_goals_for'] = home.get('avg_goals_scored', 1.5)
        features['h_roll_goals_against'] = home.get('avg_goals_conceded', 1.5)
        features['h_roll_xg_for'] = home.get('avg_xg', 1.5)
        features['h_roll_xg_against'] = home.get('avg_xg_conceded', 1.5)
        features['h_roll_shots_for'] = home.get('avg_shots', 12.0)
        features['h_roll_shots_against'] = home.get('avg_shots_conceded', 12.0)
        features['h_roll_ppda'] = home.get('avg_ppda', 10.0)
        features['h_roll_deep'] = home.get('avg_deep', 5.0)
        
        # === AWAY ROLLING STATS ===
        features['a_roll_goals_for'] = away.get('avg_goals_scored', 1.5)
        features['a_roll_goals_against'] = away.get('avg_goals_conceded', 1.5)
        features['a_roll_xg_for'] = away.get('avg_xg', 1.5)
        features['a_roll_xg_against'] = away.get('avg_xg_conceded', 1.5)
        features['a_roll_shots_for'] = away.get('avg_shots', 12.0)
        features['a_roll_shots_against'] = away.get('avg_shots_conceded', 12.0)
        features['a_roll_ppda'] = away.get('avg_ppda', 10.0)
        features['a_roll_deep'] = away.get('avg_deep', 5.0)
        
        # === HOME RECENT FORM ===
        h_form = home.get('recent_form', [1, 1, 1, 1, 1])  # Default: all draws
        if len(h_form) == 0:
            h_form = [1, 1, 1, 1, 1]
        
        features['h_recent_wins'] = h_form.count(2) / len(h_form)
        features['h_recent_draws'] = h_form.count(1) / len(h_form)
        features['h_recent_losses'] = h_form.count(0) / len(h_form)
        features['h_recent_goals_avg'] = home.get('avg_goals_scored', 1.5)
        features['h_recent_conceded_avg'] = home.get('avg_goals_conceded', 1.5)
        features['h_recent_clean_sheets'] = home.get('clean_sheets_rate', 0.3)
        
        # === AWAY RECENT FORM ===
        a_form = away.get('recent_form', [1, 1, 1, 1, 1])
        if len(a_form) == 0:
            a_form = [1, 1, 1, 1, 1]
        
        features['a_recent_wins'] = a_form.count(2) / len(a_form)
        features['a_recent_draws'] = a_form.count(1) / len(a_form)
        features['a_recent_losses'] = a_form.count(0) / len(a_form)
        features['a_recent_goals_avg'] = away.get('avg_goals_scored', 1.5)
        features['a_recent_conceded_avg'] = away.get('avg_goals_conceded', 1.5)
        features['a_recent_clean_sheets'] = away.get('clean_sheets_rate', 0.3)
        
        # === RELATIVE FEATURES ===
        features['rel_goals_for'] = features['h_roll_goals_for'] - features['a_roll_goals_for']
        features['rel_goals_against'] = features['h_roll_goals_against'] - features['a_roll_goals_against']
        features['rel_xg_for'] = features['h_roll_xg_for'] - features['a_roll_xg_for']
        features['rel_xg_against'] = features['h_roll_xg_against'] - features['a_roll_xg_against']
        features['rel_shots_for'] = features['h_roll_shots_for'] - features['a_roll_shots_for']
        features['rel_shots_against'] = features['h_roll_shots_against'] - features['a_roll_shots_against']
        features['rel_ppda'] = features['h_roll_ppda'] - features['a_roll_ppda']
        features['rel_deep'] = features['h_roll_deep'] - features['a_roll_deep']
        
        # Use the main predict endpoint logic
        result = predict_internal(features)
        
        # Add team names to response
        result['match'] = {
            'home_team': home.get('name', 'Home Team'),
            'away_team': away.get('name', 'Away Team')
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def predict_internal(features_dict):
    """Internal prediction function with Dual Hybrid Scaling"""
    # Create feature vector
    feature_vector = []
    missing = []
    for feat in feature_names:
        if feat in features_dict:
            feature_vector.append(float(features_dict[feat]))
        else:
            missing.append(feat)
            feature_vector.append(0.0)
    
    # Convert and scale
    X = np.array([feature_vector])
    X_scaled = scaler.transform(X)
    
    # 1. Ensemble Win Probs
    probs_rf = clf_rf.predict_proba(X_scaled)[0]
    probs_gb = clf_gb.predict_proba(X_scaled)[0]
    probs_1x2 = (probs_rf + probs_gb) / 2
    
    # 2. Hybrid xG (5% boost)
    xg_h = reg_h.predict(X_scaled)[0] * 1.05
    xg_a = reg_a.predict(X_scaled)[0] * 1.05
    
    # ML O/U Prob
    probs_ou_rf = clf_ou_rf.predict_proba(X_scaled)[0]
    probs_ou_gb = clf_ou_gb.predict_proba(X_scaled)[0]
    probs_ou_ml = (probs_ou_rf + probs_ou_gb) / 2
    
    # 3. DUAL GRID SCALING (Consistent with predict_v2.py)
    ML_WEIGHT = 0.7
    prob_matrix = np.outer(poisson.pmf(range(12), xg_h), poisson.pmf(range(12), xg_a))
    
    p_h_poisson = max(np.sum(np.tril(prob_matrix, -1)), 0.0001)
    p_d_poisson = max(np.sum(np.diag(prob_matrix)), 0.0001)
    p_a_poisson = max(np.sum(np.triu(prob_matrix, 1)), 0.0001)
    
    # Target 1X2
    target_h = (probs_1x2[2] * ML_WEIGHT) + (p_h_poisson * (1 - ML_WEIGHT))
    target_d = (probs_1x2[1] * ML_WEIGHT) + (p_d_poisson * (1 - ML_WEIGHT))
    target_a = (probs_1x2[0] * ML_WEIGHT) + (p_a_poisson * (1 - ML_WEIGHT))
    
    # PASS 1: Result Scaling
    for h in range(12):
        for a in range(12):
            if h > a: prob_matrix[h, a] *= (target_h / p_h_poisson)
            elif h == a: prob_matrix[h, a] *= (target_d / p_d_poisson)
            else: prob_matrix[h, a] *= (target_a / p_a_poisson)
    prob_matrix /= np.sum(prob_matrix)
    
    # PASS 2: O/U Scaling
    p_o25_p = 1.0 - sum(prob_matrix[h, a] for h in range(3) for a in range(3) if h+a <= 2)
    p_u25_p = 1.0 - p_o25_p
    target_over = (probs_ou_ml[1] * ML_WEIGHT) + (p_o25_p * (1 - ML_WEIGHT))
    target_under = 1.0 - target_over
    
    for h in range(12):
        for a in range(12):
            if h + a > 2:
                prob_matrix[h, a] *= (target_over / max(p_o25_p, 0.0001))
            else:
                prob_matrix[h, a] *= (target_under / max(p_u25_p, 0.0001))
    prob_matrix /= np.sum(prob_matrix)
    
    # Derived results
    final_h = sum(prob_matrix[h,a] for h in range(12) for a in range(12) if h > a)
    final_d = sum(prob_matrix[h,a] for h in range(12) for a in range(12) if h == a)
    final_a = sum(prob_matrix[h,a] for h in range(12) for a in range(12) if h < a)
    
    p_u25 = sum(prob_matrix[h,a] for h in range(3) for a in range(3) if h+a <= 2)
    p_o25 = 1.0 - p_u25
    
    # Scores
    indices = np.unravel_index(np.argsort(prob_matrix, axis=None)[::-1], prob_matrix.shape)
    top_scores = []
    for i in range(5):
        h, a = int(indices[0][i]), int(indices[1][i])
        top_scores.append({
            'score': f"{h}-{a}",
            'probability': round(float(prob_matrix[h, a]), 3)
        })
    
    return {
        'success': True,
        'predictions': {
            '1x2': {
                'result': ['Away Win', 'Draw', 'Home Win'][int(np.argmax([final_a, final_d, final_h]))],
                'probabilities': {
                    'away_win': round(float(final_a), 3),
                    'draw': round(float(final_d), 3),
                    'home_win': round(float(final_h), 3)
                },
                'contribution': f"ML {ML_WEIGHT*100:.0f}% / Poisson {(1-ML_WEIGHT)*100:.0f}%"
            },
            'over_under': {
                'prediction': 'Over 2.5' if p_o25 > 0.5 else 'Under 2.5',
                'probabilities': {
                    'under': round(float(p_u25), 3),
                    'over': round(float(p_o25), 3)
                }
            },
            'score': {
                'home': int(top_scores[0]['score'].split('-')[0]),
                'away': int(top_scores[0]['score'].split('-')[1]),
                'total': round(float(xg_h + xg_a), 1),
                'home_expected': round(float(xg_h), 2),
                'away_expected': round(float(xg_a), 2),
                'top_match_scores': top_scores
            }
        },
        'metadata': {
            'features_used': len(feature_names),
            'features_missing': len(missing),
            'method': 'Dual Hybrid Scaling'
        }
    }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
