import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import joblib
from scipy.stats import poisson
import os
import datetime
import xgboost # Ensure XGBoost is available for joblib loading
from model_v3.config import WINDOW_SIZE

# ==========================================
# 0. HELPER FUNCTIONS FOR ENSEMBLES
# ==========================================
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

# ==========================================
# 1. SETTINGS
# ==========================================

MODEL_PATH = os.path.join("model_v3", "football_model_v3.joblib")
DATA_PATH = os.path.join("model_v3", "global_features_v4.csv")

# --- FALLBACK CONSTANTS ---
ML_WEIGHT = 0.60  # Manual weight (Optimized via Backtest)
GLOBAL_FALLBACKS = {
    'goals': 1.2, 'xg': 1.28, 'shot': 11.4, 
    'goals_against': 1.2, 'xg_against': 1.28, 'shot_against': 11.4
}
FORM_FALLBACKS = {
    'recent_wins': 0.4, 'recent_draws': 0.2, 'recent_losses': 0.4, 
    'recent_goals_avg': 1.4, 'recent_conceded_avg': 1.4, 'recent_clean_sheets': 0.2
}

class PredictorAppV3:
    def __init__(self, root):
        self.root = root
        self.root.title("Football Match Predictor v3.0 (New Data)")
        self.root.geometry("600x750")
        
        self.model_pkg = None
        self.history_df = None
        self.leagues = []
        self.teams_by_league = {}
        
        # UI Elements
        self.create_widgets()
        
        # Load Resources
        self.load_resources()

    def create_widgets(self):
        input_frame = ttk.LabelFrame(self.root, text="Match Details", padding="10")
        input_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(input_frame, text="League:").grid(row=0, column=0, sticky="w", pady=5)
        self.league_cb = ttk.Combobox(input_frame, state="readonly")
        self.league_cb.grid(row=0, column=1, sticky="ew", pady=5)
        self.league_cb.bind("<<ComboboxSelected>>", self.update_teams)
        
        ttk.Label(input_frame, text="Home Team:").grid(row=1, column=0, sticky="w", pady=5)
        self.home_team_cb = ttk.Combobox(input_frame, state="readonly")
        self.home_team_cb.grid(row=1, column=1, sticky="ew", pady=5)
        
        ttk.Label(input_frame, text="Away Team:").grid(row=2, column=0, sticky="w", pady=5)
        self.away_team_cb = ttk.Combobox(input_frame, state="readonly")
        self.away_team_cb.grid(row=2, column=1, sticky="ew", pady=5)
        
        ttk.Label(input_frame, text="Date (YYYY-MM-DD):").grid(row=3, column=0, sticky="w", pady=5)
        self.date_entry = ttk.Entry(input_frame)
        self.date_entry.grid(row=3, column=1, sticky="ew", pady=5)
        self.date_entry.insert(0, datetime.date.today().strftime("%Y-%m-%d"))
        
        self.predict_btn = ttk.Button(input_frame, text="PREDICT RESULT", command=self.run_prediction)
        self.predict_btn.grid(row=4, column=0, columnspan=2, pady=15, sticky="ew")

        input_frame.columnconfigure(1, weight=1)

        output_frame = ttk.LabelFrame(self.root, text="Prediction Analysis (v3)", padding="10")
        output_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.result_text = tk.Text(output_frame, height=20, width=60, font=("Consolas", 10))
        self.result_text.pack(fill="both", expand=True)

    def load_resources(self):
        self.result_text.insert("end", f"Attempting to load model {MODEL_PATH}...\n")
        self.root.update()
        
        if not os.path.exists(MODEL_PATH):
            self.result_text.insert("end", f"Warning: Model file {MODEL_PATH} not found. Please train model v3 first.\n")
            return
            
        try:
            self.model_pkg = joblib.load(MODEL_PATH)
            
            if os.path.exists(DATA_PATH):
                self.history_df = pd.read_csv(DATA_PATH)
                self.history_df["date"] = pd.to_datetime(self.history_df["date"])
                
                # Organize Leagues and Teams
                if "league_name" in self.history_df.columns:
                    league_col = "league_name"
                elif "league" in self.history_df.columns:
                    league_col = "league"
                else:
                    league_col = None
                     
                if league_col:
                    grouped = self.history_df.groupby(league_col)
                    for league, group in grouped:
                        teams = sorted(set(group["team_h"].unique()) | set(group["team_a"].unique()))
                        self.teams_by_league[str(league)] = teams
                    self.leagues = sorted(self.teams_by_league.keys())
                else:
                    teams = sorted(set(self.history_df["team_h"].unique()) | set(self.history_df["team_a"].unique()))
                    self.teams_by_league["All Leagues"] = teams
                    self.leagues = ["All Leagues"]
                
                self.league_cb['values'] = self.leagues
                if self.leagues:
                    self.league_cb.current(0)
                    self.update_teams(None)
                    
                self.result_text.insert("end", "Resources loaded successfully!\n")
            else:
                self.result_text.insert("end", f"Warning: Data file {DATA_PATH} not found.\n")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load resources: {str(e)}")

    def update_teams(self, event):
        league = self.league_cb.get()
        teams = self.teams_by_league.get(league, [])
        self.home_team_cb['values'] = teams
        self.away_team_cb['values'] = teams
        if teams:
            self.home_team_cb.set('')
            self.away_team_cb.set('')

    def get_venue_stats(self, team, venue_type):
        """Get venue-specific stats (home-only or away-only) from match history."""
        df = self.history_df
        if df is None or df.empty:
            return None
        
        if venue_type == "home":
            matches = df[df["team_h"] == team].sort_values("date")
            goals_col, xg_col, ga_col = "h_goals", "h_xg", "a_goals"
        else:
            matches = df[df["team_a"] == team].sort_values("date")
            goals_col, xg_col, ga_col = "a_goals", "a_xg", "h_goals"
        
        min_games = 5
        if len(matches) < min_games:
            return None
        
        recent = matches.tail(WINDOW_SIZE)
        return {
            "venue_goals_for": recent[goals_col].mean(),
            "venue_goals_against": recent[ga_col].mean(),
            "venue_xg_for": recent[xg_col].mean() if xg_col in recent.columns else recent[goals_col].mean(),
        }

    def get_latest_rolling_stats(self, team):
        df = self.history_df
        # Find team match history
        team_matches = df[(df["team_h"] == team) | (df["team_a"] == team)].sort_values("date")
        
        window = WINDOW_SIZE
        if len(team_matches) >= window:
            last_match = team_matches.iloc[-1]
            stats = {}
            is_home = last_match["team_h"] == team
            prefix = "h_" if is_home else "a_"
            metrics = ["goals", "xg", "shot", "goals_against", "xg_against", "shot_against"]
            for m in metrics:
                col = f"{prefix}roll_{m}"
                # Handle naming variance
                if col not in last_match:
                    if m == "shot": col = f"{prefix}roll_shots_for"
                    elif m == "shot_against": col = f"{prefix}roll_shots_against"
                    elif m == "goals": col = f"{prefix}roll_goals_for"
                    elif m == "xg": col = f"{prefix}roll_xg_for"
                
                stats[f"roll_{m}"] = last_match.get(col, GLOBAL_FALLBACKS.get(m, 0))
            
            # New Volatility features
            stats["roll_goals_std"] = last_match.get(f"{prefix}roll_goals_std", 1.1)
            stats["roll_high_score_rate"] = last_match.get(f"{prefix}roll_high_score_rate", 0.2)
            
            return stats
        
        # Fallback to Season Lookup embedded in model
        if self.model_pkg and "season_lookup" in self.model_pkg:
            # Need competitor_id and year
            # We search for the team name in history to find id
            match_row = team_matches.tail(1) if not team_matches.empty else df[df["team_h"]==team].tail(1)
            if not match_row.empty:
                t_id = str(int(match_row["h"].values[0])) if match_row["team_h"].values[0] == team else str(int(match_row["a"].values[0]))
                year = str(datetime.date.today().year) # Default to current year
                lookup = self.model_pkg["season_lookup"]
                if (year, t_id) in lookup:
                    s = lookup[(year, t_id)]
                    return {
                        "roll_goals": s['scored'], "roll_xg": s['xg'], "roll_shot": 11.5,
                        "roll_goals_against": s['missed'], "roll_xg_against": s['xga'], "roll_shot_against": 11.5,
                        "roll_goals_std": 1.1, "roll_high_score_rate": 0.2
                    }
                # Try previous year if not found
                elif (str(int(year)-1), t_id) in lookup:
                    s = lookup[(str(int(year)-1), t_id)]
                    return {
                        "roll_goals": s['scored'], "roll_xg": s['xg'], "roll_shot": 11.5,
                        "roll_goals_against": s['missed'], "roll_xg_against": s['xga'], "roll_shot_against": 11.5,
                        "roll_goals_std": 1.1, "roll_high_score_rate": 0.2
                    }

        res = {f"roll_{m}": v for m, v in GLOBAL_FALLBACKS.items()}
        res.update({"roll_goals_std": 1.1, "roll_high_score_rate": 0.2})
        return res


    def get_recent_form(self, team, n=WINDOW_SIZE):
        df = self.history_df
        team_matches = df[(df["team_h"] == team) | (df["team_a"] == team)].sort_values("date")
        if team_matches.empty:
            return FORM_FALLBACKS.copy()
        
        recent = team_matches.tail(n)
        n_matches = len(recent)
        wins, draws, losses, goals, conceded, clean_sheets = 0, 0, 0, 0, 0, 0
        
        for _, match in recent.iterrows():
            if match["team_h"] == team:
                h_g, a_g = match["h_goals"], match["a_goals"]
                goals += h_g; conceded += a_g
                if h_g > a_g: wins += 1
                elif h_g == a_g: draws += 1
                else: losses += 1
                if a_g == 0: clean_sheets += 1
            else:
                h_g, a_g = match["h_goals"], match["a_goals"]
                goals += a_g; conceded += h_g
                if a_g > h_g: wins += 1
                elif a_g == h_g: draws += 1
                else: losses += 1
                if h_g == 0: clean_sheets += 1
                
        stats = {
            'recent_wins': wins / n_matches,
            'recent_draws': draws / n_matches,
            'recent_losses': losses / n_matches,
            'recent_goals_avg': goals / n_matches,
            'recent_conceded_avg': conceded / n_matches,
            'recent_clean_sheets': clean_sheets / n_matches
        }
        return stats

    def get_h2h_stats(self, h_team, a_team):
        df = self.history_df
        mask = ((df["team_h"] == h_team) & (df["team_a"] == a_team)) | \
               ((df["team_h"] == a_team) & (df["team_a"] == h_team))
        h2h_matches = df[mask]
        
        if h2h_matches.empty:
            return {
                "h2h_home_wins": 0.33, "h2h_draws": 0.33, "h2h_away_wins": 0.33,
                "h2h_avg_goals_home": 1.4, "h2h_avg_goals_away": 1.2
            }
        
        wins_h, draws, wins_a = 0, 0, 0
        goals_h, goals_a = 0, 0
        
        for _, row in h2h_matches.iterrows():
            if row["team_h"] == h_team:
                goals_h += row["h_goals"]; goals_a += row["a_goals"]
                if row["result"] == 2: wins_h += 1
                elif row["result"] == 1: draws += 1
                else: wins_a += 1
            else:
                goals_h += row["a_goals"]; goals_a += row["h_goals"]
                if row["result"] == 2: wins_a += 1 
                elif row["result"] == 1: draws += 1
                else: wins_h += 1 
        
        n = len(h2h_matches)
        return {
            "h2h_home_wins": wins_h / n, "h2h_draws": draws / n, "h2h_away_wins": wins_a / n,
            "h2h_avg_goals_home": goals_h / n, "h2h_avg_goals_away": goals_a / n
        }

    def run_prediction(self):
        h_team = self.home_team_cb.get()
        a_team = self.away_team_cb.get()
        league = self.league_cb.get()
        date_str = self.date_entry.get()
        prediction_date = pd.to_datetime(date_str)
        
        if not h_team or not a_team or not self.model_pkg:
            messagebox.showwarning("Warning", "Ready state incomplete or teams not selected.")
            return
            
        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", f"Analyzing {h_team} vs {a_team} (Model v4)...\n")
        self.root.update()
        
        h_stats = self.get_latest_rolling_stats(h_team)
        a_stats = self.get_latest_rolling_stats(a_team)
        h2h = self.get_h2h_stats(h_team, a_team)
        
        # Build Feature Vector (v4 style)
        row = {}
        row.update(h2h)
        
        # 1. Rolling Features
        metrics = ["goals", "xg", "shot", "goals_against", "xg_against", "shot_against"]
        for m in metrics:
            h_val = h_stats.get(f"roll_{m}", 0)
            a_val = a_stats.get(f"roll_{m}", 0)
            
            if "against" not in m:
                key_base = "shots_for" if m == "shot" else f"{m}_for"
            else:
                key_base = "shots_against" if m == "shot_against" else m
            
            row[f"h_roll_{key_base}"] = h_val
            row[f"a_roll_{key_base}"] = a_val
            row[f"rel_{key_base}"] = h_val - a_val

        row["h_roll_goals_std"] = h_stats.get("roll_goals_std", 1.1)
        row["a_roll_goals_std"] = a_stats.get("roll_goals_std", 1.1)
        row["h_roll_high_score_rate"] = h_stats.get("roll_high_score_rate", 0.2)
        row["a_roll_high_score_rate"] = a_stats.get("roll_high_score_rate", 0.2)

        # 2. V4-Specific Features (Elo, Fatigue, Venue)
        state = self.model_pkg.get("v4_state", {}).get(league, {})
        h_elo = state.get("elo", {}).get(h_team, 1500.0)
        a_elo = state.get("elo", {}).get(a_team, 1500.0)
        row["h_elo"] = h_elo
        row["a_elo"] = a_elo
        row["rel_elo"] = h_elo - a_elo
        
        # Calculate dynamic rest days (Temporal)
        h_last_date = state.get("last_date", {}).get(h_team)
        a_last_date = state.get("last_date", {}).get(a_team)
        
        # Fallback to most recent date in history if state is missing
        if not h_last_date:
            h_m = self.history_df[self.history_df["team_h"]==h_team].sort_values("date")
            if not h_m.empty: h_last_date = h_m.iloc[-1]["date"]
        if not a_last_date:
            a_m = self.history_df[self.history_df["team_a"]==a_team].sort_values("date")
            if not a_m.empty: a_last_date = a_m.iloc[-1]["date"]

        try:
            h_rest = (prediction_date - pd.to_datetime(h_last_date)).days if h_last_date else 15
            a_rest = (prediction_date - pd.to_datetime(a_last_date)).days if a_last_date else 15
        except:
            h_rest, a_rest = 15, 15
            
        row["h_rest"] = min(max(h_rest, 0), 20)
        row["a_rest"] = min(max(a_rest, 0), 20)

        # Venue-specific rolling goals
        h_venue = self.get_venue_stats(h_team, "home")
        a_venue = self.get_venue_stats(a_team, "away")
        row["h_venue_roll_goals"] = h_venue["venue_goals_for"] if h_venue else h_stats.get("roll_goals", 1.2)
        row["a_venue_roll_goals"] = a_venue["venue_goals_for"] if a_venue else a_stats.get("roll_goals", 1.2)
        
        # Directional Matchup Features (v4.1 - Anti-Leakage)
        row["h_att_v_a_def"] = h_stats.get("roll_goals", 1.3) - a_stats.get("roll_goals_against", 1.3)
        row["a_att_v_h_def"] = a_stats.get("roll_goals", 1.3) - h_stats.get("roll_goals_against", 1.3)
        row["rel_att_v_def"] = row["h_att_v_a_def"] - row["a_att_v_h_def"]
        row["xg_matchup_h"] = h_stats.get("roll_xg", 1.3) - a_stats.get("roll_xg_against", 1.3)
        row["xg_matchup_a"] = a_stats.get("roll_xg", 1.3) - h_stats.get("roll_xg_against", 1.3)
        row["goal_balance"] = (row["h_roll_goals_for"] - row["h_roll_goals_against"]) - (row["a_roll_goals_for"] - row["a_roll_goals_against"])
        row["h2h_draw_rate"] = h2h.get("h2h_draws", 0.33)

        input_df = pd.DataFrame([row])
        feature_names = self.model_pkg["feature_names"]
        for f in feature_names:
            if f not in input_df.columns: input_df[f] = 0.0
        input_df = input_df[feature_names]
        
        if self.model_pkg is None:
            messagebox.showerror("Error", "Model not loaded. Please ensure the model exists and restart.")
            return

        scaler = self.model_pkg["scaler"]
        X_scaled = scaler.transform(input_df)
        
        # Use best models if available (Winner from trainer comparison)
        best = self.model_pkg.get("best_models", {})
        winners_info = []
        
        if "1X2" in best:
            probs_1x2 = get_prediction(best["1X2"], X_scaled, is_proba=True)[0]
            winner_1x2 = best["1X2"].get("type", type(best["1X2"]).__name__) if type(best["1X2"]) is dict else type(best["1X2"]).__name__
            winners_info.append(f"1X2: {winner_1x2}")
        else:
            probs_rf = self.model_pkg["clf_rf"].predict_proba(X_scaled)[0]
            probs_gb = self.model_pkg["clf_gb"].predict_proba(X_scaled)[0]
            probs_1x2 = (probs_rf + probs_gb) / 2
        
        if "O/U 2.5" in best:
            probs_ou = get_prediction(best["O/U 2.5"], X_scaled, is_proba=True)[0]
            winner_ou = best["O/U 2.5"].get("type", type(best["O/U 2.5"]).__name__) if type(best["O/U 2.5"]) is dict else type(best["O/U 2.5"]).__name__
            winners_info.append(f"O/U: {winner_ou}")
        else:
            probs_ou_rf = self.model_pkg["clf_ou_rf"].predict_proba(X_scaled)[0]
            probs_ou_gb = self.model_pkg["clf_ou_gb"].predict_proba(X_scaled)[0]
            probs_ou = (probs_ou_rf + probs_ou_gb) / 2

        if "BTTS" in best:
            probs_btts = get_prediction(best["BTTS"], X_scaled, is_proba=True)[0]
            winner_btts = best["BTTS"].get("type", type(best["BTTS"]).__name__) if type(best["BTTS"]) is dict else type(best["BTTS"]).__name__
            winners_info.append(f"BTTS: {winner_btts}")
            prob_btts_yes = probs_btts[1]
        else:
            prob_btts_yes = 0.5
        
        if "Home Goals" in best:
            xg_h_raw = float(get_prediction(best["Home Goals"], X_scaled, is_proba=False)[0])
            xg_a_raw = float(get_prediction(best["Away Goals"], X_scaled, is_proba=False)[0])
            winner_reg = best["Home Goals"].get("type", type(best["Home Goals"]).__name__) if type(best["Home Goals"]) is dict else type(best["Home Goals"]).__name__
            winners_info.append(f"Goals: {winner_reg}")
        else:
            xg_h_raw = self.model_pkg["reg_h"].predict(X_scaled)[0]
            xg_a_raw = self.model_pkg["reg_a"].predict(X_scaled)[0]

        # --- DYNAMIC LAMBDA STRETCHING (Anti-Regression to Mean) ---
        # 1. Use O/U Confidence to stretch xg
        p_over = probs_ou[1]
        stretch_factor = 1.0
        if p_over > 0.65:
            stretch_factor = 1.0 + (p_over - 0.65) * 0.8 # up to ~1.28x boost
        elif p_over < 0.35:
            stretch_factor = 1.0 - (0.35 - p_over) * 0.4 # up to ~0.86x reduction
            
        # 2. Use Volatility (Standard Deviation) to expand potential
        h_vol = h_stats.get("roll_goals_std", 1.1)
        a_vol = a_stats.get("roll_goals_std", 1.1)
        vol_boost_h = max(0, (h_vol - 1.2) * 0.1)
        vol_boost_a = max(0, (a_vol - 1.2) * 0.1)

        xg_h = max(xg_h_raw * stretch_factor * (1 + vol_boost_h), 0.05)
        xg_a = max(xg_a_raw * stretch_factor * (1 + vol_boost_a), 0.05)
        
        # Dual Hybrid Logic (ML_WEIGHT imported from config)
        prob_matrix = np.outer(poisson.pmf(range(12), xg_h), poisson.pmf(range(12), xg_a))
        
        # --- HIGH SCORE CALIBRATION ---
        # If model expects a high score, bias the matrix towards (h>3 or a>3)
        if p_over > 0.7:
            # Shift weight from low scores to high scores
            for h in range(12):
                for a in range(12):
                    if h + a > 3: prob_matrix[h, a] *= 1.15
                    if h >= 3: prob_matrix[h, a] *= 1.1
                    if a >= 3: prob_matrix[h, a] *= 1.1

        p_h_poisson = max(np.sum(np.tril(prob_matrix, -1)), 0.0001)
        p_d_poisson = max(np.sum(np.diag(prob_matrix)), 0.0001)
        p_a_poisson = max(np.sum(np.triu(prob_matrix, 1)), 0.0001)
        
        target_h = (probs_1x2[2] * ML_WEIGHT) + (p_h_poisson * (1 - ML_WEIGHT))
        target_d = (probs_1x2[1] * ML_WEIGHT) + (p_d_poisson * (1 - ML_WEIGHT))
        target_a = (probs_1x2[0] * ML_WEIGHT) + (p_a_poisson * (1 - ML_WEIGHT))
        
        for h in range(12):
            for a in range(12):
                if h > a: prob_matrix[h, a] *= (target_h / p_h_poisson)
                elif h == a: prob_matrix[h, a] *= (target_d / p_d_poisson)
                else: prob_matrix[h, a] *= (target_a / p_a_poisson)
        
        prob_matrix /= np.sum(prob_matrix)
        
        # Pass 2: Over/Under
        p_o25_poisson = max(1.0 - sum(prob_matrix[h, a] for h in range(3) for a in range(3) if h+a <= 2), 0.0001)
        target_over = (probs_ou[1] * ML_WEIGHT) + (p_o25_poisson * (1 - ML_WEIGHT))
        target_under = 1.0 - target_over
        p_u25_poisson = 1.0 - p_o25_poisson
        
        for h in range(12):
            for a in range(12):
                if h + a > 2: prob_matrix[h, a] *= (target_over / p_o25_poisson)
                else: prob_matrix[h, a] *= (target_under / max(p_u25_poisson, 0.0001))
                    
        prob_matrix /= np.sum(prob_matrix)

        # Pass 3: BTTS
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

        
        final_h = sum(prob_matrix[h,a] for h in range(12) for a in range(12) if h > a)
        final_d = sum(prob_matrix[h,a] for h in range(12) for a in range(12) if h == a)
        final_a = sum(prob_matrix[h,a] for h in range(12) for a in range(12) if h < a)
        final_o25 = 1.0 - sum(prob_matrix[h,a] for h in range(3) for a in range(3) if h+a <= 2)

        # Pass 3: BTTS
        final_btts_yes = sum(prob_matrix[h,a] for h in range(1, 12) for a in range(1, 12))
        
        # Pass 4: Asian Handicap (Approximate based on prob_matrix)
        def get_ah_prob(handicap):
            # handicap is for home team (e.g., -0.5, +0.5, 0)
            p_win = sum(prob_matrix[h,a] for h in range(12) for a in range(12) if h + handicap > a)
            p_push = sum(prob_matrix[h,a] for h in range(12) for a in range(12) if h + handicap == a)
            p_loss = sum(prob_matrix[h,a] for h in range(12) for a in range(12) if h + handicap < a)
            return p_win, p_push, p_loss

        ah_lines = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
        ah_results = {}
        for line in ah_lines:
            ah_results[line] = get_ah_prob(line)

        # Display Results
        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", f"PREDICTION: {h_team} vs {a_team}\n")
        self.result_text.insert("end", f"Models used: {', '.join(winners_info)}\n")
        self.result_text.insert("end", "="*50 + "\n")
        self.result_text.insert("end", f"Date: {date_str}\n")
        self.result_text.insert("end", "="*50 + "\n\n")
        
        self.result_text.insert("end", f"Probabilities:\n")
        self.result_text.insert("end", f"  Home Win: {final_h*100:.1f}%\n")
        self.result_text.insert("end", f"  Draw:     {final_d*100:.1f}%\n")
        self.result_text.insert("end", f"  Away Win: {final_a*100:.1f}%\n")
        
        # Predicted Label
        if final_h > final_d and final_h > final_a:
            pred_label = "CHỦ NHÀ THẮNG (HOME WIN)"
        elif final_a > final_h and final_a > final_d:
            pred_label = "ĐỘI KHÁCH THẮNG (AWAY WIN)"
        else:
            pred_label = "HÒA (DRAW)"
        self.result_text.insert("end", f"  => Dự đoán: {pred_label}\n\n")
        
        self.result_text.insert("end", f"Over/Under 2.5:\n")
        self.result_text.insert("end", f"  Over:  {final_o25*100:.1f}%\n")
        self.result_text.insert("end", f"  Under: {(1-final_o25)*100:.1f}%\n\n")

        self.result_text.insert("end", f"BTTS (Both Teams To Score):\n")
        self.result_text.insert("end", f"  Yes: {final_btts_yes*100:.1f}%\n")
        self.result_text.insert("end", f"  No:  {(1-final_btts_yes)*100:.1f}%\n\n")

        self.result_text.insert("end", f"Asian Handicap (Home):\n")
        best_line = 0
        min_diff = 1.0
        for line, (w, p, l) in ah_results.items():
            # Find the line closest to 50/50 win/loss
            diff = abs(w - l)
            if diff < min_diff:
                min_diff = diff
                best_line = line
            
            sign = "+" if line > 0 else ""
            self.result_text.insert("end", f"  AH {sign}{line}: Win {w*100:.1f}% | Push {p*100:.1f}% | Loss {l*100:.1f}%\n")
        
        sign = "+" if best_line > 0 else ""
        self.result_text.insert("end", f"\n  -> Recommended AH Line: {sign}{best_line}\n\n")
        
        self.result_text.insert("end", "Most Likely Scores (Consensus-Aware):\n")
        scores_list = []
        for h in range(7):
            for a in range(7):
                p = prob_matrix[h, a]
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
                    'rank_score': p * weight
                })
        
        scores_list.sort(key=lambda x: x['rank_score'], reverse=True)
        for i in range(5):
            s = scores_list[i]
            self.result_text.insert("end", f"  {s['score']} ({s['probability']*100:.1f}%)\n")
        
        # --- SMART SCORE SYNC (Representative Scores) ---
        self.result_text.insert("end", "\nSmart Suggestions (Aligned with Trends):\n")
        
        # Find best score for specific outcomes
        def get_best_score_for_condition(cond_fn):
            best_s = (0,0)
            max_p = -1.0
            for h in range(12):
                for a in range(12):
                    if cond_fn(h, a):
                        if prob_matrix[h, a] > max_p:
                            max_p = prob_matrix[h, a]
                            best_s = (h, a)
            return best_s, max_p

        # 1. Best Score if OVER 2.5
        s_over, p_over_val = get_best_score_for_condition(lambda h, a: h + a > 2.5)
        # 2. Best Score if UNDER 2.5
        s_under, p_under_val = get_best_score_for_condition(lambda h, a: h + a <= 2.5)
        
        # Determine which one to highlight as "Primary Recommendation"
        if final_o25 > 0.5:
             self.result_text.insert("end", f"  -> [OVER 2.5 Focus] : {s_over[0]}-{s_over[1]} ({p_over_val*100:.1f}%)\n")
        else:
             self.result_text.insert("end", f"  -> [UNDER 2.5 Focus]: {s_under[0]}-{s_under[1]} ({p_under_val*100:.1f}%)\n")

        # 3. Trend by Match Result
        if final_h > final_a and final_h > final_d:
            s_win, p_win_val = get_best_score_for_condition(lambda h, a: h > a)
            self.result_text.insert("end", f"  -> [Home Win Trend] : {s_win[0]}-{s_win[1]}\n")
        elif final_a > final_h and final_a > final_d:
            s_win, p_win_val = get_best_score_for_condition(lambda h, a: h < a)
            self.result_text.insert("end", f"  -> [Away Win Trend] : {s_win[0]}-{s_win[1]}\n")
        else:
            s_draw, p_draw_val = get_best_score_for_condition(lambda h, a: h == a)
            self.result_text.insert("end", f"  -> [Draw Trend]     : {s_draw[0]}-{s_draw[1]}\n")
            
        self.result_text.see("end")

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictorAppV3(root)
    root.mainloop()
