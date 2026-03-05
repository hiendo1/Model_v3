import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import joblib
from scipy.stats import poisson
import os
import sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import datetime

# ==========================================
# 1. LOGIC (Embedded for Stability)
# ==========================================

MODEL_PATH = "football_model_v2.joblib"
DATA_PATH = "global_features_v2.csv"

# --- FALLBACK CONSTANTS (Global Medians for Unknown Teams) ---
GLOBAL_FALLBACKS = {
    'goals': 1.2, 'xg': 1.28, 'shot': 11.4, 'ppda': 10.3, 'deep': 5.2, 
    'goals_against': 1.2, 'xg_against': 1.28, 'shot_against': 11.4
}
FORM_FALLBACKS = {
    'recent_wins': 0.4, 'recent_draws': 0.2, 'recent_losses': 0.4, 
    'recent_goals_avg': 1.4, 'recent_conceded_avg': 1.4, 'recent_clean_sheets': 0.2
}

class PredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Football Match Predictor v2.0")
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
        # Frame for Inputs
        input_frame = ttk.LabelFrame(self.root, text="Match Details", padding="10")
        input_frame.pack(fill="x", padx=10, pady=5)
        
        # League Selection
        ttk.Label(input_frame, text="League:").grid(row=0, column=0, sticky="w", pady=5)
        self.league_cb = ttk.Combobox(input_frame,state="readonly")
        self.league_cb.grid(row=0, column=1, sticky="ew", pady=5)
        self.league_cb.bind("<<ComboboxSelected>>", self.update_teams)
        
        # Home Team
        ttk.Label(input_frame, text="Home Team:").grid(row=1, column=0, sticky="w", pady=5)
        self.home_team_cb = ttk.Combobox(input_frame, state="readonly")
        self.home_team_cb.grid(row=1, column=1, sticky="ew", pady=5)
        
        # Away Team
        ttk.Label(input_frame, text="Away Team:").grid(row=2, column=0, sticky="w", pady=5)
        self.away_team_cb = ttk.Combobox(input_frame, state="readonly")
        self.away_team_cb.grid(row=2, column=1, sticky="ew", pady=5)
        
        # Date
        ttk.Label(input_frame, text="Date (YYYY-MM-DD):").grid(row=3, column=0, sticky="w", pady=5)
        self.date_entry = ttk.Entry(input_frame)
        self.date_entry.grid(row=3, column=1, sticky="ew", pady=5)
        self.date_entry.insert(0, datetime.date.today().strftime("%Y-%m-%d"))
        
        # Predict Button
        self.predict_btn = ttk.Button(input_frame, text="PREDICT RESULT", command=self.run_prediction)
        self.predict_btn.grid(row=4, column=0, columnspan=2, pady=15, sticky="ew")

        input_frame.columnconfigure(1, weight=1)

        # Output Text Area
        output_frame = ttk.LabelFrame(self.root, text="Prediction Analysis", padding="10")
        output_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.result_text = tk.Text(output_frame, height=20, width=60, font=("Consolas", 10))
        self.result_text.pack(fill="both", expand=True)

    def load_resources(self):
        self.result_text.insert("end", "Loading model and history data...\n")
        self.root.update()
        
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("Error", f"Model file {MODEL_PATH} not found!")
            return
            
        try:
            self.model_pkg = joblib.load(MODEL_PATH)
            
            if os.path.exists(DATA_PATH):
                self.history_df = pd.read_csv(DATA_PATH)
                self.history_df["date"] = pd.to_datetime(self.history_df["date"])
                
                # --- Organize Leagues and Teams ---
                # Check if 'league_name' exists, if not use 'league_id' or infer
                if "league_name" in self.history_df.columns:
                     league_col = "league_name"
                elif "league_id" in self.history_df.columns:
                     league_col = "league_id"
                else:
                     league_col = None
                     
                if league_col:
                    grouped = self.history_df.groupby(league_col)
                    for league, group in grouped:
                        teams = sorted(set(group["team_h"].unique()) | set(group["team_a"].unique()))
                        self.teams_by_league[str(league)] = teams
                    self.leagues = sorted(self.teams_by_league.keys())
                else:
                    # Fallback: Just put "All Teams"
                    teams = sorted(set(self.history_df["team_h"].unique()) | set(self.history_df["team_a"].unique()))
                    self.teams_by_league["All Leagues"] = teams
                    self.leagues = ["All Leagues"]
                
                self.league_cb['values'] = self.leagues
                if self.leagues:
                    self.league_cb.current(0)
                    self.update_teams(None)
                    
                self.result_text.insert("end", "Ready!\n")
            else:
                self.result_text.insert("end", "Warning: History file not found.\n")
                
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

    # --- PREDICTION LOGIC COPY ---
    def get_latest_rolling_stats(self, team):
        df = self.history_df
        team_matches = df[(df["team_h"] == team) | (df["team_a"] == team)].sort_values("date")
        
        if team_matches.empty:
            return {f"roll_{m}": v for m, v in GLOBAL_FALLBACKS.items()}
        
        last_match = team_matches.iloc[-1]
        stats = {}
        is_home = last_match["team_h"] == team
        prefix = "h_" if is_home else "a_"
        
        metrics = ["goals", "xg", "shot", "ppda", "deep", "goals_against", "xg_against", "shot_against"]
        for m in metrics:
            col = f"{prefix}roll_{m}"
            if col in last_match:
                stats[f"roll_{m}"] = last_match[col]
            else:
                if m in ["goals", "xg", "shot"] and f"{prefix}roll_{m}_for" in last_match:
                    stats[f"roll_{m}"] = last_match[f"{prefix}roll_{m}_for"]
                else:
                    stats[f"roll_{m}"] = GLOBAL_FALLBACKS.get(m, 0)
        return stats

    def get_recent_form(self, team, n=5):
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
        
        # If fewer than 3 matches, blend with FORM_FALLBACKS for stability
        if n_matches < 3:
            for k in stats:
                stats[k] = (stats[k] * 0.5) + (FORM_FALLBACKS[k] * 0.5)
                
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
                goals_h += row["h_goals"]
                goals_a += row["a_goals"]
                if row["result"] == 2: wins_h += 1
                elif row["result"] == 1: draws += 1
                else: wins_a += 1
            else:
                goals_h += row["a_goals"]
                goals_a += row["h_goals"]
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
        date_str = self.date_entry.get()
        
        if not h_team or not a_team:
            messagebox.showwarning("Input Error", "Please select both Home and Away teams.")
            return
            
        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", f"Analyzing {h_team} vs {a_team}...\n")
        self.root.update()
        
        # Logic
        h_stats = self.get_latest_rolling_stats(h_team)
        a_stats = self.get_latest_rolling_stats(a_team)
        h2h = self.get_h2h_stats(h_team, a_team)
        h_form = self.get_recent_form(h_team)
        a_form = self.get_recent_form(a_team)
        
        if not h_stats or not a_stats:
            self.result_text.insert("end", "Error: Insufficient data for selected teams.\n")
            return

        # Build Feature Vector
        row = {}
        row.update(h2h)
        metrics = ["goals", "xg", "shot", "ppda", "deep", "goals_against", "xg_against", "shot_against"]
        for m in metrics:
            h_val = h_stats.get(f"roll_{m}", 0)
            a_val = a_stats.get(f"roll_{m}", 0)
            
            # Exact naming transform to match training all_features
            if m in ["ppda", "deep"]: key_base = m
            elif "against" not in m:
                if m == "shot": key_base = "shots_for"
                else: key_base = f"{m}_for"
            else:
                if m == "shot_against": key_base = "shots_against"
                else: key_base = m
            
            row[f"h_roll_{key_base}"] = h_val
            row[f"a_roll_{key_base}"] = a_val
            row[f"rel_{key_base}"] = h_val - a_val

        # Add recent form features
        for k, v in h_form.items():
            row[f"h_{k}"] = v
        for k, v in a_form.items():
            row[f"a_{k}"] = v

        input_df = pd.DataFrame([row])
        feature_names = self.model_pkg["feature_names"]
        for f in feature_names:
            if f not in input_df.columns: input_df[f] = 0.0
        input_df = input_df[feature_names]
        
        # Predict
        scaler = self.model_pkg["scaler"]
        X_scaled = scaler.transform(input_df)
        
        # Ensemble predictions
        probs_rf = self.model_pkg["clf_rf"].predict_proba(X_scaled)[0]
        probs_gb = self.model_pkg["clf_gb"].predict_proba(X_scaled)[0]
        probs_1x2 = (probs_rf + probs_gb) / 2
        
        probs_ou_rf = self.model_pkg["clf_ou_rf"].predict_proba(X_scaled)[0]
        probs_ou_gb = self.model_pkg["clf_ou_gb"].predict_proba(X_scaled)[0]
        probs_ou = (probs_ou_rf + probs_ou_gb) / 2
        
        # Get raw xG and boost by 5%
        xg_h_raw = self.model_pkg["reg_h"].predict(X_scaled)[0]
        xg_a_raw = self.model_pkg["reg_a"].predict(X_scaled)[0]
        xg_h = xg_h_raw * 1.05
        xg_a = xg_a_raw * 1.05
        
        # --- SCORE HYBRIDIZATION (DUAL MATRIX SCALING) ---
        # Weight settings: 0.3 means 30% Machine Learning, 70% Poisson (xG)
        ML_WEIGHT = 0.3
        
        prob_matrix = np.outer(poisson.pmf(range(12), xg_h), poisson.pmf(range(12), xg_a))
        
        p_h_poisson = np.sum(np.tril(prob_matrix, -1))
        p_d_poisson = np.sum(np.diag(prob_matrix))
        p_a_poisson = np.sum(np.triu(prob_matrix, 1))
        
        # Avoid division by zero
        p_h_poisson = max(p_h_poisson, 0.0001)
        p_d_poisson = max(p_d_poisson, 0.0001)
        p_a_poisson = max(p_a_poisson, 0.0001)
        
        # --- PASS 1: Scale to match 1X2 Targets ---
        target_h = (probs_1x2[2] * ML_WEIGHT) + (p_h_poisson * (1 - ML_WEIGHT))
        target_d = (probs_1x2[1] * ML_WEIGHT) + (p_d_poisson * (1 - ML_WEIGHT))
        target_a = (probs_1x2[0] * ML_WEIGHT) + (p_a_poisson * (1 - ML_WEIGHT))
        
        for h in range(12):
            for a in range(12):
                if h > a: prob_matrix[h, a] *= (target_h / p_h_poisson)
                elif h == a: prob_matrix[h, a] *= (target_d / p_d_poisson)
                else: prob_matrix[h, a] *= (target_a / p_a_poisson)
        
        prob_matrix /= np.sum(prob_matrix)
        
        # --- PASS 2: Scale to match O/U Target ---
        p_o25_poisson = 1.0 - sum(prob_matrix[h, a] for h in range(3) for a in range(3) if h+a <= 2)
        p_u25_poisson = 1.0 - p_o25_poisson
        
        # probs_ou[1] is Over 2.5
        target_over = (probs_ou[1] * ML_WEIGHT) + (p_o25_poisson * (1 - ML_WEIGHT))
        target_under = 1.0 - target_over
        
        for h in range(12):
            for a in range(12):
                if h + a > 2:
                    prob_matrix[h, a] *= (target_over / max(p_o25_poisson, 0.0001))
                else:
                    prob_matrix[h, a] *= (target_under / max(p_u25_poisson, 0.0001))
                    
        prob_matrix /= np.sum(prob_matrix)
        
        # Derived O/U and display probs from adjusted matrix
        p_u25 = sum(prob_matrix[h,a] for h in range(3) for a in range(3) if h+a <= 2)
        p_o25 = 1.0 - p_u25
        p_o35 = 1.0 - sum(prob_matrix[h,a] for h in range(4) for a in range(4) if h+a <= 3)

        # Final probabilities for display (synced with scaled grid)
        final_h = sum(prob_matrix[h,a] for h in range(12) for a in range(12) if h > a)
        final_d = sum(prob_matrix[h,a] for h in range(12) for a in range(12) if h == a)
        final_a = sum(prob_matrix[h,a] for h in range(12) for a in range(12) if h < a)

        # --- OUTPUT ---
        self.result_text.insert("end", "="*50 + "\n")
        self.result_text.insert("end", f"PREDICTION: {h_team} vs {a_team}\n")
        self.result_text.insert("end", f"Date: {date_str}\n")
        self.result_text.insert("end", "="*50 + "\n\n")
        
        self.result_text.insert("end", f"[DUAL HYBRID] 1X2 Probabilities (ML {ML_WEIGHT*100:.0f}% / Poisson {(1-ML_WEIGHT)*100:.0f}%):\n")
        self.result_text.insert("end", f"  Home Win: {final_h*100:.1f}%\n")
        self.result_text.insert("end", f"  Draw:     {final_d*100:.1f}%\n")
        self.result_text.insert("end", f"  Away Win: {final_a*100:.1f}%\n\n")
        
        self.result_text.insert("end", f"Expected Goals (xG + 5% boost):\n")
        self.result_text.insert("end", f"  {h_team}: {xg_h:.2f}\n")
        self.result_text.insert("end", f"  {a_team}: {xg_a:.2f}\n\n")
        
        self.result_text.insert("end", "[DUAL HYBRID] Over/Under 2.5 (Blended):\n")
        self.result_text.insert("end", f"  Over:  {p_o25*100:.1f}%\n")
        self.result_text.insert("end", f"  Under: {p_u25*100:.1f}%\n")
        
        if p_o35 > 0.35:
             self.result_text.insert("end", f"  🔥 High Scoring Alert: {p_o35*100:.1f}% chance of > 3.5 Goals\n")
        
        self.result_text.insert("end", "\nMost Likely Scores (Hybrid Matrix):\n")
        indices = np.unravel_index(np.argsort(prob_matrix, axis=None)[::-1], prob_matrix.shape)
        for i in range(5):
            h_s, a_s = indices[0][i], indices[1][i]
            prob = prob_matrix[h_s, a_s]
            self.result_text.insert("end", f"  {h_s}-{a_s} ({prob*100:.1f}%)\n")
            
        self.result_text.see("end")

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictorApp(root)
    root.mainloop()
