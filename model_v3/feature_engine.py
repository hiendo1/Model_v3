import pandas as pd
import numpy as np
from collections import defaultdict
from config import WINDOW_SIZE

def get_season_label(date):
    """Determine season based on date."""
    year = date.year
    month = date.month
    if month >= 8:
        return f"{year}-{year+1}"
    else:
        return f"{year-1}-{year}"

def create_features_v4(df, season_stats_lookup):
    """Enriched feature engineering (v4) with Elo, Venue-Specific form, and Fatigue."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "h"]) # Stable sort for consistent data splits
    
    # Basic targets
    df["result"] = np.where(df["h_goals"] > df["a_goals"], 2, 
                   np.where(df["h_goals"] == df["a_goals"], 1, 0))
    df["total_goals"] = df["h_goals"] + df["a_goals"]
    df["over_2.5"] = (df["total_goals"] > 2.5).astype(int)
    df["btts"] = ((df["h_goals"] > 0) & (df["a_goals"] > 0)).astype(int)
    
    metrics = ["goals", "xg", "shot", "goals_against", "xg_against", "shot_against"]
    
    team_history = defaultdict(list)
    h2h_history = defaultdict(list)
    elo_ratings = defaultdict(lambda: 1500.0)
    last_game_date = {} # Track fatigue
    
    features_rows = []
    
    for idx, row in df.iterrows():
        h_team, a_team = row["team_h"], row["team_a"]
        h_id, a_id = row["h"], row["a"]
        date = row["date"]
        year = date.year
        
        # 0. Fatigue (Rest Days)
        h_rest = (date - last_game_date[h_team]).days if h_team in last_game_date else 15
        a_rest = (date - last_game_date[a_team]).days if a_team in last_game_date else 15
        h_rest = min(h_rest, 15); a_rest = min(a_rest, 15)

        # 1. Elo Ratings
        h_elo = elo_ratings[h_team]
        a_elo = elo_ratings[a_team]
        
        # 2. H2H Hist
        pair_key = tuple(sorted([h_team, a_team]))
        past_h2h = h2h_history[pair_key]
        if not past_h2h:
            h2h_feat = {"h2h_home_wins": 0.33, "h2h_draws": 0.33, "h2h_away_wins": 0.33, "h2h_avg_goals_home": 1.3, "h2h_avg_goals_away": 1.2}
        else:
            wins_h, draws, wins_a, g_h, g_a = 0, 0, 0, 0, 0
            for m in past_h2h:
                if m['h_team'] == h_team:
                    g_h += m['h_g']; g_a += m['a_g']
                    if m['res'] == 2: wins_h += 1
                    elif m['res'] == 1: draws += 1
                    else: wins_a += 1
                else:
                    g_h += m['a_g']; g_a += m['h_g']
                    if m['res'] == 2: wins_a += 1
                    elif m['res'] == 1: draws += 1
                    else: wins_h += 1
            n = len(past_h2h)
            h2h_feat = {"h2h_home_wins": wins_h/n, "h2h_draws": draws/n, "h2h_away_wins": wins_a/n, "h2h_avg_goals_home": g_h/n, "h2h_avg_goals_away": g_a/n}
        h2h_history[pair_key].append({'h_team': h_team, 'a_team': a_team, 'h_g': row['h_goals'], 'a_g': row['a_goals'], 'res': row['result']})

        # 3. Rolling Stats & Venue Form
        def get_team_metrics(history, team_id, current_year, is_home=True):
            window = WINDOW_SIZE
            if len(history) >= window:
                recent = history[-window:]
                res = {f"roll_{m}": np.mean([match[m] for match in recent]) for m in metrics}
                
                # Venue-specific form (Home-only or Away-only)
                venue_matches = [m for m in history if m["is_home"] == is_home]
                if len(venue_matches) >= 3:
                    res["venue_roll_goals"] = np.mean([m["goals"] for m in venue_matches[-3:]])
                else:
                    res["venue_roll_goals"] = res["roll_goals"]

                goals_list = [match["goals"] for match in recent]
                res["roll_goals_std"] = np.std(goals_list)
                res["roll_high_score_rate"] = len([g for g in goals_list if g >= 3]) / window
                return res
            
            # Fallback: Use season-shifted stats from data_loader
            season_label = get_season_label(pd.Timestamp(f"{current_year}-01-01") if not hasattr(date, 'month') else date)
            if (season_label, team_id) in season_stats_lookup:
                s = season_stats_lookup[(season_label, team_id)]
                return {
                    "roll_goals": s['scored'], "roll_xg": s['xg'], "roll_shot": s.get('shot', 11.5),
                    "roll_goals_against": s['missed'], "roll_xg_against": s['xga'], "roll_shot_against": s.get('shot_against', 11.5),
                    "roll_goals_std": s.get('goals_std', 1.1), "roll_high_score_rate": s.get('high_score_rate', 0.2),
                    "venue_roll_goals": s['scored']
                }
            return None

        h_feats = get_team_metrics(team_history[h_team], h_id, year, is_home=True)
        a_feats = get_team_metrics(team_history[a_team], a_id, year, is_home=False)

        # 4. Update Elo & State
        def update_elo(h_elo, a_elo, result):
            k = 32
            expected_h = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
            score_h = 1.0 if result == 2 else (0.5 if result == 1 else 0.0)
            new_h = h_elo + k * (score_h - expected_h)
            new_a = a_elo + (h_elo - new_h)
            return new_h, new_a

        def safe_val(v): return v if pd.notnull(v) else 0
        h_match = {"goals": row["h_goals"], "xg": safe_val(row["h_xg"]), "shot": safe_val(row["h_shot"]), 
                   "goals_against": row["a_goals"], "xg_against": safe_val(row["a_xg"]), "shot_against": safe_val(row["a_shot"]), "is_home": True}
        a_match = {"goals": row["a_goals"], "xg": safe_val(row["a_xg"]), "shot": safe_val(row["a_shot"]), 
                   "goals_against": row["h_goals"], "xg_against": safe_val(row["h_xg"]), "shot_against": safe_val(row["h_shot"]), "is_home": False}
        team_history[h_team].append(h_match)
        team_history[a_team].append(a_match)
        elo_ratings[h_team], elo_ratings[a_team] = update_elo(h_elo, a_elo, row["result"])
        last_game_date[h_team] = date; last_game_date[a_team] = date

        if h_feats and a_feats:
            r_dict = row.to_dict()
            r_dict.update(h2h_feat)
            r_dict.update({"h_elo": h_elo, "a_elo": a_elo, "h_rest": h_rest, "a_rest": a_rest, "rel_elo": h_elo - a_elo})
            for k, v in h_feats.items():
                if k in ["roll_goals_std", "roll_high_score_rate", "venue_roll_goals"]: r_dict[f"h_{k}"] = v; continue
                name = k.replace("goals", "goals_for") if "goals" in k and "against" not in k else \
                       k.replace("xg", "xg_for") if "xg" in k and "against" not in k else \
                       k.replace("shot", "shots_for") if "shot" in k and "against" not in k else \
                       k.replace("shot_against", "shots_against").replace("goals_against", "goals_against").replace("xg_against", "xg_against")
                r_dict[f"h_{name}"] = v
            for k, v in a_feats.items():
                if k in ["roll_goals_std", "roll_high_score_rate", "venue_roll_goals"]: r_dict[f"a_{k}"] = v; continue
                name = k.replace("goals", "goals_for") if "goals" in k and "against" not in k else \
                       k.replace("xg", "xg_for") if "xg" in k and "against" not in k else \
                       k.replace("shot", "shots_for") if "shot" in k and "against" not in k else \
                       k.replace("shot_against", "shots_against").replace("goals_against", "goals_against").replace("xg_against", "xg_against")
                r_dict[f"a_{name}"] = v
            r_dict["rel_goals_for"] = h_feats["roll_goals"] - a_feats["roll_goals"]
            r_dict["rel_xg_for"] = h_feats["roll_xg"] - a_feats["roll_xg"]
            r_dict["rel_shots_for"] = h_feats["roll_shot"] - a_feats["roll_shot"]
            r_dict["h_att_v_a_def"] = h_feats["roll_goals"] - a_feats["roll_goals_against"]
            r_dict["a_att_v_h_def"] = a_feats["roll_goals"] - h_feats["roll_goals_against"]
            r_dict["rel_att_v_def"] = r_dict["h_att_v_a_def"] - r_dict["a_att_v_h_def"]
            
            r_dict["xg_matchup_h"] = h_feats["roll_xg"] - a_feats["roll_xg_against"]
            r_dict["xg_matchup_a"] = a_feats["roll_xg"] - h_feats["roll_xg_against"]
            r_dict["goal_balance"] = (h_feats["roll_goals"] - h_feats["roll_goals_against"]) - (a_feats["roll_goals"] - a_feats["roll_goals_against"])
            r_dict["h2h_draw_rate"] = h2h_feat.get("h2h_draws", 0.33)
            features_rows.append(r_dict)

    state = {
        "elo": dict(elo_ratings),
        "last_date": last_game_date
    }
    return pd.DataFrame(features_rows).dropna(), state



