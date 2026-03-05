import pandas as pd
import os
from datetime import datetime
import json

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

MATCH_DATA_PATH = os.path.join(DATA_DIR, "match_data.csv")
MATCH_INFO_PATH = os.path.join(DATA_DIR, "match_info.csv")
SEASON_DATA_PATH = os.path.join(DATA_DIR, "season.csv")

def get_team_mapping():
    """Builds a mapping of team names to their internal IDs."""
    if not os.path.exists(MATCH_DATA_PATH):
        return {}
    
    df = pd.read_csv(MATCH_DATA_PATH)
    mapping = {}
    
    # Map h and team_h
    for _, row in df.iterrows():
        mapping[row['team_h']] = row['h']
        mapping[row['team_a']] = row['a']
        
    return mapping

def import_matches(external_csv_path):
    """
    Imports matches from an external CSV and appends to internal databases.
    Expected columns (minimum): date, team_h, team_a, h_goals, a_goals, league
    Optional columns: h_xg, a_xg, h_shot, a_shot, h_shot_on_target, a_shot_on_target
    """
    print(f"\n--- IMPORTING EXTERNAL DATA FROM {os.path.basename(external_csv_path)} ---")
    
    if not os.path.exists(external_csv_path):
        print(f"[ERROR] External file not found: {external_csv_path}")
        return

    df_ext = pd.read_csv(external_csv_path)
    if df_ext.empty:
        print("[SKIP] External file is empty.")
        return

    # 1. Load existing data and mapping
    df_mdata = pd.read_csv(MATCH_DATA_PATH)
    df_minfo = pd.read_csv(MATCH_INFO_PATH)
    df_season = pd.read_csv(SEASON_DATA_PATH)
    team_map = get_team_mapping()

    # 2. Preparation for appending
    next_id = df_mdata['id'].max() + 1
    next_sched = df_mdata['schedule_id'].max() + 1
    next_season_id = df_season['id'].max() + 1
    
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    new_mdata_list = []
    new_minfo_list = []
    new_season_list = []

    count = 0
    for idx, row in df_ext.iterrows():
        # Clean team names
        t_h = row['team_h'].strip()
        t_a = row['team_a'].strip()
        
        # Get IDs (or generate if unknown)
        h_id = team_map.get(t_h)
        if not h_id:
            h_id = f"manual:{t_h.replace(' ', '_').lower()}"
            print(f"[NEW TEAM] {t_h} -> {h_id}")
            team_map[t_h] = h_id
            
        a_id = team_map.get(t_a)
        if not a_id:
            a_id = f"manual:{t_a.replace(' ', '_').lower()}"
            print(f"[NEW TEAM] {t_a} -> {a_id}")
            team_map[t_a] = a_id

        # Handle optional columns with defaults
        h_goals = int(row['h_goals'])
        a_goals = int(row['a_goals'])
        h_xg = float(row.get('h_xg', 1.0))
        a_xg = float(row.get('a_xg', 1.0))
        h_shot = int(row.get('h_shot', 10))
        a_shot = int(row.get('a_shot', 10))
        h_sot = int(row.get('h_shot_on_target', 4))
        a_sot = int(row.get('a_shot_on_target', 4))
        league = row.get('league', 'Unknown')
        date_str = row['date']
        
        try:
            dt_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            try:
                dt_obj = datetime.strptime(date_str, "%Y-%m-%d")
                date_str = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
            except:
                print(f"[ERROR] Date format invalid for match {t_h} vs {t_a}: {date_str}")
                continue

        # Create Match Data entry
        new_mdata = {
            "id": next_id + count,
            "schedule_id": next_sched + count,
            "flash_score_match_id": f"EXT_{next_id + count}",
            "h": h_id, "a": a_id,
            "date": date_str,
            "league_id": "manual",
            "season": "manual",
            "h_goals": h_goals, "a_goals": a_goals,
            "team_h": t_h, "team_a": t_a,
            "h_xg": h_xg, "a_xg": a_xg,
            "league": league,
            "h_shot": h_shot, "a_shot": a_shot,
            "h_shot_on_target": h_sot, "a_shot_on_target": a_sot,
            "created_at": now_str, "updated_at": now_str
        }
        new_mdata_list.append(new_mdata)

        # Create Match Info entry
        new_minfo = {
            "id": next_id + count,
            "schedule_id": next_sched + count,
            "flash_score_match_id": f"EXT_{next_id + count}",
            "is_result": 1,
            "datetime": date_str,
            "h_id": h_id, "h_title": t_h,
            "a_id": a_id, "a_short_title": t_a[:3].upper(),
            "goals_h": h_goals, "goals_a": a_goals,
            "xg_h": h_xg, "xg_a": a_xg,
            "created_at": now_str, "updated_at": now_str
        }
        new_minfo_list.append(new_minfo)

        # Create Season entries (both sides)
        res_h = "w" if h_goals > a_goals else "d" if h_goals == a_goals else "l"
        res_a = "l" if h_goals > a_goals else "d" if h_goals == a_goals else "w"
        
        new_season_list.append({
            "id": next_season_id + (count * 2),
            "schedule_id": next_sched + count,
            "competitor_id": h_id, "title": t_h,
            "year": dt_obj.year, "h_a": "h",
            "xg": h_xg, "xga": a_xg,
            "scored": h_goals, "missed": a_goals,
            "result": res_h, "date": date_str,
            "wins": 1 if res_h == "w" else 0,
            "draws": 1 if res_h == "d" else 0,
            "loses": 1 if res_h == "l" else 0,
            "pts": 3 if res_h == "w" else 1 if res_h == "d" else 0,
            "created_at": now_str, "updated_at": now_str
        })
        
        new_season_list.append({
            "id": next_season_id + (count * 2) + 1,
            "schedule_id": next_sched + count,
            "competitor_id": a_id, "title": t_a,
            "year": dt_obj.year, "h_a": "a",
            "xg": a_xg, "xga": h_xg,
            "scored": a_goals, "missed": h_goals,
            "result": res_a, "date": date_str,
            "wins": 1 if res_a == "w" else 0,
            "draws": 1 if res_a == "d" else 0,
            "loses": 1 if res_a == "l" else 0,
            "pts": 3 if res_a == "w" else 1 if res_a == "d" else 0,
            "created_at": now_str, "updated_at": now_str
        })

        count += 1

    # 3. Append and save
    if new_mdata_list:
        df_mdata = pd.concat([df_mdata, pd.DataFrame(new_mdata_list)], ignore_index=True)
        df_minfo = pd.concat([df_minfo, pd.DataFrame(new_minfo_list)], ignore_index=True)
        df_season = pd.concat([df_season, pd.DataFrame(new_season_list)], ignore_index=True)
        
        df_mdata.to_csv(MATCH_DATA_PATH, index=False)
        df_minfo.to_csv(MATCH_INFO_PATH, index=False)
        df_season.to_csv(SEASON_DATA_PATH, index=False)
        print(f"[SUCCESS] Appended {count} matches to databases.")
    else:
        print("[SKIP] No valid matches to import.")

if __name__ == "__main__":
    # Generate Template if it doesn't exist
    template_path = os.path.join(BASE_DIR, "external_data_template.csv")
    if not os.path.exists(template_path):
        template_df = pd.DataFrame(columns=["date", "team_h", "team_a", "h_goals", "a_goals", "league", "h_xg", "a_xg"])
        template_df.loc[0] = ["2024-03-01 19:45:00", "Arsenal FC", "Chelsea FC", 2, 0, "Premier League", 1.8, 0.5]
        template_df.to_csv(template_path, index=False)
        print(f"[INFO] Created template at {template_path}")

    # Example usage:
    # import_matches("path_to_your_friend_data.csv")
    print("Usage: import_matches('path/to/csv')")
