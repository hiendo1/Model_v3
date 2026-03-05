import pandas as pd
import os
import shutil
from datetime import datetime
from football_model_v3 import run_main_pipeline
from export_latest_stats import export_stats

def update_all_databases(new_data_csv):
    """
    Takes a single CSV with new match results and updates:
    - match_data.csv
    - match_info.csv
    - season.csv
    Then runs the pipeline to update the model and API lookup.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    
    # 1. Load New Data
    print(f"--- UPDATING RAW DATABASES FROM {new_data_csv} ---")
    df_new = pd.read_csv(new_data_csv)
    if df_new.empty:
        print("New data file is empty. Skipping.")
        return

    # Helper to get next IDs
    def get_next_ids(file_path):
        df = pd.read_csv(file_path)
        next_id = df['id'].max() + 1 if not df.empty else 1
        next_schedule = df['schedule_id'].max() + 1 if not df.empty else 1000000
        return next_id, next_schedule

    # --- 2. UPDATE match_data.csv ---
    match_data_path = os.path.join(data_dir, "match_data.csv")
    df_mdata = pd.read_csv(match_data_path)
    next_id, next_sched = get_next_ids(match_data_path)
    
    new_mdata_rows = []
    for idx, row in df_new.iterrows():
        new_row = {
            "id": next_id + idx,
            "schedule_id": next_sched + idx,
            "flash_score_match_id": f"NEW_{next_id + idx}",
            "h": row["h_id"],
            "a": row["a_id"],
            "date": row["date"],
            "league_id": "manual_entry",
            "season": "manual_entry",
            "h_goals": row["h_goals"],
            "a_goals": row["a_goals"],
            "team_h": row["team_h"],
            "team_a": row["team_a"],
            "h_xg": row["h_xg"],
            "a_xg": row["a_xg"],
            "league": row["league"],
            "h_shot": row["h_shot"],
            "a_shot": row["a_shot"],
            "h_shot_on_target": row["h_shot_on_target"],
            "a_shot_on_target": row["a_shot_on_target"],
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        new_mdata_rows.append(new_row)
    
    df_mdata = pd.concat([df_mdata, pd.DataFrame(new_mdata_rows)], ignore_index=True)
    df_mdata.to_csv(match_data_path, index=False)
    print(f"[OK] Appended {len(df_new)} matches to match_data.csv")

    # --- 3. UPDATE match_info.csv ---
    match_info_path = os.path.join(data_dir, "match_info.csv")
    df_minfo = pd.read_csv(match_info_path)
    new_minfo_rows = []
    for idx, row in df_new.iterrows():
        new_row = {
            "id": next_id + idx,
            "schedule_id": next_sched + idx,
            "flash_score_match_id": f"NEW_{next_id + idx}",
            "is_result": 1,
            "datetime": row["date"],
            "h_id": row["h_id"],
            "h_title": row["team_h"],
            "a_id": row["a_id"],
            "a_short_title": row["team_a"][:3].upper(),
            "goals_h": row["h_goals"],
            "goals_a": row["a_goals"],
            "xg_h": row["h_xg"],
            "xg_a": row["a_xg"],
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        new_minfo_rows.append(new_row)
    df_minfo = pd.concat([df_minfo, pd.DataFrame(new_minfo_rows)], ignore_index=True)
    df_minfo.to_csv(match_info_path, index=False)
    print(f"[OK] Appended {len(df_new)} matches to match_info.csv")

    # --- 4. UPDATE season.csv ---
    season_path = os.path.join(data_dir, "season.csv")
    df_season = pd.read_csv(season_path)
    s_id = df_season['id'].max() + 1 if not df_season.empty else 1
    new_season_rows = []
    
    for idx, row in df_new.iterrows():
        # Home side
        new_season_rows.append({
            "id": s_id + (idx*2),
            "schedule_id": next_sched + idx,
            "competitor_id": row["h_id"],
            "title": row["team_h"],
            "year": datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S").year,
            "h_a": "h",
            "xg": row["h_xg"],
            "xga": row["a_xg"],
            "scored": row["h_goals"],
            "missed": row["a_goals"],
            "result": "w" if row["h_goals"] > row["a_goals"] else "d" if row["h_goals"] == row["a_goals"] else "l",
            "date": row["date"],
            "wins": 1 if row["h_goals"] > row["a_goals"] else 0,
            "draws": 1 if row["h_goals"] == row["a_goals"] else 0,
            "loses": 1 if row["h_goals"] < row["a_goals"] else 0,
            "pts": 3 if row["h_goals"] > row["a_goals"] else 1 if row["h_goals"] == row["a_goals"] else 0,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        # Away side
        new_season_rows.append({
            "id": s_id + (idx*2) + 1,
            "schedule_id": next_sched + idx,
            "competitor_id": row["a_id"],
            "title": row["team_a"],
            "year": datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S").year,
            "h_a": "a",
            "xg": row["a_xg"],
            "xga": row["h_xg"],
            "scored": row["a_goals"],
            "missed": row["h_goals"],
            "result": "w" if row["a_goals"] > row["h_goals"] else "d" if row["h_goals"] == row["a_goals"] else "l",
            "date": row["date"],
            "wins": 1 if row["a_goals"] > row["h_goals"] else 0,
            "draws": 1 if row["h_goals"] == row["a_goals"] else 0,
            "loses": 1 if row["a_goals"] < row["h_goals"] else 0,
            "pts": 3 if row["a_goals"] > row["h_goals"] else 1 if row["h_goals"] == row["a_goals"] else 0,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    df_season = pd.concat([df_season, pd.DataFrame(new_season_rows)], ignore_index=True)
    df_season.to_csv(season_path, index=False)
    print(f"[OK] Appended {len(new_season_rows)} records to season.csv")

    # --- 5. RUN PIPELINE ---
    print("\n--- TRIGGERING PIPELINE UPDATE ---")
    run_main_pipeline() # Retrains and updates global_features_v3.csv
    
    print("\n--- EXPORTING LATEST STATS ---")
    export_stats() # Generates latest_stats.json
    
    # --- 6. SYNC TO API FOLDER ---
    api_dir = os.path.join(os.path.dirname(base_dir), "api_files")
    if os.path.exists(api_dir):
        shutil.copy(os.path.join(base_dir, "football_model_v3.joblib"), api_dir)
        shutil.copy(os.path.join(base_dir, "latest_stats.json"), api_dir)
        print(f"\n[SUCCESS] Deployment files copied to {api_dir}")
        print("Ready for 'git push' to Render!")
    else:
        print("[WARNING] api_files directory not found. Manual sync required.")

if __name__ == "__main__":
    # Point this to your new data file
    target_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "new_results.csv")
    if os.path.exists(target_file):
        update_all_databases(target_file)
    else:
        print(f"File not found: {target_file}")
        print("Please create new_results.csv in the model_v3 folder first.")
