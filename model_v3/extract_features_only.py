import pandas as pd
import os
from data_loader import load_integrated_data
from feature_engine import create_features_v4
from export_latest_stats import export_stats
from config import TARGET_LEAGUES


def extract_features():
    """Load data, compute features for all leagues, and export stats."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Load Data
    df_merged, season_lookup = load_integrated_data()

    # 2. Process Features (no training filter)
    print("--- 2. PROCESSING FEATURES (Extract Only) ---")
    processed_matches = []
    league_states = {}

    for league in df_merged["league"].unique():
        ldf = df_merged[df_merged["league"] == league]
        try:
            pdf, state = create_features_v4(ldf, season_lookup)
            if not pdf.empty:
                processed_matches.append(pdf)
                league_states[str(league)] = state
                print(f"  [OK] {league}: {len(pdf)} matches")
        except Exception as e:
            print(f"  [ERROR] {league}: {e}")

    if not processed_matches:
        print("No valid matches found. Exiting.")
        return

    # 3. Save global_features_v4.csv
    global_df = pd.concat(processed_matches, ignore_index=True).sort_values("date")
    features_csv_path = os.path.join(base_dir, "global_features_v4.csv")
    global_df.to_csv(features_csv_path, index=False)
    print(f"\n[OK] Features saved to {features_csv_path}")
    print(f"     Total rows: {len(global_df)} | Leagues: {global_df['league'].nunique()}")

    # 4. Export latest_stats.json
    print("\n--- 3. EXPORTING LATEST STATS ---")
    export_stats()

    print("\n[SUCCESS] Feature extraction complete (no training performed).")


if __name__ == "__main__":
    extract_features()
