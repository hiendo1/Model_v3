import pandas as pd
import joblib
import os
from data_loader import load_integrated_data
from feature_engine import create_features_v4
from model_trainer import train_and_evaluate, print_evaluation
from config import TARGET_LEAGUES

def run_main_pipeline():
    """Main entry point for training the football prediction model v3."""
    # 1. Load Data
    df_merged, season_lookup = load_integrated_data()
    
    # 2. Process Features
    print("--- 2. PROCESSING FEATURES ---")
    processed_matches = []
    matches_for_training = []
    league_states = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for league in df_merged["league"].unique():
        ldf = df_merged[df_merged["league"] == league]
        
        # 2.1 Calculate Features for ALL Valid Matches (for Stats Export)
        try:
            pdf, state = create_features_v4(ldf, season_lookup)
            
            # Any league that produces feature rows is valid for STATS (Universal Prediction)
            if not pdf.empty:
                processed_matches.append(pdf)
                league_states[str(league)] = state
            else:
                continue

            # 2.2 Filter for TRAINING (Strict Quality Control)
            null_xg = ldf["h_xg"].isnull().mean()
            
            # Skip training if xG data is poor
            if null_xg > 0.4:
                print(f"[STATS ONLY] {league}: {null_xg:.1%} missing xG (Filtered from training)")
                continue

            # Skip training if history is too short
            if len(pdf) < 100:
                print(f"[STATS ONLY] {league}: Only {len(pdf)} matches (Filtered from training)")
                continue

            # Elite/Target leagues check
            if not TARGET_LEAGUES or league in TARGET_LEAGUES:
                matches_for_training.append(pdf)
                print(f"[TRAIN + STATS] {league}: {len(pdf)} matches")
            else:
                print(f"[STATS ONLY] {league}: {len(pdf)} matches (Filtered from training)")

        except Exception as e:
            print(f"[ERROR] {league}: {e}")

    if not matches_for_training:
        print("No matches for training. Exiting.")
        return

    # 1. Global Features for Export (Includes everything valid)
    global_df_all = pd.concat(processed_matches, ignore_index=True).sort_values("date")
    features_csv_path = os.path.join(base_dir, "global_features_v4.csv")
    global_df_all.to_csv(features_csv_path, index=False)
    print(f"Generated features (all valid leagues) saved to {features_csv_path}")

    # 2. DataFrame for Training (Elite Leagues only)
    global_df_train = pd.concat(matches_for_training, ignore_index=True).sort_values("date")
    
    # 3. Train and Evaluate
    training_results = train_and_evaluate(global_df_train)
    
    # 4. Save and Report
    best = training_results["best_models"]
    
    final_pkg = {
        "best_models": best,
        "scaler": training_results["scaler"],
        "feature_names": training_results["features"],
        "season_lookup": season_lookup,
        "v4_state": league_states, # Saving the Elo/Date states per league
        "comparison": training_results["comparison"]
    }
    
    model_joblib_path = os.path.join(base_dir, "football_model_v3.joblib")
    joblib.dump(final_pkg, model_joblib_path)
    print(f"[SUCCESS] Model package saved to {model_joblib_path}")
    
    
    print_evaluation(training_results)
    print("[SUCCESS] Pipeline complete.")

if __name__ == "__main__":
    run_main_pipeline()
