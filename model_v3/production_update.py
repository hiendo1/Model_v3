import os
import sys
import subprocess
from datetime import datetime

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXTERNAL_DATA_PATH = os.path.join(BASE_DIR, "new_results.csv")

def run_command(command, description):
    print(f"\n>>> {description}...")
    try:
        # Use sys.executable to ensure we use the same python environment
        result = subprocess.run([sys.executable] + command.split(), check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed:")
        print(e.stderr)
        return False

def main():
    print("="*50)
    print(f"FOOTBALL MODEL PRODUCTION UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)

    # 1. Import External Data (if file exists)
    if os.path.exists(EXTERNAL_DATA_PATH):
        print(f"\n[STEP 1] Found {EXTERNAL_DATA_PATH}. Importing...")
        # Since import_matches is a function, we could import it, 
        # but for simplicity in a standalone script, we'll run it as a module or just call the function if we import it.
        from import_external_data import import_matches
        import_matches(EXTERNAL_DATA_PATH)
        # Optional: Archive or delete the file after import
        # os.rename(EXTERNAL_DATA_PATH, EXTERNAL_DATA_PATH + ".bak")
    else:
        print("\n[STEP 1] No new_results.csv found. Skipping data import.")

    # 2. Retrain Model (with League Filtering)
    # This will use TARGET_LEAGUES from config.py automatically
    print("\n[STEP 2] Training Model (Applying League Filters from config.py)...")
    success = run_command("football_model_v3.py", "Model Training")
    if not success:
        print("[CRITICAL] Training failed. Aborting update.")
        return

    # 3. Export Latest Stats for API
    print("\n[STEP 3] Exporting Latest Stats (including last_match_date)...")
    success = run_command("export_latest_stats.py", "Stats Export")
    if not success:
        print("[CRITICAL] Stats export failed.")
        return

    print("\n" + "="*50)
    print("PRODUCTION UPDATE COMPLETE!")
    print("Next step: Restart your API or 'git push' to Render.")
    print("="*50)

if __name__ == "__main__":
    main()
