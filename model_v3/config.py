import os

# --- MODEL PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FEATURES_CSV = os.path.join(BASE_DIR, "global_features_v3.csv")
MODEL_JOBLIB = os.path.join(BASE_DIR, "football_model_v3.joblib")

# --- DATA SPLIT SETTINGS ---
# Curated list of high-transparency, professional-grade leagues
TARGET_LEAGUES = [
    # Top 5 Europe
    "Premier League", "Ligue 1", "LaLiga", "Serie A", "Bundesliga",
    # European Major/Secondary
    "Championship", "Liga Portugal", "Eredivisie", "Super Lig", "Premiership",
    "2. Bundesliga", "Serie B", "LaLiga 2", "Pro League",
    # Elite International
    "UEFA Champions League", "UEFA Europa League", "UEFA Conference League",
    "J1 League", "Saudi Pro League", "A-League"
]

TRAIN_SPLIT_RATIO = 1

# --- MODEL HYPERPARAMETERS (Random Forest) ---
MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 15,
    "min_samples_split": 10,
    "random_state": 42,
    "n_jobs": -1
}

# --- MODEL HYPERPARAMETERS (XGBoost) ---
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0
}

# --- FEATURE SETTINGS ---
METRICS = ["goals", "xg", "shot", "goals_against", "xg_against", "shot_against"]
WINDOW_SIZE = 5
