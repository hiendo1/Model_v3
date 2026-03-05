import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor
from config import MODEL_PARAMS, XGB_PARAMS, TRAIN_SPLIT_RATIO

def train_and_evaluate(global_df):
    """Logic for splitting data, training and comparing RF vs XGBoost models."""
    # Features
    h2h = ["h2h_home_wins", "h2h_draws", "h2h_away_wins", "h2h_avg_goals_home", "h2h_avg_goals_away"]
    rel = ["rel_goals_for", "rel_xg_for", "rel_shots_for", "rel_elo"]
    context = ["h_elo", "a_elo", "h_rest", "a_rest"]
    venue = ["h_venue_roll_goals", "a_venue_roll_goals"]
    roll = [f"{p}_roll_{m}" for p in ['h', 'a'] for m in ['goals_for', 'goals_against', 'xg_for', 'xg_against', 'shots_for', 'shots_against', 'goals_std', 'high_score_rate']]
    draw_feats = ["h_att_v_a_def", "a_att_v_h_def", "rel_att_v_def", "xg_matchup_h", "xg_matchup_a", "goal_balance", "h2h_draw_rate"]
    features = h2h + rel + context + venue + roll + draw_feats
    
    split = int(len(global_df) * TRAIN_SPLIT_RATIO)
    train, test = global_df.iloc[:split], global_df.iloc[split:]
    
    X_train, X_test = train[features], test[features]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    
    # Handle empty test set for production training
    if len(test) == 0:
        print("\n[INFO] 100% Training Split: Skipping evaluation and using best-known models.")
        X_test_s = np.array([]).reshape(0, len(features))
    else:
        X_test_s = scaler.transform(X_test)

    print("\n--- 3. TRAINING & COMPARING MODELS (RF vs XGB) ---")
    
    best_models = {}
    comparison_results = {}

    tasks = [
        {"name": "1X2", "target": "result", "type": "clf"},
        {"name": "O/U 2.5", "target": "over_2.5", "type": "clf"},
        {"name": "BTTS", "target": "btts", "type": "clf"},
        {"name": "Home Goals", "target": "h_goals", "type": "reg"},
        {"name": "Away Goals", "target": "a_goals", "type": "reg"}
    ]

    for task in tasks:
        print(f"Evaluating {task['name']}...")
        y_train = train[task['target']]
        if len(test) == 0:
            # --- PRODUCTION FIT (100% Data) ---
            # Based on anti-leakage benchmark (Feb 2027):
            # 1X2 -> Ensemble (won 4/6 splits, avg 48.4%)
            # O/U 2.5 -> RF (won 4/6, avg 54.5%)
            # BTTS -> RF (won 4/6, avg 52.1%)
            # Goals -> RF (won 5/6)
            print(f"Fitting production {task['name']} model...")
            
            if task['name'] == "1X2":
                # Ensemble for 1X2 only (best performer)
                rf_params = MODEL_PARAMS.copy()
                rf_params['class_weight'] = 'balanced'
                rf = RandomForestClassifier(**rf_params)
                rf.fit(X_train_s, y_train)
                
                xgb = XGBClassifier(**XGB_PARAMS)
                from sklearn.utils.class_weight import compute_sample_weight
                sw = compute_sample_weight('balanced', y_train)
                xgb.fit(X_train_s, y_train, sample_weight=sw)
                
                best_models[task['name']] = {"type": "ensemble", "rf": rf, "xgb": xgb, "weight_rf": 0.5, "weight_xgb": 0.5, "classes_": rf.classes_}
                comparison_results[task['name']] = {"winner": "Ensemble (Production)", "rf": 0, "xgb": 0, "ens": 0}
            
            else:
                # Random Forest for O/U, BTTS, and Goals (most stable)
                m = RandomForestClassifier(**MODEL_PARAMS) if task['type'] == "clf" else RandomForestRegressor(**MODEL_PARAMS)
                if task['type'] == "clf": m.set_params(class_weight='balanced')
                m.fit(X_train_s, y_train)
                
                classes_ = m.classes_ if task['type'] == "clf" else None
                best_models[task['name']] = {"type": "single", "model": m, "classes_": classes_}
                comparison_results[task['name']] = {"winner": "Random Forest (Production)", "rf": 0, "xgb": 0}
            continue

        y_test = test[task['target']]
        
        if task['type'] == "clf":
            # RF with class balancing for better draw prediction
            rf_params = MODEL_PARAMS.copy()
            rf_params['class_weight'] = 'balanced'
            rf = RandomForestClassifier(**rf_params)
            rf.fit(X_train_s, y_train)
            rf_preds = rf.predict(X_test_s)
            rf_acc = accuracy_score(y_test, rf_preds)
            
            # XGB with sample weights for class balance
            xgb_params = XGB_PARAMS.copy()
            xgb = XGBClassifier(**xgb_params)
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights = compute_sample_weight('balanced', y_train)
            xgb.fit(X_train_s, y_train, sample_weight=sample_weights)
            xgb_preds = xgb.predict(X_test_s)
            xgb_acc = accuracy_score(y_test, xgb_preds)
            
            # Soft Voting Ensemble
            rf_probs = rf.predict_proba(X_test_s)
            xgb_probs = xgb.predict_proba(X_test_s)
            ens_probs = (rf_probs + xgb_probs) / 2.0
            ens_preds_idx = np.argmax(ens_probs, axis=1)
            classes_ = rf.classes_
            ens_preds = [classes_[i] for i in ens_preds_idx]
            ens_acc = accuracy_score(y_test, ens_preds)
            
            accs = {"Random Forest": rf_acc, "XGBoost": xgb_acc, "Ensemble": ens_acc}
            best_name = max(accs, key=accs.get)
            
            if best_name == "Ensemble":
                best_models[task['name']] = {"type": "ensemble", "rf": rf, "xgb": xgb, "weight_rf": 0.5, "weight_xgb": 0.5, "classes_": classes_}
            else:
                best_models[task['name']] = {"type": "single", "model": rf if best_name == "Random Forest" else xgb, "classes_": classes_}
                
            comparison_results[task['name']] = {"winner": best_name, "rf": rf_acc, "xgb": xgb_acc, "ens": ens_acc}
        else:
            # RF
            rf = RandomForestRegressor(**MODEL_PARAMS)
            rf.fit(X_train_s, y_train)
            rf_preds = rf.predict(X_test_s)
            rf_mae = mean_absolute_error(y_test, rf_preds)
            
            # XGB
            xgb = XGBRegressor(**XGB_PARAMS)
            xgb.fit(X_train_s, y_train)
            xgb_preds = xgb.predict(X_test_s)
            xgb_mae = mean_absolute_error(y_test, xgb_preds)
            
            # Ensemble
            ens_preds = (rf_preds + xgb_preds) / 2.0
            ens_mae = mean_absolute_error(y_test, ens_preds)
            
            maes = {"Random Forest": rf_mae, "XGBoost": xgb_mae, "Ensemble": ens_mae}
            best_name = min(maes, key=maes.get)
            
            if best_name == "Ensemble":
                best_models[task['name']] = {"type": "ensemble", "rf": rf, "xgb": xgb, "weight_rf": 0.5, "weight_xgb": 0.5}
            else:
                best_models[task['name']] = {"type": "single", "model": rf if best_name == "Random Forest" else xgb}
                
            comparison_results[task['name']] = {"winner": best_name, "rf": rf_mae, "xgb": xgb_mae, "ens": ens_mae}

    return {
        "best_models": best_models,
        "comparison": comparison_results,
        "scaler": scaler,
        "features": features,
        "results": {
            "train": train, "test": test,
            "X_train_s": X_train_s, "X_test_s": X_test_s
        }
    }

def print_evaluation(training_results):
    """Prints a structured comparison and evaluation report."""
    comp = training_results["comparison"]
    print("\n==================================")
    print("--- MODEL COMPARISON & RESULTS ---")
    print("==================================\n")
    
    for task_name, res in comp.items():
        metric = "Accuracy" if "Goals" not in task_name else "MAE"
        print(f"[{task_name}] Winner: {res['winner']}")
        print(f"  RF {metric}:  {res['rf']:.4f}")
        print(f"  XGB {metric}: {res['xgb']:.4f}")
        if "ens" in res:
            print(f"  ENS {metric}: {res['ens']:.4f}")
        print("")

    # Detailed report for the winners
    train = training_results["results"]["train"]
    test = training_results["results"]["test"]
    X_test_s = training_results["results"]["X_test_s"]
    best = training_results["best_models"]
    
    if len(test) > 0:
        print("Detailed Classification Report (Best 1X2 Model):")
        best_1x2 = best["1X2"]
        if type(best_1x2) is dict and best_1x2.get("type") == "ensemble":
            probs = best_1x2["weight_rf"] * best_1x2["rf"].predict_proba(X_test_s) + best_1x2["weight_xgb"] * best_1x2["xgb"].predict_proba(X_test_s)
            preds = [best_1x2["classes_"][i] for i in np.argmax(probs, axis=1)]
        elif type(best_1x2) is dict and best_1x2.get("type") == "single":
            preds = best_1x2["model"].predict(X_test_s)
        else:
            preds = best_1x2.predict(X_test_s) # Backward compatibility
            
        print(classification_report(test["result"], preds, target_names=["Away", "Draw", "Home"], zero_division=0))
    else:
        print("Detailed report skipped (Production Training on 100% Data)")

    ratio = TRAIN_SPLIT_RATIO
    print(f"[INFO] Data Split: {ratio*100:.1f}% Training ({len(train)}), {(1-ratio)*100:.1f}% Testing ({len(test)})")
