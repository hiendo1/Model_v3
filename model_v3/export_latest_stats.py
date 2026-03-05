import pandas as pd
import json
import os
from collections import defaultdict

# Paths (dynamic, relative to this script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_CSV = os.path.join(BASE_DIR, "global_features_v4.csv")
OUTPUT_JSON = os.path.join(os.path.dirname(BASE_DIR), "api_files", "latest_stats.json")

def export_stats():
    if not os.path.exists(FEATURES_CSV):
        print(f"Error: {FEATURES_CSV} not found. Run training first.")
        return

    print(f"Loading {FEATURES_CSV}...")
    df = pd.read_csv(FEATURES_CSV)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    team_stats = {}
    h2h_lookup = {}

    # 1. Pre-calculate absolute last match date for every team across ALL leagues
    # (Required for correct rest day calculations regardless of which league form is used)
    print("Calculating global last match dates...")
    global_last_dates = {}
    for team in set(df['team_h'].unique()) | set(df['team_a'].unique()):
        team_m = df[(df['team_h'] == team) | (df['team_a'] == team)]
        if not team_m.empty:
            global_last_dates[team] = str(team_m.iloc[-1]['date'].date())

    team_stats = {}
    h2h_lookup = {}

    # 2. Extract Latest Rolling Stats for each (team, league) pair
    # Use a combination of team and league because form can differ between competitions
    team_league_pairs = df.groupby(['team_h', 'league']).size().index.tolist() + \
                        df.groupby(['team_a', 'league']).size().index.tolist()
    team_league_pairs = sorted(list(set(team_league_pairs)))
    
    print(f"Processing {len(team_league_pairs)} team-league combinations...")

    for team, league in team_league_pairs:
        # Get matches for this team IN THIS LEAGUE
        team_matches = df[((df['team_h'] == team) | (df['team_a'] == team)) & (df['league'] == league)]
        if team_matches.empty:
            continue
            
        last_match = team_matches.iloc[-1]
        is_home = last_match['team_h'] == team
        prefix = "h_" if is_home else "a_"
        
        # v4 metrics
        metrics = ['goals_for', 'goals_against', 'xg_for', 'xg_against', 'shots_for', 'shots_against', 
                   'goals_std', 'high_score_rate', 'venue_roll_goals']
        
        stats = {
            'league': str(league),
            'team_name': str(team),
            'last_league_match_date': str(last_match['date'].date()),
            'last_match_date': global_last_dates.get(team) # Global rest day reference
        }

        for m in metrics:
            col = f"{prefix}roll_{m}"
            if col in last_match:
                stats[f"roll_{m}"] = float(last_match[col])
        
        # Unique key for API lookup
        team_stats[f"{team}|{league}"] = stats

    # 2. Extract Latest H2H for every pair
    # We group by the sorted team pair
    df['pair'] = df.apply(lambda x: "-".join(sorted([str(x['team_h']), str(x['team_a'])])), axis=1)
    
    unique_pairs = df['pair'].unique()
    print(f"Processing {len(unique_pairs)} H2H pairs...")
    
    for pair in unique_pairs:
        pair_matches = df[df['pair'] == pair]
        last_h2h = pair_matches.iloc[-1]
        
        # H2H features from the CSV: h2h_home_wins, h2h_draws, h2h_away_wins, h2h_avg_goals_home, h2h_avg_goals_away, h2h_draw_rate
        h2h_fields = ['h2h_home_wins', 'h2h_draws', 'h2h_away_wins', 'h2h_avg_goals_home', 'h2h_avg_goals_away', 'h2h_draw_rate']
        
        # IMPORTANT: H2H is direction-sensitive in the CSV (Home vs Away)
        # We store it for the specific names listed in that match
        h_team = last_h2h['team_h']
        a_team = last_h2h['team_a']
        
        stats = {f: float(last_h2h[f]) for f in h2h_fields if f in last_h2h}
        h2h_lookup[f"{h_team}|{a_team}"] = stats

    # 3. Save to JSON
    output = {
        "team_stats": team_stats,
        "h2h_stats": h2h_lookup,
        "metadata": {
            "last_updated": str(df['date'].max()),
            "total_teams": len(team_stats),
            "total_pairs": len(h2h_lookup)
        }
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Successfully exported stats to {OUTPUT_JSON}")

if __name__ == "__main__":
    export_stats()
