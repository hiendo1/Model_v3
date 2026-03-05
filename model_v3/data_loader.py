import pandas as pd
import os

def get_season_label(date):
    """Aug-Dec Y, Jan-Jul Y+1 -> 'Y-(Y+1)'"""
    y, m = date.year, date.month
    return f"{y}-{y+1}" if m >= 8 else f"{y-1}-{y}"

def load_integrated_data():
    """Loads and merges match_data, match_info, and season data."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    
    match_data_path = os.path.join(data_dir, "match_data.csv")
    match_info_path = os.path.join(data_dir, "match_info.csv")
    season_data_path = os.path.join(data_dir, "season.csv")

    print("--- 1. LOADING & INTEGRATING DATA ---")
    df_data = pd.read_csv(match_data_path)
    df_info = pd.read_csv(match_info_path)
    df_season = pd.read_csv(season_data_path)
    
    print(f"Loaded: match_data ({len(df_data)}), match_info ({len(df_info)}), season ({len(df_season)})")

    df_data['date'] = pd.to_datetime(df_data['date'])
    df_data['season_label'] = df_data['date'].apply(get_season_label)
    
    # 1.1 Calculate Aggregate Stats PER SEASON
    # This represents the "Full Profile" of a team for a completed season
    def get_team_stats(df, attr_map, team_col):
        return df.groupby(['season_label', team_col]).agg(attr_map)

    h_stats = get_team_stats(df_data, {
        'h_goals': ['mean', 'std'], 'a_goals': 'mean',
        'h_xg': 'mean', 'a_xg': 'mean', 'h_shot': 'mean', 'a_shot': 'mean'
    }, 'h')
    
    a_stats = get_team_stats(df_data, {
        'a_goals': ['mean', 'std'], 'h_goals': 'mean',
        'a_xg': 'mean', 'h_xg': 'mean', 'a_shot': 'mean', 'h_shot': 'mean'
    }, 'a')
    
    df_data['is_high_score_h'] = (df_data['h_goals'] >= 3).astype(int)
    df_data['is_high_score_a'] = (df_data['a_goals'] >= 3).astype(int)
    h_hsr = df_data.groupby(['season_label', 'h'])['is_high_score_h'].mean()
    a_hsr = df_data.groupby(['season_label', 'a'])['is_high_score_a'].mean()

    # 1.2 Build Shifted Season Lookup (ANTI-LEAKAGE)
    # Match in Season S -> Uses Stats from Season S-1
    season_lookup = {}
    
    # Get sorted unique seasons
    all_seasons = sorted(df_data['season_label'].unique())
    
    # Map raw competitor data (from season.csv) by season label too
    df_season['season_label'] = df_season.apply(lambda x: f"{int(x['year'])}-{int(x['year'])+1}", axis=1)

    for i, current_s in enumerate(all_seasons):
        # We look for the PREVIOUS season to populate the fallback for current_s
        prev_s = all_seasons[i-1] if i > 0 else None
        
        # All teams in current data
        current_teams = set(df_data[df_data['season_label'] == current_s]['h']) | \
                        set(df_data[df_data['season_label'] == current_s]['a'])
        
        for t_id in current_teams:
            stats = {'shot': 11.5, 'shot_against': 11.5, 'goals_std': 1.1, 'high_score_rate': 0.2,
                     'xg': 1.3, 'xga': 1.3, 'scored': 1.3, 'missed': 1.3, 'pts': 1.0}
            
            if prev_s:
                # 1. Get Base Stats from season.csv for PREVIOUS season
                s_prev = df_season[(df_season['season_label'] == prev_s) & (df_season['competitor_id'] == t_id)]
                if not s_prev.empty:
                    stats.update({
                        'xg': s_prev['xg'].mean(), 'xga': s_prev['xga'].mean(),
                        'scored': s_prev['scored'].mean(), 'missed': s_prev['missed'].mean(),
                        'pts': s_prev['pts'].mean()
                    })

                # 2. Get Detailed Stats from Match History for PREVIOUS season
                try:
                    h_exists = (prev_s, t_id) in h_stats.index
                    a_exists = (prev_s, t_id) in a_stats.index
                    
                    if h_exists or a_exists:
                        h_count = len(df_data[(df_data['season_label'] == prev_s) & (df_data['h'] == t_id)])
                        a_count = len(df_data[(df_data['season_label'] == prev_s) & (df_data['a'] == t_id)])
                        total = h_count + a_count
                        
                        def weighted(h_val, a_val): return (h_val * h_count + a_val * a_count) / total

                        if h_exists and a_exists:
                            stats['shot'] = weighted(h_stats.loc[(prev_s, t_id), ('h_shot', 'mean')], a_stats.loc[(prev_s, t_id), ('a_shot', 'mean')])
                            stats['shot_against'] = weighted(h_stats.loc[(prev_s, t_id), ('a_shot', 'mean')], a_stats.loc[(prev_s, t_id), ('h_shot', 'mean')])
                            stats['goals_std'] = weighted(h_stats.loc[(prev_s, t_id), ('h_goals', 'std')], a_stats.loc[(prev_s, t_id), ('a_goals', 'std')])
                            stats['high_score_rate'] = weighted(h_hsr.loc[(prev_s, t_id)], a_hsr.loc[(prev_s, t_id)])
                except: pass

            # Store mapping: CURRENT season match will use PREVIOUS season profile
            season_lookup[(current_s, t_id)] = stats

    # Merge match data for main pipeline
    df_merged = pd.merge(
        df_data, 
        df_info[['schedule_id', 'datetime', 'h_title', 'a_short_title', 'xg_h', 'xg_a']], 
        on='schedule_id', how='left', suffixes=('', '_info')
    )
    
    return df_merged, season_lookup
