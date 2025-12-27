# src/processor.py
import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self):
        pass

    def clean_names(self, name):
        """Removes asterisks and extra whitespace from team names."""
        if pd.isna(name):
            return name
        return str(name).replace('*', '').strip()

    def clean_schedule(self, df):
        """Cleans schedule data."""
        # Standardize column names
        df = df.rename(columns={
            'Date': 'date', 'Visitor/Neutral': 'visitor', 
            'PTS': 'visitor_pts', 'Home/Neutral': 'home', 
            'PTS.1': 'home_pts'
        })
        df['date'] = pd.to_datetime(df['date'])
        
        # CLEAN TEAM NAMES HERE
        df['visitor'] = df['visitor'].apply(self.clean_names)
        df['home'] = df['home'].apply(self.clean_names)
        
        # Create Target: 1 if Home Wins, 0 if Visitor Wins
        # Ensure points are numeric
        df['visitor_pts'] = pd.to_numeric(df['visitor_pts'], errors='coerce')
        df['home_pts'] = pd.to_numeric(df['home_pts'], errors='coerce')
        
        # Filter completed games (games with scores)
        completed_games = df.dropna(subset=['visitor_pts', 'home_pts']).copy()
        
        completed_games['home_win'] = np.where(completed_games['home_pts'] > completed_games['visitor_pts'], 1, 0)
        
        return completed_games

    def merge_stats(self, schedule_df, advanced_stats_df):
        """
        Merges advanced stats. 
        """
        # CLEAN STATS TEAM NAMES HERE
        advanced_stats_df = advanced_stats_df.copy()
        advanced_stats_df['Team'] = advanced_stats_df['Team'].apply(self.clean_names)
        
        # Select relevant features
        # Ensure we are using float for stats
        cols_to_convert = ['Pace', 'ORtg', 'DRtg', 'NRtg']
        for col in cols_to_convert:
            advanced_stats_df[col] = pd.to_numeric(advanced_stats_df[col], errors='coerce')

        features = advanced_stats_df[['Team'] + cols_to_convert]
        
        # Merge for Visitor
        merged = schedule_df.merge(features, left_on='visitor', right_on='Team', how='inner')
        merged = merged.rename(columns={'Pace': 'v_pace', 'ORtg': 'v_ortg', 'DRtg': 'v_drtg', 'NRtg': 'v_nrtg'})
        
        # Merge for Home
        merged = merged.merge(features, left_on='home', right_on='Team', how='inner', suffixes=('_v', '_h'))
        merged = merged.rename(columns={'Pace': 'h_pace', 'ORtg': 'h_ortg', 'DRtg': 'h_drtg', 'NRtg': 'h_nrtg'})
        
        # Drop the extra 'Team' columns from the merge
        merged = merged.drop(columns=['Team_v', 'Team_h'], errors='ignore')
        
        return merged

    def prepare_features(self, df):
        """Selects numerical features for the model."""
        feature_cols = ['v_pace', 'v_ortg', 'v_drtg', 'v_nrtg', 
                        'h_pace', 'h_ortg', 'h_drtg', 'h_nrtg']
        X = df[feature_cols]
        y = df['home_win']
        return X, y