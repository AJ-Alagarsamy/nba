import pandas as pd
import os
from datetime import datetime

class OddsProvider:
    def __init__(self, csv_path=None):
        self.csv_path = csv_path
    
    def get_todays_odds(self):
        """Get today's NBA games and moneyline odds from CSV file."""
        try:
            # If no CSV path provided, use sample data
            if self.csv_path is None or not os.path.exists(self.csv_path):
                print("CSV file not found. Using sample data.")
                return self.get_sample_odds()
            
            # Load from CSV file
            games_df = pd.read_csv(self.csv_path)
            
            # Check required columns
            required_columns = ['game_date', 'away_team', 'home_team', 'away_odds', 'home_odds']
            missing_columns = [col for col in required_columns if col not in games_df.columns]
            
            if missing_columns:
                print(f"Missing required columns in CSV: {missing_columns}")
                print("Using sample data instead.")
                return self.get_sample_odds()
            
            # Get today's date
            today = datetime.now().strftime("%Y-%m-%d")
            
            # Filter for today's games
            todays_games = games_df[games_df['game_date'] == today]
            
            if len(todays_games) == 0:
                print(f"No games found for today ({today})")
                # Try to get any games if today's date doesn't match
                print("Showing all games from CSV...")
                todays_games = games_df
            
            print(f"Loaded {len(todays_games)} games from CSV")
            
            # Rename columns to match expected format in main.py
            todays_games = todays_games.rename(columns={
                'away_team': 'visitor',
                'home_team': 'home',
                'away_odds': 'visitor_moneyline',
                'home_odds': 'home_moneyline'
            })
            
            # Select only the columns we need
            todays_games = todays_games[['visitor', 'home', 'visitor_moneyline', 'home_moneyline']]
            
            return todays_games
            
        except Exception as e:
            print(f"Error loading odds from CSV: {e}")
            print("Using sample data instead.")
            return self.get_sample_odds()
    
    def get_sample_odds(self):
        """Return sample odds data for testing."""
        today = datetime.now().strftime("%Y-%m-%d")
        
        games_data = {
            'game_date': [today] * 9,
            'away_team': ['Dallas Mavericks', 'Denver Nuggets', 'Phoenix Suns', 
                         'New York Knicks', 'Indiana Pacers', 'Milwaukee Bucks',
                         'Cleveland Cavaliers', 'Brooklyn Nets', 'Utah Jazz'],
            'home_team': ['Sacramento Kings', 'Orlando Magic', 'New Orleans Pelicans',
                         'Atlanta Hawks', 'Miami Heat', 'Chicago Bulls',
                         'Houston Rockets', 'Minnesota Timberwolves', 'San Antonio Spurs'],
            'away_odds': [-150, -180, -200, -250, +305, +130, +150, +305, +800],
            'home_odds': [+130, +160, +170, +205, -375, -150, -170, -375, -1350]
        }
        
        games_df = pd.DataFrame(games_data)
        
        # Rename columns to match expected format
        games_df = games_df.rename(columns={
            'away_team': 'visitor',
            'home_team': 'home',
            'away_odds': 'visitor_moneyline',
            'home_odds': 'home_moneyline'
        })
        
        # Select only the columns we need
        games_df = games_df[['visitor', 'home', 'visitor_moneyline', 'home_moneyline']]
        
        return games_df
    
    def american_to_decimal(self, odds):
        """
        Convert American odds to decimal odds.
        
        Args:
            odds: American odds (positive or negative integer)
            
        Returns:
            float: Decimal odds
        """
        try:
            odds = float(odds)
            
            if odds > 0:
                decimal = (odds / 100) + 1
            elif odds < 0:
                decimal = (100 / abs(odds)) + 1
            else:
                decimal = 2.0  # Even money
                
            # Cap extreme values to prevent mathematical errors
            if decimal > 101:  # For odds like +10000
                decimal = 101
            elif decimal < 1.01:  # For odds like -10000
                decimal = 1.01
                
            return decimal
        except (ValueError, TypeError, ZeroDivisionError):
            print(f"Warning: Invalid odds value '{odds}', using 2.0 (even money)")
            return 2.0
    
    def american_to_probability(self, odds):
        """
        Convert American odds to implied probability.
        
        Args:
            odds: American odds
            
        Returns:
            float: Implied probability (0 to 1)
        """
        decimal = self.american_to_decimal(odds)
        if decimal > 0:
            prob = 1 / decimal
            # Ensure probability is between 0 and 1
            return max(0.01, min(0.99, prob))
        return 0.5