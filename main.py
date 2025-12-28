import pandas as pd
import numpy as np
from src.scraper import NBAStatScraper
from src.processor import DataProcessor
from src.model import NBAModel
from src.odds import OddsProvider
import difflib

def find_best_match(name, choices):
    """Finds the closest team name match to handle naming discrepancies."""
    match = difflib.get_close_matches(name, choices, n=1, cutoff=0.6)
    return match[0] if match else None

def main():
    # Initialize components
    scraper = NBAStatScraper(year=2025)
    
    # Specify the path to your CSV file
    csv_path = "odds.csv"  # Update this to your actual CSV file path
    odds_provider = OddsProvider(csv_path=csv_path)
    
    print("="*70)
    print("NBA BETTING EV CALCULATOR")
    print("="*70)
    
    # Step 1: Scrape and prepare data
    print("\nLoading data...")
    adv_stats = scraper.scrape_advanced_stats()
    schedule = scraper.scrape_schedule()
    
    if adv_stats is None or schedule is None:
        print("Failed to scrape data. Exiting.")
        return

    processor = DataProcessor()
    historical_games = processor.clean_schedule(schedule)
    training_data = processor.merge_stats(historical_games, adv_stats)
    
    # Drop rows with missing stats
    initial_count = len(training_data)
    training_data = training_data.dropna()
    final_count = len(training_data)
    print(f"Training data: {final_count} games ({initial_count - final_count} removed due to missing stats)")
    
    if final_count == 0:
        print("No valid training data. Exiting.")
        return
    
    # Step 2: Train model
    print("\nTraining model...")
    X, y = processor.prepare_features(training_data)
    
    model = NBAModel()
    model.train(X, y)
    
    # Step 3: Get today's games and odds from CSV
    print(f"\nGetting today's games and odds from CSV: {csv_path}")
    todays_games = odds_provider.get_todays_odds()
    
    if todays_games is None or len(todays_games) == 0:
        print("No games found for today. Exiting.")
        return
    
    print(f"Found {len(todays_games)} games today")
    
    # List of all valid team names from our stats source
    stat_teams = adv_stats['Team'].tolist()
    
    results = []
    print("\n" + "="*70)
    print("CALCULATING EV FOR TODAY'S GAMES")
    print("="*70)
    
    # Step 4: Calculate EV for each game
    for game_idx, (_, row) in enumerate(todays_games.iterrows(), 1):
        print(f"\n{'='*50}")
        print(f"GAME {game_idx}: {row['visitor']} @ {row['home']}")
        print('='*50)
        
        # Use fuzzy matching to find the right team in our stats
        home_match = find_best_match(row['home'], stat_teams)
        visitor_match = find_best_match(row['visitor'], stat_teams)
        
        if not home_match or not visitor_match:
            print(f"Skipping: Could not match teams to stats database")
            print(f"Looking for: '{row['home']}' and '{row['visitor']}'")
            print(f"Available teams: {len(stat_teams)} teams in database")
            continue

        h_stats = adv_stats[adv_stats['Team'] == home_match].iloc[0]
        v_stats = adv_stats[adv_stats['Team'] == visitor_match].iloc[0]
        
        print(f"Team matching:")
        print(f"  {row['home']} -> {home_match}")
        print(f"  {row['visitor']} -> {visitor_match}")
        print(f"Team stats:")
        print(f"  {home_match}: ORtg={h_stats['ORtg']:.1f}, DRtg={h_stats['DRtg']:.1f}, NRtg={h_stats['NRtg']:.1f}")
        print(f"  {visitor_match}: ORtg={v_stats['ORtg']:.1f}, DRtg={v_stats['DRtg']:.1f}, NRtg={v_stats['NRtg']:.1f}")
        
        # Create features for prediction
        features_dict = {
            'v_pace': [v_stats['Pace']], 'v_ortg': [v_stats['ORtg']], 
            'v_drtg': [v_stats['DRtg']], 'v_nrtg': [v_stats['NRtg']],
            'h_pace': [h_stats['Pace']], 'h_ortg': [h_stats['ORtg']], 
            'h_drtg': [h_stats['DRtg']], 'h_nrtg': [h_stats['NRtg']]
        }
        
        features = pd.DataFrame(features_dict)
        
        # Add the same derived features as in training
        features['ortg_diff'] = features['h_ortg'] - features['v_ortg']
        features['drtg_diff'] = features['h_drtg'] - features['v_drtg']
        features['nrtg_diff'] = features['h_nrtg'] - features['v_nrtg']
        features['pace_diff'] = features['h_pace'] - features['v_pace']
        features['h_off_eff'] = features['h_ortg'] / (features['v_drtg'] + 0.1)
        features['v_off_eff'] = features['v_ortg'] / (features['h_drtg'] + 0.1)
        features['offensive_advantage'] = features['h_off_eff'] - features['v_off_eff']
        features['avg_ortg'] = (features['h_ortg'] + features['v_ortg']) / 2
        features['avg_drtg'] = (features['h_drtg'] + features['v_drtg']) / 2
        
        # Get model prediction
        try:
            prob_home_win = model.predict_probs(features)[0]
        except Exception as e:
            print(f"Error in prediction: {e}")
            continue
        
        # Apply probability calibration/clipping
        prob_home_win = max(0.15, min(0.85, prob_home_win))  # Cap between 15% and 85%
        prob_visitor_win = 1 - prob_home_win
        
        # Get market probabilities for comparison
        market_prob_home = odds_provider.american_to_probability(row['home_moneyline'])
        market_prob_visitor = odds_provider.american_to_probability(row['visitor_moneyline'])
        
        print(f"\nProbabilities:")
        print(f"  Model prediction: Home = {prob_home_win:.1%}, Visitor = {prob_visitor_win:.1%}")
        print(f"  Market implied:   Home = {market_prob_home:.1%}, Visitor = {market_prob_visitor:.1%}")
        
        # Calculate decimal odds
        home_dec = odds_provider.american_to_decimal(row['home_moneyline'])
        visitor_dec = odds_provider.american_to_decimal(row['visitor_moneyline'])
        
        print(f"\nOdds:")
        print(f"  Home ({row['home']}): {row['home_moneyline']} (Decimal: {home_dec:.3f})")
        print(f"  Visitor ({row['visitor']}): {row['visitor_moneyline']} (Decimal: {visitor_dec:.3f})")
        
        # Calculate Expected Value
        ev_home = (prob_home_win * home_dec) - 1
        ev_visitor = (prob_visitor_win * visitor_dec) - 1
        
        print(f"\nExpected Value (EV):")
        print(f"  Home ({row['home']}): {ev_home:.3f} ({ev_home:.1%})")
        print(f"  Visitor ({row['visitor']}): {ev_visitor:.3f} ({ev_visitor:.1%})")
        
        # Store results
        results.append({
            'Matchup': f"{row['visitor']} @ {row['home']}", 
            'Bet': row['home'], 
            'Odds': row['home_moneyline'], 
            'Model_Prob': f"{prob_home_win:.1%}",
            'Market_Prob': f"{market_prob_home:.1%}",
            'EV': round(ev_home, 4)
        })
        results.append({
            'Matchup': f"{row['visitor']} @ {row['home']}", 
            'Bet': row['visitor'], 
            'Odds': row['visitor_moneyline'], 
            'Model_Prob': f"{prob_visitor_win:.1%}",
            'Market_Prob': f"{market_prob_visitor:.1%}",
            'EV': round(ev_visitor, 4)
        })
        
        # Betting recommendation
        print(f"\nRecommendation:")
        threshold = 0.02  # 2% EV threshold for recommending a bet
        
        if ev_home > threshold and ev_home > ev_visitor:
            print(f"  BET: {row['home']} (EV: {ev_home:.1%})")
            print(f"  Model thinks they win {prob_home_win:.1%} of the time")
            print(f"  Market thinks they win {market_prob_home:.1%} of the time")
            print(f"  Value: {prob_home_win - market_prob_home:.1%} edge")
        elif ev_visitor > threshold and ev_visitor > ev_home:
            print(f"  BET: {row['visitor']} (EV: {ev_visitor:.1%})")
            print(f"  Model thinks they win {prob_visitor_win:.1%} of the time")
            print(f"  Market thinks they win {market_prob_visitor:.1%} of the time")
            print(f"  Value: {prob_visitor_win - market_prob_visitor:.1%} edge")
        else:
            print(f"  NO BET: No clear value found")
            if ev_home > 0:
                print(f"  Small positive EV on {row['home']} ({ev_home:.1%}), but below threshold")
            elif ev_visitor > 0:
                print(f"  Small positive EV on {row['visitor']} ({ev_visitor:.1%}), but below threshold")
            else:
                print(f"  Both sides have negative EV")

    # Step 5: Display final results
    if results:
        print(f"\n{'='*70}")
        print("FINAL BETTING RECOMMENDATIONS")
        print("="*70)
        
        results_df = pd.DataFrame(results)
        
        # Sort by EV (highest first)
        results_df = results_df.sort_values(by='EV', ascending=False)
        
        # Reset index for clean display
        results_df = results_df.reset_index(drop=True)
        
        # Format output
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 30)
        
        # Create a summary table
        summary_df = results_df[['Matchup', 'Bet', 'Odds', 'Model_Prob', 'Market_Prob', 'EV']].copy()
        
        # Add EV% column for easier reading
        summary_df['EV%'] = summary_df['EV'].apply(lambda x: f"{x:.1%}")
        
        # Add signal indicators
        def get_signal(val):
            if val > 0.05:
                return '+++'  # Strong positive
            elif val > 0.02:
                return '++'   # Moderate positive
            elif val > 0:
                return '+'    # Slight positive
            elif val > -0.05:
                return '-'    # Slight negative
            else:
                return '---'  # Strong negative
        
        summary_df['Signal'] = summary_df['EV'].apply(get_signal)
        
        # Reorder columns
        summary_df = summary_df[['Signal', 'Matchup', 'Bet', 'Odds', 'Model_Prob', 'Market_Prob', 'EV%', 'EV']]
        
        print(summary_df.to_string(index=False))
        
        # Summary statistics
        print(f"\n{'='*70}")
        print("SUMMARY STATISTICS")
        print("="*70)
        
        total_bets = len(results_df)
        positive_bets = results_df[results_df['EV'] > 0]
        strong_bets = results_df[results_df['EV'] > 0.05]
        moderate_bets = results_df[(results_df['EV'] > 0.02) & (results_df['EV'] <= 0.05)]
        
        print(f"Total bets analyzed: {total_bets}")
        print(f"Positive EV bets: {len(positive_bets)} ({len(positive_bets)/total_bets:.1%})")
        print(f"Strong positive EV (>5%): {len(strong_bets)}")
        print(f"Moderate positive EV (2-5%): {len(moderate_bets)}")
        
        if len(strong_bets) > 0:
            print(f"\nTOP RECOMMENDATIONS:")
            for idx, (_, bet) in enumerate(strong_bets.head(3).iterrows(), 1):
                # Extract opponent from matchup
                matchup_parts = bet['Matchup'].split(' @ ')
                if len(matchup_parts) == 2:
                    opponent = matchup_parts[0] if matchup_parts[1] == bet['Bet'] else matchup_parts[1]
                else:
                    opponent = "Opponent"
                    
                print(f"  {idx}. {bet['Bet']} vs {opponent}")
                print(f"     Odds: {bet['Odds']}")
                print(f"     Model Probability: {bet['Model_Prob']}")
                print(f"     Market Probability: {bet['Market_Prob']}")
                print(f"     Expected Value: {bet['EV']:.3f} ({bet['EV']:.1%})")
        
        # Calculate average EV
        avg_ev = results_df['EV'].mean()
        avg_positive_ev = positive_bets['EV'].mean() if len(positive_bets) > 0 else 0
        
        print(f"\nAVERAGE EV:")
        print(f"  All bets: {avg_ev:.3f} ({avg_ev:.1%})")
        if len(positive_bets) > 0:
            print(f"  Positive bets only: {avg_positive_ev:.3f} ({avg_positive_ev:.1%})")
        
        # Save to CSV for record keeping
        try:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"nba_bets_{timestamp}.csv"
            summary_df.to_csv(filename, index=False)
            print(f"\nResults saved to: {filename}")
        except Exception as e:
            print(f"\nCould not save results: {e}")
            
    else:
        print("\nNo valid games could be analyzed. Possible issues:")
        print("  Team name mismatches between odds source and stats database")
        print("  No stats available for today's teams")
        print("  Odds data format issues")

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()