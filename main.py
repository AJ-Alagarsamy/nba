import pandas as pd
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
    scraper = NBAStatScraper(year=2025)
    adv_stats = scraper.scrape_advanced_stats()
    schedule = scraper.scrape_schedule()
    
    if adv_stats is None or schedule is None:
        print("Failed to scrape data.")
        return

    processor = DataProcessor()
    historical_games = processor.clean_schedule(schedule)
    training_data = processor.merge_stats(historical_games, adv_stats)
    
    # Drop rows with missing stats
    training_data = training_data.dropna()
    X, y = processor.prepare_features(training_data)
    
    model = NBAModel()
    model.train(X, y)

    odds_provider = OddsProvider()
    todays_games = odds_provider.get_todays_odds()
    
    # List of all valid team names from our stats source
    stat_teams = adv_stats['Team'].tolist()
    
    results = []
    print("\n--- CALCULATING EV FOR TODAY'S GAMES ---")
    
    for _, row in todays_games.iterrows():
        # Use fuzzy matching to find the right team in our stats
        home_match = find_best_match(row['home'], stat_teams)
        visitor_match = find_best_match(row['visitor'], stat_teams)
        
        if not home_match or not visitor_match:
            print(f"Skipping {row['visitor']} @ {row['home']} (Match not found)")
            continue

        h_stats = adv_stats[adv_stats['Team'] == home_match]
        v_stats = adv_stats[adv_stats['Team'] == visitor_match]

        features = pd.DataFrame({
            'v_pace': v_stats['Pace'].values, 'v_ortg': v_stats['ORtg'].values, 
            'v_drtg': v_stats['DRtg'].values, 'v_nrtg': v_stats['NRtg'].values,
            'h_pace': h_stats['Pace'].values, 'h_ortg': h_stats['ORtg'].values, 
            'h_drtg': h_stats['DRtg'].values, 'h_nrtg': h_stats['NRtg'].values
        })
        
        prob_home_win = model.predict_probs(features)[0]
        prob_visitor_win = 1 - prob_home_win
        
        # Home EV
        home_dec = odds_provider.american_to_decimal(row['home_moneyline'])
        ev_home = (prob_home_win * home_dec) - 1
        
        # Visitor EV
        visitor_dec = odds_provider.american_to_decimal(row['visitor_moneyline'])
        ev_visitor = (prob_visitor_win * visitor_dec) - 1
        
        results.append({'Matchup': f"{row['visitor']} @ {row['home']}", 'Bet': row['home'], 'Odds': row['home_moneyline'], 'EV': round(ev_home, 4)})
        results.append({'Matchup': f"{row['visitor']} @ {row['home']}", 'Bet': row['visitor'], 'Odds': row['visitor_moneyline'], 'EV': round(ev_visitor, 4)})

    # Final Check: Only sort if we actually found games
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='EV', ascending=False)
        print(results_df.to_string(index=False))
    else:
        print("No betting opportunities found with current data matches.")

if __name__ == "__main__":
    main()