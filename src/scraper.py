import pandas as pd
import time
import random

class NBAStatScraper:
    def __init__(self, year=2026):
        self.year = year
        self.base_url = "https://www.basketball-reference.com"

    def scrape_advanced_stats(self):
        """Scrapes Team Advanced Stats (Pace, ORtg, DRtg, etc.)"""
        print(f"Scraping Advanced Stats for {self.year}...")
        url = f"{self.base_url}/leagues/NBA_{self.year}.html"
        
        try:
            # Table #advanced-team might need specific selection
            dfs = pd.read_html(url, header=1) 
            # Usually the advanced stats are further down, checking for specific columns
            for df in dfs:
                if 'Pace' in df.columns and 'ORtg' in df.columns:
                    # Clean up: Remove divider rows
                    df = df[df['Team'] != 'League Average']
                    return df
            print("Advanced stats table not found.")
            return None
        except Exception as e:
            print(f"Error scraping stats: {e}")
            return None

    def scrape_schedule(self):
        """Scrapes the game schedule and results."""
        print(f"Scraping Schedule for {self.year}...")
        months = ['october', 'november', 'december', 'january', 'february', 'march', 'april']
        schedule_dfs = []
        
        for month in months:
            url = f"{self.base_url}/leagues/NBA_{self.year}_games-{month}.html"
            try:
                dfs = pd.read_html(url)
                if dfs:
                    df = dfs[0]
                    schedule_dfs.append(df)
                    print(f"  - Scraped {month}")
                # Be respectful to the server
                time.sleep(random.uniform(3, 5))
            except Exception:
                # Month might not have started yet
                continue
        
        if schedule_dfs:
            full_schedule = pd.concat(schedule_dfs, ignore_index=True)
            return full_schedule
        return pd.DataFrame()

if __name__ == "__main__":
    scraper = NBAStatScraper()
    stats = scraper.scrape_advanced_stats()
    # stats.to_csv("data/raw/advanced_stats.csv")