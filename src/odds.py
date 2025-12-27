import random
import pandas as pd

class OddsProvider:
    def get_todays_odds(self):
        # All data must be indented inside the method
        data = [
            {
                'home': 'Sacramento Kings',
                'visitor': 'Dallas Mavericks',
                'home_moneyline': +124,
                'visitor_moneyline': -148
            },
            {
                'home': 'Orlando Magic',
                'visitor': 'Denver Nuggets',
                'home_moneyline': +164,
                'visitor_moneyline': -198
            },
            {
                'home': 'New Orleans Pelicans',
                'visitor': 'Phoenix Suns',
                'home_moneyline': +185,
                'visitor_moneyline': -225
            },
            {
                'home': 'Atlanta Hawks',
                'visitor': 'New York Knicks',
                'home_moneyline': +185,
                'visitor_moneyline': -225
            },
            {
                'home': 'Miami Heat',
                'visitor': 'Indiana Pacers',
                'home_moneyline': -345,
                'visitor_moneyline': +275
            },
            {
                'home': 'Chicago Bulls',
                'visitor': 'Milwaukee Bucks',
                'home_moneyline': -198,
                'visitor_moneyline': +164
            },
            {
                'home': 'Houston Rockets',
                'visitor': 'Cleveland Cavaliers',
                'home_moneyline': -198,
                'visitor_moneyline': +164
            },
            {
                'home': 'Minnesota Timberwolves',
                'visitor': 'Brooklyn Nets',
                'home_moneyline': -395,
                'visitor_moneyline': +310
            },
            {
                'home': 'San Antonio Spurs',
                'visitor': 'Utah Jazz',
                'home_moneyline': -1350,
                'visitor_moneyline': +800
            }
        ]

        return pd.DataFrame(data)

    @staticmethod
    def american_to_decimal(american_odds):
        """Converts American odds (-110, +150) to Decimal (1.91, 2.50)."""
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            # Using abs() handles the negative sign for the calculation
            return (100 / abs(american_odds)) + 1