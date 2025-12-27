# nba

nba-ev-predictor/
├── data/                  # Directory to save CSVs
│   ├── raw/
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── scraper.py         # Scrapes generic stats (Schedule, Advanced Stats)
│   ├── processor.py       # Cleans data & calculates rolling averages
│   ├── model.py           # Trains the ML model
│   └── odds.py            # (Mock) Odds retrieval
├── main.py                # Entry point
├── requirements.txt
└── README.md

## be sure to reupload data in the odds script