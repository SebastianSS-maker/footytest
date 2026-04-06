# Football Edge — Dixon-Coles Model

A Streamlit app that runs your Dixon-Coles football prediction model and surfaces value against bookmaker odds.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud (free, takes 5 minutes)

1. Push this folder to a GitHub repo
2. Go to share.streamlit.io
3. Connect your GitHub account
4. Select the repo and set main file as `app.py`
5. Hit Deploy — public URL generated instantly

## Deploy to Railway (if you want a persistent server later)

```bash
railway init
railway up
```

## What it does

- Pulls 3 seasons of Premier League data from football-data.co.uk
- Builds time-weighted Dixon-Coles attack/defense ratings
- Takes bookmaker decimal odds as input
- Outputs model probability vs implied probability
- Flags value bets where edge > 3%
- Shows scoreline probability heatmap
