import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Football Edge",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0a;
    color: #e8e8e0;
}

.stApp {
    background-color: #0a0a0a;
}

h1, h2, h3 {
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 0.05em;
    color: #e8e8e0;
}

.main-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4rem;
    letter-spacing: 0.1em;
    color: #e8e8e0;
    margin: 0;
    line-height: 1;
}

.sub-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #666;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 4px;
}

.metric-card {
    background: #141414;
    border: 0.5px solid #2a2a2a;
    border-radius: 4px;
    padding: 1.25rem;
    text-align: center;
}

.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #555;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.metric-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.2rem;
    letter-spacing: 0.05em;
    line-height: 1;
}

.value-positive {
    color: #7fff7f;
}

.value-negative {
    color: #ff7f7f;
}

.value-neutral {
    color: #e8e8e0;
}

.edge-banner {
    background: #0f1f0f;
    border: 1px solid #2a5a2a;
    border-radius: 4px;
    padding: 1rem 1.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #7fff7f;
    letter-spacing: 0.1em;
    margin: 1rem 0;
}

.no-value-banner {
    background: #1a1a1a;
    border: 0.5px solid #2a2a2a;
    border-radius: 4px;
    padding: 1rem 1.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #555;
    letter-spacing: 0.1em;
    margin: 1rem 0;
}

.stSelectbox > div > div {
    background-color: #141414 !important;
    border: 0.5px solid #2a2a2a !important;
    border-radius: 4px !important;
    color: #e8e8e0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
}

.stNumberInput > div > div > input {
    background-color: #141414 !important;
    border: 0.5px solid #2a2a2a !important;
    border-radius: 4px !important;
    color: #e8e8e0 !important;
    font-family: 'DM Mono', monospace !important;
}

.stButton > button {
    background-color: #e8e8e0 !important;
    color: #0a0a0a !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.1em !important;
    padding: 0.5rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.15s !important;
}

.stButton > button:hover {
    opacity: 0.85 !important;
}

.divider {
    border: none;
    border-top: 0.5px solid #1f1f1f;
    margin: 1.5rem 0;
}

.lambda-display {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #555;
    text-align: center;
    margin-top: 4px;
}

.section-heading {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #444;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 0.5px solid #1f1f1f;
}

.status-dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #7fff7f;
    margin-right: 8px;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data():
    urls = {
        '2022-23': 'https://www.football-data.co.uk/mmz4281/2223/E0.csv',
        '2023-24': 'https://www.football-data.co.uk/mmz4281/2324/E0.csv',
        '2024-25': 'https://www.football-data.co.uk/mmz4281/2425/E0.csv',
        '2025-26': 'https://www.football-data.co.uk/mmz4281/2526/E0.csv'
    }
    dfs = []
    for season, url in urls.items():
        df = pd.read_csv(url, usecols=['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR'])
        df['season'] = season
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    return data.dropna()


@st.cache_data(show_spinner=False)
def build_model(data: pd.DataFrame, xi: float = 0.0065, form_window: int = 10):
    df = data.copy()
    ref = df['Date'].max()
    df['w'] = np.exp(-xi * (ref - df['Date']).dt.days)

    # Hard recent-form window — last N matches per team weighted separately
    # This captures current shape independently of the full historical model
    def get_recent(team_col, goals_col, n=form_window):
        rows = []
        for team, group in df.groupby(team_col):
            recent = group.nlargest(n, 'Date')
            rows.append({
                'team': team,
                'recent_avg': recent[goals_col].mean(),
                'recent_weight': len(recent) / n  # confidence — penalise teams with few matches
            })
        return pd.DataFrame(rows).set_index('team')

    rh = get_recent('HomeTeam', 'FTHG')  # recent home goals scored
    rc = get_recent('HomeTeam', 'FTAG')  # recent home goals conceded
    ra = get_recent('AwayTeam', 'FTAG')  # recent away goals scored
    rac = get_recent('AwayTeam', 'FTHG') # recent away goals conceded

    # Full time-weighted averages (historical signal)
    def wavg(gc, tc):
        return df.groupby(tc).apply(
            lambda x: np.average(x[gc], weights=x['w'])
            if x['w'].sum() > 0 else x[gc].mean()
        )

    hs = wavg('FTHG', 'HomeTeam')
    hc = wavg('FTAG', 'HomeTeam')
    as_ = wavg('FTAG', 'AwayTeam')
    ac = wavg('FTHG', 'AwayTeam')

    avg_h, avg_a = hs.mean(), as_.mean()

    # Blend: 40% historical ratings, 60% recent form window
    # You can expose these weights in the sidebar later
    HIST_W  = 0.4
    FORM_W  = 0.6

    def blend(hist, recent_df, col):
        combined = hist.to_frame('hist').join(recent_df[['recent_avg', 'recent_weight']])
        combined['blended'] = (
            HIST_W * combined['hist'] +
            FORM_W * combined['recent_avg'] * combined['recent_weight']
        )
        # Fall back to historical if recent is missing
        combined['blended'] = combined['blended'].fillna(combined['hist'])
        return combined['blended']

    hs_b  = blend(hs,  rh,  'recent_avg')
    hc_b  = blend(hc,  rc,  'recent_avg')
    as_b  = blend(as_, ra,  'recent_avg')
    ac_b  = blend(ac,  rac, 'recent_avg')

    avg_h_b = hs_b.mean()
    avg_a_b = as_b.mean()

    teams = pd.DataFrame({
        'attack_h':  hs_b  / avg_h_b,
        'defense_h': hc_b  / avg_a_b,
        'attack_a':  as_b  / avg_a_b,
        'defense_a': ac_b  / avg_h_b,
    }).dropna()

    return teams, avg_h_b, avg_a_b

def dc_correction(hg, ag, lh, la, rho=-0.1):
    if hg == 0 and ag == 0:   return 1 - (lh * la * rho)
    elif hg == 1 and ag == 0: return 1 + (la * rho)
    elif hg == 0 and ag == 1: return 1 + (lh * rho)
    elif hg == 1 and ag == 1: return 1 - rho
    return 1.0


def predict_dc(home, away, teams, avg_h, avg_a, max_goals=7):
    lam_h = teams.loc[home, 'attack_h'] * teams.loc[away, 'defense_a'] * avg_h
    lam_a = teams.loc[away, 'attack_a'] * teams.loc[home, 'defense_h'] * avg_a

    matrix = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            raw = stats.poisson.pmf(i, lam_h) * stats.poisson.pmf(j, lam_a)
            matrix[i,j] = raw * dc_correction(i, j, lam_h, lam_a)
    matrix /= matrix.sum()

    return {
        'lambda_home': round(lam_h, 2),
        'lambda_away': round(lam_a, 2),
        'home_win': round(np.tril(matrix, -1).sum(), 4),
        'draw':     round(np.trace(matrix), 4),
        'away_win': round(np.triu(matrix,  1).sum(), 4),
        'matrix':   matrix,
    }


# ── UI ────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="margin-bottom: 2rem;">
    <div class="main-title">FOOTBALL EDGE</div>
    <div class="sub-title"><span class="status-dot"></span>Dixon-Coles Model · Premier League</div>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading match data..."):
    data = load_data()
    teams_model, avg_h, avg_a = build_model(data)

team_list = sorted(teams_model.index.tolist())

col_left, col_right = st.columns([1, 1.6], gap="large")

with col_left:
    st.markdown('<div class="section-heading">Match Setup</div>', unsafe_allow_html=True)

    home = st.selectbox("Home team", team_list,
                        index=team_list.index('Arsenal') if 'Arsenal' in team_list else 0)
    away = st.selectbox("Away team", team_list,
                        index=team_list.index('Chelsea') if 'Chelsea' in team_list else 1)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Bookmaker Odds (decimal)</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        odds_h = st.number_input("Home", value=2.10, min_value=1.01, step=0.05, format="%.2f")
    with c2:
        odds_d = st.number_input("Draw", value=3.40, min_value=1.01, step=0.05, format="%.2f")
    with c3:
        odds_a = st.number_input("Away", value=3.60, min_value=1.01, step=0.05, format="%.2f")

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("ANALYSE MATCH")

with col_right:
    if run or True:
        if home == away:
            st.warning("Select two different teams.")
        else:
            result = predict_dc(home, away, teams_model, avg_h, avg_a)

            impl_h = 1 / odds_h
            impl_d = 1 / odds_d
            impl_a = 1 / odds_a

            edge_h = result['home_win'] - impl_h
            edge_d = result['draw']     - impl_d
            edge_a = result['away_win'] - impl_a
            best_edge = max(edge_h, edge_d, edge_a)

            st.markdown(f'<div class="section-heading">{home} vs {away}</div>', unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)

            def edge_color(e):
                if e > 0.03: return "value-positive"
                if e < -0.03: return "value-negative"
                return "value-neutral"

            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Home win</div>
                    <div class="metric-value {edge_color(edge_h)}">{result['home_win']*100:.1f}%</div>
                    <div class="lambda-display">bookie {impl_h*100:.1f}% · edge {edge_h*100:+.1f}%</div>
                </div>""", unsafe_allow_html=True)

            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Draw</div>
                    <div class="metric-value {edge_color(edge_d)}">{result['draw']*100:.1f}%</div>
                    <div class="lambda-display">bookie {impl_d*100:.1f}% · edge {edge_d*100:+.1f}%</div>
                </div>""", unsafe_allow_html=True)

            with m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Away win</div>
                    <div class="metric-value {edge_color(edge_a)}">{result['away_win']*100:.1f}%</div>
                    <div class="lambda-display">bookie {impl_a*100:.1f}% · edge {edge_a*100:+.1f}%</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if best_edge > 0.03:
                outcomes = {edge_h: 'HOME WIN', edge_d: 'DRAW', edge_a: 'AWAY WIN'}
                best_label = outcomes[best_edge]
                st.markdown(f"""
                <div class="edge-banner">
                    ⚑ VALUE DETECTED · {best_label} · +{best_edge*100:.1f}% EDGE
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="no-value-banner">
                    NO SIGNIFICANT VALUE DETECTED
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="lambda-display" style="margin-bottom: 1rem;">
                xG projection — {home}: {result['lambda_home']} · {away}: {result['lambda_away']}
            </div>""", unsafe_allow_html=True)

            # Heatmap
            fig, ax = plt.subplots(figsize=(7, 4.5))
            fig.patch.set_facecolor('#0a0a0a')
            ax.set_facecolor('#0a0a0a')

            matrix_pct = result['matrix'][:6, :6] * 100

            sns.heatmap(
                matrix_pct,
                annot=True, fmt='.1f',
                cmap='Greens',
                linewidths=0.5,
                linecolor='#1a1a1a',
                ax=ax,
                cbar=False,
                xticklabels=[f"{i}" for i in range(6)],
                yticklabels=[f"{i}" for i in range(6)],
            )

            ax.set_xlabel(f"{away} goals", color='#555', fontsize=9, labelpad=8)
            ax.set_ylabel(f"{home} goals", color='#555', fontsize=9, labelpad=8)
            ax.tick_params(colors='#555', labelsize=8)
            for spine in ax.spines.values():
                spine.set_visible(False)

            ax.set_title("Scoreline probability matrix (%)",
                         color='#444', fontsize=8, pad=10,
                         fontfamily='monospace', loc='left')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
