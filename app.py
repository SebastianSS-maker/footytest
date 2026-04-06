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
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #0a0a0a; color: #e8e8e0; }
.stApp { background-color: #0a0a0a; }
h1,h2,h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 0.05em; color: #e8e8e0; }
.main-title { font-family: 'Bebas Neue', sans-serif; font-size: 4rem; letter-spacing: .1em; color: #e8e8e0; margin: 0; line-height: 1; }
.sub-title { font-family: 'DM Mono', monospace; font-size: .75rem; color: #666; letter-spacing: .2em; text-transform: uppercase; margin-top: 4px; }
.metric-card { background: #141414; border: .5px solid #2a2a2a; border-radius: 4px; padding: 1.25rem; text-align: center; }
.metric-label { font-family: 'DM Mono', monospace; font-size: .65rem; color: #555; letter-spacing: .15em; text-transform: uppercase; margin-bottom: 6px; }
.metric-value { font-family: 'Bebas Neue', sans-serif; font-size: 2.2rem; letter-spacing: .05em; line-height: 1; }
.value-positive { color: #7fff7f; }
.value-negative { color: #ff7f7f; }
.value-neutral  { color: #e8e8e0; }
.edge-banner { background: #0f1f0f; border: 1px solid #2a5a2a; border-radius: 4px; padding: 1rem 1.5rem; font-family: 'DM Mono', monospace; font-size: .8rem; color: #7fff7f; letter-spacing: .1em; margin: 1rem 0; }
.no-value-banner { background: #1a1a1a; border: .5px solid #2a2a2a; border-radius: 4px; padding: 1rem 1.5rem; font-family: 'DM Mono', monospace; font-size: .8rem; color: #555; letter-spacing: .1em; margin: 1rem 0; }
.form-wrap { margin-bottom: 8px; }
.form-bar { display: inline-block; width: 22px; height: 22px; border-radius: 3px; margin-right: 3px; text-align: center; line-height: 22px; font-size: 11px; font-weight: 500; font-family: 'DM Mono', monospace; }
.form-w { background: #1a3a1a; color: #7fff7f; }
.form-d { background: #2a2a1a; color: #ffff7f; }
.form-l { background: #3a1a1a; color: #ff7f7f; }
.fixture-row { background: #141414; border: .5px solid #2a2a2a; border-radius: 4px; padding: .75rem 1rem; margin-bottom: 8px; }
.fixture-teams { font-family: 'Bebas Neue', sans-serif; font-size: 1.1rem; letter-spacing: .05em; color: #e8e8e0; }
.fixture-meta { font-family: 'DM Mono', monospace; font-size: .65rem; color: #444; margin-top: 2px; }
.edge-pill { display: inline-block; font-family: 'DM Mono', monospace; font-size: .65rem; padding: 2px 8px; border-radius: 3px; margin-left: 8px; }
.edge-pill-pos { background: #0f2a0f; color: #7fff7f; border: .5px solid #2a5a2a; }
.edge-pill-neu { background: #1a1a1a; color: #444; border: .5px solid #2a2a2a; }
.stSelectbox > div > div { background-color: #141414 !important; border: .5px solid #2a2a2a !important; border-radius: 4px !important; color: #e8e8e0 !important; font-family: 'DM Mono', monospace !important; font-size: .85rem !important; }
.stNumberInput > div > div > input { background-color: #141414 !important; border: .5px solid #2a2a2a !important; border-radius: 4px !important; color: #e8e8e0 !important; font-family: 'DM Mono', monospace !important; }
.stButton > button { background-color: #e8e8e0 !important; color: #0a0a0a !important; border: none !important; border-radius: 4px !important; font-family: 'Bebas Neue', sans-serif !important; font-size: 1.1rem !important; letter-spacing: .1em !important; padding: .5rem 2rem !important; width: 100% !important; }
.divider { border: none; border-top: .5px solid #1f1f1f; margin: 1.5rem 0; }
.lambda-display { font-family: 'DM Mono', monospace; font-size: .75rem; color: #555; text-align: center; margin-top: 4px; }
.section-heading { font-family: 'DM Mono', monospace; font-size: .65rem; color: #444; letter-spacing: .2em; text-transform: uppercase; margin-bottom: 1rem; padding-bottom: .5rem; border-bottom: .5px solid #1f1f1f; }
.status-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #7fff7f; margin-right: 8px; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1;}50%{opacity:.3;} }
</style>
""", unsafe_allow_html=True)


# ── DATA ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    urls = {
        '2022-23': 'https://www.football-data.co.uk/mmz4281/2223/E0.csv',
        '2023-24': 'https://www.football-data.co.uk/mmz4281/2324/E0.csv',
        '2024-25': 'https://www.football-data.co.uk/mmz4281/2425/E0.csv',
        '2025-26': 'https://www.football-data.co.uk/mmz4281/2526/E0.csv',
    }
    dfs = []
    for season, url in urls.items():
        try:
            df = pd.read_csv(url, usecols=['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR'])
            df['season'] = season
            dfs.append(df)
        except Exception:
            continue
    data = pd.concat(dfs, ignore_index=True)
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')
    data['FTHG'] = pd.to_numeric(data['FTHG'], errors='coerce')
    data['FTAG'] = pd.to_numeric(data['FTAG'], errors='coerce')
    return data.dropna(subset=['Date','HomeTeam','AwayTeam','FTHG','FTAG']).reset_index(drop=True)


@st.cache_data(ttl=1800, show_spinner=False)
def load_fixtures():
    try:
        df = pd.read_csv('https://www.football-data.co.uk/fixtures.csv')
        df = df[df['Div'] == 'E0'].copy()
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        today = pd.Timestamp.now().normalize()
        df = df[df['Date'] >= today]
        for col in ['B365H','B365D','B365A']:
            df[col] = pd.to_numeric(df.get(col), errors='coerce')
        df = df.dropna(subset=['HomeTeam','AwayTeam'])
        return df.sort_values('Date').reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


# ── MODEL ─────────────────────────────────────────────────────────────────────

def build_model(data, xi=0.0065, form_window=10, hist_w=0.4):
    df = data.copy()
    ref = df['Date'].max()
    df['w'] = np.exp(-xi * (ref - df['Date']).dt.days)
    form_w = 1.0 - hist_w

    def wavg(gc, tc):
        return df.groupby(tc).apply(
            lambda x: np.average(x[gc], weights=x['w']) if x['w'].sum() > 0 else x[gc].mean()
        )

    hs  = wavg('FTHG', 'HomeTeam')
    hc  = wavg('FTAG', 'HomeTeam')
    as_ = wavg('FTAG', 'AwayTeam')
    ac  = wavg('FTHG', 'AwayTeam')

    def get_recent(team_col, goals_col):
        rows = []
        for team, group in df.groupby(team_col):
            recent = group.nlargest(form_window, 'Date')
            rows.append({
                'team': team,
                'recent_avg': recent[goals_col].mean(),
                'confidence': len(recent) / form_window,
            })
        return pd.DataFrame(rows).set_index('team')

    rh  = get_recent('HomeTeam', 'FTHG')
    rc  = get_recent('HomeTeam', 'FTAG')
    ra  = get_recent('AwayTeam', 'FTAG')
    rac = get_recent('AwayTeam', 'FTHG')

    def blend(hist, recent_df):
        combined = hist.to_frame('hist').join(recent_df)
        combined['blended'] = (
            hist_w * combined['hist'] +
            form_w * combined['recent_avg'] * combined['confidence']
        )
        return combined['blended'].fillna(combined['hist'])

    hs_b  = blend(hs,  rh)
    hc_b  = blend(hc,  rc)
    as_b  = blend(as_, ra)
    ac_b  = blend(ac,  rac)

    avg_h, avg_a = hs_b.mean(), as_b.mean()

    teams = pd.DataFrame({
        'attack_h':  hs_b  / avg_h,
        'defense_h': hc_b  / avg_a,
        'attack_a':  as_b  / avg_a,
        'defense_a': ac_b  / avg_h,
    }).dropna()

    return teams, avg_h, avg_a


def dc_correction(hg, ag, lh, la, rho=-0.1):
    if   hg==0 and ag==0: return 1-(lh*la*rho)
    elif hg==1 and ag==0: return 1+(la*rho)
    elif hg==0 and ag==1: return 1+(lh*rho)
    elif hg==1 and ag==1: return 1-rho
    return 1.0


def predict_dc(home, away, teams, avg_h, avg_a, rho=-0.1, max_goals=7):
    if home not in teams.index or away not in teams.index:
        return None
    lam_h = teams.loc[home,'attack_h'] * teams.loc[away,'defense_a'] * avg_h
    lam_a = teams.loc[away,'attack_a'] * teams.loc[home,'defense_h'] * avg_a

    matrix = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            raw = stats.poisson.pmf(i,lam_h) * stats.poisson.pmf(j,lam_a)
            matrix[i,j] = raw * dc_correction(i,j,lam_h,lam_a,rho)
    matrix /= matrix.sum()

    flat = sorted(
        [(matrix[i,j],i,j) for i in range(max_goals+1) for j in range(max_goals+1)],
        reverse=True
    )

    return {
        'lambda_home': round(lam_h,2),
        'lambda_away': round(lam_a,2),
        'home_win':    round(np.tril(matrix,-1).sum(),4),
        'draw':        round(np.trace(matrix),4),
        'away_win':    round(np.triu(matrix,1).sum(),4),
        'matrix':      matrix,
        'top5':        [(f"{i}-{j}", round(p*100,1)) for p,i,j in flat[:5]],
    }


def get_form(data, team, n=6):
    h = data[data['HomeTeam']==team][['Date','FTR']].copy()
    h['result'] = h['FTR'].map({'H':'W','D':'D','A':'L'})
    a = data[data['AwayTeam']==team][['Date','FTR']].copy()
    a['result'] = a['FTR'].map({'A':'W','D':'D','H':'L'})
    return pd.concat([h[['Date','result']],a[['Date','result']]]).sort_values('Date').tail(n)['result'].tolist()


def form_html(results, label):
    bars = "".join(f'<span class="form-bar form-{r.lower()}">{r}</span>' for r in results)
    return (f'<div class="form-wrap">'
            f'<span style="font-family:\'DM Mono\',monospace;font-size:.65rem;color:#444;margin-right:8px;">'
            f'{label[:3].upper()}</span>{bars}</div>')


def edge_color(e):
    return "value-positive" if e>0.03 else "value-negative" if e<-0.03 else "value-neutral"

def edge_col(e):
    return "#7fff7f" if e>0.03 else "#ff7f7f" if e<-0.03 else "#555"


# ── UI ────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="margin-bottom:2rem;">
  <div class="main-title">FOOTBALL EDGE</div>
  <div class="sub-title"><span class="status-dot"></span>Dixon-Coles · Live Data · Form-Weighted · Premier League</div>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading match data..."):
    data         = load_data()
    fixtures_raw = load_fixtures()

last_date     = data['Date'].max().strftime('%d %b %Y')
total_matches = len(data)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Model settings")
    xi = st.slider("Time decay (xi)", 0.001, 0.015, 0.0065, 0.001,
                   help="Higher = old matches fade faster.")
    form_window = st.slider("Form window (matches)", 3, 20, 10, 1,
                            help="How many recent matches count as current form.")
    hist_w = st.slider("History vs form weight", 0.0, 1.0, 0.4, 0.05,
                       help="0.0 = pure recent form. 1.0 = pure historical average.")
    rho = st.slider("Rho (DC correction)", -0.2, 0.0, -0.1, 0.01,
                    help="Corrects underestimation of 0-0 and 1-1 draws.")
    edge_threshold = st.slider("Edge threshold (%)", 1, 10, 3, 1,
                               help="Minimum edge to flag as value in gameweek scanner.")
    st.markdown("---")
    st.markdown(f"""
    <div style="font-family:'DM Mono',monospace;font-size:.65rem;color:#444;line-height:2.2;">
    DATA THROUGH<br><span style="color:#7fff7f">{last_date}</span><br><br>
    MATCHES LOADED<br><span style="color:#7fff7f">{total_matches:,}</span><br><br>
    CURRENT SEASON<br><span style="color:#7fff7f">2025-26 · LIVE</span>
    </div>""", unsafe_allow_html=True)

teams_model, avg_h, avg_a = build_model(data, xi=xi, form_window=form_window, hist_w=hist_w)
team_list = sorted(teams_model.index.tolist())

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["⚽  Match Analyser", "📅  Gameweek Scanner"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MATCH ANALYSER
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 1.6], gap="large")

    with col_left:
        st.markdown('<div class="section-heading">Match Setup</div>', unsafe_allow_html=True)
        home = st.selectbox("Home team", team_list,
                            index=team_list.index('Arsenal') if 'Arsenal' in team_list else 0)
        away = st.selectbox("Away team", team_list,
                            index=team_list.index('Chelsea') if 'Chelsea' in team_list else 1)
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Bookmaker Odds (decimal)</div>', unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        with c1: odds_h = st.number_input("Home", value=2.10, min_value=1.01, step=0.05, format="%.2f")
        with c2: odds_d = st.number_input("Draw", value=3.40, min_value=1.01, step=0.05, format="%.2f")
        with c3: odds_a = st.number_input("Away", value=3.60, min_value=1.01, step=0.05, format="%.2f")
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("ANALYSE MATCH")

    with col_right:
        if home == away:
            st.warning("Select two different teams.")
        else:
            result = predict_dc(home, away, teams_model, avg_h, avg_a, rho=rho)
            if result is None:
                st.warning("One or both teams not found in model.")
            else:
                impl_h, impl_d, impl_a = 1/odds_h, 1/odds_d, 1/odds_a
                edge_h = result['home_win'] - impl_h
                edge_d = result['draw']     - impl_d
                edge_a = result['away_win'] - impl_a
                best_edge = max(edge_h, edge_d, edge_a)

                fh = get_form(data, home)
                fa = get_form(data, away)

                st.markdown(f'<div class="section-heading">{home} vs {away}</div>', unsafe_allow_html=True)
                st.markdown(form_html(fh, home) + form_html(fa, away), unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                m1,m2,m3 = st.columns(3)
                for col, label, prob, edge, impl in [
                    (m1,"Home win",result['home_win'],edge_h,impl_h),
                    (m2,"Draw",    result['draw'],    edge_d,impl_d),
                    (m3,"Away win",result['away_win'],edge_a,impl_a),
                ]:
                    with col:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">{label}</div>
                            <div class="metric-value {edge_color(edge)}">{prob*100:.1f}%</div>
                            <div class="lambda-display">bookie {impl*100:.1f}%</div>
                            <div class="lambda-display" style="color:{edge_col(edge)}">edge {edge*100:+.1f}%</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                if best_edge > 0.03:
                    lmap = {edge_h:'HOME WIN',edge_d:'DRAW',edge_a:'AWAY WIN'}
                    st.markdown(f'<div class="edge-banner">⚑ VALUE DETECTED · {lmap[best_edge]} · +{best_edge*100:.1f}% EDGE</div>', unsafe_allow_html=True)
                else:
