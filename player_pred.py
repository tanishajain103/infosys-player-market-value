import pickle
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Consistent green visual system matching the website banner.
CHART_GREENS = [
    "#2F6B5B",  # light forest green
    "#4F806F",  # medium green
    "#6F9B88",  # muted green
    "#86AFA0",  # soft green
    "#A4C7B8",  # light green
    "#C0D8CE",  # pale green
]

def style_chart(fig, height=None, **layout_kwargs):
    """Apply the PlayerValue AI light-green palette consistently to Plotly charts."""
    base_layout = dict(
        height=height,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#18312B", family="Arial, sans-serif", size=13),
        title_font=dict(color="#18312B", size=18),
        legend=dict(
            font=dict(color="#18312B", size=12),
            bgcolor="rgba(255,255,255,0)",
        ),
        colorway=CHART_GREENS,
        xaxis=dict(
            gridcolor="rgba(18,63,53,.10)",
            zerolinecolor="rgba(18,63,53,.18)",
            tickfont=dict(color="#5D706A", size=11),
            title_font=dict(color="#38524A", size=12),
            linecolor="rgba(18,63,53,.18)",
        ),
        yaxis=dict(
            gridcolor="rgba(18,63,53,.10)",
            zerolinecolor="rgba(18,63,53,.18)",
            tickfont=dict(color="#5D706A", size=11),
            title_font=dict(color="#38524A", size=12),
            linecolor="rgba(18,63,53,.18)",
        ),
    )

    # Apply defaults first, then custom per-chart layout. This prevents
    # duplicate keyword errors when a chart supplies its own legend/margins.
    fig.update_layout(**base_layout)
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)

    # Explicitly recolor every trace so Plotly's default blue never appears.
    # The palette is intentionally lighter than the hero/banner green.
    for i, trace in enumerate(fig.data):
        green = CHART_GREENS[i % len(CHART_GREENS)]

        if trace.type in {"bar", "histogram", "box", "violin", "funnel"}:
            trace.update(marker=dict(color=green, line=dict(color=green)))
        elif trace.type == "scatter":
            trace.update(
                marker=dict(color=green, line=dict(color=green)),
                line=dict(color=green),
            )
        elif trace.type == "pie":
            trace.update(marker=dict(colors=CHART_GREENS, line=dict(color="#FFFFFF", width=1)))
        elif trace.type in {"area", "scattergl"}:
            trace.update(line=dict(color=green), fillcolor=green)
        else:
            if hasattr(trace, "marker") and trace.marker is not None:
                trace.marker.color = green
            if hasattr(trace, "line") and trace.line is not None:
                trace.line.color = green

    return fig

import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="PlayerValue AI | Football Market Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent

# =========================================================
# MODEL / DATA
# =========================================================
@st.cache_resource
def load_model():
    with open(BASE_DIR / "xgb_final_model.pkl", "rb") as file:
        return pickle.load(file)


@st.cache_data
def load_feature_importance():
    return pd.read_excel(BASE_DIR / "feature_importance.xlsx")


@st.cache_data
def load_players():
    return pd.read_csv(BASE_DIR / "Player_Stats.csv", encoding="latin1")


@st.cache_data
def load_market_values():
    path = BASE_DIR / "player_market_values.csv"
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


model = load_model()
feature_importance_df = load_feature_importance()
players_df = load_players()
market_values_df = load_market_values()

# =========================================================
# PLAYER NAME NORMALIZATION
# The source CSV contains a number of names where Unicode
# characters were replaced by "?" during export. Keep the
# original datasets untouched and normalize the display/lookup
# values inside the app so Search, Compare and profiles show
# the correct names.
# =========================================================
PLAYER_NAME_FIXES = {
    "Komnen Andri?": "Komnen Andrić",
    "Marko Arnautovi?": "Marko Arnautović",
    "Sr?an Babi?": "Srđan Babić",
    "Toma Ba\x9ai?": "Toma Bašić",
    "Asmir Begovi?": "Asmir Begović",
    "Bartosz Bereszy?ski": "Bartosz Bereszyński",
    "Ermin Bi?ak?i?": "Ermin Bičakčić",
    "Veljko Birman?evi?": "Veljko Birmančević",
    "Kristijan Bistrovi?": "Kristijan Bistrović",
    "Miha Bla?i?": "Miha Blažić",
    "Domagoj Bradari?": "Domagoj Bradarić",
    "Marcelo Brozovi?": "Marcelo Brozović",
    "Marcin Bu?ka": "Marcin Bułka",
    "Duje ?aleta-Car": "Duje Ćaleta-Car",
    "Vlad Chiriche?": "Vlad Chiricheș",
    "David ?olina": "David Čolina",
    "Domen ?rnigoj": "Domen Črnigoj",
    "Pawe? Dawidowicz": "Paweł Dawidowicz",
    "Ermedin Demirovi?": "Ermedin Demirović",
    "Marko Dmitrovi?": "Marko Dmitrović",
    "Bart?omiej Dr?gowski": "Bartłomiej Drągowski",
    "Edin D?eko": "Edin Džeko",
    "?ukasz Fabia?ski": "Łukasz Fabiański",
    "Bartol Franji?": "Bartol Franjić",
    "Przemys?aw Frankowski": "Przemysław Frankowski",
    "Rafa? Gikiewicz": "Rafał Gikiewicz",
    "Adrian Grbi?": "Adrian Grbić",
    "Ivo Grbi?": "Ivo Grbić",
    "Samir Handanovi?": "Samir Handanović",
    "Adam Hlo?ek": "Adam Hložek",
    "Luká\x9a Hrádeck?": "Lukáš Hrádecký",
    "Arijon Ibrahimovi?": "Arijon Ibrahimović",
    "Ivan Ili?": "Ivan Ilić",
    "Kristijan Jaki?": "Kristijan Jakić",
    "?órir Jóhann Helgason": "Þórir Jóhann Helgason",
    "Stevan Joveti?": "Stevan Jovetić",
    "Luka Jovi?": "Luka Jović",
    "Josip Juranovi?": "Josip Juranović",
    "Pavel Kade?ábek": "Pavel Kadeřábek",
    "Jakub Kami?ski": "Jakub Kamiński",
    "Marcin Kami?ski": "Marcin Kamiński",
    "Filip Kosti?": "Filip Kostić",
    "Mateo Kova?i?": "Mateo Kovačić",
    "Andrej Kramari?": "Andrej Kramarić",
    "Rade Kruni?": "Rade Krunić",
    "Marko Lazeti?": "Marko Lazetić",
    "Darko Lazovi?": "Darko Lazović",
    "Sandi Lovri?": "Sandi Lovrić",
    "Sa\x9aa Luki?": "Saša Lukić",
    "Nemanja Maksimovi?": "Nemanja Maksimović",
    "R?zvan Marin": "Răzvan Marin",
    "Adam Maru\x9ai?": "Adam Marušić",
    "Erhan Ma\x9aovi?": "Erhan Mašović",
    "Nemanja Mati?": "Nemanja Matić",
    "Nikola Milenkovi?": "Nikola Milenković",
    "Sergej Milinkovi?-Savi?": "Sergej Milinković-Savić",
    "Vanja Milinkovi?-Savi?": "Vanja Milinković-Savić",
    "Luka Milivojevi?": "Luka Milivojević",
    "Marko Milovanovi?": "Marko Milovanović",
    "Aleksandar Mitrovi?": "Aleksandar Mitrović",
    "Stefan Mitrovi?": "Stefan Mitrović",
    "Luka Modri?": "Luka Modrić",
    "Matija Nastasi?": "Matija Nastasić",
    "Mislav Or\x9ai?": "Mislav Oršić",
    "Milo\x9a Pantovi?": "Miloš Pantović",
    "Marco Pa\x9aali?": "Marco Pašalić",
    "Mario Pa\x9aali?": "Mario Pašalić",
    "Ji?í Pavlenka": "Jiří Pavlenka",
    "Ivan Peri\x9ai?": "Ivan Perišić",
    "Krzysztof Pi?tek": "Krzysztof Piątek",
    "Marin Pongra?i?": "Marin Pongračić",
    "?ukasz Por?ba": "Łukasz Poręba",
    "Sanjin Prci?": "Sanjin Prcić",
    "Nemanja Radonji?": "Nemanja Radonjić",
    "Ivan Radovanovi?": "Ivan Radovanović",
    "Ionu? Radu": "Ionuț Radu",
    "Predrag Rajkovi?": "Predrag Rajković",
    "Ivan Rakiti?": "Ivan Rakitić",
    "Ante Rebi?": "Ante Rebić",
    "Halid \x8aabanovi?": "Halid Šabanović",
    "Stefan Savi?": "Stefan Savić",
    "Haris Seferovi?": "Haris Seferović",
    "?ukasz Skorupski": "Łukasz Skorupski",
    "Hrvoje Smol?i?": "Hrvoje Smolčić",
    "Tomá\x9a Sou?ek": "Tomáš Souček",
    "Josip Stani\x9ai?": "Josip Stanišić",
    "Petar Stojanovi?": "Petar Stojanović",
    "Ivan \x8aunji?": "Ivan Šunjić",
    "Wojciech Szcz?sny": "Wojciech Szczęsny",
    "Ciprian T?t?ru?anu": "Ciprian Tătărușanu",
    "Aleksa Terzi?": "Aleksa Terzić",
    "Filip Uremovi?": "Filip Uremović",
    "Milo\x9a Veljkovi?": "Miloš Veljković",
    "Du\x9aan Vlahovi?": "Dušan Vlahović",
    "Nikola Vla\x9ai?": "Nikola Vlašić",
    "Przemys?aw Wi?niewski": "Przemysław Wiśniewski",
    "Piotr Zieli?ski": "Piotr Zieliński",
    "Szymon ?urkowski": "Szymon Żurkowski",
    "Milan ?uri?": "Milan Đurić",
    "Filip ?uri?i?": "Filip Đuričić",
}

def normalize_player_names(df):
    result = df.copy()
    if "Player" in result.columns:
        result["Player"] = result["Player"].astype(str).str.strip().replace(PLAYER_NAME_FIXES)
    return result

players_df = normalize_player_names(players_df)
market_values_df = normalize_player_names(market_values_df)

model_features = list(model.get_booster().feature_names)

# =========================================================
# ORIGINAL PREDICTION INPUTS — KEEPING THE EXISTING PIPELINE
# =========================================================
DEFAULT_VALUES = {
    "Age": 37,
    "Min": 1873,
    "Comp_Premier League": True,
    "PasTotCmp%": 65.6,
    "Rec": 10.8,
    "TouAttPen": 0.00,
    "CarTotDist": 101.5,
    "Tkl+Int": 0.00,
    "CarProg": 0.00,
    "AerWon": 0.48,
    "GcaPassLive": 0.00,
    "Recov": 0.48,
    "PasProg": 0.00,
    "90s": 20.8,
    "CrdY": 0.00,
    "AerLost": 0.05,
    "PasShoAtt": 3.70,
    "TouAtt3rd": 0.0,
    "Int": 0.00,
    "TklDef3rd": 0.00,
    "ToTkl": 0.00,
}

FIELD_HELP = {
    "Age": "Player age in years.",
    "Min": "Minutes played during the season.",
    "Comp_Premier League": "Whether the player competes in the Premier League.",
    "PasTotCmp%": "Overall pass completion percentage.",
    "Rec": "Successful receptions per 90.",
    "TouAttPen": "Touches in the opposition penalty area.",
    "CarTotDist": "Total carrying distance.",
    "Tkl+Int": "Combined tackles and interceptions.",
    "CarProg": "Progressive carries.",
    "AerWon": "Aerial duels won.",
    "GcaPassLive": "Goal-creating actions from live-ball passes.",
    "Recov": "Recoveries.",
    "PasProg": "Progressive passes.",
    "90s": "Approximate full-match equivalents played.",
    "CrdY": "Yellow cards.",
    "AerLost": "Aerial duels lost.",
    "PasShoAtt": "Passes leading to shots.",
    "TouAtt3rd": "Touches in the final third.",
    "Int": "Interceptions.",
    "TklDef3rd": "Tackles in the defensive third.",
    "ToTkl": "Tackles attempted.",
}

FEATURE_LABELS = {
    "PasTotCmp%": "Pass Completion %",
    "Comp_Premier League": "Premier League",
    "TouAttPen": "Penalty Area Touches",
    "CarTotDist": "Carry Distance",
    "Tkl+Int": "Tackles + Interceptions",
    "CarProg": "Progressive Carries",
    "AerWon": "Aerial Duels Won",
    "GcaPassLive": "Live-Pass GCA",
    "PasProg": "Progressive Passes",
    "CrdY": "Yellow Cards",
    "AerLost": "Aerial Duels Lost",
    "PasShoAtt": "Shot-Creating Passes",
    "TouAtt3rd": "Final-Third Touches",
    "TklDef3rd": "Defensive-Third Tackles",
    "ToTkl": "Tackles Attempted",
}


def pretty_feature(name):
    return FEATURE_LABELS.get(name, str(name).replace("_", " "))


def build_prediction_input(player_inputs):
    full_input_data = {feature: 0 for feature in model_features}
    for col, value in player_inputs.items():
        if col in full_input_data:
            full_input_data[col] = 1 if isinstance(value, bool) and value else value
            if isinstance(value, bool) and not value:
                full_input_data[col] = 0
    return pd.DataFrame([full_input_data])[model_features]


def predict_from_row(row):
    values = {feature: 0 for feature in model_features}
    for feature in model_features:
        if feature in row.index:
            value = row[feature]
            if pd.isna(value):
                value = 0
            if feature == "Comp_Premier League":
                value = 1 if str(row.get("Comp", "")).lower() == "premier league" else 0
            try:
                values[feature] = float(value)
            except (TypeError, ValueError):
                values[feature] = 0
    return float(model.predict(pd.DataFrame([values])[model_features])[0])


def money_eur(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"€{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"€{value / 1_000:.1f}K"
    return f"€{value:,.0f}"


def actual_market_value(player_name):
    if "Player" not in market_values_df.columns or "Market Value" not in market_values_df.columns:
        return None
    matches = market_values_df[
        market_values_df["Player"].astype(str).str.strip().str.lower()
        == str(player_name).strip().lower()
    ]
    if matches.empty:
        return None
    return matches["Market Value"].iloc[0]


def go_to(page):
    # Navigation widgets are created later in the script. Streamlit does not
    # allow changing a widget's own session-state key after that widget has
    # already been instantiated in the same run. Store a pending route here;
    # it is applied before the navigation widget is created on the next run.
    st.session_state["pending_page"] = page
    if page == "Predictor":
        st.session_state["pending_predictor_mode"] = "Manual"


# =========================================================
# GLOBAL CSS — FULL WIDTH PRODUCT WEBSITE
# =========================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --ink: #14211e;
    --muted: #68746f;
    --green: #123f35;
    --green-2: #1d5a4b;
    --lime: #d8ed83;
    --mint: #eaf2ec;
    --cream: #f7f8f5;
    --white: #ffffff;
    --line: #e2e7e3;
    --shadow: 0 14px 40px rgba(20, 40, 34, .07);
}

* { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
body { background: var(--cream); }

.stApp {
    min-height: 100vh;
    background:
        radial-gradient(circle at 94% 4%, rgba(216,237,131,.22), transparent 25rem),
        linear-gradient(180deg, #fbfcfa 0%, #f6f8f5 48%, #ffffff 100%);
}

.block-container,
[data-testid="stAppViewContainer"] .main .block-container,
[data-testid="stMainBlockContainer"] {
    width: 100% !important;
    max-width: none !important;
    margin: 0 !important;
    padding: 1rem clamp(1rem, 3vw, 4.5rem) 4rem !important;
}

#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

h1,h2,h3,h4 { font-family: 'Space Grotesk', sans-serif !important; color: var(--ink) !important; letter-spacing: -.035em; }
h1 { font-size: clamp(2.7rem, 5.2vw, 5.6rem) !important; line-height: .97 !important; }
h2 { font-size: clamp(2rem, 3vw, 3.2rem) !important; }
h3 { font-size: 1.2rem !important; }
p, li, label { color: var(--muted); }

/* Header */
.site-nav {
    width: 100%;
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:1rem;
    padding:.65rem 0 1.35rem;
    border-bottom:1px solid var(--line);
    margin-bottom:1.5rem;
}
.brand { display:flex; align-items:center; gap:1rem; min-width:max-content; }
.brand-mark {
    width:52px; height:52px; border-radius:13px; display:grid; place-items:center;
    background:var(--green); color:var(--lime); font-size:1.5rem;
    box-shadow:0 8px 22px rgba(18,63,53,.15);
}
.brand-name { font:700 1.5rem 'Space Grotesk',sans-serif; color:var(--green); }
.brand-sub { font-size:.88rem; color:var(--muted); margin-top:.1rem; }
.nav-badge { padding:.62rem 1rem; border:1px solid var(--line); border-radius:999px; color:var(--muted); font-size:.9rem; background:#fff; }

/* Hero */
.hero {
    position:relative; overflow:hidden; min-height:470px;
    border-radius:30px; padding:clamp(2rem,5vw,5.3rem);
    background:linear-gradient(135deg,#123f35 0%,#174a3d 58%,#0f362e 100%);
    color:white; box-shadow:0 20px 55px rgba(18,63,53,.16); margin:1rem 0 1.3rem;
}
.hero::before { content:""; position:absolute; width:560px;height:560px;right:-130px;top:-250px;border:1px solid rgba(216,237,131,.27);border-radius:50%;box-shadow:0 0 0 55px rgba(216,237,131,.035),0 0 0 110px rgba(216,237,131,.025); }
.hero::after { content:""; position:absolute; width:260px;height:260px;right:13%;bottom:-190px;border:1px solid rgba(255,255,255,.08);border-radius:50%; }
.hero-content { position:relative; z-index:2; max-width:760px; }
.eyebrow { color:var(--lime); text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; }
.hero h1,.hero h2,.hero p { color:white !important; }
.hero h1 { margin:.65rem 0 1.2rem; max-width:760px; }
.hero p { max-width:680px; color:#cbdad5 !important; font-size:1.04rem; line-height:1.75; }
.pill { display:inline-block; margin-top:1rem; padding:.5rem .8rem; border-radius:999px; background:rgba(216,237,131,.13); color:var(--lime); font-size:.78rem; font-weight:600; }
.hero-visual {
    position:absolute; right:6%; bottom:9%; z-index:2; width:min(31vw,390px); min-width:290px;
    background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.18); backdrop-filter:blur(12px);
    border-radius:22px; padding:1.25rem; box-shadow:0 20px 50px rgba(0,0,0,.15);
}
.visual-label { color:#b8cbc5; font-size:.72rem; text-transform:uppercase; letter-spacing:.12em; }
.visual-value { font:700 clamp(2rem,4vw,3.4rem) 'Space Grotesk',sans-serif; color:#fff; margin:.35rem 0; }
.visual-line { height:7px; border-radius:999px; background:rgba(255,255,255,.14); overflow:hidden; }
.visual-line span { display:block; width:84%; height:100%; background:var(--lime); border-radius:inherit; }
.visual-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:.7rem; margin-top:1rem; }
.visual-stat { border-top:1px solid rgba(255,255,255,.12); padding-top:.65rem; }
.visual-stat b { color:#fff; display:block; font-size:1rem; }
.visual-stat span { color:#a9beb8; font-size:.7rem; }

/* Navigation control */
div[data-testid="stRadio"] > div { gap:.55rem; flex-wrap:wrap; }
div[data-testid="stRadio"] label {
    background:rgba(255,255,255,.7); border:1px solid var(--line); border-radius:999px;
    padding:.62rem 1rem !important; margin:0 !important; transition:.15s ease;
}
div[data-testid="stRadio"] label:hover { border-color:#b7c6bf; }
div[data-testid="stRadio"] label p { color:var(--ink) !important; font-size:1.02rem !important; font-weight:600 !important; }

/* Streamlit top-right deploy control */
div[data-testid="stDeployButton"] button {
    font-size:1rem !important;
    min-height:40px !important;
    padding:.45rem .9rem !important;
}

/* Sections/cards */
.section-wrap { padding:1rem 0; }
.section-kicker { color:var(--green); text-transform:uppercase; letter-spacing:.15em; font-size:.7rem; font-weight:700; margin-top:2.8rem; margin-bottom:.55rem; }
.section-title { font:700 clamp(1.9rem,3vw,2.8rem) 'Space Grotesk',sans-serif; color:var(--ink); letter-spacing:-.035em; }
.section-copy { max-width:720px; color:var(--muted); line-height:1.7; }
.card { background:#fff; border:1px solid var(--line); border-radius:20px; padding:1.45rem; box-shadow:var(--shadow); }
.card-flat { background:#fdfefd; border:1px solid var(--line); border-radius:18px; padding:1.3rem; }
.card h3 { margin-top:0; }
.icon-box { width:44px;height:44px;display:grid;place-items:center;border-radius:13px;background:var(--mint);font-size:1.15rem;margin-bottom:.8rem; }
.card p { line-height:1.65; }

/* Stats */
.stat-card { background:#fff;border:1px solid var(--line);border-radius:18px;padding:1.25rem;box-shadow:0 8px 28px rgba(20,40,34,.045); }
.stat-number { font:700 clamp(1.65rem,2.6vw,2.2rem) 'Space Grotesk',sans-serif;color:var(--green); }
.stat-label { font-size:.8rem;color:var(--muted);margin-top:.15rem; }

/* Mode cards */
.mode-card { min-height:205px; }
.mode-tag { display:inline-block;padding:.3rem .55rem;border-radius:999px;background:#f0f5ef;color:var(--green);font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em; }

/* Controls */
div[data-testid="stButton"] > button,
.stButton > button {
    min-height:68px !important;
    height:68px !important;
    border-radius:14px !important;
    border:1px solid var(--green) !important;
    background:var(--green) !important;
    color:#ffffff !important;
    font-size:1.12rem !important;
    font-weight:800 !important;
    letter-spacing:.01em !important;
    padding:.85rem 1.6rem !important;
    box-shadow:0 8px 22px rgba(18,63,53,.12) !important;
    opacity:1 !important;
}
div[data-testid="stButton"] > button *,
.stButton > button * {
    color:#ffffff !important;
    font-size:1.12rem !important;
    font-weight:800 !important;
    opacity:1 !important;
}
div[data-testid="stButton"] > button:hover,
.stButton > button:hover {
    background:var(--green-2) !important;
    border-color:var(--green-2) !important;
    color:#ffffff !important;
    transform:translateY(-1px);
    box-shadow:0 11px 28px rgba(18,63,53,.18) !important;
}
.stFormSubmitButton > button {
    min-height:58px !important;
    height:58px !important;
    border-radius:13px !important;
    background:var(--green) !important;
    color:#ffffff !important;
    font-size:1.08rem !important;
    font-weight:800 !important;
}
.stFormSubmitButton > button * {
    color:#ffffff !important;
    font-size:1.08rem !important;
    font-weight:800 !important;
}
div[data-baseweb="input"] > div, div[data-baseweb="select"] > div, textarea { border-radius:11px !important; border-color:#d9e0dc !important; background:#fff !important; }
.stTextInput input, .stNumberInput input { color:var(--ink) !important; }

/* Homepage CTA buttons */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button {
    min-height:72px !important;
    height:72px !important;
    font-size:1.18rem !important;
    font-weight:800 !important;
}
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button * {
    font-size:1.18rem !important;
    font-weight:800 !important;
    color:#ffffff !important;
}

/* Result */
.result-card { background:var(--green); color:#fff; border-radius:24px; padding:1.7rem; min-height:240px; box-shadow:0 18px 45px rgba(18,63,53,.15); }
.result-card .eyebrow { color:var(--lime); }
.result-value { font:700 clamp(2.6rem,5vw,4.3rem) 'Space Grotesk',sans-serif; margin:.4rem 0; }
.result-muted { color:#c4d5cf; line-height:1.6; }

/* Tables */
div[data-testid="stDataFrame"] { border-radius:14px; overflow:hidden; }

/* Footer */
.site-footer { margin-top:4rem; padding:1.8rem 0 .3rem; border-top:1px solid var(--line); display:flex; justify-content:space-between; gap:1rem; color:var(--muted); font-size:.78rem; }

@media (max-width: 1050px) {
    .hero-visual { position:relative; right:auto; bottom:auto; width:100%; min-width:0; margin-top:2rem; }
    .hero { min-height:auto; }
}
@media (max-width: 700px) {
    .block-container { padding: .7rem 1rem 3rem !important; }
    .site-nav { align-items:flex-start; }
    .nav-badge { display:none; }
    .hero { border-radius:22px; padding:2rem 1.35rem; }
    .hero h1 { font-size:2.8rem !important; }
    .site-footer { flex-direction:column; }
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
<div class="site-nav">
  <div class="brand">
    <div class="brand-mark">⚽</div>
    <div>
      <div class="brand-name">PlayerValue AI</div>
      <div class="brand-sub">Football market intelligence</div>
    </div>
  </div>
  <div class="nav-badge">AI-powered • XGBoost</div>
</div>
""",
    unsafe_allow_html=True,
)

if "page" not in st.session_state:
    st.session_state["page"] = "Home"

# Apply routes requested by homepage CTA buttons BEFORE the navigation widget
# is instantiated. This avoids StreamlitAPIException for widget session state.
if "pending_page" in st.session_state:
    _pending_page = st.session_state.pop("pending_page")
    st.session_state["page"] = _pending_page
    st.session_state["main_navigation"] = _pending_page

if "pending_predictor_mode" in st.session_state:
    _pending_mode = st.session_state.pop("pending_predictor_mode")
    st.session_state["predictor_mode"] = _pending_mode
    st.session_state["predictor_mode_radio"] = _pending_mode

NAV = ["Home", "Predictor", "Players", "Analytics", "Model Insights", "About"]
current_page = st.radio(
    "Main navigation",
    NAV,
    index=NAV.index(st.session_state["page"]) if st.session_state["page"] in NAV else 0,
    horizontal=True,
    label_visibility="collapsed",
    key="main_navigation",
)
st.session_state["page"] = current_page
page = current_page

# =========================================================
# HOME
# =========================================================
if page == "Home":
    st.markdown(
        f"""
<div class="hero">
  <div class="hero-content">
    <div class="eyebrow">AI • Football Analytics • Market Intelligence</div>
    <h1>Understand what a player could be worth.</h1>
    <p>Estimate football player market value using performance statistics and the project's existing XGBoost regression model — then explore the data behind the estimate.</p>
    <span class="pill">{len(model_features)} model features • {len(players_df):,} player records</span>
  </div>
  <div class="hero-visual">
    <div class="visual-label">Example model estimate</div>
    <div class="visual-value">€42.8M</div>
    <div class="visual-label">Performance profile</div>
    <div class="visual-line"><span></span></div>
    <div class="visual-grid">
      <div class="visual-stat"><b>2,841</b><span>Minutes</span></div>
      <div class="visual-stat"><b>18</b><span>Goals</span></div>
      <div class="visual-stat"><b>11</b><span>Assists</span></div>
      <div class="visual-stat"><b>84%</b><span>Profile score</span></div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    b1, b2 = st.columns([1, 1])
    with b1:
        if st.button("⚡  Start a prediction", use_container_width=True):
            go_to("Predictor")
            st.rerun()
    with b2:
        if st.button("🔎  Explore players", use_container_width=True):
            go_to("Players")
            st.rerun()

    st.markdown('<div class="section-kicker">Project at a glance</div>', unsafe_allow_html=True)
    stat_cols = st.columns(4)
    stats = [
        (f"{len(players_df):,}", "Player records"),
        (f"{len(model_features)}", "Model features"),
        ("XGBoost", "Regression model"),
        ("Python", "Core technology"),
    ]
    for col, (num, label) in zip(stat_cols, stats):
        with col:
            st.markdown(f'<div class="stat-card"><div class="stat-number">{num}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-kicker">Ways to use PlayerValue AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Analyze a player your way.</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-copy">The original Manual, Search and Compare workflows are all retained and are now organized as a clearer product experience.</p>', unsafe_allow_html=True)

    cards = [
        ("✏️", "Manual prediction", "Enter the original player performance inputs and generate an XGBoost market-value estimate.", "PREDICTOR"),
        ("🔍", "Search player", "Find a player in the included dataset and inspect their profile, market value and model estimate.", "PLAYER LOOKUP"),
        ("⚖️", "Compare players", "Place two players side by side and compare market values, predictions and performance metrics.", "COMPARISON"),
    ]
    cols = st.columns(3)
    for col, (icon, title, body, tag) in zip(cols, cards):
        with col:
            st.markdown(f'<div class="card mode-card"><div class="icon-box">{icon}</div><span class="mode-tag">{tag}</span><h3>{title}</h3><p>{body}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-kicker">How it works</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">From player statistics to market insight.</div>', unsafe_allow_html=True)
    steps = [
        ("01", "Input", "Enter a player profile manually or select an existing player from the dataset."),
        ("02", "Predict", "The existing feature preparation pipeline sends the model-ordered data into XGBoost."),
        ("03", "Understand", "Review the estimate, performance statistics, comparisons and feature importance."),
    ]
    cols = st.columns(3)
    for col, (num, title, body) in zip(cols, steps):
        with col:
            st.markdown(f'<div class="card"><div class="eyebrow" style="color:var(--green)">{num}</div><h3>{title}</h3><p>{body}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-kicker">Machine learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Built around explainable football data.</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    for col, icon, title, body in [
        (m1, "🧠", "XGBoost regression", "The saved XGBoost model produces the market-value estimate."),
        (m2, "📊", "Performance features", "Passing, carrying, minutes, defensive and other statistics feed the model."),
        (m3, "💡", "Feature importance", "The included feature-importance data makes the model easier to inspect."),
    ]:
        with col:
            st.markdown(f'<div class="card-flat"><div class="icon-box">{icon}</div><h3>{title}</h3><p>{body}</p></div>', unsafe_allow_html=True)

# =========================================================
# PREDICTOR
# =========================================================
elif page == "Predictor":
    st.markdown('<div class="section-kicker">Prediction studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Predict, search or compare.</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-copy">All three original workflows are available here. The redesigned interface changes the presentation, not the underlying model or data pipeline.</p>', unsafe_allow_html=True)

    modes = ["Manual", "Search", "Compare"]
    if "predictor_mode" not in st.session_state:
        st.session_state["predictor_mode"] = "Manual"
    mode = st.radio("Analysis mode", modes, index=modes.index(st.session_state["predictor_mode"]), horizontal=True, label_visibility="collapsed", key="predictor_mode_radio")
    st.session_state["predictor_mode"] = mode

    # ------------------------- MANUAL -------------------------
    if mode == "Manual":
        st.markdown('<div class="section-kicker">01 · Manual</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Build a player profile.</div>', unsafe_allow_html=True)
        st.markdown('<p class="section-copy">These are the original user-facing prediction inputs. The complete trained feature vector is still created in the same model feature order.</p>', unsafe_allow_html=True)

        with st.form("prediction_form"):
            st.markdown("### Player & playing time")
            player_inputs = {}
            c = st.columns(3)
            for i, (col, default) in enumerate(list(DEFAULT_VALUES.items())[:3]):
                with c[i]:
                    if col == "Comp_Premier League":
                        player_inputs[col] = st.selectbox("Competition", [True, False], index=0 if default else 1, help=FIELD_HELP[col], key=f"manual_{col}")
                    else:
                        if col == "Age":
                            player_inputs[col] = st.number_input(
                                "Age (1–100)",
                                min_value=1,
                                max_value=100,
                                value=int(default),
                                step=1,
                                format="%d",
                                help="Enter the player's age as a whole number from 1 to 100.",
                                key=f"manual_{col}",
                            )
                        else:
                            player_inputs[col] = st.number_input(
                                pretty_feature(col),
                                value=float(default),
                                help=FIELD_HELP.get(col),
                                key=f"manual_{col}",
                            )

            st.markdown("### Passing & possession")
            c = st.columns(3)
            for i, col in enumerate(["PasTotCmp%", "Rec", "TouAttPen", "PasProg", "PasShoAtt", "TouAtt3rd"]):
                with c[i % 3]:
                    player_inputs[col] = st.number_input(pretty_feature(col), value=float(DEFAULT_VALUES[col]), help=FIELD_HELP[col], key=f"manual_{col}")

            st.markdown("### Carrying & attacking movement")
            c = st.columns(3)
            for i, col in enumerate(["CarTotDist", "CarProg", "GcaPassLive", "90s"]):
                with c[i % 3]:
                    player_inputs[col] = st.number_input(pretty_feature(col), value=float(DEFAULT_VALUES[col]), help=FIELD_HELP[col], key=f"manual_{col}")

            st.markdown("### Defensive & physical actions")
            c = st.columns(3)
            for i, col in enumerate(["Tkl+Int", "AerWon", "Recov", "CrdY", "AerLost", "Int", "TklDef3rd", "ToTkl"]):
                with c[i % 3]:
                    player_inputs[col] = st.number_input(pretty_feature(col), value=float(DEFAULT_VALUES[col]), help=FIELD_HELP[col], key=f"manual_{col}")

            submitted = st.form_submit_button("⚡  Predict Market Value", use_container_width=True)

        if submitted:
            age_value = player_inputs.get("Age")
            try:
                valid_age = (
                    age_value is not None
                    and float(age_value).is_integer()
                    and 1 <= int(age_value) <= 100
                )
            except (TypeError, ValueError):
                valid_age = False

            if not valid_age:
                st.error("Please enter a valid age: a whole number between 1 and 100.")
                st.stop()

            player_inputs["Age"] = int(age_value)
            prediction = float(model.predict(build_prediction_input(player_inputs))[0])
            st.markdown('<div class="section-kicker">Prediction result</div>', unsafe_allow_html=True)
            result, explain = st.columns([1, 1.5])
            with result:
                st.markdown(f'''
<div class="result-card">
  <div class="eyebrow">Estimated market value</div>
  <div class="result-value">{money_eur(prediction)}</div>
  <div class="result-muted">Generated from the supplied football performance profile using the existing XGBoost model.</div>
</div>
''', unsafe_allow_html=True)
            with explain:
                top = feature_importance_df.sort_values("Feature Importance Score", ascending=False).head(10).copy()
                top["Feature"] = top["Variable"].map(pretty_feature)
                fig = px.bar(
                    top.sort_values("Feature Importance Score"),
                    x="Feature Importance Score", y="Feature", orientation="h",
                    labels={"Feature Importance Score": "Importance", "Feature": ""},
                    color_discrete_sequence=CHART_GREENS,
                )
                style_chart(fig, 390, margin=dict(l=0, r=15, t=35, b=10))
                st.plotly_chart(fig, use_container_width=True)

    # ------------------------- SEARCH -------------------------
    elif mode == "Search":
        st.markdown('<div class="section-kicker">02 · Search</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Explore an existing player.</div>', unsafe_allow_html=True)
        st.markdown('<p class="section-copy">Search the included football dataset and open a player profile with their statistics, market value and model estimate.</p>', unsafe_allow_html=True)

        names = sorted(players_df["Player"].dropna().astype(str).unique()) if "Player" in players_df.columns else []
        query = st.text_input("🔍 Search player", placeholder="Type a player name…", key="player_search")
        matches = [n for n in names if not query or query.lower() in n.lower()]
        st.caption(f"{len(matches):,} matching players")
        selected = st.selectbox("Select player", matches[:300] if matches else ["No player found"], key="selected_player")

        if selected != "No player found":
            row = players_df[players_df["Player"].astype(str) == selected].iloc[0]
            predicted = predict_from_row(row)
            actual = actual_market_value(selected)

            a, b, c, d = st.columns(4)
            a.metric("Model prediction", money_eur(predicted))
            b.metric("Market value", str(actual) if actual is not None else "N/A")
            c.metric("Age", str(row.get("Age", "—")))
            d.metric("Minutes", str(row.get("Min", "—")))

            left, right = st.columns([1, 1.7])
            with left:
                st.markdown(f'''
<div class="card">
  <span class="mode-tag">PLAYER PROFILE</span>
  <h2 style="font-size:2rem !important;margin-top:.8rem">{selected}</h2>
  <p><b>Position:</b> {row.get("Pos", "—")}</p>
  <p><b>Club:</b> {row.get("Squad", "—")}</p>
  <p><b>Competition:</b> {row.get("Comp", "—")}</p>
  <p><b>90s:</b> {row.get("90s", "—")}</p>
</div>
''', unsafe_allow_html=True)
            with right:
                stat_cols = [x for x in ["Goals", "Assists", "PasTotCmp%", "Rec", "CarProg", "Tkl+Int", "AerWon", "AerLost", "TouAttPen", "PasProg", "GcaPassLive", "Recov"] if x in players_df.columns]
                if stat_cols:
                    stats_df = pd.DataFrame({"Metric": [pretty_feature(x) for x in stat_cols], "Value": [row[x] for x in stat_cols]})
                    st.markdown("### Performance statistics")
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)

            st.markdown("### Player profile snapshot")
            numeric_candidates = [x for x in ["Goals", "Assists", "Min", "PasTotCmp%", "Rec", "CarProg", "Tkl+Int", "AerWon"] if x in players_df.columns]
            if numeric_candidates:
                chart_df = pd.DataFrame({"Metric": [pretty_feature(x) for x in numeric_candidates], "Value": [pd.to_numeric(row[x], errors="coerce") for x in numeric_candidates]}).dropna()
                fig = px.bar(
                    chart_df, x="Value", y="Metric", orientation="h",
                    title="Selected performance indicators",
                    color_discrete_sequence=CHART_GREENS,
                )
                style_chart(fig, 360, margin=dict(l=0, r=10, t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)

    # ------------------------- COMPARE -------------------------
    else:
        st.markdown('<div class="section-kicker">03 · Compare</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Compare two players side by side.</div>', unsafe_allow_html=True)
        st.markdown('<p class="section-copy">Compare the same football performance indicators, actual market values and model estimates without changing the underlying data.</p>', unsafe_allow_html=True)

        names = sorted(players_df["Player"].dropna().astype(str).unique()) if "Player" in players_df.columns else []
        if len(names) >= 2:
            ca, cb = st.columns(2)
            with ca:
                player_a = st.selectbox("Player 1", names, key="compare_a")
            with cb:
                default_b = 1 if len(names) > 1 else 0
                player_b = st.selectbox("Player 2", names, index=default_b, key="compare_b")

            row_a = players_df[players_df["Player"].astype(str) == player_a].iloc[0]
            row_b = players_df[players_df["Player"].astype(str) == player_b].iloc[0]
            pred_a, pred_b = predict_from_row(row_a), predict_from_row(row_b)
            actual_a, actual_b = actual_market_value(player_a), actual_market_value(player_b)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'''
<div class="result-card"><div class="eyebrow">PLAYER 1</div><h2 style="color:#fff !important;font-size:2rem !important">{player_a}</h2><div class="result-value" style="font-size:2.5rem">{money_eur(pred_a)}</div><div class="result-muted">Model estimate<br>Market value: {actual_a if actual_a is not None else "N/A"}</div></div>
''', unsafe_allow_html=True)
            with c2:
                st.markdown(f'''
<div class="result-card"><div class="eyebrow">PLAYER 2</div><h2 style="color:#fff !important;font-size:2rem !important">{player_b}</h2><div class="result-value" style="font-size:2.5rem">{money_eur(pred_b)}</div><div class="result-muted">Model estimate<br>Market value: {actual_b if actual_b is not None else "N/A"}</div></div>
''', unsafe_allow_html=True)

            metrics = [x for x in ["Age", "Min", "90s", "Goals", "Assists", "PasTotCmp%", "Rec", "CarProg", "Tkl+Int", "AerWon", "AerLost", "TouAttPen", "PasProg", "GcaPassLive", "Recov"] if x in players_df.columns]
            compare_df = pd.DataFrame({"Metric": [pretty_feature(x) for x in metrics], player_a: [row_a[x] for x in metrics], player_b: [row_b[x] for x in metrics]})
            st.markdown("### Performance comparison")
            st.dataframe(compare_df, use_container_width=True, hide_index=True)

            numeric = compare_df.copy()
            long_df = numeric.melt(id_vars="Metric", var_name="Player", value_name="Value")
            long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")
            long_df = long_df.dropna()
            # Keep the two comparison charts side-by-side and make the values directly visible.
            chart_left, chart_right = st.columns(2, gap="large")

            with chart_left:
                if not long_df.empty:
                    fig = px.bar(
                        long_df, x="Metric", y="Value", color="Player", barmode="group",
                        title="Side-by-side performance metrics",
                        color_discrete_sequence=CHART_GREENS,
                        text="Value",
                    )
                    fig.update_traces(
                        texttemplate="%{text:.0f}",
                        textposition="outside",
                        cliponaxis=False,
                        hovertemplate="<b>%{fullData.name}</b><br>%{x}<br>Value: %{y:,.2f}<extra></extra>",
                    )
                    style_chart(
                        fig, 520,
                        barmode="group",
                        margin=dict(l=45, r=20, t=105, b=125),
                        title=dict(x=0, xanchor="left", y=0.98, yanchor="top"),
                        legend_title_text="",
                        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0),
                    )
                    fig.update_xaxes(tickangle=-35)
                    st.plotly_chart(fig, use_container_width=True)

            value_df = pd.DataFrame({"Player": [player_a, player_b], "Model Estimate": [pred_a, pred_b]})
            if actual_a is not None and actual_b is not None:
                value_df["Actual Market Value"] = [float(actual_a), float(actual_b)] if all(isinstance(x, (int, float)) for x in [actual_a, actual_b]) else [0, 0]

            with chart_right:
                fig2 = go.Figure()
                fig2.add_trace(
                    go.Bar(
                        name="Model Estimate",
                        x=value_df["Player"],
                        y=value_df["Model Estimate"],
                        marker_color=CHART_GREENS[0],
                        text=value_df["Model Estimate"],
                        texttemplate="€%{text:.2s}",
                        textposition="outside",
                        cliponaxis=False,
                        hovertemplate="<b>%{x}</b><br>Model estimate: €%{y:,.0f}<extra></extra>",
                    )
                )
                if "Actual Market Value" in value_df.columns and value_df["Actual Market Value"].sum() > 0:
                    fig2.add_trace(
                        go.Bar(
                            name="Actual Market Value",
                            x=value_df["Player"],
                            y=value_df["Actual Market Value"],
                            marker_color=CHART_GREENS[2],
                            text=value_df["Actual Market Value"],
                            texttemplate="€%{text:.2s}",
                            textposition="outside",
                            cliponaxis=False,
                            hovertemplate="<b>%{x}</b><br>Actual market value: €%{y:,.0f}<extra></extra>",
                        )
                    )
                style_chart(
                    fig2, 520,
                    barmode="group",
                    title=dict(text="Market value comparison", x=0, xanchor="left", y=0.98, yanchor="top"),
                    margin=dict(l=45, r=20, t=105, b=85),
                    legend_title_text="",
                    legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0),
                )
                st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# PLAYERS
# =========================================================
elif page == "Players":
    st.markdown('<div class="section-kicker">Player database</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Explore the football dataset.</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-copy">Search and filter the included player records before opening a player through the Predictor → Search workflow.</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Players", f"{len(players_df):,}")
    c2.metric("Statistics", f"{len(players_df.columns):,}")
    c3.metric("Competitions", f"{players_df['Comp'].nunique():,}" if "Comp" in players_df.columns else "—")
    c4.metric("Positions", f"{players_df['Pos'].nunique():,}" if "Pos" in players_df.columns else "—")

    f1, f2, f3 = st.columns([1.2, 1, 1])
    with f1:
        search = st.text_input("🔍 Search player", placeholder="e.g. Haaland")
    with f2:
        competitions = sorted(players_df["Comp"].dropna().astype(str).unique()) if "Comp" in players_df.columns else []
        selected_comp = st.selectbox("Competition", ["All"] + competitions)
    with f3:
        positions = sorted(players_df["Pos"].dropna().astype(str).unique()) if "Pos" in players_df.columns else []
        selected_pos = st.selectbox("Position", ["All"] + positions)

    filtered = players_df.copy()
    if selected_comp != "All" and "Comp" in filtered.columns:
        filtered = filtered[filtered["Comp"].astype(str) == selected_comp]
    if selected_pos != "All" and "Pos" in filtered.columns:
        filtered = filtered[filtered["Pos"].astype(str) == selected_pos]
    if search and "Player" in filtered.columns:
        filtered = filtered[filtered["Player"].astype(str).str.contains(search, case=False, na=False)]

    st.caption(f"Showing {min(len(filtered), 200):,} of {len(filtered):,} matching records")

    display_cols = [c for c in ["Player", "Nation", "Pos", "Squad", "Comp", "Age", "MP", "Min", "Goals", "Assists", "PasTotCmp%", "Rec", "CarProg"] if c in filtered.columns]
    st.dataframe(filtered[display_cols].head(200), use_container_width=True, hide_index=True, height=520)

    if "Age" in filtered.columns and "Min" in filtered.columns and not filtered.empty:
        st.markdown('<div class="section-kicker">Dataset visualization</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Age vs playing time.</div>', unsafe_allow_html=True)
        fig = px.scatter(
            filtered.head(1000),
            x="Age", y="Min",
            hover_name="Player" if "Player" in filtered.columns else None,
            title="Age vs Minutes Played",
            color_discrete_sequence=CHART_GREENS,
        )
        style_chart(fig, 440, margin=dict(l=0, r=0, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# ANALYTICS
# =========================================================
elif page == "Analytics":
    st.markdown('<div class="section-kicker">Data intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">See the dataset from different angles.</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-copy">Interactive charts provide a wider view of the football performance data used throughout the application.</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Players", f"{len(players_df):,}")
    if "Age" in players_df.columns:
        c2.metric("Average age", f"{pd.to_numeric(players_df['Age'], errors='coerce').mean():.1f}")
    else:
        c2.metric("Average age", "—")
    if "Min" in players_df.columns:
        c3.metric("Average minutes", f"{pd.to_numeric(players_df['Min'], errors='coerce').mean():,.0f}")
    else:
        c3.metric("Average minutes", "—")
    c4.metric("Competitions", f"{players_df['Comp'].nunique():,}" if "Comp" in players_df.columns else "—")

    charts = []
    if all(x in players_df.columns for x in ["Age", "Min"]):
        charts.append(("Age vs Minutes", px.scatter(
            players_df.head(1500), x="Age", y="Min",
            hover_name="Player" if "Player" in players_df.columns else None,
            color_discrete_sequence=CHART_GREENS,
        )))
    if all(x in players_df.columns for x in ["Goals", "Min"]):
        charts.append(("Goals vs Minutes", px.scatter(
            players_df.head(1500), x="Min", y="Goals",
            hover_name="Player" if "Player" in players_df.columns else None,
            color_discrete_sequence=CHART_GREENS,
        )))
    if all(x in players_df.columns for x in ["Assists", "Min"]):
        charts.append(("Assists vs Minutes", px.scatter(
            players_df.head(1500), x="Min", y="Assists",
            hover_name="Player" if "Player" in players_df.columns else None,
            color_discrete_sequence=CHART_GREENS,
        )))

    for i in range(0, len(charts), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(charts):
                title, fig = charts[i + j]
                style_chart(fig, 390, title=title, margin=dict(l=0, r=0, t=55, b=10))
                with col:
                    st.plotly_chart(fig, use_container_width=True)

    if "Pos" in players_df.columns:
        st.markdown('<div class="section-kicker">Player distribution</div>', unsafe_allow_html=True)
        pos_counts = players_df["Pos"].astype(str).value_counts().reset_index()
        pos_counts.columns = ["Position", "Players"]
        fig = px.bar(
            pos_counts.head(20), x="Position", y="Players",
            title="Players by position",
            color_discrete_sequence=CHART_GREENS,
        )
        style_chart(fig, 400)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# MODEL INSIGHTS
# =========================================================
elif page == "Model Insights":
    st.markdown('<div class="section-kicker">Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Understand what the model uses.</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-copy">Feature importance provides a useful view of which variables have the strongest contribution in the trained XGBoost model. It is not a causal explanation of player value.</p>', unsafe_allow_html=True)

    fi = feature_importance_df.copy()
    fi["Feature"] = fi["Variable"].map(pretty_feature)
    fi = fi.sort_values("Feature Importance Score", ascending=False)

    a, b, c, d = st.columns(4)
    a.metric("Algorithm", "XGBoost")
    b.metric("Features", len(model_features))
    c.metric("Importance entries", len(fi))
    d.metric("Task", "Regression")

    left, right = st.columns([1.7, 1])
    with left:
        fig = px.bar(
            fi.head(15).sort_values("Feature Importance Score"),
            x="Feature Importance Score", y="Feature", orientation="h",
            title="Top 15 feature importance scores",
            labels={"Feature Importance Score":"Importance", "Feature":""},
            color_discrete_sequence=CHART_GREENS,
        )
        style_chart(fig, 600, margin=dict(l=0, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown('''
<div class="card">
  <span class="mode-tag">MODEL WORKFLOW</span>
  <h3 style="margin-top:.8rem">From data to estimate</h3>
  <p><b>01 · Football data</b><br>Player performance statistics form the input dataset.</p>
  <p><b>02 · Feature preparation</b><br>The app builds the full model feature vector in the trained order.</p>
  <p><b>03 · XGBoost regression</b><br>The serialized model produces the market-value estimate.</p>
  <p><b>04 · Exploration</b><br>Charts and comparisons make the result easier to inspect.</p>
</div>
''', unsafe_allow_html=True)

    st.markdown('<div class="section-kicker">Feature ranking</div>', unsafe_allow_html=True)
    st.dataframe(fi[["Feature", "Feature Importance Score"]].head(30), use_container_width=True, hide_index=True)

# =========================================================
# ABOUT
# =========================================================
elif page == "About":
    st.markdown('<div class="section-kicker">About PlayerValue AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Where football data meets machine learning.</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-copy">PlayerValue AI is an academic machine-learning project that applies an XGBoost regression model to football player statistics to estimate market value. The estimates are model outputs, not official valuations.</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card"><div class="icon-box">🐍</div><h3>Python stack</h3><p>Python, Streamlit, Pandas, Plotly and the existing XGBoost model are retained.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="icon-box">📈</div><h3>Football data</h3><p>The project uses player-level performance statistics, market-value data and feature-importance data.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card"><div class="icon-box">🧠</div><h3>Machine learning</h3><p>The saved XGBoost regression model is used for inference; the redesign does not replace the model.</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-kicker">Technology</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">The existing technology, presented better.</div>', unsafe_allow_html=True)
    tech_cols = st.columns(5)
    for col, tech in zip(tech_cols, ["Python", "Streamlit", "XGBoost", "Pandas", "Plotly"]):
        with col:
            st.markdown(f'<div class="card-flat" style="text-align:center;font-weight:700;color:var(--green)">{tech}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-kicker">Project workflow</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">One application, several ways to explore.</div>', unsafe_allow_html=True)
    st.markdown('''
<div class="card">
  <p><b>Manual:</b> enter the original prediction inputs and run the existing model.</p>
  <p><b>Search:</b> find an existing player and inspect their data and model estimate.</p>
  <p><b>Compare:</b> place two players side by side using the same dataset and prediction pipeline.</p>
  <p><b>Analytics:</b> explore distributions, relationships and player-level statistics.</p>
  <p><b>Model Insights:</b> inspect the supplied feature-importance data and model workflow.</p>
</div>
''', unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
<div class="site-footer">
  <div><b style="color:var(--green)">⚽ PlayerValue AI</b><br>AI-powered football market intelligence.</div>
  <div>Predict • Explore • Compare • Understand</div>
</div>
""",
    unsafe_allow_html=True,
)
