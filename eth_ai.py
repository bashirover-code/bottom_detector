import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go
import yfinance as yf

# ============================================================
# НАСТРОЙКИ СТРАНИЦЫ
# ============================================================

st.set_page_config(
    page_title="ДЕТЕКТОР ДНА — MULTI-ASSET",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Полностью скрываем левую панель и лишние элементы
st.markdown("""
    <style>
        [data-testid="collapsedControl"] { display: none; }
        .main > div { padding-top: 0; }
        header { display: none; }
        footer { display: none; }
        .stApp { background-color: #0a0a0a; }
        .stSelectbox > div { background-color: #1a1a1a; border: 1px solid #333; border-radius: 8px; }
        .stRadio > div { gap: 8px; }
        .stRadio label { background-color: #1a1a1a; padding: 4px 16px; border-radius: 20px; color: #ccc; }
        .stRadio [data-baseweb="radio"]:checked + label { background-color: #00ff44; color: #000; font-weight: bold; }
        div[data-testid="stMetricValue"] { font-size: 2rem !important; font-weight: 700 !important; }
        div[data-testid="stMetricLabel"] { font-size: 0.75rem !important; letter-spacing: 1px; color: #aaa; }
        .stButton button { background: #00ff44; color: #000; font-weight: bold; border: none; padding: 8px 24px; border-radius: 6px; }
        .stButton button:hover { background: #00cc33; color: #000; }
        h1, h2, h3, h4, h5, h6, p, span, div { font-family: 'Times New Roman', Times, serif !important; }
        .metric-card { background-color: #111; border-radius: 12px; padding: 16px; border-left: 3px solid; margin: 8px 0; }
        .signal-card { background: linear-gradient(135deg, #0f0f1a 0%, #0a0a0f 100%); border-radius: 16px; padding: 24px; text-align: center; margin: 24px 0; border: 1px solid #2a2a3e; }
        .section-title { font-size: 0.9rem; font-weight: 600; letter-spacing: 2px; color: #00ff44; margin-bottom: 16px; border-bottom: 1px solid #2a2a2a; padding-bottom: 8px; text-transform: uppercase; }
        hr { border-color: #2a2a2a; margin: 16px 0; }
    </style>
    <meta http-equiv="refresh" content="300">
""", unsafe_allow_html=True)

# ============================================================
# ЗАГОЛОВОК
# ============================================================

st.title("⚡ ДЕТЕКТОР ДНА")
st.markdown("<p style='color: #00ff44; font-size: 0.8rem; letter-spacing: 2px;'>ПРОФЕССИОНАЛЬНАЯ СИСТЕМА АНАЛИЗА РЫНКА</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ============================================================
# 1. СПИСКИ АКТИВОВ
# ============================================================

CRYPTO_LIST = [
    "ETH", "BTC", "SOL", "ASTER", "IMX", "ZK", "FIL", "STX", "RENDER",
    "ONDO", "GRT", "CELO", "CRV", "TWT", "SUI", "APE", "ARKM", "ONE",
    "GOAT", "POL", "LINK", "UNI", "TRUMP", "ARC", "NEAR", "ALGO", "FLOCK"
]

STOCK_LIST = [
    "HIMS", "SIL", "GDX", "TSLA", "LIT", "ZM", "URA", "PLTR",
    "EWW", "BABA", "COIN", "NVDA", "SBER", "MTSS", "HEAD"
]

VETERAN_LIST = ["BTC", "ETH", "BNB", "XRP", "LTC", "ADA", "DOGE", "AAPL", "MSFT", "NVDA"]

COINGECKO_IDS = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "FIL": "filecoin",
    "LINK": "chainlink", "UNI": "uniswap", "NEAR": "near", "ALGO": "algorand",
    "GRT": "the-graph", "CRV": "curve-dao-token", "STX": "blockstack",
    "RENDER": "render-token", "ONDO": "ondo-finance", "SUI": "sui",
    "APE": "apecoin", "IMX": "immutable-x", "ZK": "zkSync", "TWT": "trust-wallet-token",
    "CELO": "celo", "ARKM": "arkham", "ONE": "harmony", "GOAT": "goat",
    "POL": "polygon", "TRUMP": "maga", "ARC": "arc", "FLOCK": "flock"
}

# ============================================================
# 2. ЯДРО СИСТЕМЫ
# ============================================================

@st.cache_data(ttl=600)
def get_coingecko_fundamentals(coin_id):
    try:
        api_key = st.secrets.get("COINGECKO_API_KEY")
        if not api_key: return None
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
        params = {"localization": "false", "market_data": "true", "x_cg_demo_api_key": api_key}
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            md = data.get("market_data", {})
            return {
                "price_usd": md.get("current_price", {}).get("usd", 0),
                "market_cap": md.get("market_cap", {}).get("usd", 0),
                "fully_diluted_valuation": md.get("fully_diluted_valuation", {}).get("usd", 0),
                "total_volume": md.get("total_volume", {}).get("usd", 0),
                "price_change_24h": md.get("price_change_percentage_24h", 0)
            }
    except:
        return None

def get_adaptive_thresholds(z_scores):
    if len(z_scores) < 30: return -1.8, 1.5
    lower = max(-3.0, min(-1.0, np.percentile(z_scores, 5)))
    upper = min(3.0, max(0.5, np.percentile(z_scores, 95)))
    return lower, upper

def get_signal(z_score, lower_thr, upper_thr, is_veteran):
    if is_veteran:
        if z_score <= -1.8: return "ЭКСТРЕМАЛЬНАЯ ПОКУПКА", "#00ff44"
        elif z_score <= -1.2: return "НАКОПЛЕНИЕ", "#88ff88"
        elif z_score >= 1.5: return "ПРОДАЖА", "#ff2200"
        else: return "НЕЙТРАЛЬНО", "#ffaa44"
    else:
        if z_score <= lower_thr: return "ЭКСТРЕМАЛЬНАЯ ПОКУПКА", "#00ff44"
        elif z_score <= lower_thr * 0.7: return "НАКОПЛЕНИЕ", "#88ff88"
        elif z_score >= upper_thr: return "ПРОДАЖА", "#ff2200"
        else: return "НЕЙТРАЛЬНО", "#ffaa44"

@st.cache_data(ttl=600)
def load_crypto_data(symbol, days=500):
    try:
        api_key = st.secrets.get("CRYPTOCOMPARE_KEY")
        if not api_key: return None
        url = "https://min-api.cryptocompare.com/data/v2/histoday"
        params = {"fsym": symbol, "tsym": "USD", "limit": days, "api_key": api_key}
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if data.get("Response") == "Success":
                raw = data["Data"]["Data"]
                df = pd.DataFrame(raw)
                df["date"] = pd.to_datetime(df["time"], unit='s')
                df["close"] = df["close"].astype(float)
                return df.sort_values("date").reset_index(drop=True)
    except:
        return None
    return None

@st.cache_data(ttl=600)
def load_stock_data(symbol, days=500):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        if df is not None and not df.empty:
            df = df.reset_index()
            df = df.rename(columns={"Date": "date", "Close": "close"})
            return df[["date", "close"]]
    except:
        return None
    return None

def calculate_metrics(df):
    if df is None or len(df) < 30: return None, None, None, None, None, None
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    mean_ret = df["returns"].mean()
    std_ret = df["returns"].std()
    df["z_score"] = (df["returns"] - mean_ret) / (std_ret + 1e-10)
    df = df.fillna(0)
    z_scores = df["z_score"].values
    lower, upper = get_adaptive_thresholds(z_scores)
    price = df["close"].iloc[-1]
    z = df["z_score"].iloc[-1]
    sensitivity = 1.5 if len(df) > 365 else 1.0
    prob = 1 / (1 + np.exp(z * sensitivity))
    confidence = min(100, len(df) / 365 * 100)
    return df, price, z, prob, confidence, (lower, upper)

# ============================================================
# 3. ИНТЕРФЕЙС ВЫБОРА АКТИВА
# ============================================================

col_type, col_asset, _ = st.columns([1, 2, 4])
with col_type:
    asset_type = st.radio("ТИП", ["КРИПТО", "АКЦИИ"], horizontal=True, label_visibility="collapsed")
with col_asset:
    if asset_type == "КРИПТО":
        selected_asset = st.selectbox("", CRYPTO_LIST, label_visibility="collapsed")
    else:
        selected_asset = st.selectbox("", STOCK_LIST, label_visibility="collapsed")

st.markdown("<hr>", unsafe_allow_html=True)

# ============================================================
# 4. ЗАГРУЗКА ДАННЫХ
# ============================================================

is_crypto = selected_asset in CRYPTO_LIST
is_veteran = selected_asset in VETERAN_LIST

if is_crypto and selected_asset in COINGECKO_IDS:
    fundamentals = get_coingecko_fundamentals(COINGECKO_IDS[selected_asset])
else:
    fundamentals = None

with st.spinner("ЗАГРУЗКА ДАННЫХ..."):
    if is_crypto:
        df = load_crypto_data(selected_asset)
    else:
        df = load_stock_data(selected_asset)

if df is None or len(df) < 30:
    st.error("НЕДОСТАТОЧНО ДАННЫХ ДЛЯ АНАЛИЗА")
    st.stop()

df, price, z, prob, confidence, (lower, upper) = calculate_metrics(df)
signal_text, signal_color = get_signal(z, lower, upper, is_veteran)

# ============================================================
# 5. МЕТРИКИ
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("💰 ЦЕНА", f"${price:,.2f}")

with c2:
    st.metric("📊 Z-SCORE", f"{z:+.2f}")

with c3:
    prob_color = "#00ff44" if prob > 0.6 else "#ffaa44" if prob > 0.4 else "#ff2200"
    st.markdown(f"""
    <div class='metric-card' style='border-left-color: {prob_color};'>
        <div style='color: #aaa; font-size: 0.7rem;'>ВЕРОЯТНОСТЬ ДНА</div>
        <div style='font-size: 2rem; font-weight: 700; color: {prob_color};'>{prob*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class='metric-card' style='border-left-color: {signal_color};'>
        <div style='color: #aaa; font-size: 0.7rem;'>СИГНАЛ</div>
        <div style='font-size: 1.5rem; font-weight: 700; color: {signal_color};'>{signal_text}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 6. ГРАФИК ЦЕНЫ — ЖИРНЫЕ ЦВЕТНЫЕ ПОЛОСЫ + ЧЁРНАЯ ЛИНИЯ
# ============================================================

st.markdown("<div class='section-title'>📈 ГРАФИК ЦЕНЫ</div>", unsafe_allow_html=True)

df_chart = df.tail(500).copy()

def get_area_color(z, lower, upper):
    if z <= lower: return "rgba(0, 255, 68, 0.5)"
    elif z <= lower * 0.7: return "rgba(68, 255, 68, 0.4)"
    elif z <= -0.5: return "rgba(136, 255, 136, 0.3)"
    elif z <= 0.5: return "rgba(136, 136, 136, 0.2)"
    elif z <= 1.2: return "rgba(255, 170, 102, 0.4)"
    elif z <= upper: return "rgba(255, 102, 68, 0.5)"
    else: return "rgba(255, 34, 0, 0.6)"

fig = go.Figure()

for i in range(len(df_chart) - 1):
    color = get_area_color(df_chart["z_score"].iloc[i], lower, upper)
    fig.add_trace(go.Scatter(
        x=[df_chart["date"].iloc[i], df_chart["date"].iloc[i+1]],
        y=[df_chart["close"].iloc[i], df_chart["close"].iloc[i+1]],
        mode='lines', line=dict(color=color, width=14),
        fill='tozeroy', fillcolor=color,
        showlegend=False, hoverinfo='skip'
    ))

fig.add_trace(go.Scatter(
    x=df_chart["date"], y=df_chart["close"],
    mode='lines', line=dict(color='#000000', width=1.2),
    name="ЦЕНА", hovertemplate='%{y:,.2f}<extra></extra>'
))

fig.update_layout(
    height=380, template="plotly_dark", margin=dict(l=0, r=0, t=10, b=10),
    xaxis_title="", yaxis_title="", yaxis_type="log" if price > 100 else "linear",
    hovermode="x unified", paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a"
)
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 7. ГРАФИК Z-SCORE
# ============================================================

st.markdown("<div class='section-title'>📉 Z-SCORE И ПОРОГИ</div>", unsafe_allow_html=True)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=df_chart["date"], y=df_chart["z_score"],
    mode='lines', name='Z-SCORE', line=dict(color='#00d4ff', width=2.5),
    fill='tozeroy', fillcolor='rgba(0, 212, 255, 0.1)'
))
fig2.add_hline(y=lower, line_dash="dash", line_color="#00ff44", line_width=2,
               annotation_text=f"ПОКУПКА ({lower:.2f})", annotation_position="right")
fig2.add_hline(y=upper, line_dash="dash", line_color="#ff4422", line_width=2,
               annotation_text=f"ПРОДАЖА ({upper:.2f})", annotation_position="right")
fig2.add_hline(y=0, line_dash="dot", line_color="#555555")

fig2.update_layout(
    height=280, template="plotly_dark", margin=dict(l=0, r=0, t=10, b=10),
    xaxis_title="", yaxis_title="", paper_bgcolor="#0a0a0a", plot_bgcolor="#0a0a0a"
)
st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# 8. ФУНДАМЕНТАЛЬНЫЕ ДАННЫЕ (опционально)
# ============================================================

if fundamentals:
    st.markdown("<div class='section-title'>🔬 ФУНДАМЕНТАЛЬНЫЕ МЕТРИКИ</div>", unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    with f1: st.metric("РЫНОЧНАЯ КАП.", f"${fundamentals['market_cap']/1e9:.2f}B")
    with f2: st.metric("FDV", f"${fundamentals['fully_diluted_valuation']/1e9:.2f}B")
    with f3: st.metric("ОБЪЁМ 24Ч", f"${fundamentals['total_volume']/1e6:.1f}M")
    with f4:
        pc = fundamentals['price_change_24h']
        color = "#00ff44" if pc > 0 else "#ff2200"
        st.markdown(f"<div><div style='color:#aaa;'>ИЗМЕНЕНИЕ 24Ч</div><div style='color:{color}; font-size:1.5rem;'>{pc:+.1f}%</div></div>", unsafe_allow_html=True)

# ============================================================
# 9. ПОДВАЛ
# ============================================================

st.markdown("<hr>", unsafe_allow_html=True)
moscow_tz = timezone(timedelta(hours=3))
moscow_time = datetime.now(moscow_tz)
st.caption(f"📅 ОБНОВЛЕНО: {moscow_time.strftime('%Y-%m-%d %H:%M:%S')} (МСК)")
st.caption("⚡ АДАПТИВНЫЙ Z-SCORE | ИСТОЧНИКИ: CRYPTOCOMPARE / YFINANCE / COINGECKO")
st.caption("⚠️ НЕ ЯВЛЯЕТСЯ ИНВЕСТИЦИОННОЙ РЕКОМЕНДАЦИЕЙ")
