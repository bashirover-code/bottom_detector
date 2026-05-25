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
    page_title="Детектор Дна",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Скрываем левую панель и лишние элементы
st.markdown("""
    <style>
        [data-testid="collapsedControl"] { display: none; }
        .main > div { padding-top: 0; }
        header { display: none; }
        footer { display: none; }
        .stSelectbox > div { background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 8px; }
        .stRadio > div { gap: 8px; }
        .stRadio label { background-color: #e8e8e8; padding: 4px 16px; border-radius: 20px; color: #333; }
        .stRadio [data-baseweb="radio"]:checked + label { background-color: #00aa44; color: #fff; font-weight: bold; }
        div[data-testid="stMetricValue"] { font-size: 2rem !important; font-weight: 700 !important; color: #222; }
        div[data-testid="stMetricLabel"] { font-size: 0.75rem !important; letter-spacing: 1px; color: #666; }
        .stButton button { background: #00aa44; color: #fff; font-weight: bold; border: none; padding: 8px 24px; border-radius: 6px; }
        .stButton button:hover { background: #008833; color: #fff; }
        h1, h2, h3, h4, h5, h6, p, span, div { font-family: 'Times New Roman', Times, serif !important; }
        .metric-card { background-color: #f5f5f5; border-radius: 12px; padding: 16px; border-left: 3px solid; margin: 8px 0; }
        .signal-card { background: linear-gradient(135deg, #f0f0f0 0%, #e8e8e8 100%); border-radius: 16px; padding: 24px; text-align: center; margin: 24px 0; border: 1px solid #ccc; }
        .section-title { font-size: 0.9rem; font-weight: 600; letter-spacing: 2px; color: #00aa44; margin-bottom: 16px; border-bottom: 1px solid #ddd; padding-bottom: 8px; text-transform: uppercase; }
        hr { border-color: #ddd; margin: 16px 0; }
        .stDataFrame { border: 1px solid #ddd; border-radius: 12px; overflow: hidden; }
    </style>
    <meta http-equiv="refresh" content="300">
""", unsafe_allow_html=True)

# ============================================================
# ЗАГОЛОВОК
# ============================================================

st.title("Детектор Дна")
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
# 2. ФУНКЦИИ
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
        if z_score <= -1.8: return "ЭКСТРЕМАЛЬНАЯ ПОКУПКА", "#00aa44"
        elif z_score <= -1.2: return "НАКОПЛЕНИЕ", "#88cc44"
        elif z_score >= 1.5: return "ПРОДАЖА", "#cc2200"
        else: return "НЕЙТРАЛЬНО", "#ccaa00"
    else:
        if z_score <= lower_thr: return "ЭКСТРЕМАЛЬНАЯ ПОКУПКА", "#00aa44"
        elif z_score <= lower_thr * 0.7: return "НАКОПЛЕНИЕ", "#88cc44"
        elif z_score >= upper_thr: return "ПРОДАЖА", "#cc2200"
        else: return "НЕЙТРАЛЬНО", "#ccaa00"

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

def call_deepseek_analysis(asset_name, price, z, prob, signal_text):
    deepseek_key = st.secrets.get("DEEPSEEK_API_KEY", "")
    if not deepseek_key:
        return "AI-анализ недоступен. Добавьте DEEPSEEK_API_KEY в Secrets."
    
    prompt = f"""Проанализируй {asset_name}:

- Цена: ${price:,.2f}
- Z-Score: {z:.2f}
- Вероятность дна: {prob*100:.1f}%
- Сигнал: {signal_text}

Напиши краткий анализ (3-4 предложения) на русском:
1. Что означает текущий сигнал
2. Рекомендация (покупать/продавать/держать)
3. Главный фактор риска"""

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {deepseek_key}", "Content-Type": "application/json"}
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Ошибка AI: {response.status_code}"
    except Exception as e:
        return f"Ошибка: {str(e)[:100]}"

# ============================================================
# 3. ИНТЕРФЕЙС ВЫБОРА АКТИВА
# ============================================================

col_type, col_asset, _ = st.columns([1, 2, 4])
with col_type:
    asset_type = st.radio("Тип", ["Криптовалюты", "Акции"], horizontal=True, label_visibility="collapsed")
with col_asset:
    if asset_type == "Криптовалюты":
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

with st.spinner("Загрузка данных..."):
    if is_crypto:
        df = load_crypto_data(selected_asset)
    else:
        df = load_stock_data(selected_asset)

if df is None or len(df) < 30:
    st.error("Недостаточно данных для анализа")
    st.stop()

df, price, z, prob, confidence, (lower, upper) = calculate_metrics(df)
signal_text, signal_color = get_signal(z, lower, upper, is_veteran)

# ============================================================
# 5. МЕТРИКИ
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Цена", f"${price:,.2f}")

with c2:
    st.metric("Z-Score", f"{z:+.2f}")

with c3:
    prob_color = "#00aa44" if prob > 0.6 else "#ccaa00" if prob > 0.4 else "#cc2200"
    st.metric("Вероятность дна", f"{prob*100:.1f}%")

with c4:
    st.markdown(f"""
    <div class='metric-card' style='border-left-color: {signal_color};'>
        <div style='color: #666; font-size: 0.7rem;'>Сигнал</div>
        <div style='font-size: 1.5rem; font-weight: 700; color: {signal_color};'>{signal_text}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 6. ГРАФИК ЦЕНЫ
# ============================================================

st.markdown("<div class='section-title'>График цены</div>", unsafe_allow_html=True)

df_chart = df.tail(500).copy()

def get_area_color(z, lower, upper):
    if z <= lower: return "rgba(0, 170, 68, 0.3)"
    elif z <= lower * 0.7: return "rgba(136, 204, 68, 0.25)"
    elif z <= -0.5: return "rgba(136, 204, 68, 0.2)"
    elif z <= 0.5: return "rgba(136, 136, 136, 0.15)"
    elif z <= 1.2: return "rgba(204, 136, 68, 0.25)"
    elif z <= upper: return "rgba(204, 68, 0, 0.3)"
    else: return "rgba(204, 34, 0, 0.4)"

fig = go.Figure()

for i in range(len(df_chart) - 1):
    color = get_area_color(df_chart["z_score"].iloc[i], lower, upper)
    fig.add_trace(go.Scatter(
        x=[df_chart["date"].iloc[i], df_chart["date"].iloc[i+1]],
        y=[df_chart["close"].iloc[i], df_chart["close"].iloc[i+1]],
        mode='lines', line=dict(color=color, width=12),
        fill='tozeroy', fillcolor=color,
        showlegend=False, hoverinfo='skip'
    ))

fig.add_trace(go.Scatter(
    x=df_chart["date"], y=df_chart["close"],
    mode='lines', line=dict(color='#000000', width=1.2),
    name="Цена"
))

fig.update_layout(
    height=380, margin=dict(l=0, r=0, t=10, b=10),
    xaxis_title="", yaxis_title="", yaxis_type="log" if price > 100 else "linear",
    hovermode="x unified", showlegend=False,
    plot_bgcolor="white", paper_bgcolor="white",
    xaxis=dict(showgrid=False, showline=True, linecolor="#ccc"),
    yaxis=dict(showgrid=True, gridcolor="#eee", showline=True, linecolor="#ccc")
)
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 7. ГРАФИК Z-SCORE
# ============================================================

st.markdown("<div class='section-title'>Z-Score и пороги</div>", unsafe_allow_html=True)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=df_chart["date"], y=df_chart["z_score"],
    mode='lines', name='Z-Score', line=dict(color='#2288cc', width=2.5),
    fill='tozeroy', fillcolor='rgba(34, 136, 204, 0.1)'
))
fig2.add_hline(y=lower, line_dash="dash", line_color="#00aa44", line_width=2,
               annotation_text=f"Покупка ({lower:.2f})", annotation_position="right")
fig2.add_hline(y=upper, line_dash="dash", line_color="#cc2200", line_width=2,
               annotation_text=f"Продажа ({upper:.2f})", annotation_position="right")
fig2.add_hline(y=0, line_dash="dot", line_color="#999999")

fig2.update_layout(
    height=280, margin=dict(l=0, r=0, t=10, b=10),
    xaxis_title="", yaxis_title="",
    plot_bgcolor="white", paper_bgcolor="white",
    xaxis=dict(showgrid=False, showline=True, linecolor="#ccc"),
    yaxis=dict(showgrid=True, gridcolor="#eee", showline=True, linecolor="#ccc")
)
st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# 8. AI-АНАЛИЗ (ВОЗВРАЩЁН)
# ============================================================

st.markdown("<div class='section-title'>AI-анализ</div>", unsafe_allow_html=True)

if st.button("Получить AI-анализ", type="primary"):
    with st.spinner("DeepSeek анализирует..."):
        analysis = call_deepseek_analysis(selected_asset, price, z, prob, signal_text)
    
    st.markdown(f"""
    <div style='background-color: #f5f5f5; padding: 20px; border-radius: 16px; margin: 10px 0; border: 1px solid #ddd;'>
        <div style='color: #222; font-size: 15px; line-height: 1.6;'>{analysis}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 9. ФУНДАМЕНТАЛЬНЫЕ ДАННЫЕ (опционально)
# ============================================================

if fundamentals:
    st.markdown("<div class='section-title'>Фундаментальные метрики</div>", unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    with f1: st.metric("Рыночная капитализация", f"${fundamentals['market_cap']/1e9:.2f}B")
    with f2: st.metric("FDV", f"${fundamentals['fully_diluted_valuation']/1e9:.2f}B")
    with f3: st.metric("Объём 24ч", f"${fundamentals['total_volume']/1e6:.1f}M")
    with f4:
        pc = fundamentals['price_change_24h']
        color = "#00aa44" if pc > 0 else "#cc2200"
        st.markdown(f"<div><div style='color:#666;'>Изменение 24ч</div><div style='color:{color}; font-size:1.5rem;'>{pc:+.1f}%</div></div>", unsafe_allow_html=True)

# ============================================================
# 10. СВОДНАЯ ТАБЛИЦА
# ============================================================

st.markdown("<div class='section-title'>Сводная таблица</div>", unsafe_allow_html=True)

all_assets = {**{c: "Криптовалюта" for c in CRYPTO_LIST}, **{s: "Акция" for s in STOCK_LIST}}

all_data = []
progress_bar = st.progress(0)
status_text = st.empty()

for i, (symbol, atype) in enumerate(all_assets.items()):
    status_text.text(f"Загрузка {symbol}...")
    if atype == "Криптовалюта":
        df_temp = load_crypto_data(symbol)
    else:
        df_temp = load_stock_data(symbol)
    
    if df_temp is not None and len(df_temp) >= 30:
        _, price, z, prob, conf, (lthr, uthr) = calculate_metrics(df_temp)
        sig, _ = get_signal(z, lthr, uthr, symbol in VETERAN_LIST)
        all_data.append({
            "Символ": symbol, "Тип": atype,
            "Цена": f"${price:,.2f}", "Z-Score": f"{z:+.2f}",
            "Вероятность": f"{prob*100:.1f}%", "Надёжность": f"{conf:.0f}%", "Сигнал": sig
        })
    progress_bar.progress((i + 1) / len(all_assets))

status_text.empty()
progress_bar.empty()

if all_data:
    st.dataframe(pd.DataFrame(all_data), use_container_width=True, hide_index=True)

# ============================================================
# 11. ПОДВАЛ
# ============================================================

st.markdown("<hr>", unsafe_allow_html=True)
moscow_tz = timezone(timedelta(hours=3))
moscow_time = datetime.now(moscow_tz)
st.caption(f"Обновлено: {moscow_time.strftime('%Y-%m-%d %H:%M:%S')} (МСК)")
st.caption("Источники: CryptoCompare / Yahoo Finance / CoinGecko")
st.caption("Адаптивный Z-Score (5-й и 95-й процентили)")
st.caption("Не является инвестиционной рекомендацией")
