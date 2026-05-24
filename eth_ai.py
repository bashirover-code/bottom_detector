import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import time
import yfinance as yf

st.set_page_config(page_title="Мульти-актив Детектор Дна", layout="wide")

st.title("📊 Мульти-актив Детектор Дна")
st.markdown("### 🚀 Определение глобальных минимумов для криптовалют и акций")

# ============================================================
# СПИСОК АКТИВОВ
# ============================================================

CRYPTO_SYMBOLS = {
    "ETH": "Ethereum", "BTC": "Bitcoin", "SOL": "Solana",
    "BNB": "Binance Coin", "XRP": "Ripple", "ADA": "Cardano",
    "DOGE": "Dogecoin", "AVAX": "Avalanche", "DOT": "Polkadot",
    "MATIC": "Polygon", "LINK": "Chainlink", "UNI": "Uniswap",
    "ATOM": "Cosmos", "LTC": "Litecoin", "FIL": "Filecoin",
    "APT": "Aptos", "ARB": "Arbitrum", "OP": "Optimism",
    "AAVE": "Aave", "MKR": "Maker", "CRV": "Curve",
    "SNX": "Synthetix", "COMP": "Compound", "GRT": "The Graph",
    "SAND": "The Sandbox", "MANA": "Decentraland", "AXS": "Axie Infinity",
    "GALA": "Gala", "KZ": "Kazakhstan Coin", "NEAR": "NEAR Protocol",
    "TON": "Toncoin", "ALGO": "Algorand", "FLOCK": "Flock.io"
}

STOCK_SYMBOLS = {
    "HIMS": "Hims & Hers Health", "SIL": "Global X Silver Miners ETF",
    "GDX": "VanEck Gold Miners ETF", "TSLA": "Tesla Inc.",
    "LIT": "Global X Lithium ETF", "ZM": "Zoom Video",
    "URA": "Global X Uranium ETF", "PLTR": "Palantir",
    "EWW": "iShares MSCI Mexico", "BABA": "Alibaba",
    "COIN": "Coinbase", "NVDA": "NVIDIA",
    "SBER": "Sberbank", "MTSS": "MTS", "HEAD": "HeadHunter",
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google",
    "AMZN": "Amazon", "META": "Meta"
}

# Специальные тикеры для некоторых акций
RUSSIAN_TICKERS = {"SBER": "SBER.ME", "MTSS": "MTSS.ME", "HEAD": "HHRU.ME"}
ALL_ASSETS = {**CRYPTO_SYMBOLS, **STOCK_SYMBOLS}

MY_ASSETS = {
    "KZ": "Kazakhstan Coin", "NEAR": "NEAR Protocol",
    "HIMS": "Hims & Hers Health", "SIL": "Global X Silver Miners ETF",
    "GDX": "VanEck Gold Miners ETF", "TSLA": "Tesla Inc.",
    "LIT": "Global X Lithium ETF", "ZM": "Zoom Video",
    "URA": "Global X Uranium ETF", "PLTR": "Palantir",
    "EWW": "iShares MSCI Mexico", "BABA": "Alibaba",
    "COIN": "Coinbase", "NVDA": "NVIDIA",
    "SBER": "Sberbank", "MTSS": "MTS", "HEAD": "HeadHunter"
}

# ============================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================

@st.cache_data(ttl=300)
def load_crypto_data(symbol, days=500):
    """Загружает данные криптовалюты с CryptoCompare API"""
    try:
        if "CRYPTOCOMPARE_KEY" in st.secrets:
            API_KEY = st.secrets["CRYPTOCOMPARE_KEY"]
            url = "https://min-api.cryptocompare.com/data/v2/histoday"
            params = {"fsym": symbol, "tsym": "USD", "limit": days, "api_key": API_KEY}
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data.get("Response") == "Success":
                    raw_data = data["Data"]["Data"]
                    df = pd.DataFrame(raw_data)
                    df["date"] = pd.to_datetime(df["time"], unit='s')
                    df["close"] = df["close"].astype(float)
                    return df.sort_values("date").reset_index(drop=True)
    except:
        pass
    return None

@st.cache_data(ttl=300)
def load_stock_data(symbol, days=500):
    """Загружает данные акций через yfinance (работает в Streamlit Cloud)"""
    try:
        ticker = RUSSIAN_TICKERS.get(symbol, symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df is not None and not df.empty:
            df = df.reset_index()
            df = df.rename(columns={"Date": "date", "Close": "close"})
            return df[["date", "close"]]
    except Exception as e:
        st.warning(f"yfinance ошибка для {symbol}: {str(e)[:100]}")
    return None

# ============================================================
# РАСЧЁТ МЕТРИК
# ============================================================

def calculate_metrics(df):
    if df is None or len(df) < 50:
        return None, None, None, None
    
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    df["z_score"] = (df["returns"] - df["returns"].rolling(30).mean()) / (df["returns"].rolling(30).std() + 1e-10)
    df = df.fillna(0)
    
    current_price = df["close"].iloc[-1]
    current_z = df["z_score"].iloc[-1]
    current_prob = 1 / (1 + np.exp(current_z * 1.5))
    
    return df, current_price, current_z, current_prob

def get_signal_color(prob):
    if prob > 0.75:
        return "🔴", "#ef4444", "ПОКУПКА"
    elif prob > 0.6:
        return "🟡", "#eab308", "НАКОПЛЕНИЕ"
    elif prob < 0.2:
        return "🟢", "#22c55e", "ПРОДАЖА"
    else:
        return "⚪", "#6b7280", "НЕЙТРАЛЬНО"

# ============================================================
# БОКОВАЯ ПАНЕЛЬ
# ============================================================

with st.sidebar:
    st.header("⚙️ Настройки")
    asset_type = st.radio("Тип активов", ["Криптовалюты", "Акции", "Все активы", "Мои активы"])
    
    if asset_type == "Криптовалюты":
        asset_list = CRYPTO_SYMBOLS
    elif asset_type == "Акции":
        asset_list = STOCK_SYMBOLS
    elif asset_type == "Мои активы":
        asset_list = MY_ASSETS
    else:
        asset_list = ALL_ASSETS
    
    selected_asset = st.selectbox(
        "Выберите актив",
        options=list(asset_list.keys()),
        format_func=lambda x: f"{x} - {asset_list[x]}"
    )
    
    st.markdown("---")
    st.caption("📡 Источники: CryptoCompare (крипто), yfinance (акции)")
    st.caption("🕐 Обновление: каждые 5 минут")

# ============================================================
# ЗАГРУЗКА
# ============================================================

is_crypto = selected_asset in CRYPTO_SYMBOLS

if is_crypto:
    with st.spinner(f"🔄 Загрузка {selected_asset}..."):
        df = load_crypto_data(selected_asset)
else:
    with st.spinner(f"🔄 Загрузка {selected_asset}..."):
        df = load_stock_data(selected_asset)

if df is None:
    st.warning(f"⚠️ Демо-данные для {selected_asset}")
    end_date = datetime.now()
    dates = [end_date - timedelta(days=i) for i in range(500, 0, -1)]
    np.random.seed(42)
    prices = [100]
    for i in range(1, len(dates)):
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.02)))
    df = pd.DataFrame({"date": dates, "close": prices})

df, current_price, current_z, current_prob = calculate_metrics(df)
signal_icon, signal_color, signal_text = get_signal_color(current_prob)

# ============================================================
# ДАШБОРД
# ============================================================

st.header(f"{selected_asset} - {asset_list[selected_asset]}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("💰 ЦЕНА", f"${current_price:,.2f}" if current_price else "—")
with col2:
    st.metric("📊 Z-SCORE", f"{current_z:.2f}" if current_z else "—")
with col3:
    st.metric("🎯 ВЕРОЯТНОСТЬ ДНА", f"{current_prob*100:.1f}%" if current_prob else "—")
with col4:
    st.markdown(
        f"<div style='background:{signal_color}20;padding:10px;border-radius:10px;text-align:center;'>"
        f"<span style='font-size:24px;'>{signal_icon}</span>"
        f"<div style='font-size:20px;font-weight:bold;color:{signal_color};'>{signal_text}</div></div>",
        unsafe_allow_html=True
    )

# ============================================================
# ГРАФИКИ
# ============================================================

st.markdown("---")
st.subheader("📈 ГРАФИК ЦЕНЫ — ЦВЕТ ПО Z-SCORE")
st.caption("🟢 Зелёный = дно | ⚪ Серый = нейтрально | 🔴 Красный = эйфория")

df_chart = df.tail(365).copy()

def get_color(z):
    if z <= -1.8: return "#00ff44"
    elif z <= -1.2: return "#44ff44"
    elif z <= -0.5: return "#88ff88"
    elif z <= 0.5: return "#cccccc"
    elif z <= 1.2: return "#ffaa66"
    elif z <= 1.8: return "#ff6644"
    else: return "#ff2200"

fig = go.Figure()
for i in range(len(df_chart) - 1):
    color = get_color(df_chart["z_score"].iloc[i])
    fig.add_trace(go.Scatter(
        x=[df_chart["date"].iloc[i], df_chart["date"].iloc[i+1]],
        y=[df_chart["close"].iloc[i], df_chart["close"].iloc[i+1]],
        mode='lines', line=dict(color=color, width=2),
        showlegend=False, hoverinfo='skip'
    ))

fig.add_trace(go.Scatter(
    x=df_chart["date"], y=df_chart["close"], mode='markers',
    marker=dict(color='rgba(0,0,0,0)', size=1),
    hoverinfo='text',
    text=[f"📅 {d.strftime('%Y-%m-%d')}<br>💰 ${p:,.2f}<br>📊 Z: {z:.2f}"
          for d, p, z in zip(df_chart["date"], df_chart["close"], df_chart["z_score"])],
    name="Цена"
))

fig.update_layout(height=500, template="plotly_dark", xaxis_title="Дата",
                  yaxis_title="Цена (USD)", yaxis_type="log" if is_crypto and current_price > 1 else "linear",
                  hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# Z-Score график
st.subheader("📉 Z-SCORE С ЗОНАМИ")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_chart["date"], y=df_chart["z_score"],
                          mode='lines', name='Z-Score', line=dict(color='#38bdf8', width=2),
                          fill='tozeroy', fillcolor='rgba(56,189,248,0.15)'))
fig2.add_hline(y=-1.8, line_dash="dash", line_color="#22c55e", annotation_text="ПОКУПКА")
fig2.add_hline(y=1.5, line_dash="dash", line_color="#ef4444", annotation_text="ПРОДАЖА")
fig2.add_hline(y=0, line_dash="dot", line_color="#6b7280")
fig2.update_layout(height=350, template="plotly_dark", yaxis_range=[-3.5, 3.5])
st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# СВОДНАЯ ТАБЛИЦА
# ============================================================

st.markdown("---")
st.subheader("📋 СВОДНАЯ ТАБЛИЦА")

display_assets = MY_ASSETS if asset_type == "Мои активы" else asset_list if asset_type != "Все активы" else MY_ASSETS

all_data = []
progress_bar = st.progress(0)
status_text = st.empty()

for i, (symbol, name) in enumerate(display_assets.items()):
    status_text.text(f"Загрузка {symbol}...")
    
    if symbol in CRYPTO_SYMBOLS:
        df_temp = load_crypto_data(symbol)
    else:
        df_temp = load_stock_data(symbol)
    
    if df_temp is not None:
        _, price, z, prob = calculate_metrics(df_temp)
        _, _, signal = get_signal_color(prob)
        all_data.append({"Символ": symbol, "Название": name, "Цена": f"${price:,.2f}" if price else "—",
                         "Z-Score": f"{z:.2f}" if z else "—",
                         "Вероятность": f"{prob*100:.1f}%" if prob else "—", "Сигнал": signal})
    progress_bar.progress((i + 1) / len(display_assets))

status_text.empty()
progress_bar.empty()

if all_data:
    st.dataframe(pd.DataFrame(all_data), use_container_width=True, hide_index=True)
else:
    st.warning("Не удалось загрузить данные")

# ============================================================
# ПОДВАЛ
# ============================================================

st.markdown("---")
st.caption(f"📅 Обновлено: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("⚠️ Не инвестиционная рекомендация")
