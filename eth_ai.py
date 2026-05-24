import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import time

st.set_page_config(page_title="Мульти-актив Детектор Дна", layout="wide")

st.title("📊 Мульти-актив Детектор Дна")
st.markdown("### 🚀 Определение глобальных минимумов для криптовалют и акций")

# ============================================================
# СПИСОК АКТИВОВ
# ============================================================

# 30 криптовалют
CRYPTO_SYMBOLS = {
    "ETH": "Ethereum",
    "BTC": "Bitcoin",
    "SOL": "Solana",
    "BNB": "Binance Coin",
    "XRP": "Ripple",
    "ADA": "Cardano",
    "DOGE": "Dogecoin",
    "AVAX": "Avalanche",
    "DOT": "Polkadot",
    "MATIC": "Polygon",
    "LINK": "Chainlink",
    "UNI": "Uniswap",
    "ATOM": "Cosmos",
    "ETC": "Ethereum Classic",
    "LTC": "Litecoin",
    "FIL": "Filecoin",
    "APT": "Aptos",
    "ARB": "Arbitrum",
    "OP": "Optimism",
    "NEAR": "NEAR Protocol",
    "AAVE": "Aave",
    "MKR": "Maker",
    "CRV": "Curve",
    "SNX": "Synthetix",
    "COMP": "Compound",
    "GRT": "The Graph",
    "SAND": "The Sandbox",
    "MANA": "Decentraland",
    "AXS": "Axie Infinity",
    "GALA": "Gala"
}

# 10 акций
STOCK_SYMBOLS = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet (Google)",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
    "META": "Meta (Facebook)",
    "NFLX": "Netflix",
    "AMD": "Advanced Micro Devices",
    "INTC": "Intel"
}

# Объединяем все активы
ALL_ASSETS = {**CRYPTO_SYMBOLS, **STOCK_SYMBOLS}

# ============================================================
# ЗАГРУЗКА ДАННЫХ ДЛЯ КРИПТОВАЛЮТ
# ============================================================

@st.cache_data(ttl=300)  # 5 минут
def load_crypto_data(symbol, vs_currency="usd", days=500):
    """Загружает данные криптовалюты с CryptoCompare API"""
    
    try:
        if "CRYPTOCOMPARE_KEY" in st.secrets:
            API_KEY = st.secrets["CRYPTOCOMPARE_KEY"]
            
            url = "https://min-api.cryptocompare.com/data/v2/histoday"
            params = {
                "fsym": symbol,
                "tsym": vs_currency.upper(),
                "limit": days,
                "api_key": API_KEY
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("Response") == "Success":
                    raw_data = data["Data"]["Data"]
                    df = pd.DataFrame(raw_data)
                    df["date"] = pd.to_datetime(df["time"], unit='s')
                    df["close"] = df["close"].astype(float)
                    df = df.sort_values("date").reset_index(drop=True)
                    
                    return df
    except:
        pass
    
    return None

# ============================================================
# ЗАГРУЗКА ДАННЫХ ДЛЯ АКЦИЙ (Alpha Vantage API)
# ============================================================

@st.cache_data(ttl=300)  # 5 минут
def load_stock_data(symbol):
    """Загружает данные акций с Alpha Vantage API"""
    
    try:
        # Alpha Vantage API (бесплатно, 5 запросов/мин)
        API_KEY = st.secrets.get("ALPHA_VANTAGE_KEY", "demo")
        
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": API_KEY,
            "outputsize": "compact"
        }
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if "Time Series (Daily)" in data:
                ts_data = data["Time Series (Daily)"]
                df = pd.DataFrame.from_dict(ts_data, orient="index")
                df = df.reset_index()
                df.columns = ["date", "open", "high", "low", "close", "volume"]
                df["date"] = pd.to_datetime(df["date"])
                df["close"] = df["close"].astype(float)
                df = df.sort_values("date").reset_index(drop=True)
                
                return df.tail(500)
    except:
        pass
    
    return None

# ============================================================
# РАСЧЁТ Z-SCORE И ВЕРОЯТНОСТИ
# ============================================================

def calculate_metrics(df):
    """Рассчитывает Z-Score и вероятность дна"""
    
    if df is None or len(df) < 50:
        return None, None, None
    
    df["returns"] = df["close"].pct_change()
    df["z_score"] = (df["returns"] - df["returns"].rolling(30).mean()) / (df["returns"].rolling(30).std() + 1e-10)
    df = df.fillna(0)
    
    current_price = df["close"].iloc[-1]
    current_z = df["z_score"].iloc[-1]
    current_prob = 1 / (1 + np.exp(current_z * 1.5))
    
    return current_price, current_z, current_prob

# ============================================================
# ЦВЕТНАЯ КАРТА ДЛЯ СИГНАЛОВ
# ============================================================

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
    
    # Выбор типа активов
    asset_type = st.radio(
        "Тип активов",
        ["Криптовалюты", "Акции", "Все активы"]
    )
    
    # Выбор конкретного актива
    if asset_type == "Криптовалюты":
        asset_list = CRYPTO_SYMBOLS
    elif asset_type == "Акции":
        asset_list = STOCK_SYMBOLS
    else:
        asset_list = ALL_ASSETS
    
    selected_asset = st.selectbox(
        "Выберите актив",
        options=list(asset_list.keys()),
        format_func=lambda x: f"{x} - {asset_list[x]}"
    )
    
    st.markdown("---")
    st.caption("📡 Источники данных:")
    st.caption("• CryptoCompare (криптовалюты)")
    st.caption("• Alpha Vantage (акции)")
    st.caption("🕐 Обновление: каждые 5 минут")
    st.caption("⚠️ Не инвестиционная рекомендация")

# ============================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================

# Определяем тип актива
is_crypto = selected_asset in CRYPTO_SYMBOLS

if is_crypto:
    with st.spinner(f"🔄 Загрузка данных {selected_asset}..."):
        df = load_crypto_data(selected_asset)
else:
    with st.spinner(f"🔄 Загрузка данных {selected_asset}..."):
        df = load_stock_data(selected_asset)

# Если данные не загрузились — демо-режим
if df is None:
    st.warning(f"⚠️ Не удалось загрузить данные для {selected_asset}. Использую демо-данные.")
    
    # Генерация демо-данных
    end_date = datetime.now()
    start_date = end_date - timedelta(days=500)
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)
    
    np.random.seed(42)
    prices = [100]
    for i in range(1, len(dates)):
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.02)))
    
    df = pd.DataFrame({"date": dates, "close": prices})

# Расчёт метрик
current_price, current_z, current_prob = calculate_metrics(df)
signal_icon, signal_color, signal_text = get_signal_color(current_prob)

# ============================================================
# ОСНОВНОЙ ДАШБОРД
# ============================================================

# Заголовок с активом
st.header(f"{selected_asset} - {asset_list[selected_asset]}")

# Карточки
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("💰 ЦЕНА", f"${current_price:,.2f}" if current_price else "—")
with col2:
    st.metric("📊 Z-SCORE", f"{current_z:.2f}" if current_z else "—")
with col3:
    st.metric("🎯 ВЕРОЯТНОСТЬ ДНА", f"{current_prob*100:.1f}%" if current_prob else "—")
with col4:
    st.markdown(
        f"""
        <div style='background: {signal_color}20; padding: 10px; border-radius: 10px; text-align: center;'>
            <span style='font-size: 24px;'>{signal_icon}</span>
            <div style='font-size: 20px; font-weight: bold; color: {signal_color};'>{signal_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# ЦВЕТНОЙ ГРАФИК ЦЕНЫ
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
        mode='lines',
        line=dict(color=color, width=2),
        showlegend=False,
        hoverinfo='skip'
    ))

fig.add_trace(go.Scatter(
    x=df_chart["date"],
    y=df_chart["close"],
    mode='markers',
    marker=dict(color='rgba(0,0,0,0)', size=1),
    hoverinfo='text',
    text=[f"📅 {d.strftime('%Y-%m-%d')}<br>💰 ${p:,.2f}<br>📊 Z: {z:.2f}" 
          for d, p, z in zip(df_chart["date"], df_chart["close"], df_chart["z_score"])],
    name="Цена"
))

fig.update_layout(
    height=500,
    template="plotly_dark",
    xaxis_title="Дата",
    yaxis_title="Цена (USD)",
    yaxis_type="log" if is_crypto else "linear",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Z-SCORE ГРАФИК
# ============================================================

st.subheader("📉 Z-SCORE С ЗОНАМИ")
st.caption("🔴 Красная зона = продажа | 🟢 Зелёная зона = покупка")

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=df_chart["date"],
    y=df_chart["z_score"],
    mode='lines',
    name='Z-Score',
    line=dict(color='#38bdf8', width=2),
    fill='tozeroy',
    fillcolor='rgba(56,189,248,0.15)'
))

fig2.add_hline(y=-1.8, line_dash="dash", line_color="#22c55e", 
               annotation_text="ПОКУПКА (-1.8σ)", annotation_position="right")
fig2.add_hline(y=-1.2, line_dash="dot", line_color="#88cc44", 
               annotation_text="Зона страха", annotation_position="right")
fig2.add_hline(y=0, line_dash="dot", line_color="#6b7280")
fig2.add_hline(y=1.5, line_dash="dash", line_color="#ef4444", 
               annotation_text="ПРОДАЖА (1.5σ)", annotation_position="right")

fig2.update_layout(
    height=350,
    template="plotly_dark",
    xaxis_title="Дата",
    yaxis_title="Z-Score",
    yaxis_range=[-3.5, 3.5]
)

st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# ТАБЛИЦА ВСЕХ АКТИВОВ
# ============================================================

st.markdown("---")
st.subheader("📋 СВОДНАЯ ТАБЛИЦА ВСЕХ АКТИВОВ")

# Собираем данные для всех активов
all_data = []

progress_bar = st.progress(0)
status_text = st.empty()

for i, (symbol, name) in enumerate(ALL_ASSETS.items()):
    status_text.text(f"Загрузка {symbol}...")
    
    if symbol in CRYPTO_SYMBOLS:
        df_temp = load_crypto_data(symbol)
    else:
        df_temp = load_stock_data(symbol)
    
    if df_temp is not None:
        price, z, prob = calculate_metrics(df_temp)
        _, _, signal = get_signal_color(prob)
        
        all_data.append({
            "Символ": symbol,
            "Название": name,
            "Цена": f"${price:,.2f}" if price else "—",
            "Z-Score": f"{z:.2f}" if z else "—",
            "Вероятность дна": f"{prob*100:.1f}%" if prob else "—",
            "Сигнал": signal
        })
    
    progress_bar.progress((i + 1) / len(ALL_ASSETS))

status_text.empty()
progress_bar.empty()

# Показываем таблицу
if all_data:
    df_all = pd.DataFrame(all_data)
    st.dataframe(df_all, use_container_width=True, hide_index=True)
else:
    st.warning("Не удалось загрузить данные")

# ============================================================
# ПОЯСНЕНИЯ
# ============================================================

st.markdown("---")
st.subheader("📋 РАСШИФРОВКА СИГНАЛОВ")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🔴 **ПОКУПКА** (Z < -1.8)
    - Экстремальное дно
    - Страх и паника на рынке
    - **Действие:** Рассмотреть покупку
    """)

with col2:
    st.markdown("""
    ### ⚪ **НЕЙТРАЛЬНО** (-0.5 < Z < 0.5)
    - Рынок спокоен
    - Нет явного сигнала
    - **Действие:** Ничего не делать
    """)

with col3:
    st.markdown("""
    ### 🟢 **ПРОДАЖА** (Z > 1.5)
    - Эйфория / Перекупленность
    - Все верят в бесконечный рост
    - **Действие:** Фиксация прибыли
    """)

# ============================================================
# ПОДВАЛ
# ============================================================

st.markdown("---")
st.caption(f"📅 Последнее обновление: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("📡 Источники: CryptoCompare (криптовалюты), Alpha Vantage (акции)")
st.caption("⚠️ Не является инвестиционной рекомендацией. Все решения принимайте самостоятельно.")
