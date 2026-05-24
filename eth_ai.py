import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="ETH Детектор Дна", layout="wide")

# ============================================================
# ТЕСТ: проверяем, видит ли Streamlit ключ
# ============================================================

try:
    if "CRYPTOCOMPARE_KEY" in st.secrets:
        st.success("✅ Ключ НАЙДЕН в Secrets")
        st.write(f"Имя ключа: CRYPTOCOMPARE_KEY")
    else:
        st.error("❌ Ключ НЕ НАЙДЕН в Secrets")
        st.write("Доступные ключи:", list(st.secrets.keys()))
except Exception as e:
    st.error(f"Ошибка при доступе к Secrets: {e}")

st.title("📊 ETH Детектор Дна")
st.markdown("### 🚀 Определение глобальных минимумов Ethereum")

# ... дальше продолжается твой код ...

# ============================================================
# МНОГОУРОВНЕВАЯ ЗАГРУЗКА ДАННЫХ
# ============================================================

@st.cache_data(ttl=3600)
def load_data():
    """Пытается загрузить данные из нескольких источников"""
    
    # ----- СПОСОБ 1: CryptoCompare (с API ключом) -----
    try:
        # Проверяем, настроен ли ключ в секретах Streamlit
        if "CRYPTOCOMPARE_KEY" in st.secrets:
            API_KEY = st.secrets["CRYPTOCOMPARE_KEY"]
            
            url = "https://min-api.cryptocompare.com/data/v2/histoday"
            params = {
                "fsym": "ETH",
                "tsym": "USD",
                "limit": 500,
                "api_key": API_KEY
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("Response") == "Success":
                    raw_data = data["Data"]["Data"]
                    df = pd.DataFrame(raw_data)
                    df["date"] = pd.to_datetime(df["time"], unit='s')
                    df["close"] = df["close"].astype(float)
                    df = df.sort_values("date").reset_index(drop=True)
                    
                    st.success("✅ Данные загружены через CryptoCompare")
                    return df, "CryptoCompare"
    except Exception as e:
        pass  # Если ошибка — идём дальше
    
    # ----- СПОСОБ 2: CoinGecko (публичный API) -----
    try:
        url = "https://api.coingecko.com/api/v3/coins/ethereum/market_chart"
        params = {"vs_currency": "usd", "days": "500", "interval": "daily"}
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            prices = []
            for ts, price in data["prices"]:
                prices.append({
                    "date": datetime.fromtimestamp(ts/1000),
                    "close": price
                })
            df = pd.DataFrame(prices)
            df = df.sort_values("date").reset_index(drop=True)
            
            st.success("✅ Данные загружены с CoinGecko API")
            return df, "CoinGecko"
    except:
        pass
    
    # ----- СПОСОБ 3: Демо-данные (если API не работают) -----
    st.warning("⚠️ API временно недоступны. Использую демо-данные.")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=500)
    
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)
    
    np.random.seed(42)
    prices = [2000]
    for i in range(1, len(dates)):
        trend = 0.0001
        noise = np.random.normal(0, 0.02)
        prices.append(prices[-1] * (1 + trend + noise))
    
    df = pd.DataFrame({"date": dates, "close": prices})
    return df, "Demo"

# Загружаем данные
with st.spinner("🔄 Загрузка данных..."):
    df, source = load_data()

# Расчёт Z-Score
df["returns"] = df["close"].pct_change()
df["z_score"] = (df["returns"] - df["returns"].rolling(30).mean()) / (df["returns"].rolling(30).std() + 1e-10)
df = df.fillna(0)

# ============================================================
# ТЕКУЩИЙ СИГНАЛ
# ============================================================

current_price = df["close"].iloc[-1]
current_z = df["z_score"].iloc[-1]
current_prob = 1 / (1 + np.exp(current_z * 1.5))

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("💰 ЦЕНА ETH", f"${current_price:,.0f}")
with col2:
    st.metric("📊 Z-SCORE", f"{current_z:.2f}")
with col3:
    st.metric("🎯 ВЕРОЯТНОСТЬ ДНА", f"{current_prob*100:.1f}%")
with col4:
    if current_z < -1.8:
        st.metric("📈 СИГНАЛ", "🔴 ПОКУПКА", delta="Экстремальное дно")
    elif current_z < -1.2:
        st.metric("📈 СИГНАЛ", "🟡 НАКОПЛЕНИЕ", delta="Присматривайся")
    elif current_z > 1.5:
        st.metric("📈 СИГНАЛ", "🟢 ПРОДАЖА", delta="Эйфория")
    else:
        st.metric("📈 СИГНАЛ", "⚪ НЕЙТРАЛЬНО", delta="Жди")

st.caption(f"📡 Источник данных: {source}")

# ============================================================
# ЦВЕТНОЙ ГРАФИК
# ============================================================

st.markdown("---")
st.subheader("📈 ETH PRICE — ЦВЕТ ПО Z-SCORE")
st.caption("🟢 Зелёный = дно | ⚪ Серый = нейтрально | 🔴 Красный = эйфория")

df_chart = df.tail(500).copy()

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

# Добавляем точки для наведения
fig.add_trace(go.Scatter(
    x=df_chart["date"],
    y=df_chart["close"],
    mode='markers',
    marker=dict(color='rgba(0,0,0,0)', size=1),
    hoverinfo='text',
    text=[f"📅 {d.strftime('%Y-%m-%d')}<br>💰 ${p:,.0f}<br>📊 Z: {z:.2f}" 
          for d, p, z in zip(df_chart["date"], df_chart["close"], df_chart["z_score"])],
    name="Цена"
))

fig.update_layout(
    height=500,
    template="plotly_dark",
    xaxis_title="Дата",
    yaxis_title="Цена (USD)",
    yaxis_type="log",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Z-SCORE ГРАФИК
# ============================================================

st.subheader("📉 COMPOSITE Z-SCORE")
st.caption("Красная зона = продажа | Зелёная зона = покупка")

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
# ПОЯСНЕНИЯ
# ============================================================

st.markdown("---")
st.subheader("📋 РАСШИФРОВКА СИГНАЛОВ")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🟢 **Z < -1.8 — ПОКУПКА**
    - Экстремальное дно
    - Страх и паника на рынке
    - **Действие:** Рассмотреть покупку
    """)

with col2:
    st.markdown("""
    ### ⚪ **-0.5 < Z < 0.5 — НЕЙТРАЛЬНО**
    - Рынок спокоен
    - Нет явного сигнала
    - **Действие:** Ничего не делать
    """)

with col3:
    st.markdown("""
    ### 🔴 **Z > 1.5 — ПРОДАЖА**
    - Эйфория / Перекупленность
    - Все верят в бесконечный рост
    - **Действие:** Фиксация прибыли
    """)

# ============================================================
# ПОДВАЛ
# ============================================================

st.markdown("---")
st.caption(f"📅 Последнее обновление: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption(f"📡 Источник: {source}")
st.caption("⚠️ Не является инвестиционной рекомендацией. Все решения принимайте самостоятельно.")
