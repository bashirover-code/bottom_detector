import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import yfinance as yf
import json

st.set_page_config(page_title="Мульти-актив Детектор Дна", layout="wide")

st.title("📊 Мульти-актив Детектор Дна")
st.markdown("### 🚀 Определение глобальных минимумов для криптовалют и акций")

# ============================================================
# ТВОЙ СПИСОК АКТИВОВ
# ============================================================

CRYPTO_LIST = [
    "ETH", "BTC", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX",
    "DOT", "MATIC", "LINK", "UNI", "ATOM", "NEAR", "TON", "ALGO", "FLOCK", "KZ"
]

STOCK_LIST = [
    "HIMS", "SIL", "GDX", "TSLA", "LIT", "ZM", "URA", "PLTR",
    "EWW", "BABA", "COIN", "NVDA", "SBER", "MTSS", "HEAD"
]

# ============================================================
# ФУНКЦИЯ ДЛЯ ВЫЗОВА CLAUDE API
# ============================================================

def call_claude_analysis(asset_name, asset_symbol, current_price, current_z, current_prob, signal_text):
    """Отправляет запрос к Claude API и возвращает анализ"""
    
    # Получаем API ключ из секретов
    claude_key = st.secrets.get("CLAUDE_API_KEY", "")
    
    if not claude_key:
        return "❌ API ключ Claude не найден. Добавь CLAUDE_API_KEY в Secrets."
    
    # Определяем тип актива
    asset_type = "криптовалюта" if asset_symbol in CRYPTO_LIST else "акция"
    
    # Формируем промпт для Claude
    prompt = f"""Ты профессиональный финансовый аналитик. Проанализируй {asset_name} ({asset_symbol}, {asset_type}):

ТЕКУЩИЕ ДАННЫЕ:
- Цена: ${current_price:,.2f}
- Z-Score: {current_z:.2f} (отклонение от нормы)
- Вероятность дна: {current_prob*100:.1f}%
- Сигнал системы: {signal_text}

Что означает Z-Score:
- Z < -1.8: экстремальное дно (хороший вход)
- Z < -1.2: зона страха (присматривайся)
- Z > 1.5: эйфория (пора продавать)
- -0.5 < Z < 0.5: нейтрально

Напиши КРАТКИЙ анализ (3-5 предложений):
1. Что означает текущий сигнал простыми словами
2. Стоит ли сейчас покупать, продавать или держать
3. Главный фактор риска, на который стоит обратить внимание

Будь конкретен и полезен. Не пиши лишнего. Используй простой русский язык."""

    # Запрос к Claude API
    url = "https://api.anthropic.com/v1/messages"
    
    headers = {
        "x-api-key": claude_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    data = {
        "model": "claude-3-haiku-20240307",  # самая дешёвая модель
        "max_tokens": 500,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result["content"][0]["text"]
        else:
            return f"❌ Ошибка API: {response.status_code}\n{response.text[:200]}"
            
    except Exception as e:
        return f"❌ Ошибка подключения: {str(e)[:100]}"

# ============================================================
# НАСТРОЙКИ ПОРОГОВ (как на скриншоте)
# ============================================================

with st.sidebar:
    st.header("⚙️ Настройки порогов")
    st.markdown("---")
    
    st.subheader("📉 ЗОНА ПОКУПКИ (ДНО)")
    tier2_threshold = st.slider("Tier-2 plateau (зелёная линия)", -2.5, -0.5, -1.8, 0.05)
    pre_reg_threshold = st.slider("Pre-reg locked (красная линия)", -2.5, -0.5, -1.5, 0.05)
    
    st.markdown("---")
    st.subheader("📈 ЗОНА ПРОДАЖИ (ЭЙФОРИЯ)")
    euphoria_threshold = st.slider("Euphoria zone", 0.5, 2.5, 1.5, 0.05)
    
    st.markdown("---")
    st.caption("📡 Источник: CryptoCompare (крипто), yfinance (акции)")
    st.caption("🕐 Обновление: каждые 5 минут")

# ============================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================

@st.cache_data(ttl=300)
def load_crypto_data(symbol, days=500):
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
    try:
        ticker = symbol
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        if df is not None and not df.empty:
            df = df.reset_index()
            df = df.rename(columns={"Date": "date", "Close": "close"})
            return df[["date", "close"]]
    except:
        pass
    return None

def calculate_metrics(df):
    if df is None or len(df) < 50:
        return None, None, None, None
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    df["z_score"] = (df["returns"] - df["returns"].rolling(30).mean()) / (df["returns"].rolling(30).std() + 1e-10)
    df = df.fillna(0)
    return df, df["close"].iloc[-1], df["z_score"].iloc[-1], 1 / (1 + np.exp(df["z_score"].iloc[-1] * 1.5))

def get_signal(z_score, pre_reg, tier2, euphoria):
    if z_score <= pre_reg:
        return "🔴 ЭКСТРЕМАЛЬНАЯ ПОКУПКА", "#ef4444"
    elif z_score <= tier2:
        return "🟡 ЗОНА НАКОПЛЕНИЯ", "#eab308"
    elif z_score >= euphoria:
        return "🟢 ЭЙФОРИЯ — ПРОДАВАЙ", "#22c55e"
    else:
        return "⚪ НЕЙТРАЛЬНО", "#6b7280"

# ============================================================
# ВЫБОР АКТИВА
# ============================================================

col1, col2 = st.columns([1, 3])
with col1:
    asset_type = st.radio("Тип", ["Криптовалюты", "Акции"])
    if asset_type == "Криптовалюты":
        selected_asset = st.selectbox("Выберите криптовалюту", CRYPTO_LIST)
    else:
        selected_asset = st.selectbox("Выберите акцию", STOCK_LIST)

with col2:
    st.markdown("---")
    st.caption("📋 Всего активов: {} криптовалют + {} акций = {} активов".format(
        len(CRYPTO_LIST), len(STOCK_LIST), len(CRYPTO_LIST) + len(STOCK_LIST)
    ))

# ============================================================
# ЗАГРУЗКА И ОТОБРАЖЕНИЕ
# ============================================================

is_crypto = selected_asset in CRYPTO_LIST

with st.spinner(f"🔄 Загрузка {selected_asset}..."):
    if is_crypto:
        df = load_crypto_data(selected_asset)
    else:
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
asset_name = selected_asset
signal_text, signal_color = get_signal(current_z, pre_reg_threshold, tier2_threshold, euphoria_threshold)

# Карточки
cola, colb, colc, cold = st.columns(4)
with cola: st.metric("💰 ЦЕНА", f"${current_price:,.2f}")
with colb: st.metric("📊 Z-SCORE", f"{current_z:.2f}")
with colc: st.metric("🎯 ВЕРОЯТНОСТЬ ДНА", f"{current_prob*100:.1f}%")
with cold: st.metric("📈 СИГНАЛ", signal_text.split("—")[0])

# Сигнал
st.markdown(f"""
<div style='background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
            padding: 25px; border-radius: 16px; margin: 20px 0; text-align: center;
            border-left: 5px solid {signal_color};'>
    <h2 style='color: {signal_color}; margin: 0;'>{signal_text}</h2>
    <p style='color: #6b7280; margin-top: 10px;'>
        Z-Score: {current_z:.2f} | 
        Вероятность: {current_prob*100:.1f}% | 
        {'📈 ТРЕНД' if abs(current_z) > 1 else '📊 ФЛЭТ'}
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# НОВАЯ КНОПКА: AI-АНАЛИЗ ОТ CLAUDE
# ============================================================

st.markdown("---")
st.subheader("🤖 AI-анализ актива")

if st.button(f"📊 Получить AI-анализ для {selected_asset}", type="primary"):
    with st.spinner(f"🧠 Claude анализирует {selected_asset}..."):
        analysis = call_claude_analysis(
            asset_name, selected_asset, 
            current_price, current_z, current_prob, 
            signal_text.split("—")[0]
        )
    
    st.markdown(f"""
    <div style='background: #1a1a2e; padding: 20px; border-radius: 16px; margin: 10px 0;'>
        <h4 style='margin-bottom: 10px;'>📈 Анализ от Claude AI</h4>
        <p style='color: #e2e8f0;'>{analysis}</p>
        <p style='color: #6b7280; font-size: 12px; margin-top: 10px;'>⚡ Модель: Claude 3 Haiku | Анализ на основе текущих данных</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# ГРАФИКИ
# ============================================================

st.markdown("---")
st.subheader("📈 ГРАФИК ЦЕНЫ — ЦВЕТ ПО Z-SCORE")
st.caption("🟢 Зелёный = дно | ⚪ Серый = нейтрально | 🔴 Красный = эйфория | **Наведи курсор — увидишь дату и цену**")

df_chart = df.tail(500).copy()

def get_color(z):
    if z <= pre_reg_threshold: return "#00ff44"
    elif z <= tier2_threshold: return "#44ff44"
    elif z <= -0.5: return "#88ff88"
    elif z <= 0.5: return "#cccccc"
    elif z <= 1.2: return "#ffaa66"
    elif z <= euphoria_threshold: return "#ff6644"
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
    x=df_chart["date"], y=df_chart["close"],
    mode='markers', marker=dict(color='rgba(0,0,0,0)', size=1),
    hoverinfo='text',
    text=[f"📅 <b>{d.strftime('%Y-%m-%d')}</b><br>💰 <b>${p:,.2f}</b><br>📊 Z-Score: <b>{z:.2f}</b>" 
          for d, p, z in zip(df_chart["date"], df_chart["close"], df_chart["z_score"])],
    name="Информация", hovertemplate='%{text}<extra></extra>'
))

fig.update_layout(height=500, template="plotly_dark", xaxis_title="Дата",
                  yaxis_title="Цена (USD)", yaxis_type="log" if current_price > 100 else "linear",
                  hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# Z-Score график
st.subheader("📉 Z-SCORE С ЗОНАМИ")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_chart["date"], y=df_chart["z_score"],
                          mode='lines', name='Z-Score', line=dict(color='#38bdf8', width=2),
                          fill='tozeroy', fillcolor='rgba(56,189,248,0.15)',
                          text=[f"📅 {d.strftime('%Y-%m-%d')}<br>📊 Z-Score: {z:.2f}" 
                                for d, z in zip(df_chart["date"], df_chart["z_score"])],
                          hovertemplate='%{text}<extra></extra>'))
fig2.add_hline(y=tier2_threshold, line_dash="dash", line_color="#22c55e",
               annotation_text=f"Tier-2 ({tier2_threshold}σ)")
fig2.add_hline(y=pre_reg_threshold, line_dash="dash", line_color="#ef4444",
               annotation_text=f"Pre-reg ({pre_reg_threshold}σ)")
fig2.add_hline(y=0, line_dash="dot", line_color="#6b7280")
fig2.add_hline(y=euphoria_threshold, line_dash="dash", line_color="#ff6644",
               annotation_text=f"Euphoria ({euphoria_threshold}σ)")
fig2.update_layout(height=350, template="plotly_dark", yaxis_range=[-3.5, 3.5])
st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# СВОДНАЯ ТАБЛИЦА
# ============================================================

st.markdown("---")
st.subheader("📋 СВОДНАЯ ТАБЛИЦА ВСЕХ АКТИВОВ")

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
    
    if df_temp is not None:
        _, price, z, prob = calculate_metrics(df_temp)
        sig, _ = get_signal(z, pre_reg_threshold, tier2_threshold, euphoria_threshold)
        all_data.append({
            "Символ": symbol, "Тип": atype,
            "Цена": f"${price:,.2f}" if price else "—",
            "Z-Score": f"{z:.2f}" if z else "—",
            "Вероятность": f"{prob*100:.1f}%" if prob else "—",
            "Сигнал": sig.split("—")[0]
        })
    progress_bar.progress((i + 1) / len(all_assets))

status_text.empty()
progress_bar.empty()

if all_data:
    st.dataframe(pd.DataFrame(all_data), use_container_width=True, hide_index=True)

# ============================================================
# ПОДВАЛ
# ============================================================

st.markdown("---")
st.caption(f"📅 Последнее обновление: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("📡 Источник: CryptoCompare (криптовалюты), yfinance (акции)")
st.caption("🤖 AI-анализ от Claude API | ⚠️ Не инвестиционная рекомендация")
