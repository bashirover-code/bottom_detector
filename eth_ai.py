import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go
import yfinance as yf

# ============================================================
# НАСТРОЙКИ СТРАНИЦЫ И АВТООБНОВЛЕНИЕ
# ============================================================

st.set_page_config(page_title="Детектор дна активов", layout="wide")

# Глобальный шрифт Times New Roman
st.markdown("""
    <meta http-equiv="refresh" content="300">
    <style>
        html, body, [class*="css"], .stMarkdown, .stMetric, .stDataFrame,
        .stButton, .stSelectbox, .stRadio, .stCaption, h1, h2, h3, h4, p, div {
            font-family: 'Times New Roman', Times, serif !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Детектор дна активов")

# ============================================================
# 1. ФИНАЛЬНЫЙ СПИСОК АКТИВОВ
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
# 2. ФУНКЦИИ ДЛЯ РАБОТЫ С COINGECKO API
# ============================================================

@st.cache_data(ttl=600)
def get_coingecko_fundamentals(coin_id):
    """Получает фундаментальные данные (нужны для системы/AI)"""
    try:
        api_key = st.secrets.get("COINGECKO_API_KEY")
        if not api_key:
            return None
        
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
        params = {
            "localization": "false", "tickers": "false", "market_data": "true",
            "community_data": "true", "developer_data": "true", "sparkline": "false",
            "x_cg_demo_api_key": api_key
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            market_data = data.get("market_data", {})
            community_data = data.get("community_data", {})
            developer_data = data.get("developer_data", {})
            
            return {
                "price_usd": market_data.get("current_price", {}).get("usd", 0),
                "market_cap": market_data.get("market_cap", {}).get("usd", 0),
                "fully_diluted_valuation": market_data.get("fully_diluted_valuation", {}).get("usd", 0),
                "total_volume": market_data.get("total_volume", {}).get("usd", 0),
                "price_change_24h": market_data.get("price_change_percentage_24h", 0),
                "ath_usd": market_data.get("ath", {}).get("usd", 0),
                "atl_usd": market_data.get("atl", {}).get("usd", 0),
                "twitter_followers": community_data.get("twitter_followers", 0),
                "github_stars": developer_data.get("stars", 0),
                "github_forks": developer_data.get("forks", 0)
            }
    except Exception as e:
        pass
    return None

# ============================================================
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ И ИНДИКАТОРЫ
# ============================================================

def calculate_rsi(df, periods=14):
    """Расчет классического RSI"""
    close_delta = df['close'].diff()
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    ma_up = up.ewm(com=periods - 1, adjust=False).mean()
    ma_down = down.ewm(com=periods - 1, adjust=False).mean()
    
    rsi = ma_up / (ma_down + 1e-10)
    rsi = 100 - (100 / (1 + rsi))
    return rsi

def get_adaptive_thresholds(z_scores):
    if len(z_scores) < 30:
        return -1.8, 1.5
    lower = np.percentile(z_scores, 5)
    upper = np.percentile(z_scores, 95)
    lower = max(-3.0, min(-1.0, lower))
    upper = min(3.0, max(0.5, upper))
    return lower, upper

def get_signal_adaptive(z_score, lower_thr, upper_thr, is_veteran):
    if is_veteran:
        if z_score <= -1.8: return "🔴 ЭКСТРЕМАЛЬНАЯ ПОКУПКА", "#ef4444"
        elif z_score <= -1.2: return "🟡 ЗОНА НАКОПЛЕНИЯ", "#eab308"
        elif z_score >= 1.5: return "🟢 ЭЙФОРИЯ — ПРОДАВАЙ", "#22c55e"
        else: return "⚪ НЕЙТРАЛЬНО", "#6b7280"
    else:
        if z_score <= lower_thr: return "🔴 ПОКУПКА (АДАПТИВ)", "#ef4444"
        elif z_score <= lower_thr * 0.7: return "🟡 ЗОНА ВНИМАНИЯ", "#eab308"
        elif z_score >= upper_thr: return "🟢 ПРОДАЖА (АДАПТИВ)", "#22c55e"
        else: return "⚪ НЕЙТРАЛЬНО", "#6b7280"

def call_deepseek_analysis(asset_name, asset_symbol, current_price, current_z, current_prob, signal_text, confidence, fundamentals):
    deepseek_key = st.secrets.get("DEEPSEEK_API_KEY", "")
    if not deepseek_key:
        return "❌ API ключ DeepSeek не найден."
    
    asset_type = "криптовалюта" if asset_symbol in CRYPTO_LIST else "акция"
    fundamental_text = ""
    if fundamentals:
        fundamental_text = f"""
ДОПОЛНИТЕЛЬНЫЕ ДАННЫЕ (CoinGecko):
- Рыночная капитализация: ${fundamentals.get('market_cap', 0):,.0f}
- Полная оценка (FDV): ${fundamentals.get('fully_diluted_valuation', 0):,.0f}
- Объём за 24ч: ${fundamentals.get('total_volume', 0):,.0f}
- Изменение за 24ч: {fundamentals.get('price_change_24h', 0):.1f}%
- ATH: ${fundamentals.get('ath_usd', 0):,.0f}
"""
    
    prompt = f"""Проанализируй {asset_name} ({asset_symbol}, {asset_type}):
ТЕКУЩИЕ ДАННЫЕ:
- Цена: ${current_price:,.2f}
- Z-Score: {current_z:.2f}
- Вероятность дна: {current_prob*100:.1f}%
- Сигнал: {signal_text}
{fundamental_text}
Напиши КРАТКИЙ анализ (4-5 предложений) на русском:
1. Что означает текущий сигнал
2. Насколько фундаментальные данные подтверждают или опровергают сигнал
3. Рекомендация (покупать/продавать/держать)
4. Главный фактор риска"""

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {deepseek_key}", "Content-Type": "application/json"}
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 600,
        "temperature": 0.7
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ Ошибка API: {response.status_code}"
    except Exception as e:
        return f"❌ Ошибка: {str(e)[:100]}"

# ============================================================
# 4. ЗАГРУЗКА ЦЕНОВЫХ ДАННЫХ
# ============================================================

@st.cache_data(ttl=600)
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

@st.cache_data(ttl=600)
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

def calculate_metrics_adaptive(df):
    if df is None or len(df) < 30:
        return None, None, None, None, None, None, None
    
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    mean_ret = df["returns"].mean()
    std_ret = df["returns"].std()
    df["z_score"] = (df["returns"] - mean_ret) / (std_ret + 1e-10)
    df["rsi"] = calculate_rsi(df, 14)
    df = df.fillna(0)
    
    z_scores = df["z_score"].values
    lower_thr, upper_thr = get_adaptive_thresholds(z_scores)
    
    current_price = df["close"].iloc[-1]
    current_z = df["z_score"].iloc[-1]
    current_rsi = df["rsi"].iloc[-1]
    sensitivity = 1.5 if len(df) > 365 else 1.0
    current_prob = 1 / (1 + np.exp(current_z * sensitivity))
    confidence = min(100, len(df) / 365 * 100)
    
    return df, current_price, current_z, current_prob, confidence, (lower_thr, upper_thr), current_rsi

# ============================================================
# 5. ИНТЕРФЕЙС
# ============================================================

with st.sidebar:
    st.header("⚙️ Настройки")
    st.markdown("---")
    asset_type = st.radio("Тип актива", ["Криптовалюты", "Акции"])
    if asset_type == "Криптовалюты":
        selected_asset = st.selectbox("Криптовалюта", CRYPTO_LIST)
    else:
        selected_asset = st.selectbox("Акция", STOCK_LIST)
    st.markdown("---")
    st.caption("🤖 Адаптивный Z-Score (процентили)")
    st.caption("📈 RSI Индикатор (классический 14)")
    st.caption("🕐 Обновление: каждые 5-10 минут")
    st.caption(f"📋 Всего активов: {len(CRYPTO_LIST) + len(STOCK_LIST)}")

# ============================================================
# 6. ЗАГРУЗКА И РАСЧЁТ
# ============================================================

is_crypto = selected_asset in CRYPTO_LIST
is_veteran = selected_asset in VETERAN_LIST

# Загружаем в фоне для AI
fundamentals = None
if is_crypto and selected_asset in COINGECKO_IDS:
    coin_id = COINGECKO_IDS[selected_asset]
    fundamentals = get_coingecko_fundamentals(coin_id)

with st.spinner(f"🔄 Загрузка цены {selected_asset}..."):
    if is_crypto:
        df = load_crypto_data(selected_asset)
    else:
        df = load_stock_data(selected_asset)

if df is None or len(df) < 30:
    st.warning(f"⚠️ Недостаточно данных для {selected_asset} (нужно минимум 30 дней).")
    st.stop()

df, current_price, current_z, current_prob, confidence, (lower_thr, upper_thr), current_rsi = calculate_metrics_adaptive(df)
signal_text, signal_color = get_signal_adaptive(current_z, lower_thr, upper_thr, is_veteran)

# ============================================================
# 7. ПРОПОРЦИОНАЛЬНАЯ ЛИНЕЙКА ИНДИКАТОРОВ (БЕЗ СИГНАЛА)
# ============================================================

st.header(f"{selected_asset} — {'криптовалюта' if is_crypto else 'акция'}")

# Горизонтальная пропорциональная панель перестроена на 4 колонки
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("💰 ЦЕНА", f"${current_price:,.2f}")

with col2:
    st.metric("📊 Z-SCORE", f"{current_z:.2f}")

with col3:
    prob_color = "#22c55e" if current_prob > 0.6 else "#eab308" if current_prob > 0.4 else "#ef4444"
    st.markdown(f"""
        <div style='background: {prob_color}15; padding: 11px; border-radius: 8px; border: 1px solid {prob_color}40; text-align: center;'>
            <p style='color: gray; margin: 0; font-size: 14px; font-weight: bold;'>ВЕРОЯТНОСТЬ ДНА</p>
            <p style='color: {prob_color}; font-size: 22px; font-weight: bold; margin: 5px 0 0 0;'>{current_prob*100:.1f}%</p>
        </div>
    """, unsafe_allow_html=True)

with col4:
    rsi_color = "#ef4444" if current_rsi <= 30 else "#22c55e" if current_rsi >= 70 else "#00d4ff"
    st.markdown(f"""
        <div style='background: {rsi_color}15; padding: 11px; border-radius: 8px; border: 1px solid {rsi_color}40; text-align: center;'>
            <p style='color: gray; margin: 0; font-size: 14px; font-weight: bold;'>RSI (14)</p>
            <p style='color: {rsi_color}; font-size: 22px; font-weight: bold; margin: 5px 0 0 0;'>{current_rsi:.1f}</p>
        </div>
    """, unsafe_allow_html=True)

# Главный текстовый блок порогов (здесь статус "Сигнал" сохранен текстом)
st.markdown(f"""
<div style='background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
            padding: 15px; border-radius: 12px; margin: 20px 0; text-align: center;
            border-left: 5px solid {signal_color};'>
    <p style='color: #9ca3af; margin: 0; font-size: 15px;'>
        <b>Текущий статус:</b> {signal_text} | Адаптивные пороги Z-Score: покупка &lt; {lower_thr:.2f} | продажа &gt; {upper_thr:.2f}
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 8. AI-АНАЛИЗ
# ============================================================

st.markdown("---")
st.subheader("🤖 AI-анализ актива")

if st.button(f"📊 Получить AI-анализ для {selected_asset}", type="primary"):
    with st.spinner("🧠 DeepSeek анализирует..."):
        analysis = call_deepseek_analysis(
            selected_asset, selected_asset, current_price, current_z,
            current_prob, signal_text.split("—")[0], confidence, fundamentals
        )
    
    analysis_html = analysis.replace('\n', '<br>').replace('•', '&bull;')
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                padding: 20px; border-radius: 16px; margin: 10px 0;
                border: 1px solid #2a2a3e;'>
        <h4 style='margin-bottom: 10px; color: #ffffff;'>📈 Анализ от DeepSeek AI</h4>
        <div style='color: #ffffff; font-size: 15px; line-height: 1.6;'>{analysis_html}</div>
        <p style='color: #888888; font-size: 12px; margin-top: 10px;'>⚡ DeepSeek Chat | Анализ на основе Z-Score + фонового фундамента</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 9. ГРАФИКИ
# ============================================================

st.markdown("---")
st.subheader("📈 ГРАФИК ЦЕНЫ")

df_chart = df.tail(500).copy()

def get_color(z, lower, upper):
    """Сочная неоновая палитра для лучшей видимости тонких линий"""
    if z <= lower: return "#00ff66"        # Сочный неоновый зеленый (дно)
    elif z <= lower * 0.7: return "#39ff14"  # Кислотный лайм
    elif z <= -0.5: return "#bfff00"         # Лимонный
    elif z <= 0.5: return "#e5e7eb"          # Чистый контрастный светло-серый (флэт)
    elif z <= 1.2: return "#ffb703"          # Сочный оранжевый
    elif z <= upper: return "#ff5500"        # Яркий рыжий
    else: return "#ff0055"                   # Сочный фуксия/огненно-красный (хай)

fig = go.Figure()
for i in range(len(df_chart) - 1):
    color = get_color(df_chart["z_score"].iloc[i], lower_thr, upper_thr)
    fig.add_trace(go.Scatter(
        x=[df_chart["date"].iloc[i], df_chart["date"].iloc[i+1]],
        y=[df_chart["close"].iloc[i], df_chart["close"].iloc[i+1]],
        mode='lines', line=dict(color=color, width=3.5),
        showlegend=False, hoverinfo='skip'
    ))

fig.add_trace(go.Scatter(
    x=df_chart["date"], y=df_chart["close"],
    mode='markers', marker=dict(color='rgba(0,0,0,0)', size=1),
    hoverinfo='text',
    text=[f"📅 <b>{d.strftime('%Y-%m-%d')}</b><br>💰 <b>${p:,.2f}</b><br>📊 Z-Score: <b>{z:.2f}</b><br>📈 RSI: <b>{r:.1f}</b>" 
          for d, p, z, r in zip(df_chart["date"], df_chart["close"], df_chart["z_score"], df_chart["rsi"])],
    name="Информация", hovertemplate='%{text}<extra></extra>'
))

# Убрано слово "дата" из xaxis_title
fig.update_layout(height=450, template="plotly_dark", xaxis_title="",
                  yaxis_title="Цена (USD)", yaxis_type="log" if current_price > 100 else "linear",
                  hovermode="x unified",
                  font=dict(family="Times New Roman, Times, serif", size=13))
st.plotly_chart(fig, use_container_width=True)

st.subheader("📉 Z-SCORE С АДАПТИВНЫМИ ПОРОГАМИ")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_chart["date"], y=df_chart["z_score"],
                          mode='lines', name='Z-Score', line=dict(color='#00d4ff', width=2.5),
                          fill='tozeroy', fillcolor='rgba(0, 212, 255, 0.15)',
                          text=[f"📅 {d.strftime('%Y-%m-%d')}<br>📊 Z-Score: {z:.2f}" 
                                for d, z in zip(df_chart["date"], df_chart["z_score"])],
                          hovertemplate='%{text}<extra></extra>'))
fig2.add_hline(y=lower_thr, line_dash="dash", line_color="#22ff55", line_width=2,
               annotation_text=f"ПОКУПКА ({lower_thr:.2f}σ)", annotation_position="right")
fig2.add_hline(y=upper_thr, line_dash="dash", line_color="#ff4422", line_width=2,
               annotation_text=f"ПРОДАЖА ({upper_thr:.2f}σ)", annotation_position="right")
fig2.add_hline(y=0, line_dash="dot", line_color="#888888")
fig2.update_layout(height=300, template="plotly_dark", yaxis_range=[-3.5, 3.5],
                   font=dict(family="Times New Roman, Times, serif", size=13))
st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# 10. СВОДНАЯ ТАБЛИЦА
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
    
    if df_temp is not None and len(df_temp) >= 30:
        _, price, z, prob, conf, (lthr, uthr), rsi_val = calculate_metrics_adaptive(df_temp)
        sig, _ = get_signal_adaptive(z, lthr, uthr, symbol in VETERAN_LIST)
        all_data.append({
            "Символ": symbol, "Тип": atype,
            "Цена": f"${price:,.2f}",
            "Z-Score": f"{z:.2f}",
            "RSI (14)": f"{rsi_val:.1f}",
            "Вероятность дна": f"{prob*100:.1f}%",
            "Сигнал": sig.split("—")[0]
        })
    progress_bar.progress((i + 1) / len(all_assets))

status_text.empty()
progress_bar.empty()

if all_data:
    st.dataframe(pd.DataFrame(all_data), use_container_width=True, hide_index=True)

# ============================================================
# 11. ПОДВАЛ
# ============================================================

moscow_tz = timezone(timedelta(hours=3))
moscow_time = datetime.now(moscow_tz)

st.markdown("---")
st.caption(f"📅 Обновлено: {moscow_time.strftime('%Y-%m-%d %H:%M:%S')} (МСК)")
st.caption("📡 Источник: CryptoCompare / yfinance / CoinGecko | 🤖 AI: DeepSeek")
st.caption("⚡ Адаптивный Z-Score + Классический RSI | 🎨 Яркие цвета графиков | ⚠️ Не инвестиционная рекомендация")
