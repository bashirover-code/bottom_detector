import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go
import yfinance as yf

# ============================================================
# НАСТРОЙКИ СТРАНИЦЫ И СИНХРОНИЗАЦИЯ
# ============================================================

st.set_page_config(page_title="Детектор дна активов v4.5", layout="wide")

# Автообновление каждые 15 минут (900 секунд)
st.markdown("""
    <meta http-equiv="refresh" content="900">
    <style>
        html, body, [class*="css"], .stMarkdown, .stMetric, .stDataFrame,
        .stButton, .stSelectbox, .stRadio, .stCaption, h1, h2, h3, h4, p, div {
            font-family: 'Times New Roman', Times, serif !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Детектор и Матрица Качества Активов v4.5")

# ============================================================
# 1. СПИСКИ АКТИВОВ И ХАРДКОД-РАНГ КАЧЕСТВА (QUALITY RATIO)
# ============================================================

CRYPTO_LIST = [
    "ETH", "BTC", "SOL", "ASTER", "IMX", "ZK", "FIL", "STX", "RENDER",
    "ONDO", "GRT", "CELO", "CRV", "TWT", "SUI", "APE", "ARKM", "ONE",
    "GOAT", "POL", "LINK", "UNI", "TRUMP", "ARC", "NEAR", "ALGO", "FLOCK"
]

STOCK_LIST = [
    "HIMS", "SIL", "GDX", "TSLA", "LIT", "ZM", "URA", "PLTR",
    "EWW", "BABA", "COIN", "NVDA", "SBER.ME", "MTSS.ME", "HEAD.ME"
]

VETERAN_LIST = ["BTC", "ETH", "LINK", "UNI", "AAPL", "MSFT", "NVDA", "TSLA"]

# Фундаментальное качество проекта (от 1 до 10) по вашей методологии
ASSET_QUALITY = {
    "BTC": 10, "ETH": 10, "NVDA": 10, "AAPL": 10, "MSFT": 10,
    "SOL": 9, "TSLA": 9, "PLTR": 9,
    "LINK": 8, "UNI": 8, "SUI": 8, "NEAR": 8, "COIN": 8, "SBER.ME": 8,
    "RENDER": 7, "STX": 7, "ONDO": 7, "GRT": 7, "HIMS": 7, "MTSS.ME": 7,
    "IMX": 6, "FIL": 6, "ARKM": 6, "POL": 6, "ALGO": 6, "GDX": 6, "URA": 6, "HEAD.ME": 6,
    "ZK": 5, "CELO": 5, "TWT": 5, "CRV": 5, "LIT": 5, "ZM": 5, "EWW": 5, "BABA": 5, "SIL": 5,
    "ASTER": 4, "ONE": 4, "GOAT": 4, "ARC": 4, "APE": 3,
    "TRUMP": 2, "FLOCK": 1
}

COINGECKO_IDS = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "FIL": "filecoin",
    "LINK": "chainlink", "UNI": "uniswap", "NEAR": "near", "ALGO": "algorand",
    "GRT": "the-graph", "CRV": "curve-dao-token", "STX": "blockstack",
    "RENDER": "render-token", "ONDO": "ondo-finance", "SUI": "sui",
    "APE": "apecoin", "IMX": "immutable-x", "ZK": "zksync", "TWT": "trust-wallet-token",
    "CELO": "celo", "ARKM": "arkham", "ONE": "harmony", "GOAT": "goat-2",
    "POL": "polygon", "TRUMP": "official-trump", "ARC": "arc", "FLOCK": "flock-colony",
    "ASTER": "astar"
}

# ============================================================
# 2. ЗАГРУЗКА ДАННЫХ С СИНХРОННЫМ КЭШЕМ (15 МИНУТ)
# ============================================================

@st.cache_data(ttl=900)
def load_crypto_data(symbol, days=550):
    if "CRYPTOCOMPARE_KEY" in st.secrets:
        try:
            key = st.secrets["CRYPTOCOMPARE_KEY"]
            url = "https://min-api.cryptocompare.com/data/v2/histoday"
            p = {"fsym": symbol, "tsym": "USD", "limit": days, "api_key": key}
            res = requests.get(url, params=p, timeout=10)
            if res.status_code == 200 and res.json().get("Response") == "Success":
                df = pd.DataFrame(res.json()["Data"]["Data"])
                df["date"] = pd.to_datetime(df["time"], unit='s')
                df = df.rename(columns={"volumeto": "volume"})
                return df[["date", "close", "volume"]].sort_values("date").reset_index(drop=True)
        except:
            pass
            
    try:
        s = yf.Ticker(f"{symbol}-USD")
        df = s.history(period=f"{days}d")
        if df is not None and not df.empty:
            df = df.reset_index().rename(columns={"Date": "date", "Close": "close", "Volume": "volume"})
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            return df[["date", "close", "volume"]].sort_values("date").reset_index(drop=True)
    except:
        return None

@st.cache_data(ttl=900)
def load_stock_data(symbol, days=550):
    try:
        s = yf.Ticker(symbol)
        df = s.history(period=f"{days}d")
        if df is not None and not df.empty:
            df = df.reset_index().rename(columns={"Date": "date", "Close": "close", "Volume": "volume"})
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            return df[["date", "close", "volume"]].sort_values("date").reset_index(drop=True)
    except:
        return None

@st.cache_data(ttl=900)
def get_market_regime():
    btc_df = load_crypto_data("BTC", days=300)
    if btc_df is not None and len(btc_df) >= 200:
        btc_df["ma200"] = btc_df["close"].rolling(window=200).mean()
        btc_price = btc_df["close"].iloc[-1]
        btc_ma200 = btc_df["ma200"].iloc[-1]
        if btc_price > btc_ma200:
            return "🟢 Бычий"
        else:
            return "🔴 Медвежий"
    return "⚪ Нейтральный"

# ============================================================
# 3. МАТЕМАТИЧЕСКИЕ ИНДИКАТОРЫ
# ============================================================

def calculate_rsi(df, periods=14):
    close_delta = df["close"].diff()
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    ma_up = up.ewm(com=periods - 1, adjust=False).mean()
    ma_down = down.ewm(com=periods - 1, adjust=False).mean()
    rsi = ma_up / (ma_down + 1e-10)
    return 100 - (100 / (1 + rsi))

def detect_rsi_divergence(df, lookback=35):
    if len(df) < lookback + 5:
        return False
    sub = df.tail(lookback).copy().reset_index(drop=True)
    
    price_mins = []
    for i in range(2, len(sub)-2):
        if sub["close"].iloc[i] < sub["close"].iloc[i-1] and sub["close"].iloc[i] < sub["close"].iloc[i-2] and \
           sub["close"].iloc[i] < sub["close"].iloc[i+1] and sub["close"].iloc[i] < sub["close"].iloc[i+2]:
            price_mins.append(i)
            
    if len(price_mins) >= 2:
        i1, i2 = price_mins[-2], price_mins[-1]
        if sub["close"].iloc[i2] < sub["close"].iloc[i1] and sub["rsi"].iloc[i2] > sub["rsi"].iloc[i1]:
            if sub["z_score"].iloc[i2] < -0.5:
                return True
    return False

def get_adaptive_thresholds(z_scores):
    if len(z_scores) < 30:
        return -1.8, 1.5
    low = np.nanpercentile(z_scores, 6)
    high = np.nanpercentile(z_scores, 94)
    return max(-3.2, min(-1.0, low)), min(3.2, max(0.6, high))

def get_signal_adaptive(z_score, low, high, is_vet):
    if is_vet:
        if z_score <= -1.8: return "🔴 СИЛЬНАЯ ЗОНА НАКОПЛЕНИЯ", "#ef4444"
        elif z_score <= -1.0: return "🟡 СТРАТЕГИЧЕСКОЕ НАКОПЛЕНИЕ", "#eab308"
        elif z_score >= 1.6: return "🟢 ОДНОЗНАЧНАЯ ФИКСАЦИЯ", "#22c55e"
        else: return "⚪ ОЖИДАНИЕ ТРИГГЕРОВ", "#6b7280"
    else:
        if z_score <= low: return "🔴 ТОЧКА ВХОДА (АДАПТИВ)", "#ef4444"
        elif z_score <= low * 0.6: return "🟡 МЯГКИЙ НАБОР", "#eab308"
        elif z_score >= high: return "🟢 ФИКСАЦИЯ ПРИБЫЛИ", "#22c55e"
        else: return "⚪ БОКОВИК / НЕЙТРАЛЬНО", "#6b7280"

# ============================================================
# 4. ДВУХФАКТОРНЫЙ МАТРИЧНЫЙ СКОРИНГ (ДНО + КАЧЕСТВО) v4.5
# ============================================================

def calculate_metrics_adaptive(symbol, df, btc_df=None, global_regime="⚪ Нейтральный"):
    if df is None or len(df) < 90:
        return (None,) * 15
        
    df = df.copy()
    
    df["ma90"] = df["close"].rolling(window=90, min_periods=30).mean()
    df["std90"] = df["close"].rolling(window=90, min_periods=30).std()
    df["z_score"] = (df["close"] - df["ma90"]) / (df["std90"] + 1e-10)
    df["rsi"] = calculate_rsi(df, 14)
    df["ma200"] = df["close"].rolling(window=200, min_periods=50).mean()
    
    df["v_mean"] = df["volume"].rolling(window=30, min_periods=10).mean()
    df["v_std"] = df["volume"].rolling(window=30, min_periods=10).std()
    df["vol_z"] = (df["volume"] - df["v_mean"]) / (df["v_std"] + 1e-10)
    
    calc_cols = ["ma90", "std90", "z_score", "rsi", "ma200", "v_mean", "v_std", "vol_z"]
    df[calc_cols] = df[calc_cols].bfill().ffill().fillna(0)
    
    current_price = df["close"].iloc[-1]
    c_z = df["z_score"].iloc[-1]
    c_rsi = df["rsi"].iloc[-1]
    c_vol_z = df["vol_z"].iloc[-1]
    c_ma200 = df["ma200"].iloc[-1]
    
    current_ath = df["close"].max()
    drawdown_pct = ((current_price - current_ath) / current_ath * 100) if current_ath > 0 else 0
    
    # Проблема №3: Окно измерения расширено до менее шумных 45 дней
    relative_strength = 0.0
    if btc_df is not None and len(btc_df) > 0:
        df_temp = df.copy()
        btc_temp = btc_df.copy()
        df_temp["d_norm"] = df_temp["date"].dt.date
        btc_temp["d_norm"] = btc_temp["date"].dt.date
        
        common_dates = np.intersect1d(df_temp['d_norm'], btc_temp['d_norm'])
        if len(common_dates) >= 45:
            df_sub = df_temp[df_temp['d_norm'].isin(common_dates)].sort_values("d_norm")
            btc_sub = btc_temp[btc_temp['d_norm'].isin(common_dates)].sort_values("d_norm")
            if len(df_sub) >= 45 and len(btc_sub) >= 45:
                asset_perf = (df_sub['close'].iloc[-1] / df_sub['close'].iloc[-45] - 1) * 100
                btc_perf = (btc_sub['close'].iloc[-1] / btc_sub['close'].iloc[-45] - 1) * 100
                relative_strength = asset_perf - btc_perf

    low_t, up_t = get_adaptive_thresholds(df["z_score"].values)
    dv_bull = detect_rsi_divergence(df, 35)
    
    reasons_checklist = []
    
    # --------------------------------------------------------
    # ИНДЕКС ДНА (Макс 100 баллов, вес в системе 40%)
    # --------------------------------------------------------
    bottom_score = 0
    
    # Решение №2: Вес просадки уменьшен до 20 баллов макс
    if drawdown_pct <= -85: 
        bottom_score += 20
        reasons_checklist.append(("📉 Экстремальная просадка от пика (>85%): +20 к Индексу Дна", True))
    elif drawdown_pct <= -70: 
        bottom_score += 15
        reasons_checklist.append(("📉 Глубокая просадка от пика (>70%): +15 к Индексу Дна", True))
    elif drawdown_pct <= -50: 
        bottom_score += 10
        reasons_checklist.append(("📉 Умеренная просадка от пика (>50%): +10 к Индексу Дна", True))
    else:
        reasons_checklist.append((" White Слабый дисконт по цене от исторического ATH", False))
        
    # RSI (Макс 25 баллов)
    if c_rsi <= 32: 
        bottom_score += 25
        reasons_checklist.append((f"⚡ RSI в зоне глубокого страха ({c_rsi:.1f}): +25 к Индексу Дна", True))
    elif c_rsi <= 42: 
        bottom_score += 15
        reasons_checklist.append((f"⚡ RSI в фазе накопления ({c_rsi:.1f}): +15 к Индексу Дна", True))
    
    # Z-Score (Макс 25 баллов)
    if c_z <= low_t: 
        bottom_score += 25
        reasons_checklist.append((f"📊 Падение Z-Score ниже адаптивного лимита ({c_z:.2f}): +25 к Индексу Дна", True))
    elif c_z < -0.8: 
        bottom_score += 12
        reasons_checklist.append((f"📊 Отрицательный Z-Score ({c_z:.2f}) ниже нормы: +12 к Индексу Дна", True))
        
    # Дивергенция (Макс 15 баллов)
    if dv_bull: 
        bottom_score += 15
        reasons_checklist.append(("🔄 Зафиксирован скрытый бычий разворот (Дивергенция RSI): +15 к Индексу Дна", True))
        
    # Климакс капитуляции объемов (Макс 15 баллов)
    if c_vol_z >= 1.5 and c_z < 0: 
        bottom_score += 15
        reasons_checklist.append((f"📦 Экстремальный объем продаж ({c_vol_z:+.1f}σ): +15 к Индексу Дна", True))
        
    bottom_score = min(bottom_score, 100)

    # --------------------------------------------------------
    # ИНДЕКС КАЧЕСТВА (Макс 100 баллов, вес в системе 60%)
    # --------------------------------------------------------
    base_quality_rank = ASSET_QUALITY.get(symbol, 5) # Ранг от 1 до 10
    quality_score = base_quality_rank * 10           # Преобразование в шкалу до 100 баллов
    
    reasons_checklist.append((f"💎 Базовый ранг качества фундамента проекта: {base_quality_rank}/10", True))

    # Решение №1: Штрафная и бонусная сетка за Альфа-силу к BTC
    if relative_strength > 10:
        quality_score += 10
        reasons_checklist.append((f"💪 Сильное опережение BTC ({relative_strength:+.1f}% за 45д): +10 к Индексу Качества", True))
    elif relative_strength > 0:
        quality_score += 5
        reasons_checklist.append((f"✅ Удержание тренда лучше BTC ({relative_strength:+.1f}% за 45д): +5 к Индексу Качества", True))
    elif relative_strength > -10:
        reasons_checklist.append((f"⏳ Умеренное скольжение за BTC ({relative_strength:+.1f}% за 45д): 0 к Индексу Качества", True))
    else:
        quality_score -= 10
        reasons_checklist.append((f"❌ Актив падает быстрее рынка ({relative_strength:+.1f}% за 45д): ШТРАФ -10 к Индексу Качества", False))

    # Решение №3: Системный макро-штраф за Медвежий цикл
    if global_regime == "🔴 Медвежий" and symbol != "BTC":
        quality_score -= 5
        reasons_checklist.append((" Bear Глобальный медвежий цикл рынка: ШТРАФ -5 к Индексу Качества", False))

    quality_score = max(0, min(quality_score, 100))

    # --------------------------------------------------------
    # ИТОГОВЫЙ СБАЛАНСИРОВАННЫЙ РЕЙТИНГ (0.6 * Качество + 0.4 * Дно)
    # --------------------------------------------------------
    final_rating = (0.6 * quality_score) + (0.4 * bottom_score)

    return df, current_price, c_z, bottom_score, quality_score, final_rating, (low_t, up_t), c_rsi, c_vol_z, dv_bull, drawdown_pct, relative_strength, current_ath, c_ma200, reasons_checklist

# ============================================================
# ПОЛУЧЕНИЕ ГЛОБАЛЬНОГО МАКРО-РЕЖИМА
# ============================================================

with st.sidebar:
    st.header("⚙️ НАСТРОЙКИ СИСТЕМЫ v4.5")
    st.markdown("---")
    market = st.radio("Сектор рынка", ["Криптовалюты", "Фондовый рынок"])
    if market == "Криптовалюты":
        asset = st.selectbox("Выбор цифрового актива", CRYPTO_LIST)
    else:
        asset = st.selectbox("Выбор акции/фонда", STOCK_LIST)
    st.markdown("---")
    st.caption("📈 **Матрица Выживаемости Капитала v4.5**")
    st.caption("Исправлен баг привязки к BTC. Формула: 60% Качество (с учетом силы к BTC за 45 дней и режима рынка) + 40% Дно.")

with st.spinner("Синхронизация глобальных индикаторов..."):
    market_regime = get_market_regime()
    btc_global_df = load_crypto_data("BTC", days=550)

# ============================================================
# РАСЧЕТЫ ПО ВЫБРАННОМУ АКТИВУ (ДЛЯ ПАСПОРТА)
# ============================================================

is_c = asset in CRYPTO_LIST
is_v = asset in VETERAN_LIST

with st.spinner(f"Загрузка потоков данных по {asset}..."):
    raw = load_crypto_data(asset) if is_c else load_stock_data(asset)

if raw is None or len(raw) < 90:
    st.error(f"❌ Недостаточно торговой истории для анализа {asset}.")
    st.stop()

df, c_price, c_z, b_score, q_score, f_rating, (low_thr, upper_thr), c_rsi, c_vol_z, dv_bull, drawdown_pct, rel_strength, current_ath, c_ma200, reasons = calculate_metrics_adaptive(asset, raw, btc_global_df if is_c else None, market_regime)
sig_t, sig_c = get_signal_adaptive(c_z, low_thr, upper_thr, is_v)

# ============================================================
# ДАШБОРД ВЫБРАННОГО АКТИВА
# ============================================================

st.header(f"📊 Паспорт актива: {asset}")
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("💰 ТЕКУЩАЯ ЦЕНА", f"${c_price:,.4f}" if c_price < 1 else f"${c_price:,.2f}")
with c2: 
    rating_color = "#22c55e" if f_rating >= 65 else "#eab308" if f_rating >= 45 else "#ef4444"
    st.markdown(f"""
        <div style='background: {rating_color}10; padding: 10px; border-radius: 8px; border: 1px solid {rating_color}30; text-align: center;'>
            <p style='color: gray; margin:0; font-size:11px; font-weight:bold;'>ИТОГОВЫЙ ИНДЕКС v4.5</p>
            <p style='color: {rating_color}; font-size:22px; font-weight:bold; margin:3px 0 0 0;'>{f_rating:.1f} / 100</p>
        </div>
    """, unsafe_allow_html=True)
with c3: st.metric("📉 ПРОСАДКА АКТИВА", f"{drawdown_pct:.1f}%")
with c4: st.metric("⚡ СИЛА К BTC (45д)", f"{rel_strength:+.1f}%" if is_c else "N/A")
with c5: st.metric("📈 RSI (14)", f"{c_rsi:.1f}")

# СТРОКА СТАТУСА
st.markdown(f"""
<div style='background: linear-gradient(135deg, #0b0f19 0%, #111827 100%); padding:14px; border-radius:10px; margin: 15px 0; border: 1px solid #1f2937;'>
    <p style='margin:0; color:#f3f4f6; font-size:14px; text-align: center;'>
        <b>Режим макро-рынка:</b> <span style='font-weight:bold;'>{market_regime}</span> &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Базовое качество проекта:</b> <span style='color:#38bdf8; font-weight:bold;'>{ASSET_QUALITY.get(asset, 5)}/10</span> &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Матрица Качества:</b> <span style='color:#38bdf8; font-weight:bold;'>{q_score}/100</span> &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Матрица Перепроданности (Дно):</b> <span style='color:#a855f7; font-weight:bold;'>{b_score}/100</span> &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Сигнал:</b> <span style='color:{sig_c}; font-weight:bold;'>{sig_t}</span>
    </p>
</div>
""", unsafe_allow_html=True)

# РАЗБОР ФАКТОРОВ ОЦЕНКИ
st.subheader("📋 Логическое обоснование оценок")
with st.container():
    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("**Конструктивные факторы (Начисление баллов):**")
        fulfilled = [r[0] for r in reasons if r[1]]
        if fulfilled:
            for item in fulfilled: st.markdown(f"- {item}")
        else: st.caption("Конструктивных триггеров не найдено.")
    with rc2:
        st.markdown("**Слабые стороны и системные ограничения (Штрафы):**")
        unfulfilled = [r[0] for r in reasons if not r[1]]
        if unfulfilled:
            for item in unfulfilled: st.markdown(f"- {item}")
        else: st.caption("Ограничений или штрафов на актив не наложено.")

# ============================================================
# 8. ИНТЕРАКТИВНЫЙ ГРАФИК ЦЕНЫ
# ============================================================
st.markdown("---")
st.subheader("📈 ИНТЕРАКТИВНЫЙ ТРЕНД (Цветовая палитра на базе Z-Score к MA90)")

df_chart = df.tail(500).copy()

def get_color(z, lower, upper):
    if z <= lower: return "#00ff66"
    elif z <= lower*0.5: return "#39ff14"
    elif z <= -0.5: return "#bfff00"
    elif z <= 0.5: return "#e5e7eb"
    elif z <= 1.2: return "#ffb703"
    elif z <= upper: return "#ff5500"
    else: return "#ff0055"

fig = go.Figure()
for i in range(len(df_chart) - 1):
    color = get_color(df_chart["z_score"].iloc[i], low_thr, upper_thr)
    fig.add_trace(go.Scatter(
        x=[df_chart["date"].iloc[i], df_chart["date"].iloc[i+1]],
        y=[df_chart["close"].iloc[i], df_chart["close"].iloc[i+1]],
        mode="lines", line=dict(color=color, width=3.5),
        showlegend=False, hoverinfo="skip"
    ))

if "ma90" in df_chart.columns:
    fig.add_trace(go.Scatter(x=df_chart["date"], y=df_chart["ma90"], mode="lines", name="MA90", line=dict(color="#ffffff", width=1.2, dash="dot"), opacity=0.4))
if "ma200" in df_chart.columns:
    fig.add_trace(go.Scatter(x=df_chart["date"], y=df_chart["ma200"], mode="lines", name="MA200", line=dict(color="#f59e0b", width=1.5, dash="dash"), opacity=0.6))

hover_texts = []
for d, p, z, r, v in zip(df_chart["date"], df_chart["close"], df_chart["z_score"], df_chart["rsi"], df_chart["vol_z"]):
    formatted_price = f"{p:,.4f}" if p < 1 else f"{p:,.2f}"
    text_item = (
        f"📅 <b>{d.strftime('%Y-%m-%d')}</b><br>"
        f"💰 <b>${formatted_price}</b><br>"
        f"📊 Z-Score(MA90): <b>{z:.2f}</b><br>"
        f"📈 RSI: <b>{r:.1f}</b><br>"
        f"📦 Объём σ: <b>{v:.2f}</b>"
    )
    hover_texts.append(text_item)

fig.add_trace(go.Scatter(
    x=df_chart["date"], y=df_chart["close"],
    mode="markers", marker=dict(color="rgba(0,0,0,0)", size=1),
    hoverinfo="text", text=hover_texts,
    name="Инфо", hovertemplate="%{text}<extra></extra>"
))

price_range = df_chart["close"].max() / (df_chart["close"].min() + 1e-10)
fig.update_layout(height=450, template="plotly_dark", xaxis_title="", yaxis_title="Цена (USD)", yaxis_type="log" if price_range > 5 else "linear", hovermode="x unified", legend=dict(orientation="h", y=1.02, x=0), font=dict(family="Times New Roman", size=13))
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 9. СКВОЗНАЯ МАТРИЦА СКОРИНГА РЫНКОВ v4.5 (ОБНОВЛЕННАЯ СЕТКА)
# ============================================================
st.markdown("---")
st.subheader("📋 СКВОЗНАЯ ИНВЕСТИЦИОННАЯ МАТРИЦА СКОРИНГА v4.5")

@st.cache_data(ttl=900)
def build_summary_table(regime):
    all_assets = {**{c: "Криптовалюта" for c in CRYPTO_LIST}, **{s: "Акция" for s in STOCK_LIST}}
    rows = []
    
    btc_df = load_crypto_data("BTC", days=550)
    
    for symbol, atype in all_assets.items():
        df_t = load_crypto_data(symbol) if atype == "Криптовалюта" else load_stock_data(symbol)
        if df_t is None or len(df_t) < 90:
            continue
            
        # Проблема №1 (РЕШЕНО): Опечатка "Кriптовалюта" полностью исправлена на "Криптовалюта"
        res = calculate_metrics_adaptive(symbol, df_t, btc_df if atype == "Криптовалюта" else None, regime)
        if res[0] is None:
            continue
            
        (_, price, z, bottom_s, quality_s, final_r, (lt, ut), rsi_v, _, _, ddown, rel_str, _, _, _) = res
        sig, _ = get_signal_adaptive(z, lt, ut, symbol in VETERAN_LIST)
        
        clean_sig = sig
        for emoji in ["🔴", "🟡", "🟢", "⚪"]:
            clean_sig = clean_sig.replace(emoji, "")
        clean_sig = clean_sig.strip()
        
        btc_strength_text = f"{rel_str:+.1f}%" if atype == "Криптовалюта" else "N/A"
        raw_quality_rank = ASSET_QUALITY.get(symbol, 5)
        
        # Проблема №2 (РЕШЕНО): Из таблицы исключены столбцы "Тип" и "Диверг."
        # Добавлены чистые столбцы "Сила к BTC" и "Качество" (в виде ранга от 1 до 10)
        rows.append({
            "Символ": symbol,
            "Цена": f"${price:,.4f}" if price < 1 else f"${price:,.2f}",
            "Z-Score": f"{z:.2f}",
            "RSI": f"{rsi_v:.1f}",
            "Просадка": f"{ddown:.1f}%",
            "Сила к BTC": btc_strength_text,
            "Индекс Дна": bottom_s,
            "Качество": f"{raw_quality_rank}/10",
            "Итог": round(final_r, 1),
            "Сигнал": clean_sig
        })
    return rows

with st.spinner("Сквозной пересчет рыночных коэффициентов альфа-силы..."):
    summary = build_summary_table(market_regime)

if summary:
    # Идеальная жесткая сортировка по Итоговому комбинированному весу (0.6 * Качество + 0.4 * Дно)
    df_summary = pd.DataFrame(summary).sort_values(by="Итог", ascending=False)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
    st.caption("💡 Матрица отсортирована по Итоговому рейтингу. Слабые, сильно упавшие щиткоины с низким рангом качества и отрицательной силой к BTC автоматически отсеиваются в подвал таблицы.")

# ============================================================
# 10. ПОДВАЛ
# ============================================================
moscow_tz = timezone(timedelta(hours=3))
moscow_time = datetime.now(moscow_tz)

st.markdown("---")
st.caption(f"📅 Синхронизация данных: {moscow_time.strftime('%Y-%m-%d %H:%M:%S')} (МСК) | Версия сборки: v4.5 (Фикс латинской опечатки в BTC-условии + Очищенный UI без шума + Окно сглаживания силы к BTC 45 дней)")
