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

st.set_page_config(page_title="Детектор дна активов v2", layout="wide")

st.markdown("""
    <meta http-equiv="refresh" content="300">
    <style>
        html, body, [class*="css"], .stMarkdown, .stMetric, .stDataFrame,
        .stButton, .stSelectbox, .stRadio, .stCaption, h1, h2, h3, h4, p, div {
            font-family: 'Times New Roman', Times, serif !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Детектор дна активов v2")

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
    "BTC":    "bitcoin",
    "ETH":    "ethereum",
    "SOL":    "solana",
    "FIL":    "filecoin",
    "LINK":   "chainlink",
    "UNI":    "uniswap",
    "NEAR":   "near",
    "ALGO":   "algorand",
    "GRT":    "the-graph",
    "CRV":    "curve-dao-token",
    "STX":    "blockstack",
    "RENDER": "render-token",
    "ONDO":   "ondo-finance",
    "SUI":    "sui",
    "APE":    "apecoin",
    "IMX":    "immutable-x",
    "ZK":     "zksync",
    "TWT":    "trust-wallet-token",
    "CELO":   "celo",
    "ARKM":   "arkham",
    "ONE":    "harmony",
    "GOAT":   "goat-2",
    "POL":    "polygon-ecosystem-token",
    "TRUMP":  "official-trump",
    "ARC":    "arc-agi",
    "FLOCK":  "flock-io",
    "ASTER":  "astar",
}

# ============================================================
# 2. COINGECKO ФУНДАМЕНТАЛ
# ============================================================

@st.cache_data(ttl=600)
def get_coingecko_fundamentals(coin_id):
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
            md = data.get("market_data", {})
            cd = data.get("community_data", {})
            dd = data.get("developer_data", {})
            return {
                "price_usd":              md.get("current_price", {}).get("usd", 0),
                "market_cap":             md.get("market_cap", {}).get("usd", 0),
                "fully_diluted_valuation":md.get("fully_diluted_valuation", {}).get("usd", 0),
                "total_volume":           md.get("total_volume", {}).get("usd", 0),
                "price_change_24h":       md.get("price_change_percentage_24h", 0),
                "ath_usd":                md.get("ath", {}).get("usd", 0),
                "atl_usd":                md.get("atl", {}).get("usd", 0),
                "twitter_followers":      cd.get("twitter_followers", 0),
                "github_stars":           dd.get("stars", 0),
                "github_forks":           dd.get("forks", 0),
            }
    except Exception:
        pass
    return None

# ============================================================
# 3. ИНДИКАТОРЫ И ДИВЕРГЕНЦИЯ
# ============================================================

def calculate_rsi(series, periods=14):
    delta = series.diff()
    up   = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up   = up.ewm(com=periods - 1, adjust=False).mean()
    ma_down = down.ewm(com=periods - 1, adjust=False).mean()
    rs  = ma_up / (ma_down + 1e-10)
    return 100 - (100 / (1 + rs))

def detect_rsi_divergence(df, lookback=35):
    """
    Поиск классической бычьей дивергенции:
    Цена обновляет локальный минимум (Lower Low), а RSI формирует более высокий минимум (Higher Low).
    """
    if len(df) < lookback + 5:
        return False
        
    sub = df.tail(lookback).copy().reset_index(drop=True)
    
    # Поиск локальных экстремумов (минимумов) в окне
    local_mins = []
    for i in range(2, len(sub) - 2):
        if sub["close"].iloc[i] < sub["close"].iloc[i-1] and sub["close"].iloc[i] < sub["close"].iloc[i-2] and \
           sub["close"].iloc[i] < sub["close"].iloc[i+1] and sub["close"].iloc[i] < sub["close"].iloc[i+2]:
            local_mins.append(i)
            
    if len(local_mins) >= 2:
        # Берем два последних значимых локальных минимума
        idx1, idx2 = local_mins[-2], local_mins[-1]
        
        price1, price2 = sub["close"].iloc[idx1], sub["close"].iloc[idx2]
        rsi1, rsi2 = sub["rsi"].iloc[idx1], sub["rsi"].iloc[idx2]
        
        # Условие бычьей дивергенции: цена упала ниже, но индикатор RSI пошел вверх
        if price2 < price1 and rsi2 > rsi1:
            # Подтверждаем, что это происходит в зоне относительной перепроданности/угнетения
            if rsi2 < 45:
                return True
    return False

def get_adaptive_thresholds(z_scores):
    if len(z_scores) < 30:
        return -1.8, 1.5
    lower = np.percentile(z_scores, 5)
    upper = np.percentile(z_scores, 95)
    lower = max(-3.0, min(-1.0, lower))
    upper = min(3.0,  max(0.5,  upper))
    return lower, upper

def get_signal_adaptive(z_score, lower_thr, upper_thr, is_veteran):
    if is_veteran:
        if   z_score <= -1.8: return "🔴 ЭКСТРЕМАЛЬНАЯ ПОКУПКА", "#ef4444"
        elif z_score <= -1.2: return "🟡 ЗОНА НАКОПЛЕНИЯ",        "#eab308"
        elif z_score >= 1.5:  return "🟢 ЭЙФОРИЯ — ПРОДАВАЙ",    "#22c55e"
        else:                 return "⚪ НЕЙТРАЛЬНО",              "#6b7280"
    else:
        attention_thr = lower_thr * 1.3
        if   z_score <= lower_thr:    return "🔴 ПОКУПКА (АДАПТИВ)",  "#ef4444"
        elif z_score <= attention_thr: return "🟡 ЗОНА ВНИМАНИЯ",     "#eab308"
        elif z_score >= upper_thr:    return "🟢 ПРОДАЖА (АДАПТИВ)", "#22c55e"
        else:                         return "⚪ НЕЙТРАЛЬНО",         "#6b7280"

# ============================================================
# 4. ЗАГРУЗКА ДАННЫХ (с объёмом + fallback на yfinance)
# ============================================================

@st.cache_data(ttl=600)
def load_crypto_data(symbol, days=500):
    try:
        if "CRYPTOCOMPARE_KEY" in st.secrets:
            API_KEY = st.secrets["CRYPTOCOMPARE_KEY"]
            url    = "https://min-api.cryptocompare.com/data/v2/histoday"
            params = {"fsym": symbol, "tsym": "USD", "limit": days, "api_key": API_KEY}
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                data = r.json()
                if data.get("Response") == "Success":
                    raw = data["Data"]["Data"]
                    df  = pd.DataFrame(raw)
                    df["date"]   = pd.to_datetime(df["time"], unit="s")
                    df["close"]  = df["close"].astype(float)
                    df["volume"] = df["volumeto"].astype(float)
                    return df[["date", "close", "volume"]].sort_values("date").reset_index(drop=True)
    except Exception:
        pass

    try:
        ticker = f"{symbol}-USD"
        end    = datetime.now()
        start  = end - timedelta(days=days)
        ydf    = yf.Ticker(ticker).history(start=start, end=end)
        if ydf is not None and not ydf.empty:
            ydf = ydf.reset_index()
            ydf = ydf.rename(columns={"Date": "date", "Close": "close", "Volume": "volume"})
            ydf["date"] = pd.to_datetime(ydf["date"]).dt.tz_localize(None)
            return ydf[["date", "close", "volume"]].sort_values("date").reset_index(drop=True)
    except Exception:
        pass
    return None

@st.cache_data(ttl=600)
def load_stock_data(symbol, days=500):
    try:
        end   = datetime.now()
        start = end - timedelta(days=days)
        ydf   = yf.Ticker(symbol).history(start=start, end=end)
        if ydf is not None and not ydf.empty:
            ydf = ydf.reset_index()
            ydf = ydf.rename(columns={"Date": "date", "Close": "close", "Volume": "volume"})
            ydf["date"] = pd.to_datetime(ydf["date"]).dt.tz_localize(None)
            return ydf[["date", "close", "volume"]].sort_values("date").reset_index(drop=True)
    except Exception:
        pass
    return None

# ============================================================
# 5. РАСЧЁТ МЕТРИК С УЧЁТОМ ДИВЕРГЕНЦИИ
# ============================================================

def calculate_metrics_adaptive(df):
    if df is None or len(df) < 30:
        return None, None, None, None, None, None, None, None, None, False

    df = df.copy()

    df["ma90"]  = df["close"].rolling(90,  min_periods=20).mean()
    df["ma200"] = df["close"].rolling(200, min_periods=50).mean()

    rolling_std90 = df["close"].rolling(90, min_periods=20).std()
    df["z_score"] = (df["close"] - df["ma90"]) / (rolling_std90 + 1e-10)

    df["rsi"] = calculate_rsi(df["close"], 14)

    if "volume" in df.columns and df["volume"].sum() > 0:
        vol_mean = df["volume"].rolling(30, min_periods=10).mean()
        vol_std  = df["volume"].rolling(30, min_periods=10).std()
        df["vol_z"] = (df["volume"] - vol_mean) / (vol_std + 1e-10)
    else:
        df["vol_z"] = 0.0

    df = df.fillna(0)

    z_scores             = df["z_score"].values
    lower_thr, upper_thr = get_adaptive_thresholds(z_scores)

    # Расчет бычьей дивергенции RSI/цены
    has_divergence = detect_rsi_divergence(df, lookback=35)

    current_price = df["close"].iloc[-1]
    current_z     = df["z_score"].iloc[-1]
    current_rsi   = df["rsi"].iloc[-1]
    current_vol_z = df["vol_z"].iloc[-1]
    current_ma200 = df["ma200"].iloc[-1]
    current_ma90  = df["ma90"].iloc[-1]

    # -------------------------------------------------------
    # СОСТАВНАЯ ВЕРОЯТНОСТЬ ДНА (Отрегулированные веса, макс 100)
    # -------------------------------------------------------
    score = 0

    # Компонент 1: Z-Score цены к MA90 (30 баллов)
    if current_z <= lower_thr:
        z_depth = abs(current_z - lower_thr)
        score += min(30, 15 + z_depth * 10)
    elif current_z <= lower_thr * 0.5:
        score += 8

    # Компонент 2: RSI (20 баллов)
    if   current_rsi <= 20: score += 20
    elif current_rsi <= 30: score += 15
    elif current_rsi <= 40: score += 8

    # Компонент 3: БЫЧЬЯ ДИВЕРГЕНЦИЯ RSI (20 баллов) — сильнейший разворотный паттерн
    if has_divergence:
        score += 20

    # Компонент 4: аномальный объём капитуляции (15 баллов)
    if   current_vol_z >= 3.0: score += 15
    elif current_vol_z >= 2.0: score += 10
    elif current_vol_z >= 1.0: score += 5

    # Компонент 5: контекст тренда — цена выше MA200 (15 баллов)
    if current_ma200 > 0 and current_price > current_ma200:
        score += 15

    current_prob = score / 100.0
    confidence = min(100, len(df) / 365 * 100)

    return (df, current_price, current_z, current_prob,
            confidence, (lower_thr, upper_thr), current_rsi,
            current_vol_z, current_ma90, has_divergence)

# ============================================================
# 6. DeepSeek AI-АНАЛИЗ
# ============================================================

def call_deepseek_analysis(symbol, current_price, current_z, current_prob,
                            signal_text, current_rsi, current_vol_z,
                            current_ma90, confidence, fundamentals, has_divergence):
    deepseek_key = st.secrets.get("DEEPSEEK_API_KEY", "")
    if not deepseek_key:
        return "❌ API ключ DeepSeek не найден."

    asset_type = "криптовалюта" if symbol in CRYPTO_LIST else "акция"

    fund_text = ""
    if fundamentals:
        fund_text = f"""
ФУНДАМЕНТАЛЬНЫЕ ДАННЫЕ (CoinGecko):
- Рыночная капитализация: ${fundamentals.get('market_cap', 0):,.0f}
- FDV: ${fundamentals.get('fully_diluted_valuation', 0):,.0f}
- Объём 24ч: ${fundamentals.get('total_volume', 0):,.0f}
- Изменение 24ч: {fundamentals.get('price_change_24h', 0):.1f}%
- ATH: ${fundamentals.get('ath_usd', 0):,.2f}
"""

    vol_comment = "аномально высокий (возможная капитуляция)" if current_vol_z >= 2 else \
                  "повышенный" if current_vol_z >= 1 else "нормальный"
                  
    div_comment = "ОБНАРУЖЕНА (Классический сильный сигнал разворота тренда вверх)" if has_divergence else "отсутствует"

    prompt = f"""Проведи глубокий разбор {symbol} ({asset_type}):

ТЕХНИЧЕСКИЕ ДАННЫЕ:
- Цена: ${current_price:,.4f}
- Z-Score (цена к MA90): {current_z:.2f}
- RSI (14): {current_rsi:.1f}
- Бычья дивергенция RSI/Цена: {div_comment}
- Аномалия объёма (z): {current_vol_z:.2f} — {vol_comment}
- MA90: ${current_ma90:,.4f}
- Составная вероятность дна: {current_prob*100:.1f}% (уверенность модели: {confidence:.0f}%)
- Сигнал: {signal_text}
{fund_text}

Методология скоринга вероятности дна:
  30% — Z-Score цены к MA90
  20% — Классический RSI
  20% — Бычья дивергенция RSI (цена обновляет локальный лоу, индикатор — нет)
  15% — Капитуляция по объемам
  15% — Положение относительно MA200

Напиши экспертный вывод (5-6 предложений) на русском языке:
1. Что означает текущий сигнал с акцентом на наличие или отсутствие дивергенции RSI.
2. Подтверждает ли структура объёма скрытый или явный выкуп дна крупным игроком.
3. Оценка контекста: это временное охлаждение бычьего тренда или опасный слом в затяжную медвежку.
4. Конкретная торговая рекомендация.
5. Критический фактор риска."""

    url     = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {deepseek_key}", "Content-Type": "application/json"}
    payload = {
        "model":       "deepseek-chat",
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  700,
        "temperature": 0.3,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        return f"❌ Ошибка API: {r.status_code}"
    except Exception as e:
        return f"❌ Ошибка: {str(e)[:100]}"

# ============================================================
# 7. ИНТЕРФЕЙС — САЙДБАР
# ============================================================

with st.sidebar:
    st.header("⚙️ Настройки")
    st.markdown("---")
    asset_type   = st.radio("Тип актива", ["Криптовалюты", "Акции"])
    if asset_type == "Криптовалюты":
        selected_asset = st.selectbox("Криптовалюта", CRYPTO_LIST)
    else:
        selected_asset = st.selectbox("Акция", STOCK_LIST)
    st.markdown("---")
    st.caption("🔢 Z-Score: цена vs MA90 (не однодневный возврат)")
    st.caption("🔄 Бычья дивергенция RSI/Цена")
    st.caption("📈 RSI 14 | 📦 Аномалия объёма | 📉 MA200-контекст")
    st.caption("🕐 Обновление каждые 5 минут")
    st.caption(f"📋 Всего активов: {len(CRYPTO_LIST) + len(STOCK_LIST)}")

# ============================================================
# 8. ЗАГРУЗКА И РАСЧЁТ
# ============================================================

is_crypto  = selected_asset in CRYPTO_LIST
is_veteran = selected_asset in VETERAN_LIST

fundamentals = None
if is_crypto and selected_asset in COINGECKO_IDS:
    fundamentals = get_coingecko_fundamentals(COINGECKO_IDS[selected_asset])

with st.spinner(f"🔄 Загрузка {selected_asset}..."):
    df = load_crypto_data(selected_asset) if is_crypto else load_stock_data(selected_asset)

if df is None or len(df) < 30:
    st.warning(f"⚠️ Недостаточно данных для {selected_asset} (минимум 30 дней).")
    st.stop()

result = calculate_metrics_adaptive(df)
if result[0] is None:
    st.warning("⚠️ Ошибка расчёта метрик.")
    st.stop()

(df, current_price, current_z, current_prob,
 confidence, (lower_thr, upper_thr), current_rsi,
 current_vol_z, current_ma90, has_divergence) = result

signal_text, signal_color = get_signal_adaptive(current_z, lower_thr, upper_thr, is_veteran)

# ============================================================
# 9. ПАНЕЛЬ МЕТРИК (5 колонок)
# ============================================================

st.header(f"{selected_asset} — {'криптовалюта' if is_crypto else 'акция'}")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("💰 ЦЕНА", f"${current_price:,.4f}" if current_price < 1 else f"${current_price:,.2f}")

with col2:
    z_color = "#ef4444" if current_z <= lower_thr else "#22c55e" if current_z >= upper_thr else "#00d4ff"
    st.markdown(f"""
        <div style='background:{z_color}15;padding:11px;border-radius:8px;
                    border:1px solid {z_color}40;text-align:center;'>
            <p style='color:gray;margin:0;font-size:13px;font-weight:bold;'>Z-SCORE (MA90)</p>
            <p style='color:{z_color};font-size:22px;font-weight:bold;margin:5px 0 0 0;'>{current_z:.2f}</p>
        </div>""", unsafe_allow_html=True)

with col3:
    prob_pct   = current_prob * 100
    prob_color = "#22c55e" if prob_pct > 60 else "#eab308" if prob_pct > 35 else "#ef4444"
    st.markdown(f"""
        <div style='background:{prob_color}15;padding:11px;border-radius:8px;
                    border:1px solid {prob_color}40;text-align:center;'>
            <p style='color:gray;margin:0;font-size:13px;font-weight:bold;'>ВЕРОЯТНОСТЬ ДНА</p>
            <p style='color:{prob_color};font-size:22px;font-weight:bold;margin:5px 0 0 0;'>{prob_pct:.1f}%</p>
        </div>""", unsafe_allow_html=True)

with col4:
    # Обозначение дивергенции прямо в плашке RSI
    rsi_text = "RSI (14)" if not has_divergence else "RSI + ДИВЕРГЕНЦИЯ ✅"
    rsi_color = "#22c55e" if has_divergence else ("#ef4444" if current_rsi <= 30 else "#00d4ff")
    st.markdown(f"""
        <div style='background:{rsi_color}15;padding:11px;border-radius:8px;
                    border:1px solid {rsi_color}40;text-align:center;'>
            <p style='color:gray;margin:0;font-size:11px;font-weight:bold;'>{rsi_text}</p>
            <p style='color:{rsi_color};font-size:22px;font-weight:bold;margin:5px 0 0 0;'>{current_rsi:.1f}</p>
        </div>""", unsafe_allow_html=True)

with col5:
    vol_color = "#ef4444" if current_vol_z >= 2 else "#eab308" if current_vol_z >= 1 else "#6b7280"
    vol_label = "КАПИТУЛЯЦИЯ" if current_vol_z >= 2 else "ПОВЫШЕН" if current_vol_z >= 1 else "НОРМА"
    st.markdown(f"""
        <div style='background:{vol_color}15;padding:11px;border-radius:8px;
                    border:1px solid {vol_color}40;text-align:center;'>
            <p style='color:gray;margin:0;font-size:13px;font-weight:bold;'>% ОБЪЁМ ({vol_label})</p>
            <p style='color:{vol_color};font-size:22px;font-weight:bold;margin:5px 0 0 0;'>{current_vol_z:.2f}σ</p>
        </div>""", unsafe_allow_html=True)

# Строка статуса
st.markdown(f"""
<div style='background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
            padding:15px;border-radius:12px;margin:20px 0;text-align:center;
            border-left:5px solid {signal_color};'>
    <p style='color:#9ca3af;margin:0;font-size:15px;'>
        <b>Статус детектора:</b> {signal_text} &nbsp;|&nbsp;
        Пороги Z-Score: покупка &lt; {lower_thr:.2f}σ &nbsp;|&nbsp; продажа &gt; {upper_thr:.2f}σ &nbsp;|&nbsp;
        Дивергенция RSI: <b>{'ОБНАРУЖЕНА ✅' if has_divergence else 'НЕТ ❌'}</b>
    </p>
</div>
""", unsafe_allow_html=True)

# Декомпозиция вероятности дна
with st.expander("🔍 Декомпозиция вероятности дна (Матричный Скоринг)"):
    s_z   = 0
    if current_z <= lower_thr:
        s_z = min(30, 15 + abs(current_z - lower_thr) * 10)
    elif current_z <= lower_thr * 0.5:
        s_z = 8
    s_rsi = 20 if current_rsi<=20 else 15 if current_rsi<=30 else 8 if current_rsi<=40 else 0
    s_div = 20 if has_divergence else 0
    s_vol = 15 if current_vol_z>=3 else 10 if current_vol_z>=2 else 5 if current_vol_z>=1 else 0
    ma200_val = df["ma200"].iloc[-1]
    s_trend = 15 if (ma200_val > 0 and current_price > ma200_val) else 0

    dc1, dc2, dc3, dc4, dc5 = st.columns(5)
    dc1.metric("Z-Score (макс 30)", f"{s_z:.0f} б.")
    dc2.metric("RSI (макс 20)",     f"{s_rsi:.0f} б.")
    dc3.metric("Бычья Диверг. (макс 20)", f"{s_div:.0f} б.")
    dc4.metric("Объём (макс 15)",   f"{s_vol:.0f} б.")
    dc5.metric("MA200-тренд (макс 15)", f"{s_trend:.0f} б.")
    st.caption(f"MA200 = ${ma200_val:,.2f} | Контекст рынка: {'Бычий откат (Здоровый) ✅' if s_trend else 'Медвежий рынок (Опасный) ⚠️'}")

# ============================================================
# 10. AI-АНАЛИЗ
# ============================================================

st.markdown("---")
st.subheader("🤖 AI-анализ актива")

if st.button(f"📊 Получить AI-анализ для {selected_asset}", type="primary"):
    with st.spinner("🧠 DeepSeek анализирует дивергенции и стаканы..."):
        analysis = call_deepseek_analysis(
            selected_asset, current_price, current_z, current_prob,
            signal_text, current_rsi, current_vol_z,
            current_ma90, confidence, fundamentals, has_divergence
        )
    analysis_html = analysis.replace('\n', '<br>').replace('•', '&bull;')
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
                padding:20px;border-radius:16px;margin:10px 0;border:1px solid #2a2a3e;'>
        <h4 style='margin-bottom:10px;color:#ffffff;'>📈 Комплексный разбор DeepSeek AI</h4>
        <div style='color:#ffffff;font-size:15px;line-height:1.6;'>{analysis_html}</div>
        <p style='color:#888888;font-size:12px;margin-top:10px;'>
            ⚡ DeepSeek Engine | Мультифакторное сканирование (Z-Score + RSI Divergence + Volume Spikes)
        </p>
    </div>""", unsafe_allow_html=True)

# ============================================================
# 11. ГРАФИКИ
# ============================================================

st.markdown("---")
st.subheader("📈 ГРАФИК ЦЕНЫ (цвет = Z-Score к MA90)")

df_chart = df.tail(500).copy()

def get_color(z, lower, upper):
    if   z <= lower:    return "#00ff66"
    elif z <= lower*0.5: return "#39ff14"
    elif z <= -0.5:     return "#bfff00"
    elif z <= 0.5:      return "#e5e7eb"
    elif z <= 1.2:      return "#ffb703"
    elif z <= upper:    return "#ff5500"
    else:               return "#ff0055"

fig = go.Figure()

for i in range(len(df_chart) - 1):
    color = get_color(df_chart["z_score"].iloc[i], lower_thr, upper_thr)
    fig.add_trace(go.Scatter(
        x=[df_chart["date"].iloc[i], df_chart["date"].iloc[i+1]],
        y=[df_chart["close"].iloc[i], df_chart["close"].iloc[i+1]],
        mode="lines", line=dict(color=color, width=3.5),
        showlegend=False, hoverinfo="skip"
    ))

if "ma90" in df_chart.columns:
    fig.add_trace(go.Scatter(
        x=df_chart["date"], y=df_chart["ma90"],
        mode="lines", name="MA90",
        line=dict(color="#ffffff", width=1.2, dash="dot"),
        opacity=0.5
    ))

if "ma200" in df_chart.columns:
    fig.add_trace(go.Scatter(
        x=df_chart["date"], y=df_chart["ma200"],
        mode="lines", name="MA200",
        line=dict(color="#f59e0b", width=1.5, dash="dash"),
        opacity=0.7
    ))

# Добавление флага дивергенции в интерактивный hover текста
hover_texts = []
for d, p, z, r, v in zip(df_chart["date"], df_chart["close"], df_chart["z_score"], df_chart["rsi"], df_chart["vol_z"]):
    formatted_price = f"{p:,.4f}" if p < 1 else f"{p:,.2f}"
    text_item = (
        f"📅 <b>{d.strftime('%Y-%m-%d')}</b><br>"
        f"💰 <b>${formatted_price}</b><br>"
        f"📊 Z(MA90): <b>{z:.2f}</b><br>"
        f"📈 RSI: <b>{r:.1f}</b><br>"
        f"📦 Объём σ: <b>{v:.2f}</b>"
    )
    hover_texts.append(text_item)

fig.add_trace(go.Scatter(
    x=df_chart["date"], y=df_chart["close"],
    mode="markers", marker=dict(color="rgba(0,0,0,0)", size=1),
    hoverinfo="text",
    text=hover_texts,
    name="Инфо", hovertemplate="%{text}<extra></extra>"
))

price_range = df_chart["close"].max() / (df_chart["close"].min() + 1e-10)
use_log     = price_range > 5

fig.update_layout(
    height=480, template="plotly_dark",
    xaxis_title="", yaxis_title="Цена (USD)",
    yaxis_type="log" if use_log else "linear",
    hovermode="x unified",
    legend=dict(orientation="h", y=1.02, x=0),
    font=dict(family="Times New Roman, Times, serif", size=13)
)
st.plotly_chart(fig, use_container_width=True)

# График Z-Score
st.subheader("📉 Z-SCORE (цена к MA90) + АДАПТИВНЫЕ ПОРОГИ")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=df_chart["date"], y=df_chart["z_score"],
    mode="lines", name="Z-Score", line=dict(color="#00d4ff", width=2.5),
    fill="tozeroy", fillcolor="rgba(0,212,255,0.12)",
    text=[f"📅 {d.strftime('%Y-%m-%d')}<br>📊 Z-Score: {z:.2f}"
          for d, z in zip(df_chart["date"], df_chart["z_score"])],
    hovertemplate="%{text}<extra></extra>"
))
fig2.add_hline(y=lower_thr, line_dash="dash", line_color="#22ff55", line_width=2,
               annotation_text=f"ПОКУПКА ({lower_thr:.2f}σ)", annotation_position="right")
fig2.add_hline(y=upper_thr, line_dash="dash", line_color="#ff4422", line_width=2,
               annotation_text=f"ПРОДАЖА ({upper_thr:.2f}σ)", annotation_position="right")
fig2.add_hline(y=0, line_dash="dot", line_color="#888888")
fig2.update_layout(
    height=280, template="plotly_dark", yaxis_range=[-4, 4],
    font=dict(family="Times New Roman, Times, serif", size=13)
)
st.plotly_chart(fig2, use_container_width=True)

# График объёма
if "volume" in df_chart.columns and df_chart["volume"].sum() > 0:
    st.subheader("📦 ОБЪЁМ + АНОМАЛИИ (σ)")
    vol_colors = ["#ef4444" if v >= 2 else "#eab308" if v >= 1 else "#374151"
                  for v in df_chart["vol_z"]]
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=df_chart["date"], y=df_chart["volume"],
        marker_color=vol_colors, name="Объём",
        text=[f"📅 {d.strftime('%Y-%m-%d')}<br>📦 Объём σ: {v:.2f}"
              for d, v in zip(df_chart["date"], df_chart["vol_z"])],
        hovertemplate="%{text}<extra></extra>"
    ))
    fig3.update_layout(
        height=220, template="plotly_dark",
        yaxis_title="Объём (USD)", xaxis_title="",
        font=dict(family="Times New Roman, Times, serif", size=13)
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("🔴 Красный = объём > 2σ (капитуляция) | 🟡 Жёлтый = объём > 1σ (повышен)")

# ============================================================
# 12. СВОДНАЯ ТАБЛИЦА ВСЕХ АКТИВОВ (Добавлена колонка Дивергенции)
# ============================================================

st.markdown("---")
st.subheader("📋 СВОДНАЯ ТАБЛИЦА ВСЕХ АКТИВОВ")

@st.cache_data(ttl=900)
def build_summary_table():
    all_assets = {**{c: "Криптовалюта" for c in CRYPTO_LIST},
                  **{s: "Акция"        for s in STOCK_LIST}}
    rows = []
    for symbol, atype in all_assets.items():
        df_t = load_crypto_data(symbol) if atype == "Криптовалюта" else load_stock_data(symbol)
        if df_t is None or len(df_t) < 30:
            continue
        res = calculate_metrics_adaptive(df_t)
        if res[0] is None:
            continue
        (_, price, z, prob, conf, (lt, ut), rsi_v, vol_z, _, div_detected) = res
        sig, _ = get_signal_adaptive(z, lt, ut, symbol in VETERAN_LIST)
        rows.append({
            "Символ":          symbol,
            "Тип":             atype,
            "Цена":            f"${price:,.4f}" if price < 1 else f"${price:,.2f}",
            "Z-Score(MA90)":   f"{z:.2f}",
            "RSI (14)":        f"{rsi_v:.1f}",
            "Дивергенция":     "✅ Да" if div_detected else "❌ Нет",
            "Объём σ":         f"{vol_z:.2f}",
            "Вер-ть дна":      f"{prob*100:.1f}%",
            "Сигнал":          sig.split("—")[0].strip(),
        })
    return rows

with st.spinner("Загрузка сводной таблицы..."):
    summary = build_summary_table()

if summary:
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

# ============================================================
# 13. ПОДВАЛ
# ============================================================

moscow_tz   = timezone(timedelta(hours=3))
moscow_time = datetime.now(moscow_tz)

st.markdown("---")
st.caption(f"📅 Обновлено: {moscow_time.strftime('%Y-%m-%d %H:%M:%S')} (МСК)")
st.caption("📡 Источник: CryptoCompare → yfinance (fallback) / CoinGecko | 🤖 AI: DeepSeek")
st.caption("⚡ Z-Score(MA90) + RSI Divergence Matrix + Объём(σ) + MA200-контекст | ⚠️ Не инвестиционная рекомендация")
