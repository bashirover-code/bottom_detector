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

st.set_page_config(page_title="Детектор дна активов v3", layout="wide")

st.markdown("""
    <meta http-equiv="refresh" content="300">
    <style>
        html, body, [class*="css"], .stMarkdown, .stMetric, .stDataFrame,
        .stButton, .stSelectbox, .stRadio, .stCaption, h1, h2, h3, h4, p, div {
            font-family: 'Times New Roman', Times, serif !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Детектор дна активов v3")

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

# Глобальные даты дна для "памяти о восстановлении"
BOTTOM_DATES = {
    "bottom1": datetime(2025, 11, 10),
    "bottom2": datetime(2026, 2, 6),
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
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            d  = r.json()
            md = d.get("market_data", {})
            cd = d.get("community_data", {})
            dd = d.get("developer_data", {})
            return {
                "price_usd":               md.get("current_price", {}).get("usd", 0),
                "market_cap":              md.get("market_cap", {}).get("usd", 0),
                "fully_diluted_valuation": md.get("fully_diluted_valuation", {}).get("usd", 0),
                "total_volume":            md.get("total_volume", {}).get("usd", 0),
                "price_change_24h":        md.get("price_change_percentage_24h", 0),
                "ath_usd":                 md.get("ath", {}).get("usd", 0),
                "atl_usd":                 md.get("atl", {}).get("usd", 0),
                "twitter_followers":       cd.get("twitter_followers", 0),
                "github_stars":            dd.get("stars", 0),
                "github_forks":            dd.get("forks", 0),
            }
    except Exception:
        pass
    return None

# ============================================================
# 3. ИНДИКАТОРЫ
# ============================================================

def calculate_rsi(series, periods=14):
    delta   = series.diff()
    up      = delta.clip(lower=0)
    down    = -1 * delta.clip(upper=0)
    ma_up   = up.ewm(com=periods - 1, adjust=False).mean()
    ma_down = down.ewm(com=periods - 1, adjust=False).mean()
    rs      = ma_up / (ma_down + 1e-10)
    return 100 - (100 / (1 + rs))


def detect_rsi_divergence(df, lookback=20):
    """
    Бычья дивергенция RSI:
    Цена обновила локальный минимум, а RSI — нет.
    Возвращает: True/False + описание.

    Алгоритм:
    1. Берём последние lookback свечей.
    2. Ищем два минимума цены (текущий и предыдущий).
    3. Сравниваем соответствующие значения RSI.
    4. Дивергенция = цена[min2] < цена[min1], RSI[min2] > RSI[min1].
    """
    if len(df) < lookback + 5:
        return False, "недостаточно данных"

    window = df.tail(lookback).copy().reset_index(drop=True)
    prices = window["close"].values
    rsis   = window["rsi"].values

    # Локальные минимумы (простой детектор: точка ниже соседей)
    local_mins = []
    for i in range(1, len(prices) - 1):
        if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
            local_mins.append(i)

    if len(local_mins) < 2:
        return False, "нет двух локальных минимумов"

    # Два последних минимума
    idx1, idx2 = local_mins[-2], local_mins[-1]
    p1, p2     = prices[idx1], prices[idx2]
    r1, r2     = rsis[idx1],   rsis[idx2]

    # Бычья дивергенция: цена ниже, RSI выше
    if p2 < p1 and r2 > r1:
        diff_price = (p1 - p2) / p1 * 100
        diff_rsi   = r2 - r1
        desc = (f"цена ↓{diff_price:.1f}%, RSI ↑{diff_rsi:.1f} п. "
                f"(цена: {p1:.4f}→{p2:.4f}, RSI: {r1:.1f}→{r2:.1f})")
        return True, desc

    # Медвежья дивергенция (для информации): цена выше, RSI ниже
    if p2 > p1 and r2 < r1:
        return False, f"медвежья дивергенция (цена ↑, RSI ↓)"

    return False, "дивергенции нет"


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
        if   z_score <= lower_thr:     return "🔴 ПОКУПКА (АДАПТИВ)",  "#ef4444"
        elif z_score <= attention_thr: return "🟡 ЗОНА ВНИМАНИЯ",      "#eab308"
        elif z_score >= upper_thr:     return "🟢 ПРОДАЖА (АДАПТИВ)",  "#22c55e"
        else:                          return "⚪ НЕЙТРАЛЬНО",          "#6b7280"

# ============================================================
# 4. ЗАГРУЗКА ДАННЫХ
# ============================================================

@st.cache_data(ttl=600)
def load_crypto_data(symbol, days=500):
    try:
        if "CRYPTOCOMPARE_KEY" in st.secrets:
            API_KEY = st.secrets["CRYPTOCOMPARE_KEY"]
            url     = "https://min-api.cryptocompare.com/data/v2/histoday"
            params  = {"fsym": symbol, "tsym": "USD", "limit": days, "api_key": API_KEY}
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
        end  = datetime.now()
        start = end - timedelta(days=days)
        ydf  = yf.Ticker(symbol).history(start=start, end=end)
        if ydf is not None and not ydf.empty:
            ydf = ydf.reset_index()
            ydf = ydf.rename(columns={"Date": "date", "Close": "close", "Volume": "volume"})
            ydf["date"] = pd.to_datetime(ydf["date"]).dt.tz_localize(None)
            return ydf[["date", "close", "volume"]].sort_values("date").reset_index(drop=True)
    except Exception:
        pass
    return None

# ============================================================
# 5. РАСЧЁТ МЕТРИК
# ============================================================

def calculate_metrics_adaptive(df):
    """
    Детектор дна v3.

    Z-Score от цены к MA90.

    Вероятность дна — составная с АДАПТИВНЫМИ ВЕСАМИ (п.2):
      Бычий рынок  (цена > MA200): Z35 + RSI25 + Vol25 + MA15 = 100
      Медвежий рынок (цена < MA200): Z30 + RSI20 + Vol40 + MA0  = 90 → нормируем к 100
        (в медвежьем рынке объём капитуляции важнее RSI и MA200)

    + RSI-дивергенция (п.4): бонус +10 баллов сверх формулы (cap 100).
    """
    if df is None or len(df) < 30:
        return None, None, None, None, None, None, None, None, None, None, None

    df = df.copy()
    df["ma90"]  = df["close"].rolling(90,  min_periods=20).mean()
    df["ma200"] = df["close"].rolling(200, min_periods=50).mean()

    rolling_std90 = df["close"].rolling(90, min_periods=20).std()
    df["z_score"] = (df["close"] - df["ma90"]) / (rolling_std90 + 1e-10)
    df["rsi"]     = calculate_rsi(df["close"], 14)

    if "volume" in df.columns and df["volume"].sum() > 0:
        vol_mean    = df["volume"].rolling(30, min_periods=10).mean()
        vol_std     = df["volume"].rolling(30, min_periods=10).std()
        df["vol_z"] = (df["volume"] - vol_mean) / (vol_std + 1e-10)
    else:
        df["vol_z"] = 0.0

    df = df.fillna(0)

    z_scores             = df["z_score"].values
    lower_thr, upper_thr = get_adaptive_thresholds(z_scores)

    current_price = df["close"].iloc[-1]
    current_z     = df["z_score"].iloc[-1]
    current_rsi   = df["rsi"].iloc[-1]
    current_vol_z = df["vol_z"].iloc[-1]
    current_ma200 = df["ma200"].iloc[-1]
    current_ma90  = df["ma90"].iloc[-1]

    # ── Контекст тренда ──────────────────────────────────────
    is_bull = (current_ma200 > 0 and current_price > current_ma200)

    # ── Компонент Z-Score ────────────────────────────────────
    w_z = 35 if is_bull else 30
    if current_z <= lower_thr:
        s_z = min(w_z, (w_z - 15) + abs(current_z - lower_thr) * 10)
    elif current_z <= lower_thr * 0.5:
        s_z = 10
    else:
        s_z = 0

    # ── Компонент RSI ────────────────────────────────────────
    w_rsi = 25 if is_bull else 20
    if   current_rsi <= 20: s_rsi = w_rsi
    elif current_rsi <= 30: s_rsi = int(w_rsi * 0.8)
    elif current_rsi <= 40: s_rsi = int(w_rsi * 0.4)
    else:                   s_rsi = 0

    # ── Компонент объём ──────────────────────────────────────
    # В медвежьем рынке вес объёмной капитуляции выше (40 vs 25)
    w_vol = 25 if is_bull else 40
    if   current_vol_z >= 3.0: s_vol = w_vol
    elif current_vol_z >= 2.0: s_vol = int(w_vol * 0.75)
    elif current_vol_z >= 1.5: s_vol = int(w_vol * 0.45)
    elif current_vol_z >= 1.0: s_vol = int(w_vol * 0.2)
    else:                      s_vol = 0

    # ── Компонент MA200-контекст ─────────────────────────────
    # В медвежьем рынке — 0 (нет смысла начислять)
    s_trend = 15 if is_bull else 0

    # ── Суммирование и нормировка ────────────────────────────
    max_score = 100 if is_bull else 90
    raw_score = s_z + s_rsi + s_vol + s_trend

    # ── Бонус: RSI-дивергенция (п.4) ────────────────────────
    div_bull, div_desc = detect_rsi_divergence(df)
    div_bonus = 10 if div_bull else 0

    total_score  = min(100, raw_score / max_score * 100 + div_bonus)
    current_prob = total_score / 100.0
    confidence   = min(100, len(df) / 365 * 100)

    # Сохраняем компоненты для декомпозиции
    df.attrs["score_components"] = {
        "s_z": s_z, "s_rsi": s_rsi, "s_vol": s_vol,
        "s_trend": s_trend, "div_bonus": div_bonus,
        "max_score": max_score, "is_bull": is_bull,
        "div_desc": div_desc, "div_bull": div_bull,
        "w_z": w_z, "w_rsi": w_rsi, "w_vol": w_vol,
    }

    return (df, current_price, current_z, current_prob,
            confidence, (lower_thr, upper_thr), current_rsi,
            current_vol_z, current_ma90, div_bull, div_desc)

# ============================================================
# 6. ПАМЯТЬ О ВОССТАНОВЛЕНИИ (п.3)
# ============================================================

def get_recovery_status(df, bottom_date, current_price):
    """
    Находит цену актива на дату глобального дна (±3 дня допуска).
    Возвращает: цену на дне, % восстановления, статус.
    """
    if df is None or len(df) < 5:
        return None, None, "нет данных"

    df_dt = df.copy()
    df_dt["date"] = pd.to_datetime(df_dt["date"])
    window = df_dt[
        (df_dt["date"] >= bottom_date - timedelta(days=3)) &
        (df_dt["date"] <= bottom_date + timedelta(days=3))
    ]
    if window.empty:
        return None, None, "вне истории"

    bottom_price = window["close"].min()
    if bottom_price <= 0:
        return None, None, "нет данных"

    recovery_pct = (current_price - bottom_price) / bottom_price * 100

    if current_price >= bottom_price:
        if recovery_pct >= 50:
            status = "✅ Восстановился"
        else:
            status = "⚠️ Частично"
    else:
        status = "❌ Не восстановился"

    return bottom_price, recovery_pct, status

# ============================================================
# 7. DeepSeek AI-АНАЛИЗ
# ============================================================

def call_deepseek_analysis(symbol, current_price, current_z, current_prob,
                            signal_text, current_rsi, current_vol_z,
                            current_ma90, confidence, fundamentals,
                            div_bull, div_desc, is_bull,
                            rec1_pct, rec2_pct):
    deepseek_key = st.secrets.get("DEEPSEEK_API_KEY", "")
    if not deepseek_key:
        return "❌ API ключ DeepSeek не найден."

    asset_type   = "криптовалюта" if symbol in CRYPTO_LIST else "акция"
    market_ctx   = "БЫЧИЙ (цена выше MA200)" if is_bull else "МЕДВЕЖИЙ (цена ниже MA200)"
    div_text     = f"ЕСТЬ бычья дивергенция RSI: {div_desc}" if div_bull else f"Нет: {div_desc}"

    rec_text = ""
    if rec1_pct is not None:
        rec_text += f"- Восстановление от дна 10.11.2025: {rec1_pct:+.1f}%\n"
    if rec2_pct is not None:
        rec_text += f"- Восстановление от дна 06.02.2026: {rec2_pct:+.1f}%\n"

    fund_text = ""
    if fundamentals:
        fund_text = f"""
ФУНДАМЕНТАЛ (CoinGecko):
- Рыночная кап.: ${fundamentals.get('market_cap', 0):,.0f}
- FDV: ${fundamentals.get('fully_diluted_valuation', 0):,.0f}
- Объём 24ч: ${fundamentals.get('total_volume', 0):,.0f}
- Изм. 24ч: {fundamentals.get('price_change_24h', 0):.1f}%
- ATH: ${fundamentals.get('ath_usd', 0):,.2f}
"""

    vol_comment = ("аномально высокий — возможная капитуляция" if current_vol_z >= 2
                   else "повышенный" if current_vol_z >= 1 else "нормальный")

    prompt = f"""Проанализируй {symbol} ({asset_type}):

ТЕХНИЧЕСКИЕ ДАННЫЕ:
- Цена: ${current_price:,.4f}
- Z-Score (цена к MA90): {current_z:.2f}
- RSI (14): {current_rsi:.1f}
- Аномалия объёма: {current_vol_z:.2f}σ — {vol_comment}
- MA90: ${current_ma90:,.4f}
- Контекст тренда: {market_ctx}
- RSI-дивергенция: {div_text}
- Составная вероятность дна: {current_prob*100:.1f}% (уверенность: {confidence:.0f}%)
- Сигнал: {signal_text}

ПАМЯТЬ О ВОССТАНОВЛЕНИИ (глобальные дна рынка):
{rec_text if rec_text else "- данные недоступны"}
{fund_text}

Методология вероятности дна:
  Бычий рынок: Z(35%) + RSI(25%) + Объём(25%) + MA200(15%)
  Медвежий рынок: Z(30%) + RSI(20%) + Объём(40%) — повышен вес капитуляции
  Бонус +10% при бычьей дивергенции RSI

Напиши КРАТКИЙ анализ (5-6 предложений) на русском:
1. Что означает текущий сигнал с учётом контекста рынка
2. RSI-дивергенция: есть ли подтверждение разворота
3. Восстановление от исторических дён: актив выкупили или нет
4. Рекомендация (покупать/ждать/продавать)
5. Главный фактор риска"""

    url     = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {deepseek_key}", "Content-Type": "application/json"}
    payload = {
        "model":       "deepseek-chat",
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  750,
        "temperature": 0.7,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        return f"❌ Ошибка API: {r.status_code}"
    except Exception as e:
        return f"❌ Ошибка: {str(e)[:100]}"

# ============================================================
# 8. САЙДБАР
# ============================================================

with st.sidebar:
    st.header("⚙️ Настройки")
    st.markdown("---")
    asset_type_sel = st.radio("Тип актива", ["Криптовалюты", "Акции"])
    if asset_type_sel == "Криптовалюты":
        selected_asset = st.selectbox("Криптовалюта", CRYPTO_LIST)
    else:
        selected_asset = st.selectbox("Акция", STOCK_LIST)
    st.markdown("---")
    st.caption("🔢 Z-Score: цена vs MA90")
    st.caption("📈 RSI 14 + дивергенция")
    st.caption("📦 Аномалия объёма (капитуляция)")
    st.caption("📉 MA200-контекст (адаптивные веса)")
    st.caption("🔁 Память восстановления: 10.11.25 / 06.02.26")
    st.caption("🕐 Обновление каждые 5 минут")
    st.caption(f"📋 Всего активов: {len(CRYPTO_LIST) + len(STOCK_LIST)}")

# ============================================================
# 9. ЗАГРУЗКА И РАСЧЁТ
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
 current_vol_z, current_ma90, div_bull, div_desc) = result

sc         = df.attrs.get("score_components", {})
is_bull    = sc.get("is_bull", True)
signal_text, signal_color = get_signal_adaptive(current_z, lower_thr, upper_thr, is_veteran)

# Память о восстановлении
bp1, rec1_pct, rec1_status = get_recovery_status(df, BOTTOM_DATES["bottom1"], current_price)
bp2, rec2_pct, rec2_status = get_recovery_status(df, BOTTOM_DATES["bottom2"], current_price)

# ============================================================
# 10. ПАНЕЛЬ МЕТРИК
# ============================================================

st.header(f"{selected_asset} — {'криптовалюта' if is_crypto else 'акция'}")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    fmt = f"${current_price:,.4f}" if current_price < 1 else f"${current_price:,.2f}"
    st.metric("💰 ЦЕНА", fmt)

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
    rsi_color = "#ef4444" if current_rsi <= 30 else "#22c55e" if current_rsi >= 70 else "#00d4ff"
    div_tag   = " 📐" if div_bull else ""
    st.markdown(f"""
        <div style='background:{rsi_color}15;padding:11px;border-radius:8px;
                    border:1px solid {rsi_color}40;text-align:center;'>
            <p style='color:gray;margin:0;font-size:13px;font-weight:bold;'>RSI (14){div_tag}</p>
            <p style='color:{rsi_color};font-size:22px;font-weight:bold;margin:5px 0 0 0;'>{current_rsi:.1f}</p>
        </div>""", unsafe_allow_html=True)

with col5:
    vol_color = "#ef4444" if current_vol_z >= 2 else "#eab308" if current_vol_z >= 1 else "#6b7280"
    vol_label = "КАПИТУЛЯЦИЯ" if current_vol_z >= 2 else "ПОВЫШЕН" if current_vol_z >= 1 else "НОРМА"
    st.markdown(f"""
        <div style='background:{vol_color}15;padding:11px;border-radius:8px;
                    border:1px solid {vol_color}40;text-align:center;'>
            <p style='color:gray;margin:0;font-size:13px;font-weight:bold;'>ОБЪЁМ ({vol_label})</p>
            <p style='color:{vol_color};font-size:22px;font-weight:bold;margin:5px 0 0 0;'>{current_vol_z:.2f}σ</p>
        </div>""", unsafe_allow_html=True)

# Строка статуса
market_label = "🟢 БЫЧИЙ (выше MA200)" if is_bull else "🔴 МЕДВЕЖИЙ (ниже MA200)"
st.markdown(f"""
<div style='background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
            padding:15px;border-radius:12px;margin:20px 0;text-align:center;
            border-left:5px solid {signal_color};'>
    <p style='color:#9ca3af;margin:0;font-size:15px;'>
        <b>Статус:</b> {signal_text} &nbsp;|&nbsp;
        Рынок: {market_label} &nbsp;|&nbsp;
        Пороги Z: покупка &lt; {lower_thr:.2f}σ &nbsp;|&nbsp; продажа &gt; {upper_thr:.2f}σ
    </p>
</div>
""", unsafe_allow_html=True)

# ── RSI-дивергенция ──────────────────────────────────────────
if div_bull:
    st.success(f"📐 Бычья дивергенция RSI: {div_desc} → +10 б. к вероятности дна")
else:
    st.caption(f"📐 RSI-дивергенция: {div_desc}")

# ── Память о восстановлении ──────────────────────────────────
st.markdown("---")
st.subheader("🔁 Память о восстановлении от глобальных дён")

rc1, rc2 = st.columns(2)

with rc1:
    st.markdown("**Дно 10.11.2025**")
    if bp1 is not None:
        color1 = "#22c55e" if "✅" in rec1_status else "#eab308" if "⚠️" in rec1_status else "#ef4444"
        pct_str = f"{rec1_pct:+.1f}%" if rec1_pct is not None else "—"
        st.markdown(f"""
        <div style='background:{color1}15;padding:12px;border-radius:8px;border:1px solid {color1}40;'>
            <p style='color:{color1};font-size:18px;font-weight:bold;margin:0;'>{rec1_status}</p>
            <p style='color:#9ca3af;margin:4px 0 0 0;font-size:13px;'>
                Цена дна: ${bp1:,.4f if bp1 < 1 else bp1:,.2f} &nbsp;|&nbsp; Изм.: <b style="color:{color1}">{pct_str}</b>
            </p>
        </div>""", unsafe_allow_html=True)
    else:
        st.caption(f"Статус: {rec1_status}")

with rc2:
    st.markdown("**Дно 06.02.2026**")
    if bp2 is not None:
        color2 = "#22c55e" if "✅" in rec2_status else "#eab308" if "⚠️" in rec2_status else "#ef4444"
        pct_str2 = f"{rec2_pct:+.1f}%" if rec2_pct is not None else "—"
        st.markdown(f"""
        <div style='background:{color2}15;padding:12px;border-radius:8px;border:1px solid {color2}40;'>
            <p style='color:{color2};font-size:18px;font-weight:bold;margin:0;'>{rec2_status}</p>
            <p style='color:#9ca3af;margin:4px 0 0 0;font-size:13px;'>
                Цена дна: ${bp2:,.4f if bp2 < 1 else bp2:,.2f} &nbsp;|&nbsp; Изм.: <b style="color:{color2}">{pct_str2}</b>
            </p>
        </div>""", unsafe_allow_html=True)
    else:
        st.caption(f"Статус: {rec2_status}")

st.caption("❌ Не восстановился = кандидат на выход из портфеля | ✅ Восстановился = прошёл тест выкупаемости")

# ── Декомпозиция вероятности дна ─────────────────────────────
with st.expander("🔍 Декомпозиция вероятности дна"):
    dc1, dc2, dc3, dc4, dc5 = st.columns(5)
    dc1.metric(f"Z-Score (макс {sc.get('w_z',35)})",   f"{sc.get('s_z',0):.0f} б.")
    dc2.metric(f"RSI (макс {sc.get('w_rsi',25)})",     f"{sc.get('s_rsi',0):.0f} б.")
    dc3.metric(f"Объём (макс {sc.get('w_vol',25)})",   f"{sc.get('s_vol',0):.0f} б.")
    dc4.metric("MA200-контекст (макс 15)",              f"{sc.get('s_trend',0):.0f} б.")
    dc5.metric("Дивергенция RSI (бонус)",               f"{sc.get('div_bonus',0):.0f} б.")
    st.caption(
        f"Режим: {'🟢 бычий' if is_bull else '🔴 медвежий'} | "
        f"Сумма до нормировки: {sc.get('s_z',0)+sc.get('s_rsi',0)+sc.get('s_vol',0)+sc.get('s_trend',0):.0f} "
        f"из {sc.get('max_score',100)} | "
        f"MA200 = ${df['ma200'].iloc[-1]:,.2f}"
    )

# ============================================================
# 11. AI-АНАЛИЗ
# ============================================================

st.markdown("---")
st.subheader("🤖 AI-анализ актива")

if st.button(f"📊 Получить AI-анализ для {selected_asset}", type="primary"):
    with st.spinner("🧠 DeepSeek анализирует..."):
        analysis = call_deepseek_analysis(
            selected_asset, current_price, current_z, current_prob,
            signal_text, current_rsi, current_vol_z, current_ma90,
            confidence, fundamentals, div_bull, div_desc, is_bull,
            rec1_pct, rec2_pct
        )
    analysis_html = analysis.replace('\n', '<br>').replace('•', '&bull;')
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
                padding:20px;border-radius:16px;margin:10px 0;border:1px solid #2a2a3e;'>
        <h4 style='margin-bottom:10px;color:#ffffff;'>📈 Анализ от DeepSeek AI</h4>
        <div style='color:#ffffff;font-size:15px;line-height:1.6;'>{analysis_html}</div>
        <p style='color:#888888;font-size:12px;margin-top:10px;'>
            ⚡ DeepSeek Chat | Z-Score(MA90) + RSI + Дивергенция + Объём + MA200 + Восстановление
        </p>
    </div>""", unsafe_allow_html=True)

# ============================================================
# 12. ГРАФИКИ
# ============================================================

st.markdown("---")
st.subheader("📈 ГРАФИК ЦЕНЫ (цвет = Z-Score к MA90)")

df_chart = df.tail(500).copy()

def get_color(z, lower, upper):
    if   z <= lower:      return "#00ff66"
    elif z <= lower*0.5:  return "#39ff14"
    elif z <= -0.5:       return "#bfff00"
    elif z <= 0.5:        return "#e5e7eb"
    elif z <= 1.2:        return "#ffb703"
    elif z <= upper:      return "#ff5500"
    else:                 return "#ff0055"

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
        line=dict(color="#ffffff", width=1.2, dash="dot"), opacity=0.5
    ))

if "ma200" in df_chart.columns:
    fig.add_trace(go.Scatter(
        x=df_chart["date"], y=df_chart["ma200"],
        mode="lines", name="MA200",
        line=dict(color="#f59e0b", width=1.5, dash="dash"), opacity=0.7
    ))

# Вертикальные маркеры глобальных дён
for label, bdate in BOTTOM_DATES.items():
    lbl_text = "🔴 Дно 10.11.25" if "1" in label else "🔴 Дно 06.02.26"
    fig.add_vline(x=bdate, line_dash="dot", line_color="#ff6b6b",
                  line_width=1.5,
                  annotation_text=lbl_text,
                  annotation_position="top left",
                  annotation_font_color="#ff6b6b",
                  annotation_font_size=11)

fig.add_trace(go.Scatter(
    x=df_chart["date"], y=df_chart["close"],
    mode="markers", marker=dict(color="rgba(0,0,0,0)", size=1),
    hoverinfo="text",
    text=[f"📅 <b>{d.strftime('%Y-%m-%d')}</b><br>"
          f"💰 <b>${p:,.4f if p < 1 else p:,.2f}</b><br>"
          f"📊 Z(MA90): <b>{z:.2f}</b><br>"
          f"📈 RSI: <b>{r:.1f}</b><br>"
          f"📦 Vol σ: <b>{v:.2f}</b>"
          for d, p, z, r, v in zip(
              df_chart["date"], df_chart["close"],
              df_chart["z_score"], df_chart["rsi"], df_chart["vol_z"])],
    name="Инфо", hovertemplate="%{text}<extra></extra>"
))

price_range = df_chart["close"].max() / (df_chart["close"].min() + 1e-10)
fig.update_layout(
    height=490, template="plotly_dark",
    xaxis_title="", yaxis_title="Цена (USD)",
    yaxis_type="log" if price_range > 5 else "linear",
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
for label, bdate in BOTTOM_DATES.items():
    fig2.add_vline(x=bdate, line_dash="dot", line_color="#ff6b6b", line_width=1.5)
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
        text=[f"📅 {d.strftime('%Y-%m-%d')}<br>📦 Vol σ: {v:.2f}"
              for d, v in zip(df_chart["date"], df_chart["vol_z"])],
        hovertemplate="%{text}<extra></extra>"
    ))
    for label, bdate in BOTTOM_DATES.items():
        fig3.add_vline(x=bdate, line_dash="dot", line_color="#ff6b6b", line_width=1.5)
    fig3.update_layout(
        height=220, template="plotly_dark",
        yaxis_title="Объём (USD)", xaxis_title="",
        font=dict(family="Times New Roman, Times, serif", size=13)
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("🔴 Красный = объём > 2σ (капитуляция) | 🟡 Жёлтый = объём > 1σ | Пунктир = даты глобальных дён")

# ============================================================
# 13. СВОДНАЯ ТАБЛИЦА
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
        (df_r, price, z, prob, conf, (lt, ut), rsi_v, vol_z, _, dv_bull, _) = res
        sig, _ = get_signal_adaptive(z, lt, ut, symbol in VETERAN_LIST)

        _, r1p, r1s = get_recovery_status(df_r, BOTTOM_DATES["bottom1"], price)
        _, r2p, r2s = get_recovery_status(df_r, BOTTOM_DATES["bottom2"], price)

        rows.append({
            "Символ":          symbol,
            "Тип":             atype,
            "Цена":            f"${price:,.4f}" if price < 1 else f"${price:,.2f}",
            "Z-Score":         f"{z:.2f}",
            "RSI":             f"{rsi_v:.1f}",
            "Div RSI":         "✅" if dv_bull else "—",
            "Объём σ":         f"{vol_z:.2f}",
            "Вер-ть дна":      f"{prob*100:.1f}%",
            "Сигнал":          sig.split("—")[0],
            "Дно 10.11.25":    f"{r1s} ({r1p:+.0f}%)" if r1p is not None else r1s,
            "Дно 06.02.26":    f"{r2s} ({r2p:+.0f}%)" if r2p is not None else r2s,
        })
    return rows

with st.spinner("Загрузка сводной таблицы..."):
    summary = build_summary_table()

if summary:
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
    st.caption("❌ Не восстановился после дна → кандидат на выход | ✅ Восстановился → прошёл тест выкупаемости")

# ============================================================
# 14. ПОДВАЛ
# ============================================================

moscow_tz   = timezone(timedelta(hours=3))
moscow_time = datetime.now(moscow_tz)

st.markdown("---")
st.caption(f"📅 Обновлено: {moscow_time.strftime('%Y-%m-%d %H:%M:%S')} (МСК)")
st.caption("📡 CryptoCompare → yfinance (fallback) / CoinGecko | 🤖 DeepSeek AI")
st.caption("⚡ Z-Score(MA90) + RSI + Дивергенция + Объём(σ) + MA200(адаптив) + Память восстановления | ⚠️ Не инвестиционная рекомендация")
