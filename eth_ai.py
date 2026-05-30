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

st.set_page_config(page_title="Детектор дна активов v4.2", layout="wide")

st.markdown("""
    <meta http-equiv="refresh" content="300">
    <style>
        html, body, [class*="css"], .stMarkdown, .stMetric, .stDataFrame,
        .stButton, .stSelectbox, .stRadio, .stCaption, h1, h2, h3, h4, p, div {
            font-family: 'Times New Roman', Times, serif !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Детектор дна активов v4.2")

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
    "EWW", "BABA", "COIN", "NVDA", "SBER.ME", "MTSS.ME", "HEAD.ME"
]

VETERAN_LIST = ["BTC", "ETH", "LINK", "UNI", "AAPL", "MSFT", "NVDA", "TSLA"]

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
# 2. ДАННЫЕ О КРИТИЧЕСКИХ ДНЯХ (ИСПЫТАНИЯ РЫНКА)
# ============================================================

CRITICAL_DATES = {
    "test_1": {"date": "2025-11-10", "desc": "Тест 10.11.2025 (Тарифы Трампа, BTC $80K)"},
    "test_2": {"date": "2026-02-06", "desc": "Тест 06.02.2026 (Капитуляция, BTC $60K)"}
}

# ============================================================
# 3. ЗАГРУЗКА ДАННЫХ С РЕЗЕРВНЫМ ИСТОЧНИКОМ
# ============================================================

@st.cache_data(ttl=600)
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

@st.cache_data(ttl=600)
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

@st.cache_data(ttl=3600)
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
        res = requests.get(url, params=params, timeout=12)
        if res.status_code == 200:
            d = res.json()
            md = d.get("market_data", {})
            return {
                "market_cap": md.get("market_cap", {}).get("usd", 0),
                "fully_diluted_valuation": md.get("fully_diluted_valuation", {}).get("usd", 0),
                "volume_24h": md.get("total_volume", {}).get("usd", 0),
                "change_24h": md.get("price_change_percentage_24h", 0),
                "ath": md.get("ath", {}).get("usd", 0),
                "atl": md.get("atl", {}).get("usd", 0),
                "twitter": d.get("community_data", {}).get("twitter_followers", 0),
                "github": d.get("developer_data", {}).get("stars", 0)
            }
    except:
        pass
    return None

# ============================================================
# 4. РЕЖИМ РЫНКА (MARKET REGIME)
# ============================================================

@st.cache_data(ttl=600)
def get_market_regime():
    btc_df = load_crypto_data("BTC", days=300)
    if btc_df is not None and len(btc_df) >= 200:
        btc_df["ma200"] = btc_df["close"].rolling(window=200).mean()
        btc_price = btc_df["close"].iloc[-1]
        btc_ma200 = btc_df["ma200"].iloc[-1]
        if btc_price > btc_ma200:
            return "🟢 BULL"
        else:
            return "🔴 BEAR"
    return "⚪ NEUTRAL"

# ============================================================
# 5. ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ И ДИВЕРГЕНЦИИ
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
# 6. КОМПЛЕКСНЫЙ АДАПТИВНЫЙ РАСЧЁТ И МАТРИЧНЫЙ СКОРИНГ V4.2
# ============================================================

def calculate_metrics_adaptive(df, btc_df=None):
    if df is None or len(df) < 90:
        return (None,) * 12
        
    df = df.copy()
    
    df["ma90"] = df["close"].rolling(window=90, min_periods=30).mean()
    df["std90"] = df["close"].rolling(window=90, min_periods=30).std()
    df["z_score"] = (df["close"] - df["ma90"]) / (df["std90"] + 1e-10)
    
    df["rsi"] = calculate_rsi(df, 14)
    df["ma200"] = df["close"].rolling(window=200, min_periods=50).mean()
    
    df["v_mean"] = df["volume"].rolling(window=30, min_periods=10).mean()
    df["v_std"] = df["volume"].rolling(window=30, min_periods=10).std()
    df["vol_z"] = (df["volume"] - df["v_mean"]) / (df["v_std"] + 1e-10)
    
    # ИСПРАВЛЕНИЕ: Заполняем пустоты только в расчетных столбцах, не трогая datetime64[ns]
    calc_cols = ["ma90", "std90", "z_score", "rsi", "ma200", "v_mean", "v_std", "vol_z"]
    df[calc_cols] = df[calc_cols].bfill().ffill().fillna(0)
    
    current_price = df["close"].iloc[-1]
    c_z = df["z_score"].iloc[-1]
    c_rsi = df["rsi"].iloc[-1]
    c_vol_z = df["vol_z"].iloc[-1]
    c_ma200 = df["ma200"].iloc[-1]
    
    current_ath = df["close"].max()
    drawdown_pct = ((current_price - current_ath) / current_ath * 100) if current_ath > 0 else 0
    
    relative_strength = 0
    if btc_df is not None and len(btc_df) > 0:
        df_temp = df.copy()
        btc_temp = btc_df.copy()
        df_temp["d_norm"] = df_temp["date"].dt.date
        btc_temp["d_norm"] = btc_temp["date"].dt.date
        
        common_dates = np.intersect1d(df_temp['d_norm'], btc_temp['d_norm'])
        if len(common_dates) >= 30:
            df_sub = df_temp[df_temp['d_norm'].isin(common_dates)].sort_values("d_norm")
            btc_sub = btc_temp[btc_temp['d_norm'].isin(common_dates)].sort_values("d_norm")
            if len(df_sub) >= 30 and len(btc_sub) >= 30:
                asset_perf = (df_sub['close'].iloc[-1] / df_sub['close'].iloc[-30] - 1) * 100
                btc_perf = (btc_sub['close'].iloc[-1] / btc_sub['close'].iloc[-30] - 1) * 100
                relative_strength = asset_perf - btc_perf

    low_t, up_t = get_adaptive_thresholds(df["z_score"].values)
    dv_bull = detect_rsi_divergence(df, 35)
    
    bottom_score = 0
    if drawdown_pct <= -85: bottom_score += 25
    elif drawdown_pct <= -70: bottom_score += 20
    elif drawdown_pct <= -50: bottom_score += 10
    elif drawdown_pct <= -30: bottom_score += 5
    
    if c_z <= low_t: bottom_score += 20
    elif c_z < -0.8: bottom_score += 10
    
    if c_rsi <= 32: bottom_score += 10
    elif c_rsi <= 42: bottom_score += 5
    
    if dv_bull: bottom_score += 10
    
    if c_vol_z >= 2.0 and c_z < 0: bottom_score += 10
    elif c_vol_z >= 0.8 and c_z < 0: bottom_score += 5
    
    if current_price >= c_ma200 and c_ma200 > 0: bottom_score += 10
    
    if relative_strength > 5: bottom_score += 15
    elif relative_strength > -2: bottom_score += 7

    bottom_score = min(bottom_score, 100)

    return df, current_price, c_z, bottom_score, (low_t, up_t), c_rsi, c_vol_z, dv_bull, drawdown_pct, relative_strength, current_ath, c_ma200

# ============================================================
# 7. ИСТОРИЧЕСКИЙ АНАЛИЗ УСТОЙЧИВОСТИ
# ============================================================

def analyze_stress_tests(df):
    results = {"t1_status": "Нет данных", "t1_perf": None, "t2_status": "Нет данных", "t2_perf": None}
    if df is None or len(df) == 0:
        return results
        
    df_temp = df.copy()
    df_temp["date_str"] = df_temp["date"].dt.strftime("%Y-%m-%d")
    
    t1_row = df_temp[df_temp["date_str"] == CRITICAL_DATES["test_1"]["date"]]
    if not t1_row.empty:
        p_t1 = t1_row["close"].values[0]
        target_date_1 = pd.Timestamp(t1_row["date"].values[0]) + pd.Timedelta(days=25)
        future_t1 = df_temp[df_temp["date"] >= target_date_1].head(1)
        if not future_t1.empty:
            p_f1 = future_t1["close"].values[0]
            change = ((p_f1 - p_t1) / p_t1) * 100
            results["t1_perf"] = change
            results["t1_status"] = "✅ Прошёл (Выкуплен)" if change >= 15 else "❌ Не восстановился"
            
    t2_row = df_temp[df_temp["date_str"] == CRITICAL_DATES["test_2"]["date"]]
    if not t2_row.empty:
        p_t2 = t2_row["close"].values[0]
        target_date_2 = pd.Timestamp(t2_row["date"].values[0]) + pd.Timedelta(days=25)
        future_t2 = df_temp[df_temp["date"] >= target_date_2].head(1)
        if not future_t2.empty:
            p_f2 = future_t2["close"].values[0]
            change = ((p_f2 - p_t2) / p_t2) * 100
            results["t2_perf"] = change
            results["t2_status"] = "✅ Прошёл (Выкуплен)" if change >= 15 else "❌ Не восстановился"
            
    return results

# ============================================================
# 8. ИНТЕГРАЦИЯ DEEPSEEK V3 AI
# ============================================================

def call_deepseek_v3(asset, price, z, bottom_score, sig, rsi, vol_z, div, stress, fund, drawdown, regime, fdv_risk):
    key = st.secrets.get("DEEPSEEK_API_KEY", "")
    if not key:
        return "❌ Ключ интеграции ИИ DeepSeek отсутствует."
        
    f_text = f"Капитализация: ${fund.get('market_cap', 0):,.0f}, Риск токеномики (FDV Risk): {fdv_risk}." if (fund and isinstance(fund, dict) and fund.get('market_cap')) else "Фундаментальные ончейн-данные отсутствуют (традиционный актив)."
        
    prompt = f"""Проведи глубокий экспресс-анализ {asset}:
МЕТРИКИ v4.2: Цена: ${price:,.4f}, Z-Score: {z:.2f}, RSI: {rsi:.1f}, Историческая просадка: {drawdown:.1f}%, Объём: {vol_z:+.1f}σ, Бычий паттерн дивергенции: {'ДА' if div else 'НЕТ'}.
Итоговый Bottom Score системы (макс 100): {bottom_score} баллов. Сигнал детектора: {sig}.
Текущий макро-режим глобального рынка: {regime}. {f_text}

ИСТОРИЯ ПРОШЛЫХ СТРЕСС-ТЕСТОВ:
- Тест тарифов 2025: {stress.get('t1_status', 'Нет данных')} ({stress.get('t1_perf', 0) if stress.get('t1_perf') else 0:.1f}%)
- Тест капитуляции 2026: {stress.get('t2_status', 'Нет данных')} ({stress.get('t2_perf', 0) if stress.get('t2_perf') else 0:.1f}%)

Напиши профессиональный вывод (4-5 предложений) на русском. Оцени: является ли просадка фундаментально оправданной, защищен ли инвестор от скрытой инфляции предложения (FDV Risk) и конкретный торговый план в рамках текущего режима {regime}."""

    u = "https://api.deepseek.com/v1/chat/completions"
    h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    d = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 550, "temperature": 0.4
    }
    try:
        r = requests.post(u, headers=h, json=d, timeout=25)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Ошибка вызова ИИ: {e}"
    return "❌ Ошибка обработки ответа сервером DeepSeek."

# ============================================================
# 9. ПАНЕЛЬ УПРАВЛЕНИЯ И ПОЛУЧЕНИЕ ГЛОБАЛЬНОГО КОНТЕКСТА
# ============================================================

with st.sidebar:
    st.header("⚙️ НАСТРОЙКИ СИСТЕМЫ v4.2")
    st.markdown("---")
    market = st.radio("Сектор рынка", ["Криптовалюты", "Фондовый рынок"])
    if market == "Криптовалюты":
        asset = st.selectbox("Выбор цифрового актива", CRYPTO_LIST)
    else:
        asset = st.selectbox("Выбор акции/фонда", STOCK_LIST)
    st.markdown("---")
    st.caption("📈 **Математическая модель v4.2**")
    st.caption("Исправлен внутренний тип данных Pandas-fillna. Стабилизировано ядро скоринга исторических просадок.")

with st.spinner("Синхронизация глобальных индикаторов макро-режима..."):
    market_regime = get_market_regime()
    btc_global_df = load_crypto_data("BTC", days=550)

# ============================================================
# 10. ИНИЦИАЛИЗАЦИЯ И КОМПЛЕКСНЫЙ АНАЛИЗ ТЕКУЩЕГО АКТИВА
# ============================================================

is_c = asset in CRYPTO_LIST
is_v = asset in VETERAN_LIST

fund = None
fdv_risk = "N/A" if not is_c else "UNKNOWN"

if is_c and asset in COINGECKO_IDS:
    fund = get_coingecko_fundamentals(COINGECKO_IDS[asset])
    if fund and isinstance(fund, dict):
        mcap = fund.get("market_cap", 0)
        fdv = fund.get("fully_diluted_valuation", 0)
        fdv_ratio = fdv / mcap if mcap > 0 else 1
        if fdv_ratio < 1.5: fdv_risk = "🟢 LOW"
        elif fdv_ratio < 3.0: fdv_risk = "🟡 MEDIUM"
        else: fdv_risk = "🔴 HIGH"

with st.spinner(f"Загрузка потоков данных по {asset}..."):
    raw = load_crypto_data(asset) if is_c else load_stock_data(asset)

if raw is None or len(raw) < 90:
    st.error(f"❌ Недостаточно данных для запуска ядра математического анализа по {asset}.")
    st.stop()

df, c_price, c_z, bottom_score, (low_thr, upper_thr), c_rsi, c_vol_z, dv_bull, drawdown_pct, relative_strength, current_ath, c_ma200 = calculate_metrics_adaptive(raw, btc_global_df if is_c else None)
sig_t, sig_c = get_signal_adaptive(c_z, low_thr, upper_thr, is_v)
stress = analyze_stress_tests(df)

# ============================================================
# ВЕРХНИЙ ДАШБОРД V4.2
# ============================================================

st.header(f"📊 Паспорт актива: {asset}")
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("💰 ТЕКУЩАЯ ЦЕНА", f"${c_price:,.4f}" if c_price < 1 else f"${c_price:,.2f}")
with c2: 
    score_c = "#22c55e" if bottom_score >= 65 else "#eab308" if bottom_score >= 35 else "#ef4444"
    st.markdown(f"""
        <div style='background: {score_c}10; padding: 10px; border-radius: 8px; border: 1px solid {score_c}30; text-align: center;'>
            <p style='color: gray; margin:0; font-size:12px; font-weight:bold;'>BOTTOM SCORE</p>
            <p style='color: {score_c}; font-size:22px; font-weight:bold; margin:3px 0 0 0;'>{bottom_score} / 100</p>
        </div>
    """, unsafe_allow_html=True)
with c3: st.metric("📉 DRAWDOWN", f"{drawdown_pct:.1f}%")
with c4: st.metric("📈 RSI (14)", f"{c_rsi:.1f}")
with c5: st.metric("📦 ОБЪЁМ Z-SCORE", f"{c_vol_z:+.2f}σ")

# ============================================================
# СТРОКА СТАТУСА V4.2
# ============================================================

st.markdown(f"""
<div style='background: linear-gradient(135deg, #0b0f19 0%, #111827 100%); padding:14px; border-radius:10px; margin: 15px 0; border: 1px solid #1f2937;'>
    <p style='margin:0; color:#f3f4f6; font-size:14px; text-align: center;'>
        <b>Market Regime:</b> <span style='font-weight:bold;'>{market_regime}</span> &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>FDV Risk:</b> <span style='font-weight:bold;'>{fdv_risk}</span> &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Drawdown:</b> <span style='color:#ef4444; font-weight:bold;'>{drawdown_pct:.1f}%</span> &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Дивергенция:</b> <span style='font-weight:bold;'>{'YES ✅' if dv_bull else 'NO ❌'}</span> &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Сигнал:</b> <span style='color:{sig_c}; font-weight:bold;'>{sig_t}</span>
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# ИСТОРИЧЕСКИЕ ИСПЫТАНИЯ
# ============================================================
st.subheader("🛡️ Устойчивость на исторических точках дна")
sc1, sc2 = st.columns(2)
with sc1:
    t1_c = "#22c55e" if "✅" in stress["t1_status"] else "#ef4444" if "❌" in stress["t1_status"] else "#6b7280"
    st.markdown(f"""
        <div style='background: #111; padding:12px; border-radius:8px; border-top: 3px solid {t1_c};'>
            <p style='color:gray; font-size:12px; margin:0;'><b>ТЕСТ 10.11.2025 (Дно тарифов Трампа)</b></p>
            <p style='font-size:16px; font-weight:bold; color:{t1_c}; margin:5px 0 0 0;'>{stress['t1_status']} {f'({stress["t1_perf"]:+.1f}%)' if stress['t1_perf'] else ''}</p>
        </div>
    """, unsafe_allow_html=True)
with sc2:
    t2_c = "#22c55e" if "✅" in stress["t2_status"] else "#ef4444" if "❌" in stress["t2_status"] else "#6b7280"
    st.markdown(f"""
        <div style='background: #111; padding:12px; border-radius:8px; border-top: 3px solid {t2_c};'>
            <p style='color:gray; font-size:12px; margin:0;'><b>ТЕСТ 06.02.2026 (Глобальная капитуляция)</b></p>
            <p style='font-size:16px; font-weight:bold; color:{t2_c}; margin:5px 0 0 0;'>{stress['t2_status']} {f'({stress["t2_perf"]:+.1f}%)' if stress['t2_perf'] else ''}</p>
        </div>
    """, unsafe_allow_html=True)

# AI АНАЛИЗ
st.markdown("---")
if st.button("🧠 Запустить нейросетевой аудит DeepSeek v3", type="primary"):
    with st.spinner("Нейросеть сканирует профили рисков токеномики..."):
        ai_res = call_deepseek_v3(
            asset=asset, price=c_price, z=c_z, bottom_score=bottom_score, sig=sig_t, 
            rsi=c_rsi, vol_z=c_vol_z, div=dv_bull, stress=stress, fund=fund, 
            drawdown=drawdown_pct, regime=market_regime, fdv_risk=fdv_risk
        )
    st.markdown(f"""
        <div style='background:#0f172a; padding:18px; border-radius:12px; border:1px solid #1e293b; margin:10px 0;'>
            <h5 style='color:#38bdf8; margin-top:0;'>🤖 Аналитическое заключение DeepSeek ИИ:</h5>
            <p style='color:#e2e8f0; line-height:1.6; font-size:14px; margin:0;'>{ai_res}</p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================
# 11. ГРАФИКИ
# ============================================================
st.markdown("---")
st.subheader("📈 ГРАФИК ЦЕНЫ (цвет = Z-Score к MA90)")

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
    fig.add_trace(go.Scatter(x=df_chart["date"], y=df_chart["ma90"], mode="lines", name="MA90", line=dict(color="#ffffff", width=1.2, dash="dot"), opacity=0.5))
if "ma200" in df_chart.columns:
    fig.add_trace(go.Scatter(x=df_chart["date"], y=df_chart["ma200"], mode="lines", name="MA200", line=dict(color="#f59e0b", width=1.5, dash="dash"), opacity=0.7))

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
    hoverinfo="text", text=hover_texts,
    name="Инфо", hovertemplate="%{text}<extra></extra>"
))

price_range = df_chart["close"].max() / (df_chart["close"].min() + 1e-10)
fig.update_layout(height=480, template="plotly_dark", xaxis_title="", yaxis_title="Цена (USD)", yaxis_type="log" if price_range > 5 else "linear", hovermode="x unified", legend=dict(orientation="h", y=1.02, x=0), font=dict(family="Times New Roman", size=13))
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 12. СВОДНАЯ ТАБЛИЦА ВСЕХ АКТИВОВ V4.2
# ============================================================
st.markdown("---")
st.subheader("📋 СВОДНАЯ МАТРИЦА АКТИВОВ v4.2")

@st.cache_data(ttl=300)
def build_summary_table():
    all_assets = {**{c: "Криптовалюта" for c in CRYPTO_LIST}, **{s: "Акция" for s in STOCK_LIST}}
    rows = []
    
    btc_df = load_crypto_data("BTC", days=550)
    
    for symbol, atype in all_assets.items():
        df_t = load_crypto_data(symbol) if atype == "Криптовалюта" else load_stock_data(symbol)
        if df_t is None or len(df_t) < 90:
            continue
            
        res = calculate_metrics_adaptive(df_t, btc_df if atype == "Криптовалюта" else None)
        if res[0] is None:
            continue
            
        (_, price, z, bottom_score_val, (lt, ut), rsi_v, vol_z, dv_bull, ddown, rel_str, _, _) = res
        sig, _ = get_signal_adaptive(z, lt, ut, symbol in VETERAN_LIST)
        
        t_fdv_risk = "N/A" if atype == "Акция" else "📜 Клик на актив"
        
        clean_sig = sig
        for emoji in ["🔴", "🟡", "🟢", "⚪"]:
            clean_sig = clean_sig.replace(emoji, "")
        clean_sig = clean_sig.strip()
        
        rows.append({
            "Символ": symbol,
            "Тип": atype,
            "Цена": f"${price:,.4f}" if price < 1 else f"${price:,.2f}",
            "Z-Score": f"{z:.2f}",
            "RSI": f"{rsi_v:.1f}",
            "Диверг.": "✅" if dv_bull else "—",
            "Drawdown": f"{ddown:.1f}%",
            "FDV Risk": t_fdv_risk,
            "Bottom Score": bottom_score_val,
            "Сигнал": clean_sig
        })
    return rows

with st.spinner("Построение сквозной матрицы ранжирования рынков..."):
    summary = build_summary_table()

if summary:
    df_summary = pd.DataFrame(summary).sort_values(by="Bottom Score", ascending=False)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
    st.caption("💡 Таблица отсортирована по убыванию Bottom Score. Чем выше балл, тем больше факторов подтверждают истинное дно.")

# ============================================================
# 13. ПОДВАЛ
# ============================================================
moscow_tz = timezone(timedelta(hours=3))
moscow_time = datetime.now(moscow_tz)

st.markdown("---")
st.caption(f"📅 Синхронизация: {moscow_time.strftime('%Y-%m-%d %H:%M:%S')} (МСК) | Архитектура Детектора: v4.2 (Pandas Datetime Fix + Global ATH Drawdown + Optimized Data Streams)")
