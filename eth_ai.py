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

st.set_page_config(page_title="Детектор дна активов v4.3", layout="wide")

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

st.title("📊 Детектор дна активов v4.3")

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
# 2. ЗАГРУЗКА ДАННЫХ С СИНХРОННЫМ КЭШЕМ (15 МИНУТ)
# ============================================================

@st.cache_data(ttl=900) # Кэш выровнен под 15 минут
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

@st.cache_data(ttl=900) # Кэш выровнен под 15 минут
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

@st.cache_data(ttl=3600) # Защита API от лимитов (1 час)
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
            }
    except:
        pass
    return None

@st.cache_data(ttl=900) # Кэш выровнен под 15 минут
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
# 3. МАТЕМАТИЧЕСКИЕ ИНДИКАТОРЫ И ДИВЕРГЕНЦИИ
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
# 4. ЯДРО МАТРИЧНОГО СКОРИНГА С ПОВЫШЕННЫМИ ВЕСАМИ V4.3
# ============================================================

def calculate_metrics_adaptive(df, btc_df=None):
    if df is None or len(df) < 90:
        return (None,) * 13
        
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
    
    # СЛОВАРЬ ДЛЯ ФОРМИРОВАНИЯ БЛОКА ПРИЧИН
    reasons_checklist = []
    bottom_score = 0
    
    # 1. ФАКТОР ПРОСАДКИ (Повышенный вес: макс 35 баллов)
    if drawdown_pct <= -85: 
        bottom_score += 35
        reasons_checklist.append(("✅ Критическая просадкa >85%", True))
    elif drawdown_pct <= -70: 
        bottom_score += 28
        reasons_checklist.append(("✅ Экстремальная просадка >70%", True))
    elif drawdown_pct <= -50: 
        bottom_score += 22
        reasons_checklist.append(("✅ Глубокая просадка >50%", True))
    elif drawdown_pct <= -30: 
        bottom_score += 12
        reasons_checklist.append(("✅ Умеренная просадка >30%", True))
    else:
        reasons_checklist.append(("❌ Незначительная просадка от пика", False))
        
    # 2. ФАКТОР RSI (Повышенный вес: макс 20 баллов)
    if c_rsi <= 32: 
        bottom_score += 20
        reasons_checklist.append((f"✅ RSI в глубокой перепроданности ({c_rsi:.1f} <= 32)", True))
    elif c_rsi <= 42: 
        bottom_score += 12
        reasons_checklist.append((f"✅ RSI в зоне накопления ({c_rsi:.1f} <= 42)", True))
    else:
        reasons_checklist.append((f"❌ RSI вне зоны перепроданности ({c_rsi:.1f})", False))
        
    # 3. ФАКТОР Z-SCORE К МA90 (Макс 15 баллов)
    if c_z <= low_t: 
        bottom_score += 15
        reasons_checklist.append((f"✅ Отклонение Z-Score ({c_z:.2f}) ниже адаптивного дна", True))
    elif c_z < -0.8: 
        bottom_score += 8
        reasons_checklist.append((f"✅ Отрицательный Z-Score ({c_z:.2f}) ниже нормы", True))
    else:
        reasons_checklist.append((f"❌ Математическое отклонение Z-Score нейтрально ({c_z:.2f})", False))
        
    # 4. БЫЧЬЯ ДИВЕРГЕНЦИЯ RSI (Макс 10 баллов)
    if dv_bull: 
        bottom_score += 10
        reasons_checklist.append(("✅ Обнаружена бычья RSI-дивергенция (скрытый разворот)", True))
    else:
        reasons_checklist.append(("❌ Сигналов дивергенции RSI не найдено", False))
        
    # 5. КЛИМАКС ОБЪЁМОВ НА ПАДЕНИИ (Макс 10 баллов)
    if c_vol_z >= 1.8 and c_z < 0: 
        bottom_score += 10
        reasons_checklist.append((f"✅ Экстремальный объём капитуляции ({c_vol_z:+.1f}σ)", True))
    elif c_vol_z >= 0.8 and c_z < 0: 
        bottom_score += 5
        reasons_checklist.append((f"✅ Повышенный торговый объём ({c_vol_z:+.1f}σ)", True))
    else:
        reasons_checklist.append(("❌ Всплеска объёмов на падении нет", False))
        
    # 6. ВЫШЕ ТРЕНДОВОЙ MA200 (Макс 5 баллов)
    if current_price >= c_ma200 and c_ma200 > 0: 
        bottom_score += 5
        reasons_checklist.append(("✅ Актив удерживает глобальный тренд MA200", True))
    else:
        reasons_checklist.append(("❌ Актив находится под MA200 (медвежий тренд)", False))
        
    # 7. RELATIVE STRENGTH (Макс 5 баллов)
    if relative_strength > 4: 
        bottom_score += 5
        reasons_checklist.append((f"✅ Сила к BTC превосходит рынок ({relative_strength:+.1f}%)", True))
    elif relative_strength > -2:
        bottom_score += 2
        reasons_checklist.append((f"✅ Относительная стабильность к BTC ({relative_strength:+.1f}%)", True))
    else:
        reasons_checklist.append((f"❌ Слабость относительно динамики BTC ({relative_strength:+.1f}%)", False))

    bottom_score = min(bottom_score, 100)

    return df, current_price, c_z, bottom_score, (low_t, up_t), c_rsi, c_vol_z, dv_bull, drawdown_pct, relative_strength, current_ath, c_ma200, reasons_checklist

# ============================================================
# 5. ИНТЕГРАЦИЯ DEEPSEEK V3 AI
# ============================================================

def call_deepseek_v3(asset, price, z, bottom_score, sig, rsi, vol_z, div, fund, drawdown, regime, fdv_risk):
    key = st.secrets.get("DEEPSEEK_API_KEY", "")
    if not key:
        return "❌ Ключ интеграции ИИ DeepSeek отсутствует."
        
    f_text = f"Капитализация: ${fund.get('market_cap', 0):,.0f}, Риск токеномики (Риск FDV): {fdv_risk}." if (fund and isinstance(fund, dict) and fund.get('market_cap')) else "Фундаментальные ончейн-данные отсутствуют (традиционный актив)."
        
    prompt = f"""Проведи глубокий экспресс-анализ {asset}:
МЕТРИКИ v4.3: Цена: ${price:,.4f}, Z-Score: {z:.2f}, RSI: {rsi:.1f}, Просадка: {drawdown:.1f}%, Объём: {vol_z:+.1f}σ, Бычий паттерн дивергенции: {'ДА' if div else 'НЕТ'}.
Итоговый Индекс дна системы (макс 100): {bottom_score} баллов. Сигнал детектора: {sig}.
Текущий Режим рынка: {regime}. {f_text}

Напиши профессиональный вывод (4-5 предложений) на русском языке. Оцени: является ли просадка фундаментально оправданной, защищен ли инвестор от скрытой инфляции предложения (Риск FDV) и конкретный торговый план в рамках текущего макро-режима."""

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
# 6. ПОЛУЧЕНИЕ ГЛОБАЛЬНОГО СТАТУСА
# ============================================================

with st.sidebar:
    st.header("⚙️ НАСТРОЙКИ СИСТЕМЫ v4.3")
    st.markdown("---")
    market = st.radio("Сектор рынка", ["Криптовалюты", "Фондовый рынок"])
    if market == "Криптовалюты":
        asset = st.selectbox("Выбор цифрового актива", CRYPTO_LIST)
    else:
        asset = st.selectbox("Выбор акции/фонда", STOCK_LIST)
    st.markdown("---")
    st.caption("📈 **Математическая модель v4.3**")
    st.caption("Частота обновления: 15 мин. Сбалансированы веса просадки и RSI. Добавлен лог обоснования оценки дна.")

with st.spinner("Синхронизация глобальных индикаторов..."):
    market_regime = get_market_regime()
    btc_global_df = load_crypto_data("BTC", days=550)

# ============================================================
# 7. ВЫЧИСЛЕНИЯ ПО ВЫБРАННОМУ АКТИВУ
# ============================================================

is_c = asset in CRYPTO_LIST
is_v = asset in VETERAN_LIST

fund = None
fdv_risk = "N/A" if not is_c else "Неизвестно"

if is_c and asset in COINGECKO_IDS:
    fund = get_coingecko_fundamentals(COINGECKO_IDS[asset])
    if fund and isinstance(fund, dict):
        mcap = fund.get("market_cap", 0)
        fdv = fund.get("fully_diluted_valuation", 0)
        fdv_ratio = fdv / mcap if mcap > 0 else 1
        if fdv_ratio < 1.5: fdv_risk = "Низкий"
        elif fdv_ratio < 3.0: fdv_risk = "Средний"
        else: fdv_risk = "Высокий"

with st.spinner(f"Загрузка потоков данных по {asset}..."):
    raw = load_crypto_data(asset) if is_c else load_stock_data(asset)

if raw is None or len(raw) < 90:
    st.error(f"❌ Недостаточно торговой истории для анализа {asset}.")
    st.stop()

df, c_price, c_z, bottom_score, (low_thr, upper_thr), c_rsi, c_vol_z, dv_bull, drawdown_pct, relative_strength, current_ath, c_ma200, reasons = calculate_metrics_adaptive(raw, btc_global_df if is_c else None)
sig_t, sig_c = get_signal_adaptive(c_z, low_thr, upper_thr, is_v)

# ============================================================
# ДАШБОРД РЕЗУЛЬТАТОВ (РУСИФИЦИРОВАННЫЙ)
# ============================================================

st.header(f"📊 Паспорт актива: {asset}")
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("💰 ТЕКУЩАЯ ЦЕНА", f"${c_price:,.4f}" if c_price < 1 else f"${c_price:,.2f}")
with c2: 
    score_c = "#22c55e" if bottom_score >= 60 else "#eab308" if bottom_score >= 35 else "#ef4444"
    st.markdown(f"""
        <div style='background: {score_c}10; padding: 10px; border-radius: 8px; border: 1px solid {score_c}30; text-align: center;'>
            <p style='color: gray; margin:0; font-size:12px; font-weight:bold;'>ИНДЕКС ДНА</p>
            <p style='color: {score_c}; font-size:22px; font-weight:bold; margin:3px 0 0 0;'>{bottom_score} / 100</p>
        </div>
    """, unsafe_allow_html=True)
with c3: st.metric("📉 ПРОСАДКА АКТИВА", f"{drawdown_pct:.1f}%")
with c4: st.metric("📈 RSI (14)", f"{c_rsi:.1f}")
with c5: st.metric("📦 ОБЪЁМ Z-SCORE", f"{c_vol_z:+.2f}σ")

# ============================================================
# РУСИФИЦИРОВАННАЯ СТРОКА СТАТУСА
# ============================================================

st.markdown(f"""
<div style='background: linear-gradient(135deg, #0b0f19 0%, #111827 100%); padding:14px; border-radius:10px; margin: 15px 0; border: 1px solid #1f2937;'>
    <p style='margin:0; color:#f3f4f6; font-size:14px; text-align: center;'>
        <b>Режим рынка:</b> <span style='font-weight:bold;'>{market_regime}</span> &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Риск FDV:</b> <span style='font-weight:bold;'>{fdv_risk}</span> &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Дивергенция:</b> <span style='font-weight:bold;'>{'Есть ✅' if dv_bull else 'Нет ❌'}</span> &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Сигнал детектора:</b> <span style='color:{sig_c}; font-weight:bold;'>{sig_t}</span>
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# НОВЫЙ ПРАКТИЧЕСКИЙ БЛОК: ПРИЧИНЫ ОЦЕНКИ ДНА
# ============================================================
st.subheader("📋 Причины формирования текущей оценки дна")
with st.container():
    rc1, rc2 = st.columns(2)
    with rc1:
        # Выводим выполненные условия (накопление силы дна)
        st.markdown("**Условия, добавившие баллы к Индексу Дна:**")
        fulfilled = [r[0] for r in reasons if r[1]]
        if fulfilled:
            for item in fulfilled:
                st.markdown(item)
        else:
            st.caption("Ни одно из условий формирования дна не выполнено.")
            
    with rc2:
        # Выводим невыполненные условия (ограничители роста индекса)
        st.markdown("**Невыполненные / Пропущенные триггеры дна:**")
        unfulfilled = [r[0] for r in reasons if not r[1]]
        for item in unfulfilled:
            st.markdown(item)

# ИИ-ИНТЕГРАЦИЯ DEEPSEEK
st.markdown("---")
if st.button("🧠 Запустить экспресс-анализ ИИ DeepSeek v3", type="primary"):
    with st.spinner("Интегрированный ИИ считывает матрицу факторов..."):
        ai_res = call_deepseek_v3(
            asset=asset, price=c_price, z=c_z, bottom_score=bottom_score, sig=sig_t, 
            rsi=c_rsi, vol_z=c_vol_z, div=dv_bull, fund=fund, 
            drawdown=drawdown_pct, regime=market_regime, fdv_risk=fdv_risk
        )
    st.markdown(f"""
        <div style='background:#0f172a; padding:18px; border-radius:12px; border:1px solid #1e293b; margin:10px 0;'>
            <h5 style='color:#38bdf8; margin-top:0;'>🤖 Аналитическое резюме ИИ DeepSeek:</h5>
            <p style='color:#e2e8f0; line-height:1.6; font-size:14px; margin:0;'>{ai_res}</p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================
# 8. ГРАФИК ЦЕНЫ
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
    fig.add_trace(go.Scatter(x=df_chart["date"], y=df_chart["ma90"], mode="lines", name="MA90", line=dict(color="#ffffff", width=1.2, dash="dot grandfathered"), opacity=0.4))
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
# 9. СВОДНАЯ ТАБЛИЦА ВСЕХ АКТИВОВ V4.3 (РУСИФИЦИРОВАННАЯ)
# ============================================================
st.markdown("---")
st.subheader("📋 СКВОЗНАЯ МАТРИЦА СКОРИНГА РЫНКОВ v4.3")

@st.cache_data(ttl=900) # Выровнено строго под 15 минут
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
            
        (_, price, z, bottom_score_val, (lt, ut), rsi_v, vol_z, dv_bull, ddown, rel_str, _, _, _) = res
        sig, _ = get_signal_adaptive(z, lt, ut, symbol in VETERAN_LIST)
        
        t_fdv_risk = "N/A" if atype == "Акция" else "📜 Карточка актива"
        
        # Очистка эмодзи для построения таблицы
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
            "Просадка": f"{ddown:.1f}%",
            "Риск FDV": t_fdv_risk,
            "Индекс Дна": bottom_score_val,
            "Сигнал": clean_sig
        })
    return rows

with st.spinner("Расчет сводных рыночных коэффициентов..."):
    summary = build_summary_table()

if summary:
    df_summary = pd.DataFrame(summary).sort_values(by="Индекс Дна", ascending=False)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
    st.caption("💡 Таблица упорядочена по убыванию Индекса Дна. Чем выше балл, тем массивнее синергия сигналов перепроданности.")

# ============================================================
# 10. ПОДВАЛ
# ============================================================
moscow_tz = timezone(timedelta(hours=3))
moscow_time = datetime.now(moscow_tz)

st.markdown("---")
st.caption(f"📅 Последняя сквозная синхронизация: {moscow_time.strftime('%Y-%m-%d %H:%M:%S')} (МСК) | Версия сборки: v4.3 (Локализация + Декомпозиция факторов скоринга + Синхронный 15-мин кэш)")
