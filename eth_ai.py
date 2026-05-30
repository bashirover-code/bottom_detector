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

st.set_page_config(page_title="Детектор качества Активов", layout="wide")

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

# Первая строчка дашборда
st.title("📊 Детектор качества Активов")

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
# 3. МАТЕМАТИЧЕСКИЕ ИНДИКАТОРЫ И ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
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

def calculate_single_rs(df, btc_df, lookup_days):
    if btc_df is None or len(df) < lookup_days or len(btc_df) < lookup_days:
        return 0.0
    df_temp = df.copy()
    btc_temp = btc_df.copy()
    df_temp["d_norm"] = df_temp["date"].dt.date
    btc_temp["d_norm"] = btc_temp["date"].dt.date
    
    common_dates = np.intersect1d(df_temp['d_norm'], btc_temp['d_norm'])
    if len(common_dates) >= lookup_days:
        df_sub = df_temp[df_temp['d_norm'].isin(common_dates)].sort_values("d_norm")
        btc_sub = btc_temp[btc_temp['d_norm'].isin(common_dates)].sort_values("d_norm")
        if len(df_sub) >= lookup_days and len(btc_sub) >= lookup_days:
            asset_perf = (df_sub['close'].iloc[-1] / df_sub['close'].iloc[-lookup_days] - 1) * 100
            btc_perf = (btc_sub['close'].iloc[-1] / btc_sub['close'].iloc[-lookup_days] - 1) * 100
            return asset_perf - btc_perf
    return 0.0

# ============================================================
# 4. АЛГОРИТМИЧЕСКИЙ МНОГОФАКТОРНЫЙ СКОРИНГ
# ============================================================

def calculate_metrics_adaptive(symbol, df, btc_df=None, global_regime="⚪ Нейтральный"):
    if df is None or len(df) < 200:
        return (None,) * 17
        
    df = df.copy()
    
    # Скользящие средние и технические индикаторы
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
    c_ma90 = df["ma90"].iloc[-1]
    ma200_30_days_ago = df["ma200"].iloc[-30] if len(df) >= 30 else c_ma200
    
    current_ath = df["close"].max()
    drawdown_pct = ((current_price - current_ath) / current_ath * 100) if current_ath > 0 else 0
    
    # Расчет ликвидности в долларовом объеме (Dollar Volume)
    df["dollar_volume"] = df["close"] * df["volume"]
    avg_dollar_volume = df["dollar_volume"].tail(30).mean()
    
    # Мультитаймфрейм расчет силы к BTC (RS30, RS90, RS180)
    rs30 = calculate_single_rs(df, btc_df, 30)
    rs90 = calculate_single_rs(df, btc_df, 90)
    rs180 = calculate_single_rs(df, btc_df, 180)
    relative_strength = (rs30 * 0.2) + (rs90 * 0.3) + (rs180 * 0.5)

    low_t, up_t = get_adaptive_thresholds(df["z_score"].values)
    
    reasons_checklist = []
    
    # --------------------------------------------------------
    # АВТОМАТИЧЕСКИЙ ИНДЕКС ДНА (Макс 100 баллов / Вес 40%)
    # --------------------------------------------------------
    bottom_score = 0
    
    # Просадка (Макс 20 баллов)
    if drawdown_pct <= -85: 
        bottom_score += 20
        reasons_checklist.append(("📉 Экстремальная уценка от ATH (>85%): +20 баллов дна", True))
    elif drawdown_pct <= -70: 
        bottom_score += 15
        reasons_checklist.append(("📉 Глубокая уценка от ATH (>70%): +15 баллов дна", True))
    elif drawdown_pct <= -50: 
        bottom_score += 10
        reasons_checklist.append(("📉 Умеренная уценка от ATH (>50%): +10 баллов дна", True))
        
    # RSI (Макс 35 баллов)
    if c_rsi <= 32: 
        bottom_score += 35
        reasons_checklist.append((f"⚡ RSI в зоне перепроданности ({c_rsi:.1f}): +35 баллов дна", True))
    elif c_rsi <= 42: 
        bottom_score += 20
        reasons_checklist.append((f"⚡ RSI в накопительном коридоре ({c_rsi:.1f}): +20 баллов дна", True))
    
    # Z-Score (Макс 30 баллов)
    if c_z <= low_t: 
        bottom_score += 30
        reasons_checklist.append((f"📊 Z-Score вышел за адаптивный минимум ({c_z:.2f}): +30 баллов дна", True))
    elif c_z < -0.8: 
        bottom_score += 15
        reasons_checklist.append((f"📊 Снижение Z-Score ниже скользящей средней ({c_z:.2f}): +15 баллов дна", True))
        
    # Капитуляция объемов (Макс 15 баллов)
    if c_vol_z >= 1.5 and c_z < 0: 
        bottom_score += 15
        reasons_checklist.append((f"📦 Пиковый объем в зоне продаж ({c_vol_z:+.1f}σ): +15 баллов дна", True))
    
    bottom_score = min(bottom_score, 100)

    # --------------------------------------------------------
    # АВТОМАТИЧЕСКАЯ МАТРИЦА КАЧЕСТВА (Макс 100 баллов / Вес 60%)
    # --------------------------------------------------------
    quality_score = 0
    
    # Фактор "Возраст проекта" полностью удален из логики модели

    # 1. Ликвидность на основе Dollar Volume (Макс 35 баллов)
    if avg_dollar_volume > 50_000_000:
        quality_score += 35
        reasons_checklist.append((f"💎 Высокая институциональная ликвидность (${avg_dollar_volume/1e6:.1f}M/день): +35 к Качеству", True))
    elif avg_dollar_volume > 5_000_000:
        quality_score += 20
        reasons_checklist.append((f"✅ Умеренная рыночная ликвидность (${avg_dollar_volume/1e6:.1f}M/день): +20 к Качеству", True))
    else:
        quality_score += 5
        reasons_checklist.append((f"⚠️ Низколиквидный актив (${avg_dollar_volume/1e6:.1f}M/день): всего +5 к Качеству", False))

    # 2. Оценка структуры тренда (Trend Structure Score) (Макс 35 баллов)
    structure_score = 0
    if current_price > c_ma90:
        structure_score += 10
    if current_price > c_ma200:
        structure_score += 15
    if c_ma200 > ma200_30_days_ago:
        structure_score += 10
    
    quality_score += structure_score
    if structure_score > 0:
        reasons_checklist.append((f"📈 Базовая структура силы тренда: +{structure_score} к Качеству", True))

    # 3. Мультитаймфрейм сила к BTC (Макс 30 баллов)
    if symbol == "BTC":
        quality_score += 30
        reasons_checklist.append(("👑 Базовый эталон рынка (BTC): максимальный балл тренда +30", True))
    else:
        if relative_strength > 10:
            quality_score += 30
            reasons_checklist.append((f"💪 Взвешенная Альфа-сила к BTC ({relative_strength:+.1f}%): +30 к Качеству", True))
        elif relative_strength > 0:
            quality_score += 18
            reasons_checklist.append((f"✅ Удержание тренда сильнее BTC ({relative_strength:+.1f}%): +18 к Качеству", True))
        elif relative_strength > -15:
            quality_score += 8
            reasons_checklist.append((f"⏳ Умеренная слабость к BTC ({relative_strength:+.1f}%): +8 к Качеству", True))
        else:
            quality_score -= 10
            reasons_checklist.append((f"❌ Глубокое отставание от индекса BTC ({relative_strength:+.1f}%): ШТРАФ -10 к Качеству", False))

    # Системный макро-штраф за Медвежий тренд общего рынка
    if global_regime == "🔴 Медвежий" and symbol != "BTC":
        quality_score -= 5
        reasons_checklist.append(("Bear Давление общего медвежьего цикла: ШТРАФ -5 к Качеству", False))

    quality_score = max(0, min(quality_score, 100))

    # --------------------------------------------------------
    # РАСЧЕТ СТАДИИ ЦИКЛА И МНОЖИТЕЛЯ ТРЕНДА (TREND MULTIPLIER)
    # --------------------------------------------------------
    # Вычисление Recovery Score (Скорость отскока со дна за последние 365 дней)
    cycle_low = df["close"].tail(365).min()
    if (current_ath - cycle_low) > 0:
        recovery_score = ((current_price - cycle_low) / (current_ath - cycle_low)) * 100
    else:
        recovery_score = 0.0

    # Классификация стадий цикла актива по RSI, MA90, MA200, Z-score и Recovery Score
    if c_rsi < 35 and c_z < -1.5:
        cycle_stage = "Капитуляция"
    elif current_price > c_ma90 and current_price > c_ma200 and c_ma200 > ma200_30_days_ago:
        if c_rsi > 72:
            cycle_stage = "Перегрев"
        else:
            cycle_stage = "Тренд"
    elif current_price > c_ma90 and (recovery_score > 15 or c_z > 0.5):
        cycle_stage = "Разворот"
    else:
        cycle_stage = "Накопление"

    # Применение штрафов и бонусов структуры тренда к финальному рейтингу
    if structure_score >= 30:
        trend_multiplier = 1.10
        reasons_checklist.append(("🚀 Премиум-множитель за идеальный аптренд: ×1.10 к итогу", True))
    elif structure_score >= 10:
        trend_multiplier = 0.95
        reasons_checklist.append(("⏳ Мягкий штраф за неопределенный тренд: ×0.95 к итогу", False))
    else:
        trend_multiplier = 0.80
        reasons_checklist.append(("❌ Жесткий штраф за отсутствие структуры и падение: ×0.80 к итогу", False))

    # Расчет базового рейтинга (0.6 * Качество + 0.4 * Дно)
    base_rating = (0.6 * quality_score) + (0.4 * bottom_score)
    
    # Применение множителя структуры тренда
    final_rating = base_rating * trend_multiplier
    final_rating = max(0, min(final_rating, 100))

    return df, current_price, c_z, bottom_score, quality_score, final_rating, (low_t, up_t), c_rsi, c_vol_z, drawdown_pct, relative_strength, current_ath, c_ma200, avg_dollar_volume, structure_score, recovery_score, cycle_stage, reasons_checklist

# ============================================================
# ПОЛУЧЕНИЕ ГЛОБАЛЬНОГО МАКРО-РЕЖИМА
# ============================================================

with st.sidebar:
    st.header("⚙️ НАСТРОЙКИ СИСТЕМЫ")
    st.markdown("---")
    market = st.radio("Сектор рынка", ["Криптовалюты", "Фондовый рынок"])
    if market == "Криптовалюты":
        asset = st.selectbox("Выбор цифрового актива", CRYPTO_LIST)
    else:
        asset = st.selectbox("Выбор акции/фонда", STOCK_LIST)
    st.markdown("---")
    st.caption("🚀 **Квантовый Скринер структуры тренда**")
    st.caption("Автоматическая фильтрация стадий рынка и расчет силы возврата от минимумов без субъективной оценки.")

with st.spinner("Синхронизация макро-показателей рынка..."):
    market_regime = get_market_regime()
    btc_global_df = load_crypto_data("BTC", days=550)

# ============================================================
# ВЫЧИСЛЕНИЯ ПО ВЫБРАННОМУ АКТИВУ
# ============================================================

is_c = asset in CRYPTO_LIST
is_v = asset in VETERAN_LIST

with st.spinner(f"Запуск квантовой модели для {asset}..."):
    raw = load_crypto_data(asset) if is_c else load_stock_data(asset)

if raw is None or len(raw) < 200:
    st.error(f"❌ Недостаточно данных для анализа {asset} (требуется минимум 200 дней истории).")
    st.stop()

df, c_price, c_z, b_score, q_score, f_rating, (low_thr, upper_thr), c_rsi, c_vol_z, drawdown_pct, rel_strength, current_ath, c_ma200, dollar_vol, trend_score, recovery_score, cycle_stage, reasons = calculate_metrics_adaptive(asset, raw, btc_global_df if is_c else None, market_regime)
sig_t, sig_c = get_signal_adaptive(c_z, low_thr, upper_thr, is_v)

# ============================================================
# ДАШБОРД ВЫБРАННОГО АКТИВА
# ============================================================

# Вторая строчка дашборда
st.header(f"📊 актив: {asset}")

c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("💰 ТЕКУЩАЯ ЦЕНА", f"${c_price:,.4f}" if c_price < 1 else f"${c_price:,.2f}")
with c2: 
    rating_color = "#22c55e" if f_rating >= 65 else "#eab308" if f_rating >= 45 else "#ef4444"
    st.markdown(f"""
        <div style='background: {rating_color}10; padding: 10px; border-radius: 8px; border: 1px solid {rating_color}30; text-align: center;'>
            <p style='color: gray; margin:0; font-size:11px; font-weight:bold;'>ИТОГОВЫЙ ИНДЕКС</p>
            <p style='color: {rating_color}; font-size:22px; font-weight:bold; margin:3px 0 0 0;'>{f_rating:.1f} / 100</p>
        </div>
    """, unsafe_allow_html=True)
with c3: st.metric("📉 ПРОСАДКА АКТИВА", f"{drawdown_pct:.1f}%")
with c4: st.metric("⚡ СИЛА К BTC (ВЗВЕШ.)", f"{rel_strength:+.1f}%" if is_c else "N/A")
with c5: st.metric("🔄 RECOVERY SCORE", f"{recovery_score:.1f}%")

# СТРОКА СТАТУСА: Очищена от индекса дна
st.markdown(f"""
<div style='background: linear-gradient(135deg, #0b0f19 0%, #111827 100%); padding:14px; border-radius:10px; margin: 15px 0; border: 1px solid #1f2937;'>
    <p style='margin:0; color:#f3f4f6; font-size:14px; text-align: center;'>
        <b>Итоговый индекс:</b> <span style='color:{rating_color}; font-weight:bold;'>{f_rating:.1f}/100</span> &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Качество:</b> <span style='color:#38bdf8; font-weight:bold;'>{q_score}/100</span> &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Сила к BTC:</b> <span style='color:#fb923c; font-weight:bold;'>{rel_strength:+.1f}%</span> &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Режим рынка:</b> <span style='font-weight:bold;'>{market_regime}</span> &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Технический Сигнал:</b> <span style='color:{sig_c}; font-weight:bold;'>{sig_t}</span>
    </p>
</div>
""", unsafe_allow_html=True)

# ЛОГИЧЕСКОЕ ОБОСНОВАНИЕ
st.subheader("📋 Квантовый разбор факторов тренда и ликвидности")
with st.container():
    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("**Факторы начисления баллов и силы:**")
        fulfilled = [r[0] for r in reasons if r[1]]
        for item in fulfilled: st.markdown(f"- {item}")
        if detect_rsi_divergence(df, 35):
            st.markdown("- 🔄 *Информативно: На графике присутствует Бычья Дивергенция RSI (без влияния на скоринг).*")
    with rc2:
        st.markdown("**Примененные штрафы и фильтры структуры:**")
        unfulfilled = [r[0] for r in reasons if not r[1]]
        if unfulfilled:
            for item in unfulfilled: st.markdown(f"- {item}")
        else: st.caption("Штрафы тренда и ограничения ликвидности отсутствуют.")

# ============================================================
# 5. ИНТЕРАКТИВНЫЙ ГРАФИК
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
        f"📦 Долл. Объём: <b>{v:.2f}</b>"
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
# 6. СКВОЗНАЯ ИНВЕСТИЦИОННАЯ МАТРИЦА СКОРИНГА
# ============================================================
st.markdown("---")
st.subheader("📋 СКВОЗНАЯ АЛГОРИТМИЧЕСКАЯ ИНВЕСТИЦИОННАЯ МАТРИЦА СКОРИНГА")

@st.cache_data(ttl=900)
def build_summary_table(regime):
    all_assets = {**{c: "Криптовалюта" for c in CRYPTO_LIST}, **{s: "Акция" for s in STOCK_LIST}}
    rows = []
    
    btc_df = load_crypto_data("BTC", days=550)
    
    for symbol, atype in all_assets.items():
        df_t = load_crypto_data(symbol) if atype == "Криптовалюта" else load_stock_data(symbol)
        if df_t is None or len(df_t) < 200:
            continue
            
        res = calculate_metrics_adaptive(symbol, df_t, btc_df if atype == "Криптовалюта" else None, regime)
        if res[0] is None:
            continue
            
        (_, _, _, _, quality_s, final_r, _, _, _, ddown, rel_str, _, _, dollar_vol, _, _, cycle_stage, _) = res
        
        btc_strength_text = f"{rel_str:+.1f}%" if atype == "Криптовалюта" else "N/A"
        
        # Пересборка таблицы строго по вашему ТЗ: Символ, Просадка, RS BTC, Ликвидность, Стадия цикла, Качество, Итог
        rows.append({
            "Символ": symbol,
            "Просадка": f"{ddown:.1f}%",
            "RS BTC": btc_strength_text,
            "Ликвидность": f"${dollar_vol / 1e6:.1f}M",
            "Стадия цикла": cycle_stage,
            "Качество": int(quality_s),
            "Итог": round(final_r, 1)
        })
    return rows

with st.spinner("Генерация инвестиционной таблицы..."):
    summary = build_summary_table(market_regime)

if summary:
    df_summary = pd.DataFrame(summary).sort_values(by="Итог", ascending=False)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
    st.caption("💡 Матрица отсортирована по Итогу. Наличие сильной структуры тренда умножает итоговый балл, а активы без ликвидности в затяжном падении пессимизируются коэффициентом ×0.80.")

# ============================================================
# 7. ПОДВАЛ
# ============================================================
moscow_tz = timezone(timedelta(hours=3))
moscow_time = datetime.now(moscow_tz)

st.markdown("---")
st.caption(f"📅 Синхронизация данных: {moscow_time.strftime('%Y-%m-%d %H:%M:%S')} (МСК) | Ядро: Квантовый Алгоритм (Dollar Volume | Мультитаймфрейм Альфа к BTC | Тренд-мультипликатор структуры тренда | Фильтр стадий цикла)")
