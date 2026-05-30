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

st.set_page_config(page_title="Двухфакторная Матрица Активов", layout="wide")

# Автообновление каждые 15 минут
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
st.title("🏛️ Двухфакторная Инвестиционная Матрица")

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

def get_adaptive_thresholds(z_scores):
    if len(z_scores) < 30:
        return -1.8, 1.5
    low = np.nanpercentile(z_scores, 6)
    high = np.nanpercentile(z_scores, 94)
    return max(-3.2, min(-1.0, low)), min(3.2, max(0.6, high))

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
# 4. ДВУХФАКТОРНАЯ МОДЕЛЬ: КАЧЕСТВО vs ПОТЕНЦИАЛ ВХОДА
# ============================================================

def calculate_two_factor_matrix(symbol, df, btc_df=None):
    if df is None or len(df) < 200:
        return (None,) * 19
        
    df = df.copy()
    
    # Расчет базовых индикаторов тренда
    df["ma90"] = df["close"].rolling(window=90, min_periods=30).mean()
    df["std90"] = df["close"].rolling(window=90, min_periods=30).std()
    df["z_score"] = (df["close"] - df["ma90"]) / (df["std90"] + 1e-10)
    df["rsi"] = calculate_rsi(df, 14)
    df["ma200"] = df["close"].rolling(window=200, min_periods=50).mean()
    
    df["v_mean"] = df["volume"].rolling(window=30, min_periods=10).mean()
    df["v_std"] = df["volume"].rolling(window=30, min_periods=10).std()
    df["vol_z"] = (df["volume"] - df["v_mean"]) / (df["v_std"] + 1e-10)
    
    df[list(df.columns[-8:])] = df[list(df.columns[-8:])].bfill().ffill().fillna(0)
    
    current_price = df["close"].iloc[-1]
    c_z = df["z_score"].iloc[-1]
    c_rsi = df["rsi"].iloc[-1]
    c_ma200 = df["ma200"].iloc[-1]
    c_ma90 = df["ma90"].iloc[-1]
    ma200_30_days_ago = df["ma200"].iloc[-30] if len(df) >= 30 else c_ma200
    
    current_ath = df["close"].max()
    drawdown_pct = ((current_price - current_ath) / current_ath * 100) if current_ath > 0 else 0
    
    # Ликвидность (Базовое Качество)
    df["dollar_volume"] = df["close"] * df["volume"]
    avg_dollar_volume = df["dollar_volume"].tail(30).mean()
    low_t, up_t = get_adaptive_thresholds(df["z_score"].values)
    
    reasons_checklist = []

    # 1. КОМПОНЕНТ: QUALITY VOLUME (Макс 100)
    if avg_dollar_volume > 50_000_000:
        quality_vol_score = 100
        liq_risk = "Низкий"
    elif avg_dollar_volume > 5_000_000:
        quality_vol_score = 70
        liq_risk = "Средний"
    else:
        quality_vol_score = 25
        liq_risk = "Высокий"

    # 2. КОМПОНЕНТ: STRENGTH (Чистая сила к BTC, макс 100)
    rs30 = calculate_single_rs(df, btc_df, 30)
    rs90 = calculate_single_rs(df, btc_df, 90)
    rs180 = calculate_single_rs(df, btc_df, 180)
    relative_strength = (rs30 * 0.2) + (rs90 * 0.3) + (rs180 * 0.5)

    if relative_strength > 40: rs_score = 100
    elif relative_strength > 15: rs_score = 80
    elif relative_strength > 0: rs_score = 65
    elif relative_strength > -20: rs_score = 40
    else: rs_score = 10
    
    # 3. КОМПОНЕНТ: RECOVERY И ВАШ OPPORTUNITY SCORE
    cycle_low = df["close"].tail(365).min()
    if (current_ath - cycle_low) > 0:
        recovery_score = ((current_price - cycle_low) / (current_ath - cycle_low)) * 100
    else:
        recovery_score = 0.0
        
    opportunity =
    0.4 * drawdown_score +
    0.3 * rs_recovery_score +
    0.3 * accumulation_score

    # Многофакторный расчет Strength (избегаем насыщения 100 баллов)
    strength_score = (quality_vol_score * 0.3) + (rs_score * 0.5) + (recovery_score * 0.2)
    strength_score = max(0, min(strength_score, 100))

    # 4. КОМПОНЕНТ: STRUCTURE (Структура тренда, макс 100)
    structure_raw = 0
    if current_price > c_ma90: structure_raw += 10
    if current_price > c_ma200: structure_raw += 15
    if c_ma200 > ma200_30_days_ago: structure_raw += 10
    structure_score = int((structure_raw / 35) * 100) if structure_raw > 0 else 0

    # 5. КОМПОНЕНТ: BOTTOM (Перепроданность)
    bottom_score = 0
    if drawdown_pct <= -70: bottom_score += 30
    elif drawdown_pct <= -50: bottom_score += 15
    if c_rsi <= 35: bottom_score += 40
    elif c_rsi <= 45: bottom_score += 20
    if c_z <= low_t: bottom_score += 30
    bottom_score = min(bottom_score, 100)

    # --------------------------------------------------------
    # РАЗДЕЛЕНИЕ НА ДВА НЕЗАВИСИМЫХ РЕЙТИНГА СИСТЕМЫ
    # --------------------------------------------------------
    
    # А. ИТОГОВОЕ КАЧЕСТВО АКТИВА (Институциональная сила)
    quality_rating = (0.45 * quality_vol_score) + (0.35 * strength_score) + (0.20 * structure_score)
    
    # Б. ПОТЕНЦИАЛ ВХОДА (Тайминг и Своевременность)
    entry_rating = (0.60 * opportunity_score) + (0.40 * bottom_score)
    
    # Корректирующий тренд-мультипликатор для Потенциала (смягчает вход против сильного падающего тренда)
    if structure_raw == 0:
        entry_rating *= 0.85
        reasons_checklist.append(("⚠️ Потенциал входа снижен: Актив находится в жестком даунтренде под скользящими", False))

    # ЖЕСТКИЙ ФИЛЬТР: Штраф за мусорность для обоих рейтингов
    if avg_dollar_volume < 1_000_000:
        quality_rating *= 0.7
        entry_rating *= 0.7
        reasons_checklist.append(("🚨 КРИТИЧЕСКИЙ РИСК: Ликвидность < $1М. Применен штраф мусорности ×0.7", False))

    quality_rating = max(0, min(quality_rating, 100))
    entry_rating = max(0, min(entry_rating, 100))

    # --------------------------------------------------------
    # 6-СТУПЕНЧАТЫЙ ПРЕЦИЗИОННЫЙ КЛАССИФИКАТОР СТАДИЙ ЦИКЛА
    # --------------------------------------------------------
    if c_rsi < 35 and c_z < -1.5:
        cycle_stage = "Капитуляция"
    elif current_price > c_ma90 and current_price > c_ma200:
        if c_rsi > 72 or c_z > up_t:
            cycle_stage = "Перегрев"
        elif c_ma200 > ma200_30_days_ago and c_rsi > 58:
            cycle_stage = "Тренд"
        else:
            cycle_stage = "Ранний тренд"
    elif current_price > c_ma90 and (recovery_score > 15 or c_z > 0.5):
        cycle_stage = "Разворот"
    else:
        cycle_stage = "Накопление"

    return df, current_price, c_z, quality_rating, entry_rating, (low_t, up_t), c_rsi, df["vol_z"].iloc[-1], drawdown_pct, relative_strength, current_ath, avg_dollar_volume, strength_score, recovery_score, opportunity_score, cycle_stage, liq_risk, reasons_checklist

# ============================================================
# ИНИЦИАЛИЗАЦИЯ И ИНТЕРФЕЙС SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ УПРАВЛЕНИЕ МАТРИЦЕЙ")
    st.markdown("---")
    market = st.radio("Сектор рынка", ["Криптовалюты", "Фондовый рынок"])
    if market == "Криптовалюты":
        asset = st.selectbox("Выбор цифрового актива", CRYPTO_LIST)
    else:
        asset = st.selectbox("Выбор акции/фонда", STOCK_LIST)
    st.markdown("---")
    st.caption("🏛️ **Институциональная Архитектура**")
    st.caption("• Качество и Потенциал входа разделены.\n• Интегрирован Opportunity Score (100 - Recovery).\n• Защита от перегретых топ-активов.")

with st.spinner("Расчет глобального макро-режима..."):
    market_regime = get_market_regime()
    btc_global_df = load_crypto_data("BTC", days=550)

# ============================================================
# ВЫЧИСЛЕНИЯ ПО ВЫБРАННОМУ АКТИВУ
# ============================================================

is_c = asset in CRYPTO_LIST
is_v = asset in VETERAN_LIST

with st.spinner(f"Анализ двухфакторных весов для {asset}..."):
    raw = load_crypto_data(asset) if is_c else load_stock_data(asset)

if raw is None or len(raw) < 200:
    st.error(f"❌ Недостаточно данных для анализа {asset} (требуется минимум 200 дней истории).")
    st.stop()

df, c_price, c_z, q_rating, e_rating, (low_thr, upper_thr), c_rsi, c_vol_z, drawdown_pct, rel_strength, current_ath, dollar_vol, strength_s, recovery_score, opportunity_score, cycle_stage, liq_risk, reasons = calculate_two_factor_matrix(asset, raw, btc_global_df if is_c else None)

# ============================================================
# ДАШБОРД ВЫБРАННОГО АКТИВА
# ============================================================

st.header(f"📊 Спецификация актива: {asset}")

c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("💰 ТЕКУЩАЯ ЦЕНА", f"${c_price:,.4f}" if c_price < 1 else f"${c_price:,.2f}")
with c2: 
    q_color = "#22c55e" if q_rating >= 70 else "#eab308" if q_rating >= 45 else "#ef4444"
    st.markdown(f"""
        <div style='background: {q_color}10; padding: 10px; border-radius: 8px; border: 1px solid {q_color}30; text-align: center;'>
            <p style='color: gray; margin:0; font-size:10px; font-weight:bold;'>ИТОГОВОЕ КАЧЕСТВО</p>
            <p style='color: {q_color}; font-size:20px; font-weight:bold; margin:3px 0 0 0;'>{q_rating:.1f}</p>
        </div>
    """, unsafe_allow_html=True)
with c3: 
    e_color = "#22c55e" if e_rating >= 70 else "#eab308" if e_rating >= 45 else "#ef4444"
    st.markdown(f"""
        <div style='background: {e_color}10; padding: 10px; border-radius: 8px; border: 1px solid {e_color}30; text-align: center;'>
            <p style='color: gray; margin:0; font-size:10px; font-weight:bold;'>ПОТЕНЦИАЛ ВХОДА</p>
            <p style='color: {e_color}; font-size:20px; font-weight:bold; margin:3px 0 0 0;'>{e_rating:.1f}</p>
        </div>
    """, unsafe_allow_html=True)
with c4: st.metric("🔄 RECOVERY SCORE", f"{recovery_score:.1f}%")
with c5: st.metric("🎯 OPPORTUNITY SCORE", f"{opportunity_score:.1f}")

# ИНФОРМАЦИОННАЯ СТРОКА СТАТУСА
st.markdown(f"""
<div style='background: linear-gradient(135deg, #0b0f19 0%, #111827 100%); padding:12px; border-radius:10px; margin: 15px 0; border: 1px solid #1f2937;'>
    <p style='margin:0; color:#f3f4f6; font-size:13px; text-align: center;'>
        <b>Итоговое Качество (Альфа):</b> <span style='color:{q_color}; font-weight:bold;'>{q_rating:.1f}/100</span> &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Потенциал Входа (Тайминг):</b> <span style='color:{e_color}; font-weight:bold;'>{e_rating:.1f}/100</span> &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Ликвидность:</b> <span style='color:#38bdf8; font-weight:bold;'>${dollar_vol/1e6:.1f}M/день</span> &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Стадия цикла:</b> <span style='color:#a855f7; font-weight:bold;'>{cycle_stage}</span> &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Режим рынка:</b> <span style='font-weight:bold;'>{market_regime}</span>
    </p>
</div>
""", unsafe_allow_html=True)

if reasons:
    st.markdown("**Заметки риск-модели по активу:**")
    for item in reasons:
        st.markdown(f"- {item[0]}")

# ============================================================
# 5. ГРАФИК
# ============================================================
st.markdown("---")
df_chart = df.tail(500).copy()

def get_color(z, lower, upper):
    if z <= lower: return "#00ff66"
    elif z <= -0.5: return "#bfff00"
    elif z <= 0.5: return "#e5e7eb"
    elif z <= upper: return "#ff5500"
    else: return "#ff0055"

fig = go.Figure()
for i in range(len(df_chart) - 1):
    color = get_color(df_chart["z_score"].iloc[i], low_thr, upper_thr)
    fig.add_trace(go.Scatter(
        x=[df_chart["date"].iloc[i], df_chart["date"].iloc[i+1]],
        y=[df_chart["close"].iloc[i], df_chart["close"].iloc[i+1]],
        mode="lines", line=dict(color=color, width=3), showlegend=False
    ))

fig.update_layout(height=380, template="plotly_dark", xaxis_title="", yaxis_title="Цена (USD)", font=dict(family="Times New Roman", size=12))
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 6. СКВОЗНАЯ ИНВЕСТИЦИОННАЯ МАТРИЦА (ПОЛНОЕ СООТВЕТСТВИЕ ТЗ)
# ============================================================
st.markdown("---")
st.subheader("📋 ДВУХФАКТОРНАЯ ИНВЕСТИЦИОННАЯ МАТРИЦА УПРАВЛЕНИЯ КАПИТАЛОМ")

@st.cache_data(ttl=900)
def build_summary_table(regime):
    all_assets = {**{c: "Криптовалюта" for c in CRYPTO_LIST}, **{s: "Акция" for s in STOCK_LIST}}
    rows = []
    
    btc_df = load_crypto_data("BTC", days=550)
    
    for symbol, atype in all_assets.items():
        df_t = load_crypto_data(symbol) if atype == "Криптовалюта" else load_stock_data(symbol)
        if df_t is None or len(df_t) < 200:
            continue
            
        res = calculate_two_factor_matrix(symbol, df_t, btc_df if atype == "Криптовалюта" else None)
        if res[0] is None:
            continue
            
        (_, _, _, quality_r, entry_r, _, _, _, ddown, rel_str, _, dollar_vol, _, recovery_s, opportunity_s, cycle_stage, _, _) = res
        
        btc_strength_text = f"{rel_str:+.1f}%" if atype == "Криптовалюта" else "N/A"
        
        # Пересборка структуры таблицы СТРОГО под требования профессионального управления:
        rows.append({
            "Символ": symbol,
            "Итоговое качество": round(quality_r, 1),
            "Потенциал входа": round(entry_r, 1),
            "Стадия цикла": cycle_stage,
            "Просадка": f"{ddown:.1f}%",
            "RS BTC": btc_strength_text,
            "Ликвидность": f"${dollar_vol / 1e6:.1f}M",
            "Recovery": f"{recovery_s:.1f}%",
            "Opportunity": round(opportunity_s, 1)
        })
    return rows

with st.spinner("Расчет и линеаризация портфельных весов..."):
    summary = build_summary_table(market_regime)

if summary:
    df_summary = pd.DataFrame(summary).sort_values(by="Потенциал входа", ascending=False)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
    st.caption("💡 Стратегия: Покупаем активы с высоким показателем «Итоговое качество», у которых прямо сейчас максимальный «Потенциал входа».")

# ============================================================
# 7. ПОДВАЛ
# ============================================================
moscow_tz = timezone(timedelta(hours=3))
moscow_time = datetime.now(moscow_tz)

st.markdown("---")
st.caption(f"📅 Синхронизация данных: {moscow_time.strftime('%Y-%m-%d %H:%M:%S')} (МСК) | Раздельные контуры управления: Альфа (Quality) и Тайминг (Entry) | Интегрирован Opportunity Индекс")
