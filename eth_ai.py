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

st.set_page_config(page_title="Институциональная Двухфакторная Матрица", layout="wide")

st.markdown("""
    <meta http-equiv="refresh" content="900">
    <style>
        html, body, [class*="css"], .stMarkdown, .stMetric, .stDataFrame,
        .stButton, .stSelectbox, .stRadio, .stCaption, h1, h2, h3, h4, p, div {
            font-family: 'Times New Roman', Times, serif !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🏛️ Двухфакторная Инвестиционная Матрица v3")

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
# 2. ЗАГРУЗКА ДАННЫХ
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
        return "🟢 Бычий" if btc_price > btc_ma200 else "🔴 Медвежий"
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
    if len(z_scores) < 30: return -1.8, 1.5
    low = np.nanpercentile(z_scores, 6)
    high = np.nanpercentile(z_scores, 94)
    return max(-3.2, min(-1.0, low)), min(3.2, max(0.6, high))

def calculate_single_rs(df, btc_df, lookup_days):
    if btc_df is None or len(df) < lookup_days or len(btc_df) < lookup_days:
        return 0.0
    df_temp, btc_temp = df.copy(), btc_df.copy()
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
# 4. ДВУХФАКТОРНАЯ МОДЕЛЬ С УМНЫМ OPPORTUNITY SCORE
# ============================================================

def calculate_two_factor_matrix(symbol, df, btc_df=None):
    if df is None or len(df) < 200:
        return (None,) * 20
        
    df = df.copy()
    
    # 0. Инженерный расчет метрик тренда
    df["ma90"] = df["close"].rolling(window=90, min_periods=30).mean()
    df["std90"] = df["close"].rolling(window=90, min_periods=30).std()
    df["z_score"] = (df["close"] - df["ma90"]) / (df["std90"] + 1e-10)
    df["rsi"] = calculate_rsi(df, 14)
    df["ma200"] = df["close"].rolling(window=200, min_periods=50).mean()
    df["dollar_volume"] = df["close"] * df["volume"]
    
    df[list(df.columns[-6:])] = df[list(df.columns[-6:])].bfill().ffill().fillna(0)
    
    current_price = df["close"].iloc[-1]
    c_z = df["z_score"].iloc[-1]
    c_rsi = df["rsi"].iloc[-1]
    c_ma200 = df["ma200"].iloc[-1]
    c_ma90 = df["ma90"].iloc[-1]
    ma200_30_days_ago = df["ma200"].iloc[-30] if len(df) >= 30 else c_ma200
    
    current_ath = df["close"].max()
    drawdown_pct = ((current_price - current_ath) / current_ath * 100) if current_ath > 0 else 0
    avg_dollar_volume = df["dollar_volume"].tail(30).mean()
    low_t, up_t = get_adaptive_thresholds(df["z_score"].values)
    
    reasons_checklist = []

    # ФАКТОР 1: QUALITY VOLUME
    if avg_dollar_volume > 50_000_000:
        quality_vol_score, liq_risk = 100, "Низкий"
    elif avg_dollar_volume > 5_000_000:
        quality_vol_score, liq_risk = 70, "Средний"
    else:
        quality_vol_score, liq_risk = 25, "Высокий"

    # ФАКТОР 2: ЧИСТАЯ СИЛА К ВТС (Мультитаймфрейм)
    rs30 = calculate_single_rs(df, btc_df, 30)
    rs90 = calculate_single_rs(df, btc_df, 90)
    rs180 = calculate_single_rs(df, btc_df, 180)
    relative_strength = (rs30 * 0.2) + (rs90 * 0.3) + (rs180 * 0.5)

    if relative_strength > 40: rs_score = 100
    elif relative_strength > 15: rs_score = 80
    elif relative_strength > 0: rs_score = 65
    elif relative_strength > -20: rs_score = 40
    else: rs_score = 10
    
    # Расчет Recovery Score от годового дна
    cycle_low = df["close"].tail(365).min()
    recovery_score = (((current_price - cycle_low) / (current_ath - cycle_low)) * 100) if (current_ath - cycle_low) > 0 else 0.0

    # Динамическая Сила (Strength Score) во избежание ложного насыщения
    strength_score = (quality_vol_score * 0.3) + (rs_score * 0.5) + (recovery_score * 0.2)
    strength_score = max(0, min(strength_score, 100))

    # ФАКТОР 3: STRUCTURE
    structure_raw = 0
    if current_price > c_ma90: structure_raw += 10
    if current_price > c_ma200: structure_raw += 15
    if c_ma200 > ma200_30_days_ago: structure_raw += 10
    structure_score = int((structure_raw / 35) * 100) if structure_raw > 0 else 0

    # ФАКТОР 4: BOTTOM SCORE (Чистая глубина локального страха)
    bottom_score = 0
    if drawdown_pct <= -70: bottom_score += 40
    elif drawdown_pct <= -45: bottom_score += 20
    if c_rsi <= 38: bottom_score += 40
    elif c_rsi <= 46: bottom_score += 20
    if c_z <= low_t: bottom_score += 20
    bottom_score = min(bottom_score, 100)

    # ============================================================
    # МАТЕМАТИЧЕСКАЯ РЕАЛИЗАЦИЯ УМНОГО OPPORTUNITY SCORE ПО ТЗ:
    # ============================================================
    
    # 1. Drawdown Score (40%): Линейная шкала от просадки. -90% просадки -> 100 баллов
    drawdown_score = min(100, abs(drawdown_pct) * 1.11)
    
    # 2. RS Recovery Score (30%): Возможность взять силу по хорошей цене. 
    # Если актив сильный (rs_score высок), но ещё не улетел в космос по общему Recovery
    rs_recovery_score = max(0, min(100, rs_score * (1 - (recovery_score / 100))))
    
    # 3. Accumulation Score (30%): Сколько дней за последние 60 дней актив спал в накоплении (Z-Score < 0)
    recent_z = df["z_score"].tail(60).values
    accumulation_days = np.sum(recent_z < 0.2)
    accumulation_score = (accumulation_days / 60) * 100
    
    # ИТОГОВАЯ СБОРКА УМНОГО OPPORTUNITY SCORE
    opportunity_score = (0.4 * drawdown_score) + (0.3 * rs_recovery_score) + (0.3 * accumulation_score)
    opportunity_score = max(0, min(opportunity_score, 100))

    # ============================================================
    # РАСЧЕТ ИТОГОВЫХ ДВУХФАКТОРНЫХ РЕЙТИНГОВ
    # ============================================================
    
    # А. Итоговое качество (Альфа-сила актива)
    quality_rating = (0.45 * quality_vol_score) + (0.35 * strength_score) + (0.20 * structure_score)
    
    # Б. Потенциал входа (Своевременность / Смарт-тайминг)
    entry_rating = (0.60 * opportunity_score) + (0.40 * bottom_score)
    
    # Защитный коэффициент даунтренда для Потенциала входа
    if structure_raw == 0:
        entry_rating *= 0.85
        reasons_checklist.append(("⚠️ Потенциал ограничен: Покупка ведется против сильного нисходящего макро-тренда", False))

    # Жесткий фильтр мусорности
    if avg_dollar_volume < 1_000_000:
        quality_rating *= 0.7
        entry_rating *= 0.7
        reasons_checklist.append(("🚨 ФИЛЬТР МУСОРНОСТИ: Долларовый оборот < $1М. Штраф ×0.7 ко всем метрикам", False))

    quality_rating = max(0, min(quality_rating, 100))
    entry_rating = max(0, min(entry_rating, 100))

    # 6-ступенчатый классификатор стадий цикла
    if c_rsi < 35 and c_z < -1.5:
        cycle_stage = "Капитуляция"
    elif current_price > c_ma90 and current_price > c_ma200:
        if c_rsi > 72 or c_z > up_t: cycle_stage = "Перегрев"
        elif c_ma200 > ma200_30_days_ago and c_rsi > 58: cycle_stage = "Тренд"
        else: cycle_stage = "Ранний тренд"
    elif current_price > c_ma90 and (recovery_score > 15 or c_z > 0.5):
        cycle_stage = "Разворот"
    else:
        cycle_stage = "Накопление"

    return df, current_price, c_z, quality_rating, entry_rating, (low_t, up_t), c_rsi, drawdown_pct, relative_strength, current_ath, avg_dollar_volume, strength_score, recovery_score, opportunity_score, drawdown_score, rs_recovery_score, accumulation_score, cycle_stage, liq_risk, reasons_checklist

# ============================================================
# ИНТЕРФЕЙС И ОПЕРАЦИОННАЯ СРЕДА
# ============================================================

with st.sidebar:
    st.header("⚙️ УПРАВЛЕНИЕ МАТРИЦЕЙ")
    st.markdown("---")
    market = st.radio("Сектор рынка", ["Криптовалюты", "Фондовый рынок"])
    asset = st.selectbox("Выбор актива", CRYPTO_LIST if market == "Криптовалюты" else STOCK_LIST)
    st.markdown("---")
    st.caption("🏛️ **Умный Квантовый Скоринг**")
    st.caption("• Интегрирован многофакторный Opportunity Score:\n  [0.4*Drawdown + 0.3*RS_Rec + 0.3*Accum]")
    st.caption("• Полная изоляция Альфы и инвестиционного Тайминга.")

with st.spinner("Расчет макро-показателей рынка..."):
    market_regime = get_market_regime()
    btc_global_df = load_crypto_data("BTC", days=550)

is_c = asset in CRYPTO_LIST
with st.spinner(f"Анализ весов для {asset}..."):
    raw = load_crypto_data(asset) if is_c else load_stock_data(asset)

if raw is None or len(raw) < 200:
    st.error(f"❌ Недостаточно данных для анализа {asset}.")
    st.stop()

df, c_price, c_z, q_rating, e_rating, (low_thr, upper_thr), c_rsi, drawdown_pct, rel_strength, current_ath, dollar_vol, strength_s, recovery_score, opportunity_score, dd_s, rs_rec_s, accum_s, cycle_stage, liq_risk, reasons = calculate_two_factor_matrix(asset, raw, btc_global_df if is_c else None)

# ============================================================
# ВЫВОД ИНФОРМАЦИОННЫХ ПАНЕЛЕЙ И КАРТОЧЕК
# ============================================================

st.header(f"📊 Спецификация актива: {asset}")

c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("💰 ТЕКУЩАЯ ЦЕНА", f"${c_price:,.4f}" if c_price < 1 else f"${c_price:,.2f}")
with c2: 
    q_color = "#22c55e" if q_rating >= 70 else "#eab308" if q_rating >= 45 else "#ef4444"
    st.markdown(f"<div style='background: {q_color}10; padding: 10px; border-radius: 8px; border: 1px solid {q_color}30; text-align: center;'><p style='color: gray; margin:0; font-size:10px; font-weight:bold;'>ИТОГОВОЕ КАЧЕСТВО</p><p style='color: {q_color}; font-size:20px; font-weight:bold; margin:3px 0 0 0;'>{q_rating:.1f}</p></div>", unsafe_allow_html=True)
with c3: 
    e_color = "#22c55e" if e_rating >= 70 else "#eab308" if e_rating >= 45 else "#ef4444"
    st.markdown(f"<div style='background: {e_color}10; padding: 10px; border-radius: 8px; border: 1px solid {e_color}30; text-align: center;'><p style='color: gray; margin:0; font-size:10px; font-weight:bold;'>ПОТЕНЦИАЛ ВХОДА</p><p style='color: {e_color}; font-size:20px; font-weight:bold; margin:3px 0 0 0;'>{e_rating:.1f}</p></div>", unsafe_allow_html=True)
with c4: st.metric("🎯 SMART OPPORTUNITY", f"{opportunity_score:.1f}")
with c5: st.metric("🔄 RECOVERY SCORE", f"{recovery_score:.1f}%")

st.markdown(f"""
<div style='background: linear-gradient(135deg, #0b0f19 0%, #111827 100%); padding:12px; border-radius:10px; margin: 15px 0; border: 1px solid #1f2937;'>
    <p style='margin:0; color:#f3f4f6; font-size:13px; text-align: center;'>
        <b>Декомпозиция Opportunity Score:</b> &nbsp;&nbsp;
        Просадка (40%): <span style='color:#38bdf8; font-weight:bold;'>{dd_s:.1f}</span> &nbsp;&nbsp;|&nbsp;&nbsp;
        RS Отскок (30%): <span style='color:#fb923c; font-weight:bold;'>{rs_rec_s:.1f}</span> &nbsp;&nbsp;|&nbsp;&nbsp;
        Дни Накопления (30%): <span style='color:#a855f7; font-weight:bold;'>{accum_s:.1f}</span> &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Стадия:</b> <span style='font-weight:bold; color:#eab308;'>{cycle_stage}</span>
    </p>
</div>
""", unsafe_allow_html=True)

if reasons:
    for item in reasons: st.markdown(f"- {item[0]}")

# ============================================================
# 5. ГРАФИК ТРЕНДА
# ============================================================
st.markdown("---")
df_chart = df.tail(500).copy()
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_chart["date"], y=df_chart["close"], mode="lines", line=dict(color="#3b82f6", width=2), name="Цена"))
fig.update_layout(height=350, template="plotly_dark", xaxis_title="", yaxis_title="Цена (USD)", font=dict(family="Times New Roman", size=12))
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 6. СКВОЗНАЯ ИНВЕСТИЦИОННАЯ МАТРИЦА (ПОЛНОЕ СООТВЕТСТВИЕ ТЗ)
# ============================================================
st.markdown("---")
st.subheader("📋 СКВОЗНАЯ ИНВЕСТИЦИОННАЯ МАТРИЦА С УМНЫМ OPPORTUNITY SCORE")

@st.cache_data(ttl=900)
def build_summary_table(regime):
    all_assets = {**{c: "Криптовалюта" for c in CRYPTO_LIST}, **{s: "Акция" for s in STOCK_LIST}}
    rows = []
    btc_df = load_crypto_data("BTC", days=550)
    
    for symbol, atype in all_assets.items():
        df_t = load_crypto_data(symbol) if atype == "Криптовалюта" else load_stock_data(symbol)
        if df_t is None or len(df_t) < 200: continue
            
        res = calculate_two_factor_matrix(symbol, df_t, btc_df if atype == "Криптовалюта" else None)
        if res[0] is None: continue
            
        (_, _, _, quality_r, entry_r, _, _, ddown, rel_str, _, dollar_vol, _, recovery_s, opportunity_s, _, _, _, cycle_stage, _, _) = res
        
        rows.append({
            "Символ": symbol,
            "Итоговое качество": round(quality_rating, 1) if symbol == asset else round(quality_r, 1),
            "Потенциал входа": round(entry_rating, 1) if symbol == asset else round(entry_r, 1),
            "Стадия цикла": cycle_stage,
            "Просадка": f"{ddown:.1f}%",
            "RS BTC": f"{rel_str:+.1f}%" if atype == "Криптовалюта" else "N/A",
            "Ликвидность": f"${dollar_vol / 1e6:.1f}M",
            "Recovery": f"{recovery_s:.1f}%",
            "Opportunity": round(opportunity_score, 1) if symbol == asset else round(opportunity_s, 1)
        })
    return rows

with st.spinner("Генерация многофакторной матрицы скоринга..."):
    summary = build_summary_table(market_regime)

if summary:
    df_summary = pd.DataFrame(summary).sort_values(by="Потенциал входа", ascending=False)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
    st.caption("💡 Матрица ранжирована по Потенциалу входа. Благодаря многофакторному Opportunity Score, в топ поднимаются фундаментально сильные активы, прошедшие длительную фазу накопления.")

# ============================================================
# 7. ПОДВАЛ
# ============================================================
moscow_time = datetime.now(timezone(timedelta(hours=3)))
st.markdown("---")
st.caption(f"📅 Обновлено: {moscow_time.strftime('%Y-%m-%d %H:%M:%S')} (МСК) | Ядро: Многофакторный Opportunity Score [0.4*Drawdown + 0.3*RS_Rec + 0.3*Accum]")
