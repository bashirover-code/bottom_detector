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

st.set_page_config(page_title="Инвестиционная матрица", layout="wide")

st.markdown("""
    <meta http-equiv="refresh" content="900">
    <style>
        html, body, [class*="css"], .stMarkdown, .stMetric, .stDataFrame,
        .stButton, .stSelectbox, .stRadio, .stCaption, h1, h2, h3, h4, p, div {
            font-family: 'Times New Roman', Times, serif !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🏛️ Инвестиционная матрица")

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

# ============================================================
# 2. ЗАГРУЗКА ДАННЫХ В РЕАЛЬНОМ ВРЕМЕНИ
# ============================================================

@st.cache_data(ttl=900)
def load_crypto_data(symbol, days=730): # Увеличили глубину для расчета 180д форварда
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
def load_stock_data(symbol, days=730):
    try:
        s = yf.Ticker(symbol)
        df = s.history(period=f"{days}d")
        if df is not None and not df.empty:
            df = df.reset_index().rename(columns={"Date": "date", "Close": "close", "Volume": "volume"})
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            return df[["date", "close", "volume"]].sort_values("date").reset_index(drop=True)
    except:
        return None

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
# 4. ДВУХФАКТОРНАЯ МОДЕЛЬ (МЯГКОЕ ОГРАНИЧЕНИЕ КАЧЕСТВА)
# ============================================================

def calculate_two_factor_matrix(symbol, df, btc_df=None):
    if df is None or len(df) < 200:
        return (None,) * 20
        
    df = df.copy()
    
    df["ma90"] = df["close"].rolling(window=90, min_periods=30).mean()
    df["std90"] = df["close"].rolling(window=90, min_periods=30).std()
    df["z_score"] = (df["close"] - df["ma90"]) / (df["std90"] + 1e-10)
    df["rsi"] = calculate_rsi(df, 14)
    df["ma200"] = df["close"].rolling(window=200, min_periods=50).mean()
    df["dollar_volume"] = df["close"] * df["volume"]
    
    target_cols = ["ma90", "std90", "z_score", "rsi", "ma200", "dollar_volume"]
    df[target_cols] = df[target_cols].bfill().ffill().fillna(0)
    
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

    if avg_dollar_volume > 50_000_000:
        quality_vol_score, liq_risk = 100, "Низкий"
    elif avg_dollar_volume > 5_000_000:
        quality_vol_score, liq_risk = 70, "Средний"
    else:
        quality_vol_score, liq_risk = 25, "Высокий"

    rs30 = calculate_single_rs(df, btc_df, 30)
    rs90 = calculate_single_rs(df, btc_df, 90)
    rs180 = calculate_single_rs(df, btc_df, 180)
    relative_strength = (rs30 * 0.2) + (rs90 * 0.3) + (rs180 * 0.5)

    if relative_strength > 40: rs_score = 100
    elif relative_strength > 15: rs_score = 80
    elif relative_strength > 0: rs_score = 65
    elif relative_strength > -20: rs_score = 40
    else: rs_score = 10
    
    cycle_low = df["close"].tail(365).min()
    recovery_score = (((current_price - cycle_low) / (current_ath - cycle_low)) * 100) if (current_ath - cycle_low) > 0 else 0.0

    strength_score = (quality_vol_score * 0.3) + (rs_score * 0.5) + (recovery_score * 0.2)
    strength_score = max(0, min(strength_score, 100))

    structure_raw = 0
    if current_price > c_ma90: structure_raw += 10
    if current_price > c_ma200: structure_raw += 15
    if c_ma200 > ma200_30_days_ago: structure_raw += 10
    structure_score = int((structure_raw / 35) * 100) if structure_raw > 0 else 0

    bottom_score = 0
    if drawdown_pct <= -70: bottom_score += 40
    elif drawdown_pct <= -45: bottom_score += 20
    if c_rsi <= 38: bottom_score += 40
    elif c_rsi <= 46: bottom_score += 20
    if c_z <= low_t: bottom_score += 20
    bottom_score = min(bottom_score, 100)

    # Предварительное базовое качество
    quality_rating = (0.45 * quality_vol_score) + (0.35 * strength_score) + (0.20 * structure_score)
    
    # Премия лидера
    if relative_strength > 50:
        quality_rating += 15
    elif relative_strength > 30:
        quality_rating += 10

    # ============================================================
    # ВНЕДРЕНИЕ МЯГКОГО ОГРАНИЧЕНИЯ КАЧЕСТВА ПО ТЗ
    # ============================================================
    quality_rating = min(95.0, quality_rating)

    # Взвешенный Opportunity Score с участием Quality
    drawdown_score = min(100, abs(drawdown_pct) * 1.11)
    opportunity_score = (0.5 * drawdown_score) + (0.2 * rs_score) + (0.3 * quality_rating)
    opportunity_score = max(0, min(opportunity_score, 100))

    entry_rating = (0.60 * opportunity_score) + (0.40 * bottom_score)
    
    if structure_raw == 0:
        entry_rating *= 0.85

    if avg_dollar_volume < 1_000_000:
        quality_rating *= 0.7
        entry_rating *= 0.7

    # Фильтр мусора
    if quality_rating < 20:
        entry_rating *= 0.5
    elif quality_rating < 25:
        entry_rating *= 0.7

    quality_rating = max(0, min(quality_rating, 100))
    entry_rating = max(0, min(entry_rating, 100))

    # Зональные решения
    if quality_rating < 40:
        decision = "❌ Игнор"
    elif quality_rating > 70 and entry_rating > 60:
        decision = "⭐ Покупка"
    elif quality_rating > 70 and 40 <= entry_rating <= 60:
        decision = "👁 Наблюдение"
    elif 40 <= quality_rating <= 70 and entry_rating > 60:
        decision = "⚠ Спекуляция"
    else:
        decision = "⚪ Удержание"

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

    return df, current_price, c_z, quality_rating, entry_rating, (low_t, up_t), c_rsi, drawdown_pct, relative_strength, current_ath, avg_dollar_volume, strength_score, opportunity_score, drawdown_score, rs_score, cycle_stage, liq_risk, reasons_checklist, decision

# ============================================================
# ГЕНЕРАЦИЯ СКВОЗНОЙ БАЗЫ ДАННЫХ (ТЕКУЩЕЕ СОСТОЯНИЕ)
# ============================================================

@st.cache_data(ttl=900)
def get_complete_market_state():
    all_assets = {**{c: "Криптовалюта" for c in CRYPTO_LIST}, **{s: "Акция" for s in STOCK_LIST}}
    rows = []
    btc_df = load_crypto_data("BTC", days=550)
    
    for symbol, atype in all_assets.items():
        df_t = load_crypto_data(symbol) if atype == "Криптовалюта" else load_stock_data(symbol)
        if df_t is None or len(df_t) < 200: continue
        res = calculate_two_factor_matrix(symbol, df_t, btc_df if atype == "Криптовалюта" else None)
        if res[0] is None: continue
        
        rows.append({
            "Символ": symbol, "Тип": atype, "Качество": res[3], "Потенциал": res[4], "Решение": res[18],
            "Стадия": res[15], "Просадка": res[7], "Сила": res[8], "Объем": res[10], "Opportunity": res[12]
        })
    return pd.DataFrame(rows)

with st.spinner("Синхронизация глобального слепка рынка..."):
    df_global_state = get_complete_market_state()

# ============================================================
# ЭТАП №3: РАСЧЕТ ИНДЕКСА КАЧЕСТВА РЫНКА
# ============================================================
if not df_global_state.empty:
    top10_q = df_global_state.sort_values(by="Качество", ascending=False).head(10)["Качество"].mean()
    
    if top10_q >= 80: regime_text, regime_emoji = "Бычий рынок 🐂", "🟢"
    elif top10_q >= 60: regime_text, regime_emoji = "Рост 📈", "🍏"
    elif top10_q >= 40: regime_text, regime_emoji = "Нейтрально ⚖️", "🟡"
    elif top10_q >= 20: regime_text, regime_emoji = "Медвежий рынок 🐻", "🔴"
    else: regime_text, regime_emoji = "Капитуляция 💥", "💀"
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%); padding: 18px; border-radius: 12px; border: 1px solid #4338ca; text-align: center; margin-bottom: 20px;'>
        <span style='color: #a5b4fc; font-size: 13px; font-weight: bold; letter-spacing: 1px;'>🏛️ ИНДЕКС КАЧЕСТВА РЫНКА (MEAN TOP-10 QUALITY)</span>
        <h1 style='color: #ffffff; margin: 5px 0; font-size: 42px;'>{top10_q:.1f} <span style='font-size: 24px; color: #64748b;'>/ 100</span></h1>
        <p style='margin: 0; font-size: 16px; color: #f8fafc;'>Статус макро-структуры: <b>{regime_emoji} {regime_text}</b></p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# ЭТАП №2: СТАТИСТИКА РЕШЕНИЙ В СИДБАРЕ
# ============================================================
with st.sidebar:
    st.header("⚙️ УПРАВЛЕНИЕ")
    market = st.radio("Сектор рынка", ["Криптовалюты", "Фондовый рынок"])
    asset = st.selectbox("Выбор актива", CRYPTO_LIST if market == "Криптовалюты" else STOCK_LIST)
    
    st.markdown("---")
    st.subheader("📊 СТАТИСТИКА РЕШЕНИЙ")
    if not df_global_state.empty:
        counts = df_global_state["Решение"].value_counts()
        c_buy = counts.get("⭐ Покупка", 0)
        c_spec = counts.get("⚠ Спекуляция", 0)
        c_hold = counts.get("⚪ Удержание", 0)
        c_ign = counts.get("❌ Игнор", 0)
        
        st.markdown(f"**⭐ Покупка:** `{c_buy} акт.`")
        st.markdown(f"**⚠ Спекуляция:** `{c_spec} акт.`")
        st.markdown(f"**⚪ Удержание:** `{c_hold} акт.`")
        st.markdown(f"**❌ Игнор:** `{c_ign} акт.`")
        
        # Индикатор климата
        if c_buy > 5: climate = "🟢 Рынок хороший. Время искать лонги."
        elif (c_buy + c_spec) > 3: climate = "🟡 Рынок средний. Работаем точечно."
        else: climate = "🔴 Рынок плохой. Максимальная осторожность."
        st.caption(climate)

# РЕНДЕРИНГ КАРТОЧКИ ВЫБРАННОГО АКТИВА
is_c = asset in CRYPTO_LIST
btc_global_df = load_crypto_data("BTC", days=550)
raw_asset = load_crypto_data(asset) if is_c else load_stock_data(asset)

if raw_asset is not None and len(raw_asset) >= 200:
    df_a, c_price, c_z, q_rating, e_rating, _, _, drawdown_pct, rel_strength, _, _, _, opportunity_score, dd_s, rs_s, cycle_stage, _, _, main_decision = calculate_two_factor_matrix(asset, raw_asset, btc_global_df if is_c else None)
    
    st.subheader(f"📊 актив: {asset}")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("💰 ТЕКУЩАЯ ЦЕНА", f"${c_price:,.4f}" if c_price < 1 else f"${c_price:,.2f}")
    with c2: st.metric("🧬 ИТОГОВОЕ КАЧЕСТВО", f"{q_rating:.1f}")
    with c3: st.metric("🎯 ПОТЕНЦИАЛ ВХОДА", f"{e_rating:.1f}")
    with c4: st.metric("⚖️ СИСТЕМНОЕ РЕШЕНИЕ", main_decision)

# ============================================================
# СКВОЗНАЯ ТАБЛИЦА ТЕКУЩЕГО РАНЖИРОВАНИЯ
# ============================================================
st.markdown("---")
st.subheader("📋 СКВОЗНАЯ ТАБЛИЦА РАНЖИРОВАНИЯ АКТИВОВ")
if not df_global_state.empty:
    df_view = df_global_state.sort_values(by="Потенциал", ascending=False).copy()
    df_view.columns = ["Символ", "Тип", "Итоговое качество", "Потенциал входа", "Решение", "Стадия цикла", "Просадка", "RS BTC", "Ликвидность", "Opportunity"]
    
    # Форматирование отображения
    df_view["Просадка"] = df_view["Просадка"].map(lambda x: f"{x:.1f}%")
    df_view["RS BTC"] = df_view["RS BTC"].map(lambda x: f"{x:+.1f}%" if isinstance(x, (int, float)) else "N/A")
    df_view["Ликвидность"] = df_view["Ликвидность"].map(lambda x: f"${x / 1e6:.1f}M")
    df_view["Итоговое качество"] = df_view["Итоговое качество"].round(1)
    df_view["Потенциал входа"] = df_view["Потенциал входа"].round(1)
    df_view["Opportunity"] = df_view["Opportunity"].round(1)
    
    st.dataframe(df_view.drop(columns=["Тип"]), use_container_width=True, hide_index=True)

# ============================================================
# ЭТАП №1: ТАБЛИЦА ЭФФЕКТИВНОСТИ СИГНАЛОВ (ИСТОРИЧЕСКИЙ АУДИТ)
# ============================================================
st.markdown("---")
with st.expander("📈 ИСТОРИЧЕСКИЙ АУДИТ И ЭФФЕКТИВНОСТЬ СИГНАЛОВ (WALK-FORWARD ПРОВЕРКА)"):
    st.markdown("##### Моделирование слепых сигналов со смещением на 180 дней назад")
    
    @st.cache_data(ttl=3600)
    def run_historical_audit():
        all_assets = {**{c: "Криптовалюта" for c in CRYPTO_LIST}, **{s: "Акция" for s in STOCK_LIST}}
        audit_rows = []
        btc_full = load_crypto_data("BTC", days=730)
        
        for symbol, atype in all_assets.items():
            df_full = load_crypto_data(symbol) if atype == "Криптовалюта" else load_stock_data(symbol)
            if df_full is None or len(df_full) < 380: continue
            
            # Точка генерации исторического сигнала (180 дней назад от конца датасета)
            total_days = len(df_full)
            t_idx = total_days - 180
            
            if t_idx < 200: continue
            
            # Срез данных "на тот момент"
            df_past = df_full.iloc[:t_idx].reset_index(drop=True)
            past_date = df_past["date"].iloc[-1]
            
            # Срез BTC на тот момент
            btc_past = btc_full[btc_full["date"] <= past_date].reset_index(drop=True) if btc_full is not None else None
            
            # Считаем матрицу "вслепую" в прошлом
            res_past = calculate_two_factor_matrix(symbol, df_past, btc_past)
            if res_past[0] is None: continue
            
            past_decision = res_past[18]
            entry_price = res_past[1]
            
            # Цены в будущем (через 30, 90, 180 дней)
            price_30 = df_full["close"].iloc[min(t_idx + 30, total_days - 1)]
            price_90 = df_full["close"].iloc[min(t_idx + 90, total_days - 1)]
            price_180 = df_full["close"].iloc[-1] # ровно сегодняшний день
            
            perf_30 = (price_30 / entry_price - 1) * 100
            perf_90 = (price_90 / entry_price - 1) * 100
            perf_180 = (price_180 / entry_price - 1) * 100
            
            audit_rows.append({
                "Дата": past_date.strftime("%Y-%m-%d"),
                "Актив": symbol,
                "Решение": past_decision,
                "Цена входа": round(entry_price, 4),
                "Через 30 дней": perf_30,
                "Через 90 дней": perf_90,
                "Через 180 дней": perf_180
            })
        return pd.DataFrame(audit_rows)

    df_audit = run_historical_audit()
    
    if not df_audit.empty:
        df_audit_view = df_audit.copy()
        # Красивый вывод процентов
        for c in ["Через 30 дней", "Через 90 дней", "Через 180 дней"]:
            df_audit_view[c] = df_audit_view[c].map(lambda x: f"{x:+.1f}%")
            
        st.dataframe(df_audit_view, use_container_width=True, hide_index=True)
        
        # РАСЧЕТ СРЕДНЕЙ ЭФФЕКТИВНОСТИ СИГНАЛОВ
        st.markdown("##### 📊 МАТЕМАТИЧЕСКОЕ МАТОЖИДАНИЕ СИСТЕМЫ (Средняя доходность за 180 дней)")
        
        stat_cols = st.columns(4)
        decisions_to_check = [
            ("⭐ Покупка", stat_cols[0], "🟢"),
            ("⚠ Спекуляция", stat_cols[1], "🟡"),
            ("⚪ Удержание", stat_cols[2], "⚪"),
            ("❌ Игнор", stat_cols[3], "🔴")
        ]
        
        for dec_title, col, emoji in decisions_to_check:
            sub_df = df_audit[df_audit["Решение"] == dec_title]
            with col:
                if not sub_df.empty:
                    mean_perf = sub_df["Через 180 дней"].mean()
                    st.metric(f"{emoji} {dec_title}", f"{mean_perf:+.1f}%", f"Выборок: {len(sub_df)}")
                else:
                    st.metric(f"{emoji} {dec_title}", "N/A", "Нет сигналов")
    else:
        st.info("Недостаточно данных для проведения глубокого исторического форвард-теста.")

# ============================================================
# 7. ПОДВАЛ
# ============================================================
moscow_time = datetime.now(timezone(timedelta(hours=3)))
st.markdown("---")
st.caption(f"📅 Синхронизация данных: {moscow_time.strftime('%Y-%m-%d %H:%M:%S')} (МСК) | Внедрен лимит Качества (95), Индекс Рынка и Форвард-таблица эффективности сигналов.")
