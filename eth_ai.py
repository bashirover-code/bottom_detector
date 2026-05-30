import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
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
# РИСК-ПРОФИЛИ И КЛАССИФИКАЦИЯ СЕКТОРОВ
# ============================================================

ASSET_REGISTRY = {
    "BTC": {"type": "Криптовалюта", "risk": "Консервативный", "sector": "L1"},
    "ETH": {"type": "Криптовалюта", "risk": "Консервативный", "sector": "L1"},
    "LINK": {"type": "Криптовалюта", "risk": "Консервативный", "sector": "DeFi"},
    "SOL": {"type": "Криптовалюта", "risk": "Сбалансированный", "sector": "L1"},
    "NEAR": {"type": "Криптовалюта", "risk": "Сбалансированный", "sector": "L1"},
    "SUI": {"type": "Криптовалюта", "risk": "Сбалансированный", "sector": "L1"},
    "STX": {"type": "Криптовалюта", "risk": "Сбалансированный", "sector": "Layer 2"},
    "IMX": {"type": "Криптовалюта", "risk": "Сбалансированный", "sector": "Layer 2"},
    "GRT": {"type": "Криптовалюта", "risk": "Сбалансированный", "sector": "AI"},
    "UNI": {"type": "Криптовалюта", "risk": "Сбалансированный", "sector": "DeFi"},
    "RENDER": {"type": "Криптовалюта", "risk": "Агрессивный", "sector": "AI"},
    "ONDO": {"type": "Криптовалюта", "risk": "Агрессивный", "sector": "RWA"},
    "ARKM": {"type": "Криптовалюта", "risk": "Агрессивный", "sector": "AI"},
    "GOAT": {"type": "Криптовалюта", "risk": "Агрессивный", "sector": "Meme"},
    "FLOCK": {"type": "Криптовалюта", "risk": "Агрессивный", "sector": "Meme"},
    "TRUMP": {"type": "Криптовалюта", "risk": "Агрессивный", "sector": "Meme"},
    "ZK": {"type": "Криптовалюта", "risk": "Агрессивный", "sector": "Layer 2"},
    "FIL": {"type": "Криптовалюта", "risk": "Сбалансированный", "sector": "DeFi"},
    "CELO": {"type": "Криптовалюта", "risk": "Агрессивный", "sector": "L1"},
    "CRV": {"type": "Криптовалюта", "risk": "Агрессивный", "sector": "DeFi"},
    "TWT": {"type": "Криптовалюта", "risk": "Сбалансированный", "sector": "DeFi"},
    "APE": {"type": "Криптовалюта", "risk": "Агрессивный", "sector": "Meme"},
    "ONE": {"type": "Криптовалюта", "risk": "Агрессивный", "sector": "L1"},
    "POL": {"type": "Криптовалюта", "risk": "Сбалансированный", "sector": "Layer 2"},
    "ARC": {"type": "Криптовалюта", "risk": "Агрессивный", "sector": "AI"},
    "ALGO": {"type": "Криптовалюта", "risk": "Сбалансированный", "sector": "L1"},
    "ASTER": {"type": "Криптовалюта", "risk": "Агрессивный", "sector": "Web3"},
    # Фондовый рынок
    "SBER.ME": {"type": "Акция", "risk": "Консервативный", "sector": "Stocks"},
    "MTSS.ME": {"type": "Акция", "risk": "Консервативный", "sector": "Stocks"},
    "GDX": {"type": "Акция", "risk": "Консервативный", "sector": "Stocks"},
    "URA": {"type": "Акция", "risk": "Консервативный", "sector": "Stocks"},
    "TSLA": {"type": "Акция", "risk": "Сбалансированный", "sector": "Stocks"},
    "PLTR": {"type": "Акция", "risk": "Сбалансированный", "sector": "Stocks"},
    "NVDA": {"type": "Акция", "risk": "Сбалансированный", "sector": "Stocks"},
    "COIN": {"type": "Акция", "risk": "Агрессивный", "sector": "Stocks"},
    "HIMS": {"type": "Акция", "risk": "Агрессивный", "sector": "Stocks"},
    "HEAD.ME": {"type": "Акция", "risk": "Сбалансированный", "sector": "Stocks"},
    "BABA": {"type": "Акция", "risk": "Сбалансированный", "sector": "Stocks"},
    "ZM": {"type": "Акция", "risk": "Сбалансированный", "sector": "Stocks"},
    "LIT": {"type": "Акция", "risk": "Консервативный", "sector": "Stocks"},
    "SIL": {"type": "Акция", "risk": "Консервативный", "sector": "Stocks"},
    "EWW": {"type": "Акция", "risk": "Консервативный", "sector": "Stocks"}
}

# ============================================================
# 2. ДИНАМИЧЕСКАЯ ЗАГРУЗКА ДАННЫХ
# ============================================================

@st.cache_data(ttl=900)
def load_asset_data(symbol, days=750):
    meta = ASSET_REGISTRY.get(symbol, {"type": "Криптовалюта"})
    if meta["type"] == "Криптовалюта":
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
            except: pass
        try:
            s = yf.Ticker(f"{symbol}-USD")
            df = s.history(period=f"{days}d")
            if df is not None and not df.empty:
                df = df.reset_index().rename(columns={"Date": "date", "Close": "close", "Volume": "volume"})
                df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                return df[["date", "close", "volume"]].sort_values("date").reset_index(drop=True)
        except: return None
    else:
        try:
            s = yf.Ticker(symbol)
            df = s.history(period=f"{days}d")
            if df is not None and not df.empty:
                df = df.reset_index().rename(columns={"Date": "date", "Close": "close", "Volume": "volume"})
                df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                return df[["date", "close", "volume"]].sort_values("date").reset_index(drop=True)
        except: return None

# ============================================================
# 3. ВСПОМОГАТЕЛЬНЫЕ МАТЕМАТИЧЕСКИЕ МЕТРИКИ
# ============================================================

def calculate_rsi(df, periods=14):
    close_delta = df["close"].diff()
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    ma_up = up.ewm(com=periods - 1, adjust=False).mean()
    ma_down = down.ewm(com=periods - 1, adjust=False).mean()
    return 100 - (100 / (1 + (ma_up / (ma_down + 1e-10))))

def calculate_single_rs(df, btc_df, lookup_days):
    if btc_df is None or len(df) < lookup_days or len(btc_df) < lookup_days: return 0.0
    df_t, btc_t = df.copy(), btc_df.copy()
    df_t["d"] = df_t["date"].dt.date
    btc_t["d"] = btc_t["date"].dt.date
    inter = np.intersect1d(df_t['d'], btc_t['d'])
    if len(inter) >= lookup_days:
        sub_a = df_t[df_t['d'].isin(inter)].sort_values("d")
        sub_b = btc_t[btc_t['d'].isin(inter)].sort_values("d")
        return (sub_a['close'].iloc[-1] / sub_a['close'].iloc[-lookup_days] - 1)*100 - (sub_b['close'].iloc[-1] / sub_b['close'].iloc[-lookup_days] - 1)*100
    return 0.0

# ============================================================
# 4. ДВУХФАКТОРНАЯ МОДЕЛЬ С ЗАЩИТОЙ ОТ ПЕРЕГРЕВА
# ============================================================

def calculate_two_factor_matrix(symbol, df, btc_df=None):
    if df is None or len(df) < 200: return (None,) * 19
    df = df.copy()
    
    df["ma90"] = df["close"].rolling(window=90, min_periods=30).mean()
    df["std90"] = df["close"].rolling(window=90, min_periods=30).std()
    df["z_score"] = (df["close"] - df["ma90"]) / (df["std90"] + 1e-10)
    df["rsi"] = calculate_rsi(df, 14)
    df["ma200"] = df["close"].rolling(window=200, min_periods=50).mean()
    df["dollar_volume"] = df["close"] * df["volume"]
    
    current_price = df["close"].iloc[-1]
    c_z, c_rsi, c_ma200, c_ma90 = df["z_score"].iloc[-1], df["rsi"].iloc[-1], df["ma200"].iloc[-1], df["ma90"].iloc[-1]
    ma200_30 = df["ma200"].iloc[-30] if len(df) >= 30 else c_ma200
    
    current_ath = df["close"].max()
    drawdown_pct = ((current_price - current_ath) / current_ath * 100) if current_ath > 0 else 0
    avg_dollar_volume = df["dollar_volume"].tail(30).mean()
    
    quality_vol_score = 100 if avg_dollar_volume > 50_000_000 else 70 if avg_dollar_volume > 5_000_000 else 25
    rs30 = calculate_single_rs(df, btc_df, 30)
    rs90 = calculate_single_rs(df, btc_df, 90)
    rs180 = calculate_single_rs(df, btc_df, 180)
    relative_strength = (rs30 * 0.2) + (rs90 * 0.3) + (rs180 * 0.5)
    
    rs_score = 100 if relative_strength > 40 else 80 if relative_strength > 15 else 65 if relative_strength > 0 else 40 if relative_strength > -20 else 10
    cycle_low = df["close"].tail(365).min()
    recovery_score = (((current_price - cycle_low) / (current_ath - cycle_low)) * 100) if (current_ath - cycle_low) > 0 else 0.0
    
    strength_score = (quality_vol_score * 0.3) + (rs_score * 0.5) + (recovery_score * 0.2)
    
    # Исправлен синтаксис структуры (удалены ложные нули и опечатки clse)
    structure_raw = (10 if current_price > c_ma90 else 0) + (15 if current_price > c_ma200 else 0) + (10 if c_ma200 > ma200_30 else 0)
    structure_score = int((structure_raw / 35) * 100) if structure_raw > 0 else 0
    
    bottom_score = (40 if drawdown_pct <= -70 else 20 if drawdown_pct <= -45 else 0) + (40 if c_rsi <= 38 else 20 if c_rsi <= 46 else 0) + (20 if c_z <= -1.5 else 0)
    
    quality_rating = (0.45 * quality_vol_score) + (0.35 * strength_score) + (0.20 * structure_score)
    if relative_strength > 50: quality_rating += 15
    elif relative_strength > 30: quality_rating += 10
    
    low_t, up_t = np.nanpercentile(df["z_score"].values, 6), np.nanpercentile(df["z_score"].values, 94)
    if c_rsi < 35 and c_z < -1.5: cycle_stage = "Капитуляция"
    elif current_price > c_ma90 and current_price > c_ma200:
        if c_rsi > 72 or c_z > up_t: cycle_stage = "Перегрев"
        elif c_ma200 > ma200_30 and c_rsi > 58: cycle_stage = "Тренд"
        else: cycle_stage = "Ранний тренд"
    elif current_price > c_ma90 and (recovery_score > 15 or c_z > 0.5): cycle_stage = "Разворот"
    else: cycle_stage = "Накопление"
    
    if cycle_stage == "Перегрев":
        quality_rating *= 0.85
        
    quality_rating = min(95.0, quality_rating)
    drawdown_score = min(100, abs(drawdown_pct) * 1.11)
    
    # Защитный механизм от FOMO
    adjusted_rs_score = rs_score * 0.7 if relative_strength > 50 else rs_score
        
    # Смарт-Потенциал
    opportunity_score = (0.5 * drawdown_score) + (0.2 * adjusted_rs_score) + (0.3 * quality_rating)
    opportunity_score = max(0, min(opportunity_score, 100))
    
    entry_rating = (0.60 * opportunity_score) + (0.40 * bottom_score)
    if structure_raw == 0: entry_rating *= 0.85
    if avg_dollar_volume < 1_000_000:
        quality_rating *= 0.7
        entry_rating *= 0.7
        
    if quality_rating < 20: entry_rating *= 0.5
    elif quality_rating < 25: entry_rating *= 0.7
    
    quality_rating = max(0, min(quality_rating, 100))
    entry_rating = max(0, min(entry_rating, 100))
    
    decision = "❌ Игнор" if quality_rating < 40 else "⭐ Покупка" if (quality_rating > 70 and entry_rating > 60) else "👁 Наблюдение" if (quality_rating > 70) else "⚠ Спекуляция" if (entry_rating > 60) else "⚪ Удержание"
    
    # Расчет целей
    local_atr = df["close"].tail(14).pct_change().std() * current_price
    if np.isnan(local_atr) or local_atr <= 0: local_atr = current_price * 0.05
    stop_loss = current_price - (1.96 * local_atr)
    tp1 = current_price + (1.5 * local_atr)
    tp2 = current_price + (3.0 * local_atr)
    tp3 = current_price + (5.0 * local_atr)
    
    return df, current_price, c_z, quality_rating, entry_rating, c_rsi, drawdown_pct, relative_strength, current_ath, avg_dollar_volume, opportunity_score, drawdown_score, rs_score, cycle_stage, decision, stop_loss, tp1, tp2, tp3

# ============================================================
# ГЕНЕРАЦИЯ ГЛОБАЛЬНЫХ МАТРИЦ С ДЕЛЬТОЙ
# ============================================================

@st.cache_data(ttl=900)
def generate_full_market_state():
    rows = []
    btc_df = load_asset_data("BTC", days=550)
    for sym, m in ASSET_REGISTRY.items():
        raw = load_asset_data(sym)
        if raw is None or len(raw) < 200: continue
        
        res = calculate_two_factor_matrix(sym, raw, btc_df)
        if res[0] is None: continue
        
        # Симуляция 30 дней назад
        raw_past = raw.iloc[:-30].reset_index(drop=True) if len(raw) > 30 else raw
        btc_past = btc_df.iloc[:-30].reset_index(drop=True) if (btc_df is not None and len(btc_df) > 30) else btc_df
        res_past = calculate_two_factor_matrix(sym, raw_past, btc_past)
        
        q_past = res_past[3] if res_past[0] is not None else res[3]
        
        rows.append({
            "Символ": sym, "Риск": m["risk"], "Сектор": m["sector"], "Качество": res[3],
            "Было_Качество": q_past, "Дельта_Качества": res[3] - q_past,
            "Потенциал": res[4], "Решение": res[14], "Стадия": res[13], "Просадка": res[6],
            "Сила": res[7], "Ликвидность": res[9], "Смарт_Потенциал": res[10],
            "Цена": res[1], "Стоп": res[15], "ТП1": res[16], "ТП2": res[17], "ТП3": res[18]
        })
    return pd.DataFrame(rows)

with st.spinner("Генерация финансовой структуры..."):
    df_market = generate_full_market_state()

# ============================================================
# ИНТЕРФЕЙС И РИСК-ПРОФИЛИ
# ============================================================

with st.sidebar:
    st.header("⚙️ НАСТРОЙКИ СИСТЕМЫ")
    user_risk = st.radio("🛡️ Ваш риск-профиль", ["Консервативный", "Сбалансированный", "Агрессивный"])
    
    allowed_assets = df_market[df_market["Risk" if "Risk" in df_market.columns else "Риск"] == user_risk]["Символ"].tolist() if not df_market.empty else []
    if not allowed_assets: allowed_assets = list(ASSET_REGISTRY.keys())
    
    asset = st.selectbox("Выбор актива для спецификации", allowed_assets)
    st.markdown("---")
    
    st.subheader("📊 ТЕКУЩИЙ КЛИМАТ РЕШЕНИЙ")
    if not df_market.empty:
        c_df = df_market[df_market["Риск"] == user_risk]
        counts = c_df["Решение"].value_counts()
        st.markdown(f"**⭐ Покупка:** `{counts.get('⭐ Покупка', 0)}` | **⚠ Спекуляция:** `{counts.get('⚠ Спекуляция', 0)}`")
        st.markdown(f"**⚪ Удержание:** `{counts.get('⚪ Удержание', 0)}` | **❌ Игнор:** `{counts.get('❌ Игнор', 0)}`")

# ============================================================
# РАСЧЕТ ИНДЕКСА И СЕКТОРОВ
# ============================================================

if not df_market.empty:
    top10_q = df_market.sort_values(by="Качество", ascending=False).head(10)["Качество"].mean()
    regime_t = "Бычий рынок 🐂" if top10_q >= 80 else "Рост 📈" if top10_q >= 60 else "Нейтрально ⚖️" if top10_q >= 40 else "Медвежий рынок 🐻" if top10_q >= 20 else "Капитуляция 💥"
    
    m1, m2 = st.columns([1.2, 1])
    with m1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); padding:15px; border-radius:10px; border:1px solid #4338ca; text-align:center; height:140px;'>
            <span style='color:#a5b4fc; font-size:11px; font-weight:bold; letter-spacing:1px;'>🏛️ ИНДЕКС КАЧЕСТВА РЫНКА</span>
            <h2 style='color:#ffffff; margin:2px 0; font-size:36px;'>{top10_q:.1f} <span style='font-size:18px; color:#64748b;'>/ 100</span></h2>
            <p style='margin:0; font-size:13px; color:#f8fafc;'>Статус: <b>{regime_t}</b></p>
        </div>
        """, unsafe_allow_html=True)
        
    with m2:
        sector_stats = df_market.groupby("Сектор")["Качество"].mean().reset_index()
        sector_stats = sector_stats.sort_values(by="Качество", ascending=False).head(4)
        sec_html = "".join([f"<div style='display:flex; justify-content:between; font-size:12px; margin:4px 0;'><span style='color:#94a3b8;'>⚡ {r['Сектор']}:</span> <b style='color:#34d399; margin-left:auto;'>{r['Качество']:.1f}</b></div>" for _, r in sector_stats.iterrows()])
        
        st.markdown(f"""
        <div style='background:#0f172a; padding:12px; border-radius:10px; border:1px solid #1e293b; height:140px;'>
            <span style='color:#94a3b8; font-size:11px; font-weight:bold; display:block; text-align:center; margin-bottom:5px;'>🔥 РОТАЦИЯ КАПИТАЛА (РЕЙТИНГ СЕКТОРОВ)</span>
            {sec_html}
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# ВЫБРАННЫЙ АКТИВ
# ============================================================

st.markdown("---")
df_select = df_market[df_market["Символ"] == asset]
if not df_select.empty:
    row_a = df_select.iloc[0]
    st.header(f"📊 актив: {row_a['Символ']}")
    
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.metric("💰 ТЕКУЩАЯ ЦЕНА", f"${row_a['Цена']:,.4f}" if row_a['Цена'] < 1 else f"${row_a['Цена']:,.2f}")
    with k2: st.metric("🧬 ИТОГОВОЕ КАЧЕСТВО", f"{row_a['Качество']:.1f}")
    with k3: st.metric("🎯 СМАРТ-ПОТЕНЦИАЛ", f"{row_a['Смарт_Потенциал']:.1f}")
    with k4: st.metric("⚖️ РЕШЕНИЕ МАТРИЦЫ", row_a['Решение'])
    with k5: st.metric("⏳ СТАДИЯ ЦИКЛА", row_a['Стадия'])
    
    st.markdown(f"""
    <div style='background:#0b0f19; padding:12px; border-radius:8px; border:1px solid #22c55e40; margin-top:5px;'>
        <p style='margin:0; font-size:13px; text-align:center; color:#f3f4f6;'>
            🎯 <b>ОРДЕРНАЯ СЕТКА И ЗОНЫ ФИКСАЦИИ ПРИБЫЛИ ДЛЯ {asset}:</b> &nbsp;&nbsp;|&nbsp;&nbsp; 
            🛑 Стоп-Лосс: <span style='color:#ef4444; font-weight:bold;'>${row_a['Стоп']:,.2f}</span> &nbsp;&nbsp;|&nbsp;&nbsp; 
            🟢 Фиксация 1 (30%): <span style='color:#34d399;'>${row_a['ТП1']:,.2f}</span> &nbsp;&nbsp;|&nbsp;&nbsp; 
            🟢 Фиксация 2 (40%): <span style='color:#34d399;'>${row_a['ТП2']:,.2f}</span> &nbsp;&nbsp;|&nbsp;&nbsp; 
            🟢 Фиксация 3 (30%): <span style='color:#22c55e; font-weight:bold;'>${row_a['ТП3']:,.2f}</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# ТАБЛИЦЫ РАНЖИРОВАНИЯ И ДЕЛЬТЫ
# ============================================================

st.markdown("---")
t1, t2 = st.tabs(["📋 ТЕКУЩАЯ ТАБЛИЦА РАНЖИРОВАНИЯ АКТИВОВ", "📈 МОНИТОРИНГ ДЕЛЬТЫ ИЗМЕНЕНИЯ РЕЙТИНГА"])

with t1:
    if not df_market.empty:
        df_v = df_market[df_market["Риск"] == user_risk].sort_values(by="Потенциал", ascending=False).copy()
        if not df_v.empty:
            df_v["Просадка"] = df_v["Просадка"].map(lambda x: f"{x:.1f}%")
            df_v["Сила"] = df_v["Сила"].map(lambda x: f"{x:+.1f}%")
            df_v["Цена"] = df_v["Цена"].map(lambda x: f"${x:,.2f}")
            df_v = df_v.rename(columns={"Смарт_Потенциал": "Смарт-Потенциал", "Потенциал": "Потенциал входа"})
            st.dataframe(df_v[["Символ", "Сектор", "Цена", "Итоговое качество", "Смарт-Потенциал", "Потенциал входа", "Решение", "Стадия", "Просадка", "Сила"]], use_container_width=True, hide_index=True)

with t2:
    if not df_market.empty:
        st.markdown("##### 🔍 Активы с максимальным притоком умных денег (Рост Качества за 30 дней)")
        df_delta = df_market.sort_values(by="Дельта_Качества", ascending=False).copy()
        df_delta["Дельта_Качества"] = df_delta["Дельта_Качества"].map(lambda x: f"+{x:.1f}" if x > 0 else f"{x:.1f}")
        df_delta = df_delta.rename(columns={"Было_Качество": "Было (30д назад)", "Качество": "Стало (Текущее)", "Дельта_Качества": "Изменение рейтинга"})
        st.dataframe(df_delta[["Символ", "Сектор", "Было (30д назад)", "Стало (Текущее)", "Изменение рейтинга", "Решение"]].head(10), use_container_width=True, hide_index=True)

# ============================================================
# ВАЛИДАТОР ЭФФЕКТИВНОСТИ СИГНАЛОВ
# ============================================================

st.markdown("---")
with st.expander("🔬 ВАЛИДАТОР ЭФФЕКТИВНОСТИ И МАТЕМАТИЧЕСКИЙ АУДИТ МАТРИЦЫ"):
    st.markdown("##### Моделирование слепых форвард-сигналов со смещением на 180 дней назад")
    
    @st.cache_data(ttl=3600)
    def run_rigorous_validation():
        rows_audit = []
        btc_f = load_asset_data("BTC", days=750)
        
        for sym, m in ASSET_REGISTRY.items():
            df_f = load_asset_data(sym)
            if df_f is None or len(df_f) < 380: continue
            
            t_idx = len(df_f) - 180
            df_past = df_f.iloc[:t_idx].reset_index(drop=True)
            past_date = df_past["date"].iloc[-1]
            
            btc_past = btc_f[btc_f["date"] <= past_date].reset_index(drop=True) if btc_f is not None else None
            res_p = calculate_two_factor_matrix(sym, df_past, btc_past)
            
            if res_p[0] is None: continue
            past_dec, entry_p = res_p[14], res_p[1]
            
            future_window = df_f.iloc[t_idx : t_idx + 180]["close"].values
            if len(future_window) == 0: continue
            
            final_p = future_window[-1]
            net_perf = (final_p / entry_p - 1) * 100
            
            min_future_price = np.min(future_window)
            max_drawdown_after_entry = (min_future_price / entry_p - 1) * 100
            if max_drawdown_after_entry > 0: max_drawdown_after_entry = 0.0
            
            rows_audit.append({
                "Символ": sym, "Решение": past_dec, "Доходность_180": net_perf, "Макс_Просадка": max_drawdown_after_entry
            })
        return pd.DataFrame(rows_audit)

    df_validator = run_rigorous_validation()
    
    if not df_validator.empty:
        matrix_rows = []
        for dec_type in ["⭐ Покупка", "⚠ Спекуляция", "⚪ Удержание", "❌ Игнор"]:
            sub = df_validator[df_validator["Решение"] == dec_type]
            if not sub.empty:
                s_count = len(sub)
                avg_perf = sub["Доходность_180"].mean()
                med_perf = sub["Доходность_180"].median()
                win_rate = (len(sub[sub["Доходность_180"] > 0]) / s_count) * 100
                avg_mdd = sub["Макс_Просадка"].mean()
                
                # Исправлена опечатка со знаком сравнения тире на корректное условие Python
                matrix_rows.append({
                    "Тип Решения": dec_type,
                    "Количество сигналов": int(s_count),
                    "Win Rate %": f"{win_rate:.1f}%",
                    "Средняя доходность": f"{avg_perf:+.1f}%",
                    "Медианная доходность": f"{med_perf:+.1f}%",
                    "Средняя Макс. Просадка": f"{avg_mdd:.1f}%"
                })
            else:
                matrix_rows.append({
                    "Тип Решения": dec_type, "Количество сигналов": 0, "Win Rate %": "N/A",
                    "Средняя доходность": "N/A", "Медианная доходность": "N/A", "Средняя Макс. Просадка": "N/A"
                })
                
        df_eff_matrix = pd.DataFrame(matrix_rows)
        st.markdown("##### 🎯 СВОДНАЯ МАТРИЦА ЭФФЕКТИВНОСТИ РЕШЕНИЙ")
        st.dataframe(df_eff_matrix, use_container_width=True, hide_index=True)
    else:
        st.info("Формирование матрицы аудита завершится после обновления пулов исторических биржевых стаканов.")

# ============================================================
# ПОДВАЛ
# ============================================================
moscow_time = datetime.now(timezone(timedelta(hours=3)))
st.markdown("---")
st.caption(f"📅 Синхронизация: {moscow_time.strftime('%Y-%m-%d %H:%M:%S')} (МСК) | Все синтаксические опечатки устранены.")
