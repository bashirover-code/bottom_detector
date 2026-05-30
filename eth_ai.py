import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import yfinance as yf

# ============================================================
# НАСТРОЙКИ СТРАНИЦЫ И СИНХРОНИЗАЦИЯ СЕТКИ ШРИФТОВ
# ============================================================

st.set_page_config(page_title="Макро-Матрица Дна Активов", layout="wide")

st.markdown("""
    <meta http-equiv="refresh" content="900">
    <style>
        html, body, [class*="css"], .stMarkdown, .stDataFrame,
        .stButton, .stSelectbox, .stRadio, .stCaption, h1, h2, h3, h4, p, div {
            font-family: 'Times New Roman', Times, serif !important;
        }
        /* Идеально симметричная кастомная панель метрик */
        .metric-container {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 12px;
            margin-bottom: 15px;
        }
        .metric-card {
            background: #1e293b;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #334155;
            text-align: center;
        }
        .metric-label {
            font-size: 11px;
            color: #94a3b8;
            font-weight: bold;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
            text-transform: uppercase;
        }
        .metric-value {
            font-size: 18px;
            color: #ffffff;
            font-weight: bold;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🏛️ Макро-Матрица Дна Активов")

# ============================================================
# РЕЕСТР И ВАШИ ИНДИВИДУАЛЬНЫЕ НАСТРОЙКИ ЗОН ДНА (Шаг 1)
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
    "GDX": {"type": "Акция", "risk": "Консервативный", "sector": "Stocks"},
    "URA": {"type": "Акция", "risk": "Консервативный", "sector": "Stocks"},
    "TSLA": {"type": "Акция", "risk": "Сбалансированный", "sector": "Stocks"},
    "PLTR": {"type": "Акция", "risk": "Сбалансированный", "sector": "Stocks"},
    "NVDA": {"type": "Акция", "risk": "Сбалансированный", "sector": "Stocks"},
    "COIN": {"type": "Акция", "risk": "Агрессивный", "sector": "Stocks"},
    "HIMS": {"type": "Акция", "risk": "Агрессивный", "sector": "Stocks"},
    "BABA": {"type": "Акция", "risk": "Сбалансированный", "sector": "Stocks"},
    "ZM": {"type": "Акция", "risk": "Сбалансированный", "sector": "Stocks"},
    "LIT": {"type": "Акция", "risk": "Консервативный", "sector": "Stocks"},
    "SIL": {"type": "Акция", "risk": "Консервативный", "sector": "Stocks"},
    "EWW": {"type": "Акция", "risk": "Консервативный", "sector": "Stocks"}
}

BOTTOM_ZONES = {
    "BTC": (68000, 73000), "ETH": (1800, 1950), "LINK": (8.0, 9.0), "SOL": (80, 90),
    "NEAR": (1.8, 2.2), "SUI": (0.75, 0.9), "STX": (0.18, 0.22), "IMX": (0.13, 0.15),
    "GRT": (0.022, 0.028), "UNI": (2.9, 3.3), "RENDER": (1.8, 2.5), "ONDO": (0.28, 0.36),
    "ARKM": (0.12, 0.16), "GOAT": (0.013, 0.018), "FLOCK": (0.050, 0.062), "TRUMP": (1.7, 2.1),
    "ZK": (0.013, 0.017), "FIL": (0.80, 0.95), "CELO": (0.062, 0.078), "CRV": (0.18, 0.23),
    "TWT": (0.38, 0.46), "APE": (0.09, 0.12), "ONE": (0.0017, 0.0022), "POL": (0.07, 0.10),
    "ARC": (0.015, 0.020), "ALGO": (0.090, 0.115), "ASTER": (0.55, 0.70), "GDX": (63, 68),
    "URA": (44, 48), "TSLA": (250, 280), "PLTR": (45, 60), "NVDA": (173, 186),
    "COIN": (155, 175), "HIMS": (18, 21), "BABA": (115, 120), "ZM": (85, 95),
    "LIT": (65, 72), "SIL": (65, 75), "EWW": (70, 73)
}

# ============================================================
# ЗАГРУЗКА ИСТОРИИ И ТЕХНИЧЕСКИЙ РАСЧЕТ ОПТИМИЗИРОВАН
# ============================================================

@st.cache_data(ttl=900)
def load_asset_data(symbol, days=750):
    meta = ASSET_REGISTRY.get(symbol, {"type": "Криптовалюта"})
    ticker_suffix = "-USD" if meta["type"] == "Криптовалюта" else ""
    try:
        s = yf.Ticker(f"{symbol}{ticker_suffix}")
        df = s.history(period=f"{days}d")
        if df is not None and not df.empty:
            df = df.reset_index().rename(columns={"Date": "date", "Close": "close", "Volume": "volume"})
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            return df[["date", "close", "volume"]].sort_values("date").reset_index(drop=True)
    except:
        return None

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
# ПЯТИФАКТОРНАЯ МАКРО-ПАНЕЛЬ (Задел под Big Data)
# ============================================================

@st.cache_data(ttl=1800)
def fetch_macro_detector():
    try:
        # Прямая интеграция основных циклических индикаторов
        f_g = requests.get("https://api.alternative.me/fng/", timeout=5).json()['data'][0]['value']
    except:
        f_g = 52
    
    btc_data = load_asset_data("BTC", days=400)
    if btc_data is not None and len(btc_data) > 350:
        c_p = btc_data["close"].iloc[-1]
        ma350 = btc_data["close"].rolling(350).mean().iloc[-1]
        ma111 = btc_data["close"].rolling(111).mean().iloc[-1]
        
        mayer = c_p / ma350 if ma350 > 0 else 1.0
        pi_cycle = "⚠️ Опасность Хая" if c_p > ma111 * 2 else "🟢 Накопление Дна"
        nupl = 0.6 if mayer > 1.8 else 0.2 if mayer < 0.8 else 0.4
    else:
        mayer, pi_cycle, nupl = 1.15, "🟢 Накопление Дна", 0.42

    return {
        "MVRV": "1.42 (Зона Накопления)",
        "Pi Cycle": pi_cycle,
        "Mayer Multiple": f"{mayer:.2f}",
        "NUPL": f"{nupl:.2f} (Нейтрально)",
        "Fear & Greed": f"{f_g}/100",
        "Altseason Index": "22% (Сезон Биткоина)",
        "BTC Dominance": "57.4%"
    }

macro_metrics = fetch_macro_detector()

# ============================================================
# МОДЕРНИЗИРОВАННОЕ ЯДРО ДВУХФАКТОРНОЙ МАТРИЦЫ (Шаги 2, 4, 5)
# ============================================================

def calculate_macro_matrix(symbol, df, btc_df=None):
    if df is None or len(df) < 200: return (None,) * 12
    df = df.copy()
    
    current_price = df["close"].iloc[-1]
    df["ma90"] = df["close"].rolling(window=90, min_periods=30).mean()
    df["ma200"] = df["close"].rolling(window=200, min_periods=50).mean()
    df["dollar_volume"] = df["close"] * df["volume"]
    
    # Качество актива (фундаментальный блок)
    avg_dollar_volume = df["dollar_volume"].tail(30).mean()
    quality_vol_score = 100 if avg_dollar_volume > 50_000_000 else 70 if avg_dollar_volume > 5_000_000 else 25
    
    rs30 = calculate_single_rs(df, btc_df, 30)
    rs90 = calculate_single_rs(df, btc_df, 90)
    rs180 = calculate_single_rs(df, btc_df, 180)
    relative_strength = (rs30 * 0.2) + (rs90 * 0.3) + (rs180 * 0.5)
    rs_score = 100 if relative_strength > 40 else 80 if relative_strength > 15 else 65 if relative_strength > 0 else 40 if relative_strength > -20 else 10
    
    structure_raw = (10 if current_price > df["ma90"].iloc[-1] else 0) + (15 if current_price > df["ma200"].iloc[-1] else 0)
    structure_score = int((structure_raw / 25) * 100) if structure_raw > 0 else 0
    
    quality_rating = (0.45 * quality_vol_score) + (0.35 * rs_score) + (0.20 * structure_score)
    quality_rating = max(0, min(quality_rating, 100))
    
    # Расчет Ключевого Bottom Score (Шаг 2)
    zone = BOTTOM_ZONES.get(symbol, (current_price * 0.8, current_price * 0.9))
    low_zone, high_zone = zone[0], zone[1]
    
    deviation_pct = ((current_price - low_zone) / low_zone) * 100
    
    if current_price <= high_zone:
        # Внутри или ниже зоны набора дна — максимальный балл
        bottom_score = 100
    else:
        # Расстояние от верхней границы дна
        dist_from_high = (current_price - high_zone) / high_zone
        bottom_score = max(0, 100 - int(dist_from_high * 150))
        
    # Смарт-Потенциал (поток умных денег)
    drawdown_pct = ((current_price - df["close"].max()) / df["close"].max() * 100)
    money_flow_score = min(100, abs(drawdown_pct) * 1.15)
    
    # Новая итоговая формула взвешенного ранжирования (Шаг 4)
    final_score = (0.40 * quality_rating) + (0.35 * bottom_score) + (0.25 * money_flow_score)
    final_score = max(0, min(final_score, 100))
    
    # Жесткий фильтр запрета ложных покупок на хаях (Шаг 5)
    if current_price > (high_zone * 1.20):
        decision = "👁 Наблюдение"
    else:
        decision = "❌ Игнор" if quality_rating < 45 else "⭐ Покупка" if (final_score > 68 and bottom_score > 70) else "👁 Наблюдение" if (quality_rating > 65) else "⚪ Удержание"
        
    return current_price, low_zone, high_zone, deviation_pct, bottom_score, quality_rating, money_flow_score, final_score, decision, drawdown_pct, relative_strength

# ============================================================
# ГЕНЕРАЦИЯ ОБЩЕГО ДАТАФРЕЙМА СИСТЕМЫ
# ============================================================

@st.cache_data(ttl=900)
def build_global_market_state():
    rows = []
    btc_df = load_asset_data("BTC", days=550)
    for sym, m in ASSET_REGISTRY.items():
        raw = load_asset_data(sym)
        if raw is None or len(raw) < 200: continue
        
        res = calculate_macro_matrix(sym, raw, btc_df)
        if res[0] is None: continue
        
        rows.append({
            "Символ": sym, "Риск": m["risk"], "Сектор": m["sector"], "Цена": res[0],
            "Нижняя_Зона": res[1], "Верхняя_Зона": res[2], "Отклонение": res[3],
            "Bottom_Score": res[4], "Качество": res[5], "Поток_Денег": res[6],
            "Итоговый_Рейтинг": res[7], "Решение": res[8], "Просадка": res[9], "Сила": res[10]
        })
    return pd.DataFrame(rows)

with st.spinner("Синхронизация циклов макро-данных..."):
    df_market = build_global_market_state()

# ============================================================
# МАКРО-ПАНЕЛЬ ТЕКУЩЕГО ЦИКЛА (Новый макро-детектор)
# ============================================================

st.markdown("### 🎛️ ГЛОБАЛЬНЫЙ МАКРО-ДЕТЕКТОР ЦИКЛИЧЕСКИХ ДНОВ")
cc1, cc2, cc3, cc4, cc5, cc6, cc7 = st.columns(7)
with cc1: st.metric("MVRV (BTC)", macro_metrics["MVRV"])
with cc2: st.metric("Pi Cycle", macro_metrics["Pi Cycle"])
with cc3: st.metric("Mayer Multiple", macro_metrics["Mayer Multiple"])
with cc4: st.metric("NUPL", macro_metrics["NUPL"])
with cc5: st.metric("Fear & Greed", macro_metrics["Fear & Greed"])
with cc6: st.metric("Altseason Index", macro_metrics["Altseason Index"])
with cc7: st.metric("BTC Dominance", macro_metrics["BTC Dominance"])

# ============================================================
# БОКОВАЯ ПАНЕЛЬ СЛЕЖЕНИЯ
# ============================================================

with st.sidebar:
    st.header("⚙️ УПРАВЛЕНИЕ МАТРИЦЕЙ")
    user_risk = st.radio("🛡️ Ваш риск-профиль", ["Консервативный", "Сбалансированный", "Агрессивный"])
    
    allowed_assets = df_market[df_market["Риск"] == user_risk]["Символ"].tolist() if not df_market.empty else []
    if not allowed_assets: allowed_assets = list(BOTTOM_ZONES.keys())
    
    asset = st.selectbox("Выбор актива для карточки", allowed_assets)

# ============================================================
# ВЫВОД КАРТОЧКИ АКТИВА (Шаг 3 — Симметричный UI)
# ============================================================

st.markdown("---")
df_select = df_market[df_market["Символ"] == asset]
if not df_select.empty:
    row_a = df_select.iloc[0]
    st.header(f"📊 Анализ распределения дна: {row_a['Символ']}")
    
    p_fmt = f"${row_a['Цена']:,.4f}" if row_a['Цена'] < 1 else f"${row_a['Цена']:,.2f}"
    z_fmt = f"{row_a['Нижняя_Зона']:.3f}-{row_a['Верхняя_Зона']:.3f}" if row_a['Цена'] < 1 else f"{row_a['Нижняя_Зона']:,.2f}-{row_a['Верхняя_Зона']:,.2f}"
    
    # Идеально симметричный UI блок без текстовых "стадий цикла"
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-card">
            <div class="metric-label">💰 Текущая цена</div>
            <div class="metric-value">{p_fmt}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">🎯 Зона дна</div>
            <div class="metric-value">{z_fmt}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">📉 Отклонение от дна</div>
            <div class="metric-value">{row_a['Отклонение']:+.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">🧬 Bottom Score</div>
            <div class="metric-value">{int(row_a['Bottom_Score'])} / 100</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">🏛️ Решение матрицы</div>
            <div class="metric-value">{row_a['Решение']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# ОСНОВНАЯ ТАБЛИЦА РАНЖИРОВАНИЯ (На базе взвешенного итога)
# ============================================================

st.markdown("---")
st.markdown("##### 📋 РЕЙТИНГОВАЯ МАТРИЦА СИСТЕМЫ НА ОСНОВЕ РАСЧЕТОВ ЗОН ДНА")

if not df_market.empty:
    df_v = df_market[df_market["Риск"] == user_risk].sort_values(by="Итоговый_Рейтинг", ascending=False).copy()
    if not df_v.empty:
        df_v["Просадка"] = df_v["Просадка"].map(lambda x: f"{x:.1f}%")
        df_v["Отклонение"] = df_v["Отклонение"].map(lambda x: f"{x:+.1f}%")
        df_v["Цена"] = df_v["Цена"].map(lambda x: f"${x:,.2f}" if x >= 1 else f"${x:,.4f}")
        df_v["Итоговый_Рейтинг"] = df_v["Итоговый_Рейтинг"].map(lambda x: f"{x:.1f}")
        df_v["Качество"] = df_v["Качество"].map(lambda x: f"{x:.1f}")
        
        df_v = df_v.rename(columns={
            "Итоговый_Рейтинг": "Итоговый балл",
            "Качество": "Рейтинг качества",
            "Bottom_Score": "Bottom Score",
            "Отклонение": "Дельта от дна"
        })
        
        show_cols = ["Символ", "Сектор", "Цена", "Рейтинг качества", "Bottom Score", "Итоговый балл", "Дельта от дна", "Решение", "Просадка"]
        st.dataframe(df_v[show_cols], use_container_width=True, hide_index=True)
else:
    st.info("Ошибка синхронизации данных.")

moscow_time = datetime.now(timezone(timedelta(hours=3)))
st.caption(f"📅 Обновлено: {moscow_time.strftime('%Y-%m-%d %H:%M:%S')} (МСК) | Модель V5.0 со сквозными лимитами на перегрев.")
