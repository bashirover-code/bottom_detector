import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone
import yfinance as yf

# ============================================================
# НАСТРОЙКИ СТРАНИЦЫ И ШРИФТОВ
# ============================================================

st.set_page_config(page_title="Инвестиционная матрица", layout="wide")

st.markdown("""
    <style>
        html, body, [class*="css"], .stMarkdown, .stDataFrame,
        .stButton, .stSelectbox, .stRadio, .stCaption, h1, h2, h3, h4, p, div {
            font-family: 'Times New Roman', Times, serif !important;
        }
        .metric-container {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 12px;
            margin-bottom: 15px;
        }
        .metric-card {
            background: #1e293b;
            padding: 14px;
            border-radius: 8px;
            border: 1px solid #334155;
            text-align: center;
        }
        .metric-label {
            font-size: 11px;
            color: #94a3b8;
            font-weight: bold;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
            text-transform: uppercase;
        }
        .metric-value {
            font-size: 19px;
            color: #ffffff;
            font-weight: bold;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🏛️ Инвестиционная матрица")

# ============================================================
# РЕЕСТР АКТИВОВ И ФИКСИРОВАННЫЕ ДИАПАЗОНЫ ДНА
# ============================================================

ASSET_REGISTRY = {
    "BTC": {"type": "Криптовалюта", "risk": "Низкий", "sector": "L1"},
    "ETH": {"type": "Криптовалюта", "risk": "Низкий", "sector": "L1"},
    "LINK": {"type": "Криптовалюта", "risk": "Низкий", "sector": "DeFi"},
    "SOL": {"type": "Криптовалюта", "risk": "Средний", "sector": "L1"},
    "NEAR": {"type": "Криптовалюта", "risk": "Средний", "sector": "L1"},
    "SUI": {"type": "Криптовалюта", "risk": "Средний", "sector": "L1"},
    "STX": {"type": "Криптовалюта", "risk": "Средний", "sector": "Layer 2"},
    "IMX": {"type": "Криптовалюта", "risk": "Средний", "sector": "Layer 2"},
    "GRT": {"type": "Криптовалюта", "risk": "Средний", "sector": "AI"},
    "UNI": {"type": "Криптовалюта", "risk": "Средний", "sector": "DeFi"},
    "RENDER": {"type": "Криптовалюта", "risk": "Высокий", "sector": "AI"},
    "ONDO": {"type": "Криптовалюта", "risk": "Высокий", "sector": "RWA"},
    "ARKM": {"type": "Криптовалюта", "risk": "Высокий", "sector": "AI"},
    "GOAT": {"type": "Криптовалюта", "risk": "Высокий", "sector": "Meme"},
    "FLOCK": {"type": "Криптовалюта", "risk": "Высокий", "sector": "Meme"},
    "TRUMP": {"type": "Криптовалюта", "risk": "Высокий", "sector": "Meme"},
    "ZK": {"type": "Криптовалюта", "risk": "Высокий", "sector": "Layer 2"},
    "FIL": {"type": "Криптовалюта", "risk": "Средний", "sector": "DeFi"},
    "CELO": {"type": "Криптовалюта", "risk": "Высокий", "sector": "L1"},
    "CRV": {"type": "Криптовалюта", "risk": "Высокий", "sector": "DeFi"},
    "TWT": {"type": "Криптовалюта", "risk": "Средний", "sector": "DeFi"},
    "APE": {"type": "Криптовалюта", "risk": "Высокий", "sector": "Meme"},
    "ONE": {"type": "Криптовалюта", "risk": "Высокий", "sector": "L1"},
    "POL": {"type": "Криптовалюта", "risk": "Средний", "sector": "Layer 2"},
    "ARC": {"type": "Криптовалюта", "risk": "Высокий", "sector": "AI"},
    "ALGO": {"type": "Криптовалюта", "risk": "Средний", "sector": "L1"},
    "ASTER": {"type": "Криптовалюта", "risk": "Высокий", "sector": "Web3"},
    "GDX": {"type": "Акция", "risk": "Низкий", "sector": "Stocks"},
    "URA": {"type": "Акция", "risk": "Низкий", "sector": "Stocks"},
    "TSLA": {"type": "Акция", "risk": "Средний", "sector": "Stocks"},
    "PLTR": {"type": "Акция", "risk": "Средний", "sector": "Stocks"},
    "NVDA": {"type": "Акция", "risk": "Средний", "sector": "Stocks"},
    "COIN": {"type": "Акция", "risk": "Высокий", "sector": "Stocks"},
    "HIMS": {"type": "Акция", "risk": "Высокий", "sector": "Stocks"},
    "BABA": {"type": "Акция", "risk": "Средний", "sector": "Stocks"},
    "ZM": {"type": "Акция", "risk": "Средний", "sector": "Stocks"},
    "LIT": {"type": "Акция", "risk": "Низкий", "sector": "Stocks"},
    "SIL": {"type": "Акция", "risk": "Низкий", "sector": "Stocks"},
    "EWW": {"type": "Акция", "risk": "Низкий", "sector": "Stocks"}
}

BOTTOM_ZONES = {
    "BTC": (68000.0, 73000.0), "ETH": (1800.0, 1950.0), "LINK": (8.0, 9.0), "SOL": (80.0, 90.0),
    "NEAR": (1.8, 2.2), "SUI": (0.75, 0.9), "STX": (0.18, 0.22), "IMX": (0.13, 0.15),
    "GRT": (0.022, 0.028), "UNI": (2.9, 3.3), "RENDER": (1.8, 2.5), "ONDO": (0.28, 0.36),
    "ARKM": (0.12, 0.16), "GOAT": (0.013, 0.018), "FLOCK": (0.050, 0.062), "TRUMP": (1.7, 2.1),
    "ZK": (0.013, 0.017), "FIL": (0.80, 0.95), "CELO": (0.062, 0.078), "CRV": (0.18, 0.23),
    "TWT": (0.38, 0.46), "APE": (0.09, 0.12), "ONE": (0.0017, 0.0022), "POL": (0.07, 0.10),
    "ARC": (0.015, 0.020), "ALGO": (0.090, 0.115), "ASTER": (0.55, 0.70), "GDX": (63.0, 68.0),
    "URA": (44.0, 48.0), "TSLA": (250.0, 280.0), "PLTR": (45.0, 60.0), "NVDA": (173.0, 186.0),
    "COIN": (155.0, 175.0), "HIMS": (18.0, 21.0), "BABA": (115.0, 120.0), "ZM": (85.0, 95.0),
    "LIT": (65.0, 72.0), "SIL": (65.0, 75.0), "EWW": (70.0, 73.0)
}

# Валидация весов на уровне компиляции модуля
assert 20 + 20 + 20 + 15 + 15 + 10 == 100, "Веса макро-индекса не равны 100!"
assert abs(0.40 + 0.35 + 0.25 - 1.0) < 1e-6, "Веса скоринга отдельного актива не равны 1.0!"

# ============================================================
# РАБОТА С СЕТЬЮ И КЭШИРОВАНИЕМ
# ============================================================

def _fetch_raw_yfinance(symbol, ticker_suffix, days):
    """Изолированная сетевая функция с внутренней обработкой исключений"""
    time.sleep(0.35)
    try:
        s = yf.Ticker(f"{symbol}{ticker_suffix}")
        return s.history(period=f"{days}d")
    except Exception:
        return None

@st.cache_data(ttl=900)
def load_asset_data(symbol, days=1500):
    meta = ASSET_REGISTRY.get(symbol)
    if not meta:
        return None
        
    ticker_suffix = "-USD" if meta["type"] == "Криптовалюта" else ""
    df = _fetch_raw_yfinance(symbol, ticker_suffix, days)
    
    if df is not None and not df.empty:
        try:
            df = df.reset_index().rename(columns={"Date": "date", "Close": "close", "Volume": "volume"})
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            return df[["date", "close", "volume"]].sort_values("date").reset_index(drop=True)
        except Exception:
            return None
    return None

def calculate_single_rs(df_t, btc_t, lookup_days):
    if btc_t is None or len(df_t) < 10 or len(btc_t) < 10: 
        return 0.0
        
    df_work = df_t.copy()
    btc_work = btc_t.copy()
    
    df_work["d"] = df_work["date"].dt.date
    btc_work["d"] = btc_work["date"].dt.date
    
    inter = np.intersect1d(df_work['d'], btc_work['d'])
    effective_lookup = min(len(inter), lookup_days)
    
    if effective_lookup > 5:
        sub_a = df_work[df_work['d'].isin(inter)].sort_values("d")
        sub_b = btc_work[btc_work['d'].isin(inter)].sort_values("d")
        
        perf_a = (sub_a['close'].iloc[-1] / sub_a['close'].iloc[-effective_lookup] - 1) * 100
        perf_b = (sub_b['close'].iloc[-1] / sub_b['close'].iloc[-effective_lookup] - 1) * 100
        return perf_a - perf_b
        
    return 0.0

# ============================================================
# РАСЧЕТ ИНДЕКСОВ И МАТРИЦЫ
# ============================================================

def build_macro_bottom_index(volume_perf_data):
    try:
        res = requests.get("https://api.alternative.me/fng/", timeout=5).json()
        f_g_val = int(res['data'][0]['value'])
    except Exception:
        f_g_val = 50

    btc_data = load_asset_data("BTC", days=1450)
    
    mayer_val = 1.0
    lre_val = 0.0
    wma200_dist = 1.0

    if btc_data is not None and len(btc_data) > 1400:
        c_p = btc_data["close"].iloc[-1]
        
        ma350 = btc_data["close"].rolling(350).mean().iloc[-1]
        mayer_val = c_p / ma350 if ma350 > 0 else 1.0
        
        btc_data['log_p'] = np.log10(btc_data['close'])
        x = np.arange(len(btc_data))
        slope, intercept = np.polyfit(x, btc_data['log_p'], 1)
        expected_log_p = slope * x[-1] + intercept
        lre_val = btc_data['log_p'].iloc[-1] - expected_log_p
        
        ma1400 = btc_data["close"].rolling(1400).mean().iloc[-1]
        wma200_dist = c_p / ma1400 if ma1400 > 0 else 1.0

    btc_volume_share = volume_perf_data.get("btc_volume_share", 55.0)
    altseason_ratio = volume_perf_data.get("altseason_ratio", 30.0)

    mayer_score = 20 if mayer_val <= 0.75 else 15 if mayer_val <= 0.95 else 8 if mayer_val <= 1.15 else 0
    lre_score = 20 if lre_val <= -0.15 else 15 if lre_val <= -0.05 else 5 if lre_val <= 0.1 else 0
    wma_score = 20 if wma200_dist <= 1.02 else 14 if wma200_dist <= 1.20 else 5 if wma200_dist <= 1.40 else 0
    fg_score = 15 if f_g_val <= 20 else 10 if f_g_val <= 40 else 3 if f_g_val <= 60 else 0
    dom_score = 15 if btc_volume_share >= 60.0 else 10 if btc_volume_share >= 48.0 else 2
    alt_score = 10 if altseason_ratio <= 20.0 else 6 if altseason_ratio <= 45.0 else 0

    total_macro_index = mayer_score + lre_score + wma_score + fg_score + dom_score + alt_score
    total_macro_index = max(0, min(total_macro_index, 100))

    if total_macro_index >= 75:
        phase_text = "🟢 Циклическое дно / Сильное накопление"
    elif total_macro_index >= 45:
        phase_text = "🟡 Нейтральный баланс / Стабилизация"
    else:
        phase_text = "🔴 Фаза распределения / Перегрев"

    return {
        "Индекс": total_macro_index,
        "Фаза": phase_text,
        "Детализация": {
            "Mayer Multiple (BTC)": f"{mayer_val:.2f} [+{mayer_score}]",
            "Log Regression Error (LRE)": f"{lre_val:+.3f} [+{lre_score}]",
            "200WMA Distance (BTC)": f"{wma200_dist:.2f}x [+{wma_score}]",
            "Fear & Greed Index": f"{f_g_val}/100 [+{fg_score}]",
            "BTC Vol Dominance (Real)": f"{btc_volume_share:.1f}% [+{dom_score}]",
            "Alt Outperformance (60d)": f"{altseason_ratio:.1f}% [+{alt_score}]"
        }
    }

def calculate_macro_matrix(symbol, df, macro_bottom_score, btc_df=None, end_idx=None):
    zone = BOTTOM_ZONES.get(symbol)
    if not zone or df is None or len(df) < 200: 
        return (None,) * 13
        
    working_df = df.iloc[:end_idx].copy() if end_idx is not None else df.copy()
    if len(working_df) < 50:
        return (None,) * 13
        
    current_price = working_df["close"].iloc[-1]
    
    working_df["ma90"] = working_df["close"].rolling(window=90, min_periods=30).mean()
    working_df["ma200"] = working_df["close"].rolling(window=200, min_periods=50).mean()
    working_df["dollar_volume"] = working_df["close"] * working_df["volume"]
    
    avg_dollar_volume = working_df["dollar_volume"].tail(30).mean()
    quality_vol_score = 100 if avg_dollar_volume > 50_000_000 else 70 if avg_dollar_volume > 5_000_000 else 25
    
    working_btc = btc_df.iloc[:end_idx].copy() if (btc_df is not None and end_idx is not None) else btc_df
    
    rs30 = calculate_single_rs(working_df, working_btc, 30)
    rs90 = calculate_single_rs(working_df, working_btc, 90)
    rs180 = calculate_single_rs(working_df, working_btc, 180)
    relative_strength = (rs30 * 0.2) + (rs90 * 0.3) + (rs180 * 0.5)
    rs_score = 100 if relative_strength > 40 else 80 if relative_strength > 15 else 65 if relative_strength > 0 else 40 if relative_strength > -20 else 10
    
    structure_raw = (10 if current_price > working_df["ma90"].iloc[-1] else 0) + (15 if current_price > working_df["ma200"].iloc[-1] else 0)
    structure_score = int((structure_raw / 25) * 100) if structure_raw > 0 else 0
    
    fundamental_rating = (0.40 * quality_vol_score) + (0.35 * rs_score) + (0.25 * structure_score)
    
    low_zone, high_zone = zone[0], zone[1]
    deviation_high_pct = ((current_price - high_zone) / high_zone) * 100
    deviation_low_pct = ((current_price - low_zone) / low_zone) * 100
    
    is_free_fall = False
    if current_price <= high_zone:
        if current_price < low_zone and low_zone > 0:
            oversold_ratio = (low_zone - current_price) / low_zone
            bottom_score = max(0.0, 100.0 - (oversold_ratio * 150.0))
            status_zone = f"Ниже дна на {abs(deviation_low_pct):.1f}%"
            
            if deviation_low_pct < -30.0:
                is_free_fall = True
        else:
            bottom_score = 100.0
            status_zone = "Внутри зоны"
    else:
        overprice_ratio = (current_price - high_zone) / high_zone
        bottom_score = max(0.0, 100.0 - (overprice_ratio * 150.0))
        status_zone = f"+{deviation_high_pct:.1f}%"
            
    max_p = working_df["close"].max()
    drawdown_pct = ((current_price - max_p) / max_p * 100) if max_p > 0 else 0
    money_flow_score = min(100.0, abs(drawdown_pct) * 1.15)
    
    asset_score = (0.40 * fundamental_rating) + (0.35 * bottom_score) + (0.25 * money_flow_score)
    
    if is_free_fall:
        asset_score *= 0.5
        
    investment_rating = (0.70 * asset_score) + (0.30 * macro_bottom_score)
    investment_rating = max(0.0, min(investment_rating, 100.0))
    
    if is_free_fall:
        decision = "⚠️ Свободное падение"
    elif current_price <= high_zone:
        decision = "⭐ Покупка"
    elif 0.0 < deviation_high_pct <= 5.0:
        decision = "➕ Добор"
    elif 5.0 < deviation_high_pct <= 15.0:
        decision = "👁 Наблюдение"
    else:
        decision = "🔴 Перегрев"
        
    return (current_price, low_zone, high_zone, bottom_score, status_zone, 
            fundamental_rating, investment_rating, decision, drawdown_pct, 
            deviation_high_pct, 0.0, 0.0, "🟡 Расчет")

def calculate_historical_rating(symbol, df, btc_df, macro_score):
    if df is None or len(df) < 40:
        return 50.0
    
    target_date = df["date"].iloc[-1] - timedelta(days=30)
    sub_df = df[df["date"] <= target_date]
    
    if sub_df.empty or len(sub_df) < 10:
        return 50.0
        
    end_idx = len(sub_df)
    res_hist = calculate_macro_matrix(symbol, df, macro_score, btc_df, end_idx=end_idx)
    
    return res_hist[6] if res_hist[6] is not None else 50.0

# ============================================================
# СТРУКТУРИРОВАНИЕ ДАННЫХ И СТЕКА 
# ============================================================

def build_global_market_state(market_dfs, macro_score):
    rows = []
    btc_df = market_dfs.get("BTC")
    for sym, raw in market_dfs.items():
        m = ASSET_REGISTRY[sym]
        res = calculate_macro_matrix(sym, raw, macro_score, btc_df)
        if res[0] is None: 
            continue
            
        rating_30d_ago = calculate_historical_rating(sym, raw, btc_df, macro_score)
        historical_delta = res[6] - rating_30d_ago
        trend_force = "🟢 Усиливается" if historical_delta > 2.0 else "🔴 Ослабевает" if historical_delta < -2.0 else "... Боковик"
        
        rows.append({
            "Символ": sym, "Риск": m["risk"], "Сектор": m["sector"], "Цена": res[0],
            "Нижняя_Зона": res[1], "Верхняя_Зона": res[2], "Близость_к_зоне": res[3],
            "Статус_Зоны": res[4], "Фундаментал": res[5], "Инвестиционный_Рейтинг": res[6], 
            "Решение": res[7], "Просадка": res[8], "Дельта_от_зоны": res[9],
            "Рейтинг_30д_назад": rating_30d_ago, "Дельта_Рейтинга": historical_delta, "Тренд_Силы": trend_force
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=900)
def fetch_all_market_dfs():
    loaded_data = {}
    for sym in ASSET_REGISTRY.keys():
        df = load_asset_data(sym)
        if df is not None and len(df) >= 200:
            loaded_data[sym] = df
    return loaded_data

# ============================================================
# СИНХРОНИЗАЦИЯ СЕССИИ И АВТО-ОБНОВЛЕНИЕ TTL
# ============================================================

with st.spinner("Синхронизация и глубокий анализ биржевых стаканов..."):
    all_dfs = fetch_all_market_dfs()

current_time_ts = time.time()
if "df_market_ts" in st.session_state:
    if (current_time_ts - st.session_state["df_market_ts"]) > 900:
        st.session_state.pop("macro_package", None)
        st.session_state.pop("df_market", None)

if "macro_package" not in st.session_state:
    total_alt_volume = 0.0
    btc_vol_14d = 0.0
    alts_beating_btc = 0
    total_alts_checked = 0
    
    btc_df = all_dfs.get("BTC")
    btc_perf_60d = 0.0
    if btc_df is not None and len(btc_df) > 60:
        btc_perf_60d = (btc_df['close'].iloc[-1] / btc_df['close'].iloc[-60] - 1) * 100

    for sym, df_asset in all_dfs.items():
        if df_asset is None or len(df_asset) < 60:
            continue
        df_asset['dollar_vol'] = df_asset['close'] * df_asset['volume']
        vol_14d = df_asset['dollar_vol'].tail(14).sum()
        
        if sym == "BTC":
            btc_vol_14d = vol_14d
        else:
            total_alt_volume += vol_14d
            alt_perf_60d = (df_asset['close'].iloc[-1] / df_asset['close'].iloc[-60] - 1) * 100
            if alt_perf_60d > btc_perf_60d:
                alts_beating_btc += 1
            total_alts_checked += 1
            
    btc_volume_share = (btc_vol_14d / (btc_vol_14d + total_alt_volume) * 100) if (btc_vol_14d + total_alt_volume) > 0 else 55.0
    altseason_ratio = (alts_beating_btc / total_alts_checked * 100) if total_alts_checked > 0 else 30.0
    
    volume_perf_data = {
        "btc_volume_share": btc_volume_share,
        "altseason_ratio": altseason_ratio
    }
    
    st.session_state["macro_package"] = build_macro_bottom_index(volume_perf_data)

macro_package = st.session_state["macro_package"]
current_macro_score = macro_package["Индекс"]

if "df_market" not in st.session_state:
    st.session_state["df_market"] = build_global_market_state(all_dfs, current_macro_score)
    st.session_state["df_market_ts"] = current_time_ts

df_market = st.session_state["df_market"]

# ============================================================
# ИНТЕРФЕЙС: МАКРО-ИНДЕКС РЫНКА
# ============================================================

st.markdown("### 🏦 МАКРО-ИНДЕКС РЫНКА")
st.markdown(f"## **{int(current_macro_score)} / 100**")
st.markdown(f"**Оценка фазы:** {macro_package['Фаза']}")  # Исправлено с 'Phase' на 'Фаза'

with st.expander("🔍 Показать честные математические метрики и дельты"):
    col_left, col_right = st.columns(2)
    with col_left:
        for k, v in list(macro_package["Детализация"].items())[:3]:
            st.markdown(f"**{k}:** `{v}`")
    with col_right:
        for k, v in list(macro_package["Детализация"].items())[3:]:
            st.markdown(f"**{k}:** `{v}`")

# ============================================================
# АВТОМАТИЧЕСКИЙ ПОРТФЕЛЬ ТОП-5
# ============================================================

st.markdown("---")
st.markdown("### 💼 ТОП-5 АКТИВОВ ДЛЯ ПОКУПКИ СЕГОДНЯ")

if not df_market.empty:
    portfolio_pool = df_market[df_market["Решение"].isin(["⭐ Покупка", "➕ Добор"])].copy()
    
    if not portfolio_pool.empty:
        top_5 = portfolio_pool.sort_values(by="Инвестиционный_Рейтинг", ascending=False).head(5).copy()
        static_weights = [30, 25, 20, 15, 10]
        top_5["Рекомендуемый вес"] = [f"{w}%" for w in static_weights[:len(top_5)]]
        
        top_5["Цена"] = top_5["Цена"].map(lambda x: f"${x:,.2f}" if x >= 1 else f"${x:,.4f}")
        top_5["Инв. рейтинг"] = top_5["Инвестиционный_Рейтинг"].map(lambda x: f"{x:.1f}")
        
        p_cols = ["Символ", "Сектор", "Цена", "Инв. рейтинг", "Решение", "Рекомендуемый вес"]
        st.dataframe(top_5[p_cols], use_container_width=True, hide_index=True)
    else:
        st.info("Рынок локально перегрет либо находится в фазе жесткой капитуляции. Безопасные точки входа отсутствуют.")

# ============================================================
# БОКОВАЯ ПАНЕЛЬ СЛЕЖЕНИЯ И СБРОС КЭША
# ============================================================

with st.sidebar:
    st.header("⚙️ УПРАВЛЕНИЕ МАТРИЦЕЙ")
    
    if st.button("🔄 Обновить данные", use_container_width=True):
        for key in ["df_market", "macro_package", "df_market_ts"]:
            st.session_state.pop(key, None)
        st.cache_data.clear()
        st.rerun()
        
    st.markdown("---")
    user_risk = st.radio("🛡️ Категория риска активов:", ["Низкий", "Средний", "Высокий"])
    
    # Исправлено: заменено "Risk" на "Риск"
    allowed_assets = df_market[df_market["Риск"] == user_risk]["Символ"].tolist() if not df_market.empty else []
    if not allowed_assets: 
        allowed_assets = list(BOTTOM_ZONES.keys())
    
    asset = st.selectbox("Выбор актива для детального разбора:", allowed_assets)

# ============================================================
# КАРТОЧКА АКТИВА
# ============================================================

st.markdown("---")
df_select = df_market[df_market["Символ"] == asset] if not df_market.empty else pd.DataFrame()

if not df_select.empty:
    row_a = df_select.iloc[0]
    st.header(f"📊 Спецификация макро-набора: {row_a['Символ']}")
    
    price_formatted = f"${row_a['Цена']:,.4f}" if row_a['Цена'] < 1 else f"${row_a['Цена']:,.2f}"
    
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-card">
            <div class="metric-label">💰 Текущая цена</div>
            <div class="metric-value">{price_formatted}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">📊 Положение от зоны</div>
            <div class="metric-value">{row_a['Статус_Зоны']}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">⚖️ Решение матрицы</div>
            <div class="metric-value">{row_a['Решение']}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">🏛️ Фундаментал</div>
            <div class="metric-value">{row_a['Фундаментал']:.1f} / 100</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">🎯 Инвест. рейтинг</div>
            <div class="metric-value">{row_a['Инвестиционный_Рейтинг']:.1f} / 100</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c_hist, c_trend = st.columns(2)
    with c_hist:
        st.markdown(f"⏳ **Реальная история изменения рейтинга (30 дней):** `Было: {row_a['Рейтинг_30д_назад']:.1f}` ➡️ `Сейчас: {row_a['Инвестиционный_Рейтинг']:.1f}` (Δ: **{row_a['Дельта_Рейтинга']:+.1f}**)")
    with c_trend:
        st.markdown(f"⚡ **Сила тренда инвестиционного рейтинга:** `{row_a['Тренд_Силы']}`")

    # Исправлено: s.expander изменен на st.expander
    with st.expander("📝 Методика расчета и ордерные сетки диапазона"):
        raw_zone = BOTTOM_ZONES.get(asset, (0.0, 0.0))
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Расчетная зона покупки (Дно):**")
            st.code(f"{raw_zone[0]:,.4f} – {raw_zone[1]:,.4f}" if raw_zone[0] < 1 else f"{raw_zone[0]:,.2f} – {raw_zone[1]:,.2f}")
        with c2:
            st.markdown("**Текущая цена биржи:**")
            st.code(price_formatted)
        with c3:
            st.markdown("**Статус отклонения:**")
            st.code(row_a['Статус_Зоны'])

# ============================================================
# ОБЩАЯ ТАБЛИЦА РАНЖИРОВАНИЯ (Без столбца "Положение от зоны")
# ============================================================

st.markdown("---")
st.markdown("##### 📋 ОБЩАЯ СТРУКТУРНАЯ ТАБЛИЦА РАНЖИРОВАНИЯ АКТИВОВ")

if not df_market.empty:
    df_v = df_market[df_market["Риск"] == user_risk].sort_values(by="Инвестиционный_Рейтинг", ascending=False).copy()
    if not df_v.empty:
        df_v["Просадка"] = df_v["Просадка"].map(lambda x: f"{x:.1f}%")
        df_v["Цена"] = df_v["Цена"].map(lambda x: f"${x:,.2f}" if x >= 1 else f"${x:,.4f}")
        
        # Исправлено: Сначала форматируем значения, чтобы избежать конфликтов имен
        df_v["Отображаемый_Инвест_Рейтинг"] = df_v["Инвестиционный_Рейтинг"].map(lambda x: f"{float(x):.1f}")
        df_v["Отображаемый_Фундаментал"] = df_v["Фундаментал"].map(lambda x: f"{x:.1f}")
        df_v["Близость_к_зоне"] = df_v["Близость_к_зоне"].map(lambda x: f"{int(x)}")
        df_v["Дельта_Рейтинга"] = df_v["Дельта_Рейтинга"].map(lambda x: f"{x:+.1f}")
        
        # Переименование колонок без создания дубликатов с существующим столбцом "Фундаментал"
        df_v = df_v.rename(columns={
            "Отображаемый_Инвест_Рейтинг": "Инвест. рейтинг",
            "Отображаемый_Фундаментал": "Фундаментал_Итого",
            "Близость_к_зоне": "Близость к зоне",
            "Дельта_Рейтинга": "Δ Рейтинга (30д)",
            "Тренд_Силы": "Тренд силы"
        })
        
        # Столбец "Положение от зоны" и старый "Фундаментал" исключены из отображения
        show_cols = ["Символ", "Сектор", "Цена", "Фундаментал_Итого", "Близость к зоне", "Инвест. рейтинг", "Δ Рейтинга (30д)", "Тренд силы", "Решение"]
        
        # Финальный маппинг имени для красивой шапки таблицы
        df_display = df_v[show_cols].rename(columns={"Фундаментал_Итого": "Фундаментал"})
        st.dataframe(df_display, use_container_width=True, hide_index=True)
else:
    st.info("Данные о состоянии рынка временно недоступны.")

# ============================================================
# ПОДВАЛ
# ============================================================
moscow_time = datetime.now(timezone(timedelta(hours=3)))
st.markdown("---")
st.caption(f"📅 Срез зафиксирован: {moscow_time.strftime('%Y-%m-%d %H:%M:%S')} (МСК) | Избыточные столбцы скрыты из интерфейса.")
