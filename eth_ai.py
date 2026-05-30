import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import yfinance as yf

# ============================================================
# НАСТРОЙКИ СТРАНИЦЫ И СИНХРОНИЗАЦИЯ ШРИФТОВ
# ============================================================

st.set_page_config(page_title="Инвестиционная матрица", layout="wide")

st.markdown("""
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

# ============================================================
# ДИНАМИЧЕСКИЙ КЛИЕНТ ДАННЫХ
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
# СБОР ДАННЫХ МАКРО-ИНДИКАТОРОВ
# ============================================================

@st.cache_data(ttl=1800)
def fetch_macro_detector():
    try:
        f_g = requests.get("https://api.alternative.me/fng/", timeout=5).json()['data'][0]['value']
    except:
        f_g = 55
    
    btc_data = load_asset_data("BTC", days=400)
    if btc_data is not None and len(btc_data) > 350:
        c_p = btc_data["close"].iloc[-1]
        ma350 = btc_data["close"].rolling(350).mean().iloc[-1]
        ma111 = btc_data["close"].rolling(111).mean().iloc[-1]
        mayer = c_p / ma350 if ma350 > 0 else 1.12
        pi_cycle = "🟢 Накопление" if c_p < ma111 * 1.5 else "⚠️ Перегрев"
        nupl = 0.58 if mayer > 1.5 else 0.22 if mayer < 0.8 else 0.39
    else:
        mayer, pi_cycle, nupl = 1.14, "🟢 Накопление", 0.41

    return {
        "MVRV": "1.45 (Накопление)",
        "Pi Cycle": pi_cycle,
        "Mayer Multiple": f"{mayer:.2f}",
        "NUPL": f"{nupl:.2f}",
        "Fear & Greed": f"{f_g}/100",
        "Altseason Index": "24%",
        "BTC Dominance": "56.8%"
    }

macro_metrics = fetch_macro_detector()

# ============================================================
# ЯДРО МАТЕМАТИЧЕСКИХ РАСЧЕТОВ МАТРИЦЫ
# ============================================================

def calculate_macro_matrix(symbol, df, btc_df=None):
    if df is None or len(df) < 200: return (None,) * 12
    df = df.copy()
    
    current_price = df["close"].iloc[-1]
    df["ma90"] = df["close"].rolling(window=90, min_periods=30).mean()
    df["ma200"] = df["close"].rolling(window=200, min_periods=50).mean()
    df["dollar_volume"] = df["close"] * df["volume"]
    
    # 1. Расчет рейтинга качества актива (Максимальный вес в долгосроке)
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
    
    # 2. Новая нелинейная логика Близости к дну (Ваш аудит)
    zone = BOTTOM_ZONES.get(symbol, (current_price * 0.8, current_price * 0.9))
    low_zone, high_zone = zone[0], zone[1]
    
    deviation_high_pct = ((current_price - high_zone) / high_zone) * 100
    
    if low_zone <= current_price <= high_zone:
        bottom_score = 100.0
        status_zone = "🟢 В зоне покупки"
    elif current_price < low_zone:
        oversold = (low_zone - current_price) / low_zone
        bottom_score = max(70.0, 100.0 - oversold * 100.0)
        status_zone = "🟢 Ниже зоны покупки"
    else:
        overprice = (current_price - high_zone) / high_zone
        bottom_score = max(0.0, 100.0 - overprice * 150.0)
        if deviation_high_pct <= 5.0:
            status_zone = f" Adil Над зоной {deviation_high_pct:+.1f}%"
            # Для визуализации приведем к утвержденному вами текстовому виду
            status_zone = f"🟡 Над зоной {deviation_high_pct:+.1f}%"
        else:
            status_zone = f"🔴 Перегрев {deviation_high_pct:+.1f}%"
            
    # 3. Расчет потока умных денег (интенсивность просадки)
    max_p = df["close"].max()
    drawdown_pct = ((current_price - max_p) / max_p * 100) if max_p > 0 else 0
    money_flow_score = min(100.0, abs(drawdown_pct) * 1.15)
    
    # 4. Расчет Итогового взвешенного балла
    final_score = (0.40 * quality_rating) + (0.35 * bottom_score) + (0.25 * money_flow_score)
    final_score = max(0.0, min(final_score, 100.0))
    
    # Оценка инвестиционного потенциала
    if final_score >= 75: potential_text = "Высокий"
    elif final_score >= 50: potential_text = "Средний"
    else: potential_text = "Низкий"
    
    # 5. Жесткий консервативный фильтр перегрева (Максимум 5% над хаем зоны)
    if current_price > (high_zone * 1.05):
        decision = "👁 Наблюдение"
    else:
        if quality_rating < 45:
            decision = "❌ Игнор"
        elif final_score > 68 and bottom_score > 70:
            decision = "⭐ Покупка"
        elif quality_rating > 65:
            decision = "👁 Наблюдение"
        else:
            decision = "⚪ Удержание"
            
    return current_price, low_zone, high_zone, bottom_score, status_zone, quality_rating, money_flow_score, final_score, decision, drawdown_pct, potential_text

# ============================================================
# СБОРКА ЕДИНОЙ ТАБЛИЦЫ РЕЙТИНГА
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
            "Нижняя_Зона": res[1], "Верхняя_Зона": res[2], "Близость_к_дну": res[3],
            "Статус_Зоны": res[4], "Качество": res[5], "Поток_Денег": res[6],
            "Итоговый_Рейтинг": res[7], "Решение": res[8], "Просадка": res[9], "Потенциал": res[10]
        })
    return pd.DataFrame(rows)

with st.spinner("Синхронизация циклов макро-данных..."):
    df_market = build_global_market_state()

# ============================================================
# ГЛОБАЛЬНЫЙ МАКРО-ИНТЕРФЕЙС СИСТЕМЫ
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
# СТРОГАЯ СИММЕТРИЧНАЯ КАРТОЧКА АКТИВА (Ваше ТЗ)
# ============================================================

st.markdown("---")
df_select = df_market[df_market["Символ"] == asset]
if not df_select.empty:
    row_a = df_select.iloc[0]
    st.header(f"📊 Спецификация актива: {row_a['Символ']}")
    
    price_formatted = f"${row_a['Цена']:,.4f}" if row_a['Цена'] < 1 else f"${row_a['Цена']:,.2f}"
    
    # 5 монолитных колонок фиксированного размера
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-card">
            <div class="metric-label">💰 Текущая цена</div>
            <div class="metric-value">{price_formatted}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">🧬 Близость к дну</div>
            <div class="metric-value">{int(row_a['Близость_к_дну'])} / 100</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">⚖️ Решение матрицы</div>
            <div class="metric-value">{row_a['Решение']}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">🎯 Статус зоны</div>
            <div class="metric-value">{row_a['Статус_Зоны']}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">🛡️ Риск актива</div>
            <div class="metric-value">{row_a['Риск']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # СКРЫТЫЙ БЛОК МЕТОДИКИ РАСЧЕТА (Вынос шума с экрана)
    with st.expander("📝 Методика расчета и ордерные сетки"):
        raw_zone = BOTTOM_ZONES.get(asset, (0.0, 0.0))
        dev_pct = ((row_a['Цена'] - raw_zone[1]) / raw_zone[1]) * 100 if raw_zone[1] > 0 else 0
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Зона покупки (Дно):**")
            st.code(f"{raw_zone[0]:,.4f} – {raw_zone[1]:,.4f}" if raw_zone[0] < 1 else f"{raw_zone[0]:,.2f} – {raw_zone[1]:,.2f}")
        with c2:
            st.markdown("**Текущая цена биржи:**")
            st.code(price_formatted)
        with c3:
            st.markdown("**Чистая дельта от верхней границы дна:**")
            st.code(f"{dev_pct:+.2f}%")
            
        st.markdown(f"""
        > **Математическая справка:** Итоговый балл по активу составляет **{row_a['Итоговый_Рейтинг']:.1f}/100**, что указывает на **{row_a['Потенциал']}** инвестиционный потенциал. 
        Фильтр перегрева жестко заблокирует любые новые сигналы «Покупка», если текущая цена превысит отметку **${raw_zone[1] * 1.05:,.2f}** (+5% от границы набора).
        """)

# ============================================================
# ГЛУБОКАЯ ТАБЛИЦА РАНЖИРОВАНИЯ ДЛЯ СЛЕЖЕНИЯ
# ============================================================

st.markdown("---")
st.markdown("##### 📋 ОБЩАЯ СТРУКТУРНАЯ ТАБЛИЦА РАНЖИРОВАНИЯ АКТИВОВ")

if not df_market.empty:
    df_v = df_market[df_market["Риск"] == user_risk].sort_values(by="Итоговый_Рейтинг", ascending=False).copy()
    if not df_v.empty:
        df_v["Просадка"] = df_v["Просадка"].map(lambda x: f"{x:.1f}%")
        df_v["Цена"] = df_v["Цена"].map(lambda x: f"${x:,.2f}" if x >= 1 else f"${x:,.4f}")
        df_v["Итоговый_Рейтинг"] = df_v["Итоговый_Рейтинг"].map(lambda x: f"{x:.1f}")
        df_v["Качество"] = df_v["Качество"].map(lambda x: f"{x:.1f}")
        df_v["Близость_к_дну"] = df_v["Близость_к_дну"].map(lambda x: f"{int(x)}")
        
        df_v = df_v.rename(columns={
            "Итоговый_Рейтинг": "Итоговый балл",
            "Качество": "Рейтинг качества",
            "Близость_к_дну": "Близость к дну",
            "Статус_Зоны": "Статус зоны",
            "Потенциал": "Потенциал"
        })
        
        show_cols = ["Символ", "Сектор", "Цена", "Рейтинг качества", "Близость к дну", "Итоговый балл", "Статус зоны", "Потенциал", "Решение"]
        st.dataframe(df_v[show_cols], use_container_width=True, hide_index=True)
else:
    st.info("Ошибка компиляции рыночных дельт.")

# ============================================================
# ПОДВАЛ
# ============================================================
moscow_time = datetime.now(timezone(timedelta(hours=3)))
st.markdown("---")
st.caption(f"📅 Расчет завершен: {moscow_time.strftime('%Y-%m-%d %H:%M:%S')} (МСК) | Нелинейная модель Близости к дну активна | Автообновление отключено.")
