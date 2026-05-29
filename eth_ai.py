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

st.set_page_config(page_title="Детектор дна активов v3", layout="wide")

st.markdown("""
    <meta http-equiv="refresh" content="300">
    <style>
        html, body, [class*="css"], .stMarkdown, .stMetric, .stDataFrame,
        .stButton, .stSelectbox, .stRadio, .stCaption, h1, h2, h3, h4, p, div {
            font-family: 'Times New Roman', Times, serif !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Детектор дна активов v3")

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
# 3. ФУНКЦИИ ДЛЯ COINGECKO
# ============================================================

@st.cache_data(ttl=900)
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
                "fdv": md.get("fully_diluted_valuation", {}).get("usd", 0),
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
# 4. РАСЧЁТ ТЕХНИЧЕСКИХ ИНДИКАТОРОВ
# ============================================================

def calculate_rsi(df, periods=14):
    close_delta = df["close"].diff()
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    ma_up = up.ewm(com=periods - 1, adjust=False).mean()
    ma_down = down.ewm(com=periods - 1, adjust=False).mean()
    rsi = ma_up / (ma_down + 1e-10)
    return 100 - (100 / (1 + rsi))

def detect_rsi_divergence(df, lookback=30):
    if len(df) < lookback + 5:
        return False
    sub = df.tail(lookback).copy().reset_index(drop=True)
    
    # Поиск локальных минимумов цены
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
        if z_score <= -1.8: return "🔴 ХАРДКОРНАЯ ПОКУПКА", "#ef4444"
        elif z_score <= -1.0: return "🟡 СТРАТЕГИЧЕСКОЕ НАКОПЛЕНИЕ", "#eab308"
        elif z_score >= 1.6: return "🟢 ОДНОЗНАЧНАЯ ФИКСАЦИЯ", "#22c55e"
        else: return "⚪ ОЖИДАНИЕ ТРИГГЕРОВ", "#6b7280"
    else:
        if z_score <= low: return "🔴 ТОЧКА ВХОДА (АДАПТИВ)", "#ef4444"
        elif z_score <= low * 0.6: return "🟡 МЯГКИЙ НАБОР", "#eab308"
        elif z_score >= high: return "🟢 ФИКСАЦИЯ ПРИБЫЛИ", "#22c55e"
        else: return "⚪ БОКОВИК / НЕЙТРАЛЬНО", "#6b7280"

# ============================================================
# 5. ИСТОРИЧЕСКИЙ АНАЛИЗ УСТОЙЧИВОСТИ (ПАСПОРТ МОНЕТЫ)
# ============================================================

def analyze_stress_tests(df):
    results = {
        "t1_status": "Нет данных", "t1_perf": None,
        "t2_status": "Нет данных", "t2_perf": None
    }
    if df is None or len(df) == 0:
        return results
        
    df_temp = df.copy()
    df_temp["date_str"] = df_temp["date"].dt.strftime("%Y-%m-%d")
    
    # Тест 1 (10.11.2025)
    t1_row = df_temp[df_temp["date_str"] == CRITICAL_DATES["test_1"]["date"]]
    if not t1_row.empty:
        p_t1 = t1_row["close"].values[0]
        # Проверяем цену через 25 дней (восстановление)
        future_t1 = df_temp[df_temp["date"] == (t1_row["date"].values[0] + np.timedelta64(25, 'D'))]
        if not future_t1.empty:
            p_f1 = future_t1["close"].values[0]
            change = ((p_f1 - p_t1) / p_t1) * 100
            results["t1_perf"] = change
            results["t1_status"] = "✅ Прошёл (Выкуплен)" if change >= 15 else "❌ Не восстановился"
            
    # Тест 2 (06.02.2026)
    t2_row = df_temp[df_temp["date_str"] == CRITICAL_DATES["test_2"]["date"]]
    if not t2_row.empty:
        p_t2 = t2_row["close"].values[0]
        # Проверяем цену через 25 дней
        future_t2 = df_temp[df_temp["date"] == (t2_row["date"].values[0] + np.timedelta64(25, 'D'))]
        if not future_t2.empty:
            p_f2 = future_t2["close"].values[0]
            change = ((p_f2 - p_t2) / p_t2) * 100
            results["t2_perf"] = change
            results["t2_status"] = "✅ Прошёл (Выкуплен)" if change >= 15 else "❌ Не восстановился"
            
    return results

# ============================================================
# 6. КОМПЛЕКСНЫЙ АДАПТИВНЫЙ РАСЧЁТ
# ============================================================

def calculate_metrics_adaptive(df):
    if df is None or len(df) < 90:
        return (None,) * 9
        
    df = df.copy()
    
    # Тренды и математическое дно (Z-Score на MA90)
    df["ma90"] = df["close"].rolling(window=90, min_periods=30).mean()
    df["std90"] = df["close"].rolling(window=90, min_periods=30).std()
    df["z_score"] = (df["close"] - df["ma90"]) / (df["std90"] + 1e-10)
    
    df["rsi"] = calculate_rsi(df, 14)
    df["ma200"] = df["close"].rolling(window=200, min_periods=50).mean()
    
    # Объёмный анализ отклонений
    df["v_mean"] = df["volume"].rolling(window=30, min_periods=10).mean()
    df["v_std"] = df["volume"].rolling(window=30, min_periods=10).std()
    df["vol_z"] = (df["volume"] - df["v_mean"]) / (df["v_std"] + 1e-10)
    
    df = df.fillna(method="bfill").fillna(0)
    
    # Текущие экстремумы
    c_price = df["close"].iloc[-1]
    c_z = df["z_score"].iloc[-1]
    c_rsi = df["rsi"].iloc[-1]
    c_vol_z = df["vol_z"].iloc[-1]
    c_ma200 = df["ma200"].iloc[-1]
    
    low_t, up_t = get_adaptive_thresholds(df["z_score"].values)
    dv_bull = detect_rsi_divergence(df, 35)
    
    # Весовая мультифакторная модель вероятности дна
    w_score = 0
    if c_z <= low_t: w_score += 35
    elif c_z < -0.8: w_score += 15
    
    if c_rsi <= 32: w_score += 25
    elif c_rsi <= 42: w_score += 12
    
    if c_vol_z >= 2.0 and c_z < 0: w_score += 25
    elif c_vol_z >= 0.8 and c_z < 0: w_score += 10
    
    if c_price >= c_ma200: w_score += 15  # Запас прочности бычьего рынка
    
    prob = w_score / 100.0
    confidence = min(100, int(len(df) / 450 * 100))
    
    return df, c_price, c_z, prob, confidence, (low_t, up_t), c_rsi, c_vol_z, dv_bull

# ============================================================
# 7. ЗАГРУЗКА ДАННЫХ С РЕЗЕРВНЫМ ИСТОЧНИКОМ
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
            return df[["date", "close", "volume"]]
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
            return df[["date", "close", "volume"]]
    except:
        return None

# ============================================================
# 8. ИНТЕГРАЦИЯ ДЕП СИК
# ============================================================

def call_deepseek_v3(asset, price, z, prob, sig, rsi, vol_z, div, stress, fund):
    key = st.secrets.get("DEEPSEEK_API_KEY", "")
    if not key:
        return "❌ Ключ интеграции ИИ DeepSeek отсутствует."
        
    f_text = ""
    if fund:
        f_text = f"Капитализация: ${fund['market_cap']:,.0f}, Объем 24ч: ${fund['volume_24h']:,.0f}, ATH: ${fund['ath']:,.0f}."
        
    prompt = f"""Проведи глубокий экспресс-анализ {asset}:
МЕТРИКИ: Цена: ${price:,.4f}, Z-Score к MA90: {z:.2f}, RSI: {rsi:.1f}, Объёмное отклонение: {vol_z:+.1f}σ, Бычья дивергенция RSI: {'ДА' if div else 'НЕТ'}.
Итоговая вероятность истинного дна системы: {prob*100:.1f}%. Сигнал: {sig}.
{f_text}
ИСТОРИЯ СТРЕСС-ТЕСТОВ РЫНКА (10.11.25 и 06.02.26):
- Результат теста ноября 2025: {stress['t1_status']} ({stress['t1_perf'] if stress['t1_perf'] else 0:.1f}%)
- Результат теста февраля 2026: {stress['t2_status']} ({stress['t2_perf'] if stress['t2_perf'] else 0:.1f}%)

Напиши профессиональный вывод (4-5 предложений) на русском. Оцени: прошёл ли актив проверку на выкупаемость исторически, подтверждено ли дно текущими объемами и дивергенцией, и конкретный торговый план."""

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
# 9. СЕНДБАР / ПАНЕЛЬ УПРАВЛЕНИЯ
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
    st.caption("📈 **Математическая модель v3**")
    st.caption("Интегрированы исторические точки стресс-тестов: 10.11.2025 и 06.02.2026 для оценки долгосрочной живучести активов.")

# ============================================================
# 10. ИНИЦИАЛИЗАЦИЯ И РАСЧЁТ ДАННЫХ
# ============================================================

is_c = asset in CRYPTO_LIST
is_v = asset in VETERAN_LIST

fund = None
if is_c and asset in COINGECKO_IDS:
    fund = get_coingecko_fundamentals(COINGECKO_IDS[asset])

with st.spinner("Синхронизация потоков данных..."):
    raw = load_crypto_data(asset) if is_c else load_stock_data(asset)

if raw is None or len(raw) < 90:
    st.error(f"❌ Недостаточно данных для запуска ядра математического анализа по {asset}.")
    st.stop()

df, c_price, c_z, prob, conf, (low_thr, upper_thr), c_rsi, c_vol_z, dv_bull = calculate_metrics_adaptive(raw)
sig_t, sig_c = get_signal_adaptive(c_z, low_thr, upper_thr, is_v)
stress = analyze_stress_tests(df)

# Вывод основных виджетов
st.header(f"📊 Паспорт актива: {asset}")
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("💰 ТЕКУЩАЯ ЦЕНА", f"${c_price:,.4f}" if c_price < 1 else f"${c_price:,.2f}")
with c2: st.metric("📊 Z-SCORE (MA90)", f"{c_z:.2f}")
with c3:
    p_c = "#22c55e" if prob >= 0.65 else "#eab308" if prob >= 0.35 else "#ef4444"
    st.markdown(f"""
        <div style='background: {p_c}10; padding: 10px; border-radius: 8px; border: 1px solid {p_c}30; text-align: center;'>
            <p style='color: gray; margin:0; font-size:12px; font-weight:bold;'>ВЕРОЯТНОСТЬ ДНА</p>
            <p style='color: {p_c}; font-size:22px; font-weight:bold; margin:3px 0 0 0;'>{prob*100:.1f}%</p>
        </div>
    """, unsafe_allow_html=True)
with c4:
    st.metric("📈 RSI (14)", f"{c_rsi:.1f}", delta="ДИВЕРГЕНЦИЯ ✅" if dv_bull else None, delta_color="inverse")

st.markdown(f"""
<div style='background: linear-gradient(135deg, #111827 0%, #1f2937 100%); padding:12px; border-radius:10px; margin: 15px 0; border-left: 5px solid {sig_c};'>
    <p style='margin:0; color:#f3f4f6; font-size:14px;'>
        <b>Рекомендация детектора:</b> <span style='color:{sig_c}; font-weight:bold;'>{sig_t}</span> 
        | Адаптивный коридор нормы: {low_thr:.2f}σ — {upper_thr:.2f}σ
    </p>
</div>
""", unsafe_allow_html=True)

# Исторический аудит выкупаемости
st.subheader("🛡️ Результаты стресс-тестов на выкупаемость рынка")
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

# ============================================================
# ИИ ВЫВОД
# ============================================================
st.markdown("---")
if st.button("🧠 Запустить нейросетевой аудит DeepSeek v3", type="primary"):
    with st.spinner("Нейросеть считывает маркеры устойчивости..."):
        ai_res = call_deepseek_v3(asset, c_price, c_z, prob, sig_t, c_rsi, c_vol_z, dv_bull, stress, fund)
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
    color = get_color(df_chart["z_score"].iloc[i], lower_thr, upper_thr)
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

# ИСПРАВЛЕНИЕ ОШИБКИ ВЫЧИСЛЕНИЯ СТРОКИ ШАБЛОНА ПОДСКАЗКИ HINTS
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
    hoverinfo="text",
    text=hover_texts,
    name="Инфо", hovertemplate="%{text}<extra></extra>"
))

price_range = df_chart["close"].max() / (df_chart["close"].min() + 1e-10)
fig.update_layout(height=480, template="plotly_dark", xaxis_title="", yaxis_title="Цена (USD)", yaxis_type="log" if price_range > 5 else "linear", hovermode="x unified", legend=dict(orientation="h", y=1.02, x=0), font=dict(family="Times New Roman", size=13))
st.plotly_chart(fig, use_container_width=True)

# Раздел Z-Score
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_chart["date"], y=df_chart["z_score"], mode="lines", name="Z-Score", line=dict(color="#00d4ff", width=2), fill="tozeroy", fillcolor="rgba(0, 212, 255, 0.05)"))
fig2.add_hline(y=lower_thr, line_dash="dash", line_color="#22c55e", annotation_text=f"ДНО ({lower_thr:.2f}σ)")
fig2.add_hline(y=upper_thr, line_dash="dash", line_color="#ef4444", annotation_text=f"ПИК ({upper_thr:.2f}σ)")
fig2.update_layout(height=240, template="plotly_dark", yaxis_range=[-3.5, 3.5], font=dict(family="Times New Roman", size=13))
st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# 12. СВОДНАЯ ТАБЛИЦА ВСЕХ АКТИВОВ
# ============================================================
st.markdown("---")
st.subheader("📋 СВОДНАЯ МАТРИЦА АКТИВОВ")

@st.cache_data(ttl=300)
def build_summary_table():
    all_assets = {**{c: "Криптовалюта" for c in CRYPTO_LIST}, **{s: "Акция" for s in STOCK_LIST}}
    rows = []
    for symbol, atype in all_assets.items():
        df_t = load_crypto_data(symbol) if atype == "Криптовалюта" else load_stock_data(symbol)
        if df_t is None or len(df_t) < 30:
            continue
        res = calculate_metrics_adaptive(df_t)
        if res[0] is None:
            continue
        (_, price, z, prob, conf, (lt, ut), rsi_v, vol_z, dv_bull) = res
        sig, _ = get_signal_adaptive(z, lt, ut, symbol in VETERAN_LIST)
        st_res = analyze_stress_tests(df_t)
        
        r1s = "—" if "Нет" in st_res["t1_status"] else "✅" if "Прошёл" in st_res["t1_status"] else "❌"
        r2s = "—" if "Нет" in st_res["t2_status"] else "✅" if "Прошёл" in st_res["t2_status"] else "❌"
        r1p = st_res["t1_perf"]
        r2p = st_res["t2_perf"]
        
        rows.append({
            "Символ": symbol,
            "Тип": atype,
            "Цена": f"${price:,.4f}" if price < 1 else f"${price:,.2f}",
            "Z-Score": f"{z:.2f}",
            "RSI": f"{rsi_v:.1f}",
            "Div RSI": "✅" if dv_bull else "—",
            "Объём σ": f"{vol_z:.2f}",
            "Вер-ть дна": f"{prob*100:.1f}%",
            "Сигнал": sig.split(" ")[1] if " " in sig else sig,
            "Дно 10.11.25": f"{r1s} ({r1p:+.0f}%)" if r1p is not None else r1s,
            "Дно 06.02.26": f"{r2s} ({r2p:+.0f}%)" if r2p is not None else r2s,
        })
    return rows

with st.spinner("Загрузка сводной таблицы..."):
    summary = build_summary_table()

if summary:
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
    st.caption("❌ Не восстановился после дна → кандидат на выход | ✅ Восстановился → прошёл тест выкупаемости")

# ============================================================
# 13. ПОДВАЛ
# ============================================================
moscow_tz = timezone(timedelta(hours=3))
moscow_time = datetime.now(moscow_tz)

st.markdown("---")
st.caption(f"📅 Время обновления расчётов: {moscow_time.strftime('%Y-%m-%d %H:%M:%S')} (МСК) | Версия Детектора: v3.0 (Матричный скоринг + Паспорт устойчивости)")
