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
# 2. ДАННЫЕ О КРИТИЧЕСКИХ ДНЯХ
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
# 5. ИСТОРИЧЕСКИЙ АНАЛИЗ УСТОЙЧИВОСТИ
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
        future_t2 = df_temp[df_temp["date"] == (t2_row["date"].values[0] + np.timedelta64(25, 'D'))]
        if not future_t2.empty:
            p_f2 = future_t2["close"].values[0]
            change = ((p_f2 - p_t2) / p_t2) * 100
            results["t2_perf"] = change
            results["t2_status"] = "✅ Прошёл (Выкуплен)" if change >= 15 else "❌ Не восстановился"
            
    return results

# ============================================================
# 6. ОСНОВНОЙ РАСЧЁТ (ИСПРАВЛЕНА ОШИБКА fillna)
# ============================================================

def calculate_metrics_adaptive(df):
    if df is None or len(df) < 90:
        return (None,) * 9
        
    df = df.copy()
    
    df["ma90"] = df["close"].rolling(window=90, min_periods=30).mean()
    df["std90"] = df["close"].rolling(window=90, min_periods=30).std()
    df["z_score"] = (df["close"] - df["ma90"]) / (df["std90"] + 1e-10)
    
    df["rsi"] = calculate_rsi(df, 14)
    df["ma200"] = df["close"].rolling(window=200, min_periods=50).mean()
    
    df["v_mean"] = df["volume"].rolling(window=30, min_periods=10).mean()
    df["v_std"] = df["volume"].rolling(window=30, min_periods=10).std()
    df["vol_z"] = (df["volume"] - df["v_mean"]) / (df["v_std"] + 1e-10)
    
    # ИСПРАВЛЕНО: убран method="bfill"
    df = df.bfill().fillna(0)
    
    c_price = df["close"].iloc[-1]
    c_z = df["z_score"].iloc[-1]
    c_rsi = df["rsi"].iloc[-1]
    c_vol_z = df["vol_z"].iloc[-1]
    c_ma200 = df["ma200"].iloc[-1]
    
    low_t, up_t = get_adaptive_thresholds(df["z_score"].values)
    dv_bull = detect_rsi_divergence(df, 35)
    
    w_score = 0
    if c_z <= low_t: w_score += 35
    elif c_z < -0.8: w_score += 15
    
    if c_rsi <= 32: w_score += 25
    elif c_rsi <= 42: w_score += 12
    
    if c_vol_z >= 2.0 and c_z < 0: w_score += 25
    elif c_vol_z >= 0.8 and c_z < 0: w_score += 10
    
    if c_price >= c_ma200: w_score += 15
    
    prob = w_score / 100.0
    confidence = min(100, int(len(df) / 450 * 100))
    
    return df, c_price, c_z, prob, confidence, (low_t, up_t), c_rsi, c_vol_z, dv_bull

# ============================================================
# 7. ЗАГРУЗКА ДАННЫХ
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
    return None

# ============================================================
# 8. AI-АНАЛИЗ (DeepSeek)
# ============================================================

def call_deepseek_v3(asset, price, z, prob, sig, rsi, vol_z, div, stress, fund):
    key = st.secrets.get("DEEPSEEK_API_KEY", "")
    if not key:
        return "❌ Ключ DeepSeek не найден."
        
    f_text = ""
    if fund:
        f_text = f"Капитализация: ${fund['market_cap']:,.0f}, Объем 24ч: ${fund['volume_24h']:,.0f}, ATH: ${fund['ath']:,.0f}."
        
    prompt = f"""Проведи анализ {asset}:
Цена: ${price:,.4f}, Z-Score: {z:.2f}, RSI: {rsi:.1f}, Объём σ: {vol_z:+.1f}, Дивергенция: {'ДА' if div else 'НЕТ'}.
Вероятность дна: {prob*100:.1f}%. Сигнал: {sig}. {f_text}
Стресс-тесты: 10.11.25: {stress['t1_status']} ({stress['t1_perf'] if stress['t1_perf'] else 0:.1f}%)
06.02.26: {stress['t2_status']} ({stress['t2_perf'] if stress['t2_perf'] else 0:.1f}%)

Напиши краткий вывод (3-4 предложения) на русском."""

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    data = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "max_tokens": 500, "temperature": 0.5}
    try:
        r = requests.post(url, headers=headers, json=data, timeout=25)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
    except:
        pass
    return "❌ Ошибка AI"

# ============================================================
# 9. ИНТЕРФЕЙС
# ============================================================

with st.sidebar:
    st.header("⚙️ НАСТРОЙКИ")
    market = st.radio("Сектор", ["Криптовалюты", "Фондовый рынок"])
    if market == "Криптовалюты":
        asset = st.selectbox("Актив", CRYPTO_LIST)
    else:
        asset = st.selectbox("Актив", STOCK_LIST)
    st.caption("Математическая модель v3")
    st.caption("Стресс-тесты: 10.11.2025 и 06.02.2026")

is_crypto = asset in CRYPTO_LIST
is_veteran = asset in VETERAN_LIST

fund = None
if is_crypto and asset in COINGECKO_IDS:
    fund = get_coingecko_fundamentals(COINGECKO_IDS[asset])

with st.spinner("Загрузка..."):
    raw = load_crypto_data(asset) if is_crypto else load_stock_data(asset)

if raw is None or len(raw) < 90:
    st.error(f"Недостаточно данных для {asset}")
    st.stop()

result = calculate_metrics_adaptive(raw)
if result[0] is None:
    st.error("Ошибка расчёта метрик")
    st.stop()

df, price, z, prob, conf, (low, high), rsi, vol_z, div = result
sig_text, sig_color = get_signal_adaptive(z, low, high, is_veteran)
stress = analyze_stress_tests(df)

st.header(f"📊 {asset}")

c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("💰 ЦЕНА", f"${price:,.2f}" if price > 1 else f"${price:,.4f}")
with c2: st.metric("📊 Z-SCORE", f"{z:.2f}")
with c3: st.metric("📈 RSI", f"{rsi:.1f}", delta="Дивергенция ✅" if div else None)
with c4: st.metric("🎯 ВЕРОЯТНОСТЬ ДНА", f"{prob*100:.1f}%")

st.markdown(f"""
<div style='background:#111827; padding:12px; border-radius:8px; margin:15px 0; border-left:5px solid {sig_color};'>
    <b>Сигнал:</b> <span style='color:{sig_color}'>{sig_text}</span>
</div>
""", unsafe_allow_html=True)

st.subheader("🛡️ СТРЕСС-ТЕСТЫ ВЫКУПАЕМОСТИ")
col_a, col_b = st.columns(2)
with col_a:
    t1c = "#22c55e" if "✅" in stress["t1_status"] else "#ef4444"
    st.markdown(f"**10.11.2025**<br><span style='color:{t1c}'>{stress['t1_status']}</span>", unsafe_allow_html=True)
with col_b:
    t2c = "#22c55e" if "✅" in stress["t2_status"] else "#ef4444"
    st.markdown(f"**06.02.2026**<br><span style='color:{t2c}'>{stress['t2_status']}</span>", unsafe_allow_html=True)

st.markdown("---")
if st.button("🤖 AI-АНАЛИЗ (DeepSeek)"):
    with st.spinner("DeepSeek анализирует..."):
        analysis = call_deepseek_v3(asset, price, z, prob, sig_text, rsi, vol_z, div, stress, fund)
    st.info(analysis)

# ============================================================
# 10. ГРАФИКИ
# ============================================================

st.subheader("📈 ГРАФИК ЦЕНЫ + MA90 + MA200")
df_chart = df.tail(500).copy()

def get_color(z_val, low_val, high_val):
    if z_val <= low_val: return "#00ff66"
    elif z_val <= low_val * 0.6: return "#39ff14"
    elif z_val <= -0.5: return "#bfff00"
    elif z_val <= 0.5: return "#e5e7eb"
    elif z_val <= 1.2: return "#ffb703"
    elif z_val <= high_val: return "#ff5500"
    else: return "#ff0055"

fig = go.Figure()
for i in range(len(df_chart)-1):
    col = get_color(df_chart["z_score"].iloc[i], low, high)
    fig.add_trace(go.Scatter(
        x=[df_chart["date"].iloc[i], df_chart["date"].iloc[i+1]],
        y=[df_chart["close"].iloc[i], df_chart["close"].iloc[i+1]],
        mode="lines", line=dict(color=col, width=3.5), showlegend=False, hoverinfo="skip"
    ))

if "ma90" in df_chart.columns:
    fig.add_trace(go.Scatter(x=df_chart["date"], y=df_chart["ma90"], mode="lines", name="MA90", line=dict(color="white", width=1.2, dash="dot"), opacity=0.5))
if "ma200" in df_chart.columns:
    fig.add_trace(go.Scatter(x=df_chart["date"], y=df_chart["ma200"], mode="lines", name="MA200", line=dict(color="#f59e0b", width=1.5, dash="dash"), opacity=0.7))

hover_list = []
for d, p, z_val, r_val, v_val in zip(df_chart["date"], df_chart["close"], df_chart["z_score"], df_chart["rsi"], df_chart["vol_z"]):
    p_str = f"{p:,.2f}" if p > 1 else f"{p:,.4f}"
    hover_list.append(f"📅 {d.strftime('%Y-%m-%d')}<br>💰 ${p_str}<br>📊 Z: {z_val:.2f}<br>📈 RSI: {r_val:.1f}<br>📦 Объём σ: {v_val:.2f}")
fig.add_trace(go.Scatter(x=df_chart["date"], y=df_chart["close"], mode="markers", marker=dict(color="rgba(0,0,0,0)", size=1), hoverinfo="text", text=hover_list, name="Инфо"))

price_range = df_chart["close"].max() / (df_chart["close"].min() + 1e-10)
fig.update_layout(height=450, template="plotly_dark", yaxis_type="log" if price_range > 5 else "linear", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.subheader("📉 Z-SCORE + АДАПТИВНЫЕ ПОРОГИ")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_chart["date"], y=df_chart["z_score"], mode="lines", name="Z-Score", line=dict(color="#00d4ff", width=2), fill="tozeroy", fillcolor="rgba(0,212,255,0.05)"))
fig2.add_hline(y=low, line_dash="dash", line_color="#22c55e", annotation_text=f"ДНО ({low:.2f}σ)")
fig2.add_hline(y=high, line_dash="dash", line_color="#ef4444", annotation_text=f"ПИК ({high:.2f}σ)")
fig2.update_layout(height=250, template="plotly_dark", yaxis_range=[-3.5, 3.5])
st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# 11. СВОДНАЯ ТАБЛИЦА
# ============================================================

st.markdown("---")
st.subheader("📋 СВОДНАЯ ТАБЛИЦА")

@st.cache_data(ttl=300)
def build_summary():
    all_assets = {**{c: "Криптовалюта" for c in CRYPTO_LIST}, **{s: "Акция" for s in STOCK_LIST}}
    rows = []
    for sym, typ in all_assets.items():
        dft = load_crypto_data(sym) if typ == "Криптовалюта" else load_stock_data(sym)
        if dft is None or len(dft) < 30:
            continue
        res = calculate_metrics_adaptive(dft)
        if res[0] is None:
            continue
        _, pr, zz, prb, _, (lt, ut), rs, vz, dv = res
        sg, _ = get_signal_adaptive(zz, lt, ut, sym in VETERAN_LIST)
        sts = analyze_stress_tests(dft)
        t1r = "✅" if "Прошёл" in sts["t1_status"] else ("❌" if "Не восст" in sts["t1_status"] else "—")
        t2r = "✅" if "Прошёл" in sts["t2_status"] else ("❌" if "Не восст" in sts["t2_status"] else "—")
        rows.append({
            "Символ": sym, "Тип": typ, "Цена": f"${pr:,.2f}" if pr > 1 else f"${pr:,.4f}",
            "Z": f"{zz:.2f}", "RSI": f"{rs:.1f}", "Вер-ть": f"{prb*100:.1f}%",
            "Сигнал": sg.split()[1] if len(sg.split()) > 1 else sg,
            "Тест 10.11": t1r, "Тест 06.02": t2r
        })
    return rows

with st.spinner("Загрузка таблицы..."):
    summ = build_summary()
if summ:
    st.dataframe(pd.DataFrame(summ), use_container_width=True, hide_index=True)

# ============================================================
# 12. ПОДВАЛ
# ============================================================

moscow_tz = timezone(timedelta(hours=3))
moscow_time = datetime.now(moscow_tz)
st.markdown("---")
st.caption(f"Обновлено: {moscow_time.strftime('%Y-%m-%d %H:%M:%S')} (МСК) | Детектор v3.0")
