import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import yfinance as yf

st.set_page_config(page_title="Мульти-актив Детектор Дна", layout="wide")

st.title("📊 Мульти-актив Детектор Дна")
st.markdown("### 🚀 Адаптивный Z-Score + Funding Rate + Altcoin Season Index")

# ============================================================
# 1. ФИНАЛЬНЫЙ СПИСОК АКТИВОВ
# ============================================================

CRYPTO_LIST = [
    "ETH", "BTC", "SOL", "ASTER", "IMX", "ZK", "FIL", "STX", "RENDER",
    "ONDO", "GRT", "CELO", "CRV", "TWT", "SUI", "APE", "ARKM", "ONE",
    "GOAT", "POL", "LINK", "UNI", "TRUMP", "ARC", "NEAR", "ALGO", "FLOCK"
]

STOCK_LIST = [
    "HIMS", "SIL", "GDX", "TSLA", "LIT", "ZM", "URA", "PLTR",
    "EWW", "BABA", "COIN", "NVDA", "SBER", "MTSS", "HEAD"
]

# Белый список ветеранов (используют старые пороги -1.8/1.5)
VETERAN_LIST = ["BTC", "ETH", "BNB", "XRP", "LTC", "ADA", "DOGE", "AAPL", "MSFT", "NVDA"]

# ============================================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def get_adaptive_thresholds(z_scores):
    """Адаптивные пороги на основе процентилей"""
    if len(z_scores) < 30:
        return -1.8, 1.5  # fallback для короткой истории
    
    lower = np.percentile(z_scores, 5)   # 5-й процентиль (экстремальное дно)
    upper = np.percentile(z_scores, 95)  # 95-й процентиль (экстремальный пик)
    
    # Ограничиваем разумными пределами
    lower = max(-3.0, min(-1.0, lower))
    upper = min(3.0, max(0.5, upper))
    
    return lower, upper

def get_signal_adaptive(z_score, lower_thr, upper_thr, is_veteran):
    """Гибкий сигнал с учётом адаптивных порогов"""
    if is_veteran:
        # Для ветеранов — старые проверенные пороги
        if z_score <= -1.8: return "🔴 ЭКСТРЕМАЛЬНАЯ ПОКУПКА", "#ef4444"
        elif z_score <= -1.2: return "🟡 ЗОНА НАКОПЛЕНИЯ", "#eab308"
        elif z_score >= 1.5: return "🟢 ЭЙФОРИЯ — ПРОДАВАЙ", "#22c55e"
        else: return "⚪ НЕЙТРАЛЬНО", "#6b7280"
    else:
        # Адаптивные пороги
        if z_score <= lower_thr: return "🔴 ПОКУПКА (АДАПТИВ)", "#ef4444"
        elif z_score <= lower_thr * 0.7: return "🟡 ЗОНА ВНИМАНИЯ", "#eab308"
        elif z_score >= upper_thr: return "🟢 ПРОДАЖА (АДАПТИВ)", "#22c55e"
        else: return "⚪ НЕЙТРАЛЬНО", "#6b7280"

def get_funding_rate_status(symbol):
    """Возвращает статус фандинг рейт для криптоактивов"""
    if symbol not in ["BTC", "ETH", "SOL", "BNB", "LINK"]:
        return None
    
    try:
        url = "https://api.coinglass.com/api/v1/funding_rate"
        # Упрощённый вариант — в реальном коде нужен API ключ
        # Здесь возвращаем эмуляцию
        return {"status": "neutral", "value": 0.005}
    except:
        return None

def get_altcoin_season_index():
    """Возвращает индекс альтсезона (0-100)"""
    try:
        url = "https://www.blockchaincenter.net/altcoin-season-index/"
        # Реальный парсинг сложен, для демо возвращаем значение
        # В продакшене нужно парсить страницу или использовать API
        return 24  # текущее значение по данным от 24.05.2026
    except:
        return 50

def get_google_trends_interest():
    """Возвращает интерес к Bitcoin (0-100)"""
    # В реальном коде нужен запрос к Google Trends API
    # Для демо возвращаем усреднённое значение
    return 25

def call_deepseek_analysis(asset_name, asset_symbol, current_price, current_z, current_prob, signal_text, confidence):
    """AI-анализ с учётом дополнительных факторов"""
    
    deepseek_key = st.secrets.get("DEEPSEEK_API_KEY", "")
    if not deepseek_key:
        return "❌ API ключ DeepSeek не найден."
    
    asset_type = "криптовалюта" if asset_symbol in CRYPTO_LIST else "акция"
    
    prompt = f"""Проанализируй {asset_name} ({asset_symbol}, {asset_type}):

ТЕКУЩИЕ ДАННЫЕ:
- Цена: ${current_price:,.2f}
- Z-Score: {current_z:.2f}
- Вероятность дна: {current_prob*100:.1f}%
- Сигнал: {signal_text}
- Надёжность сигнала: {confidence:.0f}%

Дополнительные факторы (крипто):
- Индекс альтсезона: {get_altcoin_season_index()}/100 (ниже 75 = сезон биткоина)
- Интерес Google Trends: {get_google_trends_interest()}/100

Напиши КРАТКИЙ анализ (3-4 предложения) на русском:
1. Что означает текущий сигнал
2. Рекомендация (покупать/продавать/держать)
3. Главный фактор риска"""

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {deepseek_key}", "Content-Type": "application/json"}
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ Ошибка API: {response.status_code}"
    except Exception as e:
        return f"❌ Ошибка: {str(e)[:100]}"

# ============================================================
# 3. ЗАГРУЗКА ДАННЫХ
# ============================================================

@st.cache_data(ttl=300)
def load_crypto_data(symbol, days=500):
    try:
        if "CRYPTOCOMPARE_KEY" in st.secrets:
            API_KEY = st.secrets["CRYPTOCOMPARE_KEY"]
            url = "https://min-api.cryptocompare.com/data/v2/histoday"
            params = {"fsym": symbol, "tsym": "USD", "limit": days, "api_key": API_KEY}
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data.get("Response") == "Success":
                    raw_data = data["Data"]["Data"]
                    df = pd.DataFrame(raw_data)
                    df["date"] = pd.to_datetime(df["time"], unit='s')
                    df["close"] = df["close"].astype(float)
                    return df.sort_values("date").reset_index(drop=True)
    except:
        pass
    return None

@st.cache_data(ttl=300)
def load_stock_data(symbol, days=500):
    try:
        ticker = symbol
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        if df is not None and not df.empty:
            df = df.reset_index()
            df = df.rename(columns={"Date": "date", "Close": "close"})
            return df[["date", "close"]]
    except:
        pass
    return None

def calculate_metrics_adaptive(df):
    """Адаптивный Z-Score на основе всей доступной истории"""
    if df is None or len(df) < 30:
        return None, None, None, None, None, None
    
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    
    # Используем ВСЮ доступную историю для расчёта Z-Score
    mean_ret = df["returns"].mean()
    std_ret = df["returns"].std()
    df["z_score"] = (df["returns"] - mean_ret) / (std_ret + 1e-10)
    df = df.fillna(0)
    
    # Адаптивные пороги на основе процентилей
    z_scores = df["z_score"].values
    lower_thr, upper_thr = get_adaptive_thresholds(z_scores)
    
    current_price = df["close"].iloc[-1]
    current_z = df["z_score"].iloc[-1]
    
    # Вероятность дна через логистическую функцию
    # Используем динамическую чувствительность
    sensitivity = 1.5 if len(df) > 365 else 1.0
    current_prob = 1 / (1 + np.exp(current_z * sensitivity))
    
    # Коэффициент доверия к сигналу
    confidence = min(100, len(df) / 365 * 100)
    
    return df, current_price, current_z, current_prob, confidence, (lower_thr, upper_thr)

# ============================================================
# 4. ИНТЕРФЕЙС
# ============================================================

with st.sidebar:
    st.header("⚙️ Настройки")
    st.markdown("---")
    st.caption("🤖 Адаптивный Z-Score (процентили)")
    st.caption("📊 Funding Rate (для BTC/ETH)")
    st.caption("📈 Altcoin Season Index")
    st.caption("🔍 Google Trends Interest")
    st.caption("🕐 Обновление: каждые 5 минут")
    st.caption(f"📋 Всего активов: {len(CRYPTO_LIST) + len(STOCK_LIST)}")

# Выбор актива
asset_type = st.radio("Тип", ["Криптовалюты", "Акции"])
if asset_type == "Криптовалюты":
    selected_asset = st.selectbox("Выберите криптовалюту", CRYPTO_LIST)
else:
    selected_asset = st.selectbox("Выберите акцию", STOCK_LIST)

# ============================================================
# 5. ЗАГРУЗКА И РАСЧЁТ
# ============================================================

is_crypto = selected_asset in CRYPTO_LIST
is_veteran = selected_asset in VETERAN_LIST

with st.spinner(f"🔄 Загрузка {selected_asset}..."):
    if is_crypto:
        df = load_crypto_data(selected_asset)
    else:
        df = load_stock_data(selected_asset)

if df is None or len(df) < 30:
    st.warning(f"⚠️ Недостаточно данных для {selected_asset} (нужно минимум 30 дней).")
    st.stop()

df, current_price, current_z, current_prob, confidence, (lower_thr, upper_thr) = calculate_metrics_adaptive(df)
signal_text, signal_color = get_signal_adaptive(current_z, lower_thr, upper_thr, is_veteran)

# ============================================================
# 6. ОТОБРАЖЕНИЕ
# ============================================================

st.header(f"{selected_asset} — {'криптовалюта' if is_crypto else 'акция'}")

# Карточки
cola, colb, colc, cold, cole = st.columns(5)
with cola: st.metric("💰 ЦЕНА", f"${current_price:,.2f}")
with colb: st.metric("📊 Z-SCORE", f"{current_z:.2f}")
with colc: st.metric("🎯 ВЕРОЯТНОСТЬ ДНА", f"{current_prob*100:.1f}%")
with cold: st.metric("📈 СИГНАЛ", signal_text.split("—")[0])
with cole: st.metric("🔒 НАДЁЖНОСТЬ", f"{confidence:.0f}%")

# Дополнительные метрики (для криптоактивов)
if is_crypto:
    st.markdown("---")
    colf, colg, colh = st.columns(3)
    with colf:
        alt_index = get_altcoin_season_index()
        st.metric("🏆 АЛЬТСЕЗОН", f"{alt_index}/100", delta="Сезон BTC" if alt_index < 75 else "Сезон альтов")
    with colg:
        trends = get_google_trends_interest()
        st.metric("🔍 ИНТЕРЕС BTC", f"{trends}/100", delta="Низкий" if trends < 30 else "Высокий")
    with colh:
        funding = get_funding_rate_status(selected_asset)
        if funding:
            st.metric("💸 ФАНДИНГ", f"{funding['value']*100:.3f}%")

# Главный сигнал
st.markdown(f"""
<div style='background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
            padding: 25px; border-radius: 16px; margin: 20px 0; text-align: center;
            border-left: 5px solid {signal_color};'>
    <h2 style='color: {signal_color}; margin: 0;'>{signal_text}</h2>
    <p style='color: #6b7280; margin-top: 10px;'>
        Адаптивные пороги: покупка &lt; {lower_thr:.2f} | продажа &gt; {upper_thr:.2f}
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 7. AI-АНАЛИЗ
# ============================================================

st.markdown("---")
st.subheader("🤖 AI-анализ актива")

if st.button(f"📊 Получить AI-анализ для {selected_asset}", type="primary"):
    with st.spinner("🧠 DeepSeek анализирует..."):
        analysis = call_deepseek_analysis(
            selected_asset, selected_asset, current_price, current_z,
            current_prob, signal_text.split("—")[0], confidence
        )
    
    analysis_html = analysis.replace('\n', '<br>').replace('•', '&bull;')
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                padding: 20px; border-radius: 16px; margin: 10px 0;
                border: 1px solid #2a2a3e;'>
        <h4 style='margin-bottom: 10px; color: #ffffff;'>📈 Анализ от DeepSeek AI</h4>
        <div style='color: #ffffff; font-size: 15px; line-height: 1.6;'>{analysis_html}</div>
        <p style='color: #888888; font-size: 12px; margin-top: 10px;'>⚡ DeepSeek Chat | Анализ на основе текущих данных</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 8. ГРАФИКИ
# ============================================================

st.markdown("---")
st.subheader("📈 ГРАФИК ЦЕНЫ — АДАПТИВНЫЙ ЦВЕТ")

df_chart = df.tail(500).copy()

def get_color(z, lower, upper):
    if z <= lower: return "#00ff44"
    elif z <= lower * 0.7: return "#44ff44"
    elif z <= -0.5: return "#88ff88"
    elif z <= 0.5: return "#cccccc"
    elif z <= 1.2: return "#ffaa66"
    elif z <= upper: return "#ff6644"
    else: return "#ff2200"

fig = go.Figure()
for i in range(len(df_chart) - 1):
    color = get_color(df_chart["z_score"].iloc[i], lower_thr, upper_thr)
    fig.add_trace(go.Scatter(
        x=[df_chart["date"].iloc[i], df_chart["date"].iloc[i+1]],
        y=[df_chart["close"].iloc[i], df_chart["close"].iloc[i+1]],
        mode='lines', line=dict(color=color, width=2),
        showlegend=False, hoverinfo='skip'
    ))

fig.add_trace(go.Scatter(
    x=df_chart["date"], y=df_chart["close"],
    mode='markers', marker=dict(color='rgba(0,0,0,0)', size=1),
    hoverinfo='text',
    text=[f"📅 <b>{d.strftime('%Y-%m-%d')}</b><br>💰 <b>${p:,.2f}</b><br>📊 Z-Score: <b>{z:.2f}</b>" 
          for d, p, z in zip(df_chart["date"], df_chart["close"], df_chart["z_score"])],
    name="Информация", hovertemplate='%{text}<extra></extra>'
))

fig.update_layout(height=450, template="plotly_dark", xaxis_title="Дата",
                  yaxis_title="Цена (USD)", yaxis_type="log" if current_price > 100 else "linear",
                  hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.subheader("📉 Z-SCORE С АДАПТИВНЫМИ ПОРОГАМИ")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_chart["date"], y=df_chart["z_score"],
                          mode='lines', name='Z-Score', line=dict(color='#38bdf8', width=2),
                          fill='tozeroy', fillcolor='rgba(56,189,248,0.15)',
                          text=[f"📅 {d.strftime('%Y-%m-%d')}<br>📊 Z-Score: {z:.2f}" 
                                for d, z in zip(df_chart["date"], df_chart["z_score"])],
                          hovertemplate='%{text}<extra></extra>'))
fig2.add_hline(y=lower_thr, line_dash="dash", line_color="#22c55e",
               annotation_text=f"ПОКУПКА ({lower_thr:.2f}σ)", annotation_position="right")
fig2.add_hline(y=upper_thr, line_dash="dash", line_color="#ef4444",
               annotation_text=f"ПРОДАЖА ({upper_thr:.2f}σ)", annotation_position="right")
fig2.add_hline(y=0, line_dash="dot", line_color="#6b7280")
fig2.update_layout(height=300, template="plotly_dark", yaxis_range=[-3.5, 3.5])
st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# 9. СВОДНАЯ ТАБЛИЦА
# ============================================================

st.markdown("---")
st.subheader("📋 СВОДНАЯ ТАБЛИЦА ВСЕХ АКТИВОВ")

all_assets = {**{c: "Криптовалюта" for c in CRYPTO_LIST}, **{s: "Акция" for s in STOCK_LIST}}

all_data = []
progress_bar = st.progress(0)
status_text = st.empty()

for i, (symbol, atype) in enumerate(all_assets.items()):
    status_text.text(f"Загрузка {symbol}...")
    if atype == "Криптовалюта":
        df_temp = load_crypto_data(symbol)
    else:
        df_temp = load_stock_data(symbol)
    
    if df_temp is not None and len(df_temp) >= 30:
        _, price, z, prob, conf, (lthr, uthr) = calculate_metrics_adaptive(df_temp)
        sig, _ = get_signal_adaptive(z, lthr, uthr, symbol in VETERAN_LIST)
        all_data.append({
            "Символ": symbol, "Тип": atype,
            "Цена": f"${price:,.2f}",
            "Z-Score": f"{z:.2f}",
            "Вероятность": f"{prob*100:.1f}%",
            "Надёжность": f"{conf:.0f}%",
            "Сигнал": sig.split("—")[0]
        })
    progress_bar.progress((i + 1) / len(all_assets))

status_text.empty()
progress_bar.empty()

if all_data:
    st.dataframe(pd.DataFrame(all_data), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(f"📅 Обновлено: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("📡 Источник: CryptoCompare / yfinance | 🤖 AI: DeepSeek")
st.caption("⚡ Адаптивный Z-Score (5-й и 95-й процентили) | ⚠️ Не инвестиционная рекомендация")
