import streamlit as st
import pandas as pd
import numpy as np
import pathlib
import plotly.graph_objects as go

# ----------- Загрузка данных с защитой -----------
@st.cache_data(ttl=1800)
def load_history(symbol: str, period: str = "6mo", interval: str = "1d"):
    # 1) Попробовать Yahoo Finance
    try:
        import yfinance as yf
        df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
        if df is not None and not df.empty:
            df = df.reset_index().rename(columns=str.title)
            return df[["Date", "Open", "High", "Low", "Close", "Volume"]], "yahoo"
    except Exception:
        pass

    # 2) Попробовать локальный CSV
    base = pathlib.Path(__file__).resolve().parent
    demo = base / "data" / "demo" / f"{symbol.lower()}_demo.csv"
    if demo.exists():
        df = pd.read_csv(demo)
        return df[["Date", "Open", "High", "Low", "Close", "Volume"]], "demo-csv"

    # 3) Автогенерация синтетических данных
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=40)
    rng = np.random.RandomState(abs(hash(symbol)) % 10_000)
    price = 470.0
    rows = []
    for d in dates:
        drift = rng.uniform(-2.0, 2.4)
        o = price + rng.uniform(-1.0, 1.0)
        c = max(1.0, o + drift)
        h = max(o, c) + rng.uniform(0.1, 1.5)
        l = min(o, c) - rng.uniform(0.1, 1.5)
        v = int(rng.uniform(20_000_000, 80_000_000))
        rows.append([d, round(o, 2), round(h, 2), round(l, 2), round(c, 2), v])
        price = c
    df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    return df, "auto-demo"


# ----------- Анализ сигнала (упрощённая версия) -----------
def get_signal(df: pd.DataFrame):
    if df.empty:
        return "WAIT", "Недостаточно данных"
    last_close = df["Close"].iloc[-1]
    prev_close = df["Close"].iloc[-2]
    if last_close > prev_close:
        return "BUY", f"Цена растёт ({prev_close} → {last_close})"
    elif last_close < prev_close:
        return "SHORT", f"Цена падает ({prev_close} → {last_close})"
    else:
        return "WAIT", "Цена без изменений"


# ----------- UI Streamlit -----------
st.set_page_config(page_title="US Stocks — Demo", layout="centered")
st.title("📈 US Stocks — Demo (всегда работает)")

tickers_input = st.text_input("Tickers", value="AAPL,MSFT,NVDA").upper()
tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
symbol = st.selectbox("Выбери тикер", tickers)

if st.button("Сгенерировать сигнал"):
    df, src_type = load_history(symbol)
    if df is None or df.empty:
        st.error("❌ Нет данных")
    else:
        sig, reason = get_signal(df)
        st.success(f"Сигнал: **{sig}** — {reason}  _(данные: {src_type})_")

        # Мини-график
        fig = go.Figure(data=[go.Candlestick(
            x=df["Date"],
            open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"]
        )])
        fig.update_layout(height=400, title=f"График {symbol} ({src_type})")
        st.plotly_chart(fig, use_container_width=True)
