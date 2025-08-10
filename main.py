import os, pathlib, sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---- Настройки по умолчанию ----
DEFAULT_TICKERS = "QQQ,AAPL,MSFT,NVDA"

# ---- Загрузка данных: Yahoo -> локальный CSV (demo) ----
@st.cache_data(ttl=1800)
def load_history(symbol: str, period: str = "6mo", interval: str = "1d") -> tuple[pd.DataFrame, str]:
    # 1) Yahoo
    try:
        import yfinance as yf
        df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
        if df is not None and not df.empty:
            df = df.reset_index().rename(columns=str.title)
            return df[["Date","Open","High","Low","Close","Volume"]], "yahoo"
    except Exception:
        pass
    # 2) demo CSV
    base = pathlib.Path(__file__).resolve().parent
    demo = base / "data" / "demo" / f"{symbol.lower()}_demo.csv"
    if demo.exists():
        df = pd.read_csv(demo)
        return df[["Date","Open","High","Low","Close","Volume"]], "demo-csv"
    raise RuntimeError("Нет данных ни из Yahoo, ни из demo CSV")

# ---- Индикаторы/уровни (упрощённые, безопасные) ----
def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    w = min(window, max(5, len(df)//2))
    return tr.rolling(w).mean().fillna(tr.ewm(span=w, adjust=False).mean())

def landmarks(df: pd.DataFrame) -> pd.DataFrame:
    h1, l1, c1 = df["High"].shift(1), df["Low"].shift(1), df["Close"].shift(1)
    key_mark  = (h1 + l1 + c1) / 3.0           # «ключевая ценовая отметка»
    upper     = 2*key_mark - l1                # «верхняя зона сопротивления»
    lower     = 2*key_mark - h1                # «нижняя зона спроса»
    return pd.DataFrame({"key_mark":key_mark, "upper_zone":upper, "lower_zone":lower})

def compute_signal(df: pd.DataFrame, horizon: str):
    df = df.copy()
    df["ATR14"] = atr(df, 14)
    df = pd.concat([df, landmarks(df)], axis=1)
    x = df.iloc[-1]

    px = float(x["Close"])
    a  = float(x["ATR14"] or 0.0)
    KM = float(x["key_mark"] or px)
    UZ = float(x["upper_zone"] or px)
    LZ = float(x["lower_zone"] or px)

    # пресеты горизонта
    if horizon == "short":
        scale = 2.0; tp1m, tp2m, slm = 0.5, 1.0, 0.8
    elif horizon == "position":
        scale = 3.5; tp1m, tp2m, slm = 0.8, 1.8, 1.2
    else:  # swing
        scale = 2.8; tp1m, tp2m, slm = 0.6, 1.4, 1.0

    score = 0.5 if a <= 0 else max(0.0, min(1.0, 0.5 + (px - KM)/(scale*a)))
    action = "BUY" if score >= 0.6 else ("SHORT" if score <= 0.4 else "WAIT")

    if action == "BUY":
        entry = px; tp1 = max(px+tp1m*a, UZ); tp2 = max(px+tp2m*a, KM+(UZ-KM)*1.5); sl = px - slm*a
    elif action == "SHORT":
        entry = px; tp1 = min(px-tp1m*a, LZ); tp2 = min(px-tp2m*a, KM-(KM-LZ)*1.5); sl = px + slm*a
    else:
        entry = px; tp1 = px+(tp1m+0.2)*a; tp2 = px+(tp2m+0.4)*a; sl = px-(slm-0.2)*a

    return {
        "action": action, "confidence": round(float(score),2),
        "entry": round(entry,2), "tp1": round(tp1,2), "tp2": round(tp2,2), "sl": round(sl,2),
        "key_mark": round(KM,2), "upper_zone": round(UZ,2), "lower_zone": round(LZ,2),
        "atr": round(a,2)
    }

def plot_levels(df: pd.DataFrame, c: dict) -> go.Figure:
    fig = go.Figure([go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"])])
    lines = {"Entry":c["entry"], "TP1":c["tp1"], "TP2":c["tp2"], "SL":c["sl"],
             "Ключевая отметка":c["key_mark"], "Верхняя зона сопротивления":c["upper_zone"], "Нижняя зона спроса":c["lower_zone"]}
    colors = {"Entry":"#2563eb","TP1":"#16a34a","TP2":"#16a34a","SL":"#dc2626",
              "Ключевая отметка":"#6b7280","Верхняя зона сопротивления":"#f59e0b","Нижняя зона спроса":"#10b981"}
    for label, y in lines.items():
        fig.add_hline(y=y, line_width=1, line_dash="dot", line_color=colors.get(label, "#999"),
                      annotation_text=label, annotation_position="top left")
    x0 = df["Date"].iloc[-min(len(df), 30)]
    x1 = df["Date"].iloc[-1]
    fig.add_shape(type="rect", x0=x0, x1=x1, y0=min(c["entry"], c["tp2"]), y1=max(c["entry"], c["tp2"]),
                  fillcolor="rgba(34,197,94,0.08)", line=dict(width=0), layer="below")
    fig.add_shape(type="rect", x0=x0, x1=x1, y0=min(c["sl"], c["entry"]), y1=max(c["sl"], c["entry"]),
                  fillcolor="rgba(239,68,68,0.08)", line=dict(width=0), layer="below")
    fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=420, showlegend=False)
    return fig

# ---- UI ----
st.set_page_config(page_title="AI Trading — One-file Fallback", layout="wide")
st.title("AI Trading — One-file Demo")
st.caption("Аварийная версия без импортов из core/config. Работает с Yahoo или demo CSV.")

tickers = st.text_input("Tickers", value=DEFAULT_TICKERS).upper()
symbols = [t.strip() for t in tickers.split(",") if t.strip()]
h_map = {"Краткосрок":"short","Среднесрок":"swing","Долгосрок":"position"}
hz_ui = st.selectbox("Горизонт", list(h_map.keys()), index=1)
horizon = h_map[hz_ui]
symbol = st.selectbox("Тикер", symbols, index=0)

colA, colB = st.columns([1,2])
with colA:
    if st.button("Сгенерировать сигнал"):
        try:
            df, source = load_history(symbol)
            st.session_state["df"] = df
            st.session_state["source"] = source
            st.session_state["sig"] = compute_signal(df, horizon)
        except Exception as e:
            st.error(str(e))

with colB:
    sig = st.session_state.get("sig")
    df  = st.session_state.get("df")
    src = st.session_state.get("source","—")
    if sig and df is not None:
        st.subheader(f"{symbol} — {hz_ui} | source: {src}")
        color = "#16a34a" if sig["action"]=="BUY" else ("#dc2626" if sig["action"]=="SHORT" else "#6b7280")
        st.markdown(f"<div style='display:inline-block;padding:6px 10px;border-radius:8px;background:{color};color:white;font-weight:600'>{sig['action']}</div>", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Entry", sig["entry"]); m2.metric("TP1", sig["tp1"])
        m3.metric("TP2", sig["tp2"]);    m4.metric("SL",  sig["sl"])
        d1, d2, d3 = st.columns(3)
        d1.metric("Ключевая отметка", sig["key_mark"])
        d2.metric("Верхняя зона сопротивления", sig["upper_zone"])
        d3.metric("Нижняя зона спроса", sig["lower_zone"])
        st.metric("Confidence", sig["confidence"])

        st.plotly_chart(plot_levels(df.tail(60), sig), use_container_width=True)

        note = (
            f"Цена {'выше' if sig['entry']>=sig['key_mark'] else 'ниже'} ключевой отметки; "
            f"волатильность (ATR≈{sig['atr']}) учтена. План: вход {sig['entry']} • "
            f"стоп {sig['sl']} • цели {sig['tp1']}/{sig['tp2']}. "
            f"Наблюдаем реакцию в диапазоне {sig['lower_zone']}–{sig['upper_zone']}."
        )
        st.write(note)
        with st.expander("Последние строки данных"):
            st.dataframe(df.tail(12))
    else:
        st.info("Нажмите «Сгенерировать сигнал». Если Yahoo недоступен — возьмём data/demo/qqq_demo.csv.")
