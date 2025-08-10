import streamlit as st
import pandas as pd
import numpy as np
import pathlib
import plotly.graph_objects as go

# ---------- ДАННЫЕ: Yahoo -> CSV -> автоген -----------
@st.cache_data(ttl=1800)
def load_history(symbol: str, period: str = "6mo", interval: str = "1d"):
    # 1) Yahoo Finance
    try:
        import yfinance as yf
        df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
        if df is not None and not df.empty:
            df = df.reset_index().rename(columns=str.title)
            return df[["Date","Open","High","Low","Close","Volume"]], "yahoo"
    except Exception:
        pass
    # 2) Локальный CSV
    base = pathlib.Path(__file__).resolve().parent
    demo = base / "data" / "demo" / f"{symbol.lower()}_demo.csv"
    if demo.exists():
        df = pd.read_csv(demo)
        return df[["Date","Open","High","Low","Close","Volume"]], "demo-csv"
    # 3) Автогенерация (всегда есть)
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
        rows.append([d, round(o,2), round(h,2), round(l,2), round(c,2), v])
        price = c
    df = pd.DataFrame(rows, columns=["Date","Open","High","Low","Close","Volume"])
    return df, "auto-demo"

# ---------- ИНДИКАТОРЫ/УРОВНИ ----------
def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = np.maximum(h-l, np.maximum((h-pc).abs(), (l-pc).abs()))
    w = min(window, max(5, len(df)//2))
    return tr.rolling(w).mean().fillna(tr.ewm(span=w, adjust=False).mean())

def landmarks(df: pd.DataFrame) -> pd.DataFrame:
    h1, l1, c1 = df["High"].shift(1), df["Low"].shift(1), df["Close"].shift(1)
    key_mark  = (h1 + l1 + c1) / 3.0            # ключевая отметка (без слова pivot)
    upper     = 2*key_mark - l1                 # верхняя зона сопротивления
    lower     = 2*key_mark - h1                 # нижняя зона спроса
    return pd.DataFrame({"key_mark":key_mark, "upper_zone":upper, "lower_zone":lower})

def compute_levels(df: pd.DataFrame, horizon: str) -> dict:
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
    else:  # swing (среднесрок)
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
        "price": round(px,2), "atr": round(a,2),
        "key_mark": round(KM,2), "upper_zone": round(UZ,2), "lower_zone": round(LZ,2),
        "action": action, "confidence": round(float(score),2),
        "entry": round(entry,2), "tp1": round(tp1,2), "tp2": round(tp2,2), "sl": round(sl,2)
    }

def build_note(symbol: str, hz_ui: str, c: dict) -> str:
    side = "покупку" if c["action"]=="BUY" else ("шорт" if c["action"]=="SHORT" else "ожидание")
    bias = "выше" if c["price"] >= c["key_mark"] else "ниже"
    tone = ("сбалансированный риск/потенциал" if 0.45 <= c["confidence"] <= 0.55
            else "повышенная уверенность" if c["confidence"] > 0.55 else "консервативный сценарий")
    horizon_note = {"Краткосрок":"3–10 торговых дней","Среднесрок":"3–8 недель","Долгосрок":"1–3 месяца"}[hz_ui]
    return (f"Для {symbol} цена {bias} ключевой ценовой отметки, что формирует сценарий на {side}. "
            f"Волатильность (ATR≈{c['atr']}) учтена в уровнях; {tone}. "
            f"План: вход {c['entry']} • стоп {c['sl']} • цели {c['tp1']}/{c['tp2']}. "
            f"Контекст: {horizon_note}. Следим за реакцией в диапазоне {c['lower_zone']}–{c['upper_zone']}.")

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

# ---------- UI ----------
st.set_page_config(page_title="US Stocks — Demo", layout="wide")
st.title("📈 US Stocks — Demo (с уровнями и разбором)")
tickers_input = st.text_input("Tickers", value="QQQ,AAPL,MSFT,NVDA").upper()
tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
hz_map = {"Краткосрок":"short","Среднесрок":"swing","Долгосрок":"position"}
hz_ui = st.selectbox("Горизонт", list(hz_map.keys()), index=1)
horizon = hz_map[hz_ui]
symbol = st.selectbox("Выбери тикер", tickers, index=0)

if st.button("Сгенерировать сигнал"):
    df, src = load_history(symbol)
    levels = compute_levels(df, horizon)
    # бейдж
    color = "#16a34a" if levels["action"]=="BUY" else ("#dc2626" if levels["action"]=="SHORT" else "#6b7280")
    st.markdown(f"<div style='display:inline-block;padding:6px 10px;border-radius:8px;background:{color};color:white;font-weight:600'>{levels['action']}</div>", unsafe_allow_html=True)
    # метрики
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Entry", levels["entry"]); c2.metric("TP1", levels["tp1"])
    c3.metric("TP2", levels["tp2"]);    c4.metric("SL",  levels["sl"])
    d1,d2,d3 = st.columns(3)
    d1.metric("Ключевая отметка", levels["key_mark"])
    d2.metric("Верхняя зона сопротивления", levels["upper_zone"])
    d3.metric("Нижняя зона спроса", levels["lower_zone"])
    st.metric("Confidence", levels["confidence"])
    # график
    st.plotly_chart(plot_levels(df.tail(60), levels), use_container_width=True)
    # текст
    st.write(build_note(symbol, hz_ui, levels))
    # таблица
    with st.expander("Последние строки данных"):
        st.dataframe(df.tail(12))
