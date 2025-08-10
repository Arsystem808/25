from __future__ import annotations
from typing import Literal
import pandas as pd
import numpy as np
from core.schemas import SignalModel, Horizon

def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    w = min(window, max(5, len(df)//2))
    return tr.rolling(w).mean().fillna(tr.ewm(span=w, adjust=False).mean())

def _landmarks(df: pd.DataFrame) -> pd.DataFrame:
    h1, l1, c1 = df["High"].shift(1), df["Low"].shift(1), df["Close"].shift(1)
    key_mark = (h1 + l1 + c1) / 3.0
    upper_zone = 2*key_mark - l1
    lower_zone = 2*key_mark - h1
    return pd.DataFrame({"key_mark":key_mark, "upper_zone":upper_zone, "lower_zone":lower_zone})

class StrategyEngine:
    def compute(self, df: pd.DataFrame, symbol: str, horizon: Horizon) -> SignalModel:
        if df is None or df.empty:
            raise ValueError("Empty dataframe in StrategyEngine.compute")
        df = df.copy()
        df["ATR14"] = _atr(df, 14)
        df = pd.concat([df, _landmarks(df)], axis=1)
        latest = df.iloc[-1]

        px = float(latest["Close"])
        a  = float(latest["ATR14"] or 0.0)
        KM = float(latest["key_mark"] or px)
        UZ = float(latest["upper_zone"] or px)
        LZ = float(latest["lower_zone"] or px)

        if horizon == "short":
            scale = 2.0; tp_mult1, tp_mult2, sl_mult = 0.5, 1.0, 0.8
        elif horizon == "position":
            scale = 3.5; tp_mult1, tp_mult2, sl_mult = 0.8, 1.8, 1.2
        else:
            scale = 2.8; tp_mult1, tp_mult2, sl_mult = 0.6, 1.4, 1.0

        score = 0.5 if a <= 0 else max(0.0, min(1.0, 0.5 + (px - KM) / (scale * a)))
        action: Literal["BUY","SHORT","WAIT"]
        action = "BUY" if score >= 0.6 else ("SHORT" if score <= 0.4 else "WAIT")

        if action == "BUY":
            entry = px
            tp1 = max(px + tp_mult1*a, UZ)
            tp2 = max(px + tp_mult2*a, KM + (UZ - KM)*1.5)
            sl  = px - sl_mult*a
        elif action == "SHORT":
            entry = px
            tp1 = min(px - tp_mult1*a, LZ)
            tp2 = min(px - tp_mult2*a, KM - (KM - LZ)*1.5)
            sl  = px + sl_mult*a
        else:
            entry = px
            tp1 = px + (tp_mult1+0.2)*a
            tp2 = px + (tp_mult2+0.4)*a
            sl  = px - (sl_mult-0.2)*a

        note = self._note(symbol, horizon, score, a, KM, UZ, LZ, entry, tp1, tp2, sl)

        return SignalModel(
            symbol=symbol, horizon=horizon, action=action, confidence=float(round(score,2)),
            entry=round(entry,2), tp1=round(tp1,2), tp2=round(tp2,2), sl=round(sl,2),
            key_mark=round(KM,2), upper_zone=round(UZ,2), lower_zone=round(LZ,2),
            rationale=note
        )

    def _note(self, symbol: str, horizon: Horizon, score: float, atr: float, KM: float, UZ: float, LZ: float,
              entry: float, tp1: float, tp2: float, sl: float) -> str:
        side = "покупку" if score >= 0.6 else ("шорт" if score <= 0.4 else "ожидание")
        bias = "выше" if entry >= KM else "ниже"
        tone = (
            "сбалансированный риск/потенциал" if 0.45 <= score <= 0.55 else
            "повышенная уверенность" if score > 0.55 else
            "консервативный сценарий"
        )
        hz = {"short":"3–10 торговых дней","swing":"3–8 недель","position":"1–3 месяца"}[horizon]
        return (
            f"Для {symbol} цена {bias} ключевой ценовой отметки, что формирует сценарий на {side}. "
            f"Волатильность учтена в расчётах уровней (ATR≈{atr:.2f}); {tone}. "
            f"План: вход {entry:.2f} • стоп {sl:.2f} • цели {tp1:.2f}/{tp2:.2f}. "
            f"Контекст: {hz}. Наблюдаем за реакцией в диапазоне {LZ:.2f}–{UZ:.2f}."
        )
