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
        if df.empty:
            raise ValueError("no data")

        # 1) Предрасчёты/фичи (оставь только то, что нужно твоей стратегии)
        df = df.copy()
        # пример: твои фичи/паттерны
        # df["my_feature"] = ...

        # 2) Твоя логика определения направления и уверенности
        # --- ВСТАВЬ СВОИ ПРАВИЛА ---
        action = "BUY"      # "SHORT" | "WAIT"
        confidence = 0.68   # 0..1

        # 3) Уровни риска/таргеты (по твоим правилам)
        price = float(df["Close"].iloc[-1])
        # примеры; замени своей формулой
        entry = price
        tp1   = price + 2.0
        tp2   = price + 5.0
        sl    = price - 1.5

        # 4) Ключевая отметка/зоны (используй любые твои ориентиры)
        key_mark   = price - 0.5
        upper_zone = price + 3.0
        lower_zone = price - 2.5

        # 5) Текст для UI (без раскрытия внутренней математики)
        rationale = (
            f"Сценарий на {'покупку' if action=='BUY' else ('шорт' if action=='SHORT' else 'ожидание')}. "
            f"План: вход {entry:.2f} • стоп {sl:.2f} • цели {tp1:.2f}/{tp2:.2f}. "
            f"Слежение за диапазоном {lower_zone:.2f}–{upper_zone:.2f}."
        )

        # 6) Валидация монотонности уровней (полезно для тестов)
        if action == "BUY":
            assert tp2 >= tp1 >= entry >= sl
        elif action == "SHORT":
            assert tp2 <= tp1 <= entry <= sl

        return SignalModel(
            symbol=symbol, horizon=horizon, action=action, confidence=float(round(confidence,2)),
            entry=round(entry,2), tp1=round(tp1,2), tp2=round(tp2,2), sl=round(sl,2),
            key_mark=round(key_mark,2), upper_zone=round(upper_zone,2), lower_zone=round(lower_zone,2),
            rationale=rationale
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
