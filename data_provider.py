from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import os, requests
import pandas as pd

@dataclass
class FetchResult:
    df: pd.DataFrame
    source: str

class DataProvider:
    def __init__(self, polygon_api_key: str | None = None):
        self.polygon_api_key = polygon_api_key

    def history(self, symbol: str, period: str = "6mo", interval: str = "1d") -> FetchResult:
        if self.polygon_api_key:
            try:
                df = self._polygon_history(symbol, interval=interval, lookback_days=180 if period.endswith("mo") else 365)
                if df is not None and not df.empty:
                    return FetchResult(df=df, source="polygon")
            except Exception:
                pass
        try:
            import yfinance as yf
            df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
            if df is not None and not df.empty:
                df = df.reset_index().rename(columns=str.title)
                df = df[["Date","Open","High","Low","Close","Volume"]]
                return FetchResult(df=df, source="yahoo")
        except Exception:
            pass
        demo_path = os.path.join(os.path.dirname(__file__), "..", "data", "demo", f"{symbol.lower()}_demo.csv")
        demo_path = os.path.abspath(demo_path)
        if os.path.exists(demo_path):
            df = pd.read_csv(demo_path)
            return FetchResult(df=df[["Date","Open","High","Low","Close","Volume"]], source="demo-csv")
        raise RuntimeError("No data available from Polygon, Yahoo or demo CSV.")

    def _polygon_history(self, symbol: str, interval: str = "1d", lookback_days: int = 180) -> pd.DataFrame | None:
        if interval != "1d":
            raise NotImplementedError("Polygon demo supports only 1d interval here.")
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=lookback_days)
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}?adjusted=true&sort=asc&limit=50000&apiKey={self.polygon_api_key}"
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return None
        js = r.json()
        if not js or "results" not in js or not js["results"]:
            return None
        rows = []
        for it in js["results"]:
            rows.append({
                "Date": pd.to_datetime(it["t"], unit="ms").tz_localize("UTC").tz_convert(None),
                "Open": it["o"], "High": it["h"], "Low": it["l"], "Close": it["c"], "Volume": it.get("v", 0)
            })
        df = pd.DataFrame(rows)
        return df
