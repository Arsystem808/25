from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    polygon_api_key: str | None = None
    default_tickers: list[str] = ["QQQ","AAPL","MSFT","NVDA"]
    default_horizon: str = "swing"

def get_settings() -> Settings:
    key = os.getenv("POLYGON_API_KEY") or None
    tickers_env = os.getenv("DEFAULT_TICKERS", "QQQ,AAPL,MSFT,NVDA")
    default_tickers = [t.strip().upper() for t in tickers_env.split(",") if t.strip()]
    horizon = os.getenv("DEFAULT_HORIZON", "swing").strip().lower()
    return Settings(polygon_api_key=key, default_tickers=default_tickers, default_horizon=horizon)
