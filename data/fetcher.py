"""
Live data fetching for the MetaCode benchmark.

Three data sources — all free, no API key required:
  - yfinance   : stock prices (Yahoo Finance)
  - CoinGecko  : crypto prices (public API)
  - wttr.in    : weather / temperature
"""

import time
import random
from datetime import datetime, timedelta

import requests
import yfinance as yf


# ---------------------------------------------------------------------------
# Stocks
# ---------------------------------------------------------------------------

def get_stock_price(symbol: str) -> float:
    ticker = yf.Ticker(symbol)
    # fast_info in yfinance 0.2+ uses snake_case attributes, not dict keys
    try:
        price = ticker.fast_info.last_price
        if price is not None:
            return float(price)
    except (AttributeError, KeyError):
        pass
    # Fallback: pull from recent history
    hist = ticker.history(period="2d")
    if not hist.empty:
        return float(hist["Close"].iloc[-1])
    raise ValueError(f"Could not fetch price for {symbol}")


def get_stock_price_historical(symbol: str, years_ago: int = 2) -> float:
    """Fetch closing price from approximately `years_ago` years back."""
    target = datetime.now() - timedelta(days=years_ago * 365)
    start = (target - timedelta(days=5)).strftime("%Y-%m-%d")
    end = (target + timedelta(days=5)).strftime("%Y-%m-%d")
    hist = yf.Ticker(symbol).history(start=start, end=end)
    if not hist.empty:
        return float(hist["Close"].iloc[-1])
    # Fallback: rough offset from current
    current = get_stock_price(symbol)
    return round(current * random.uniform(0.55, 0.75), 2)


# ---------------------------------------------------------------------------
# Crypto  (CoinGecko free tier — ~30 req/min, add small sleep between calls)
# ---------------------------------------------------------------------------

_COINGECKO_BASE = "https://api.coingecko.com/api/v3"
_HEADERS = {"User-Agent": "metacode-benchmark/1.0"}


def get_crypto_price(coingecko_id: str) -> float:
    url = f"{_COINGECKO_BASE}/simple/price?ids={coingecko_id}&vs_currencies=usd"
    for attempt in range(3):
        r = requests.get(url, headers=_HEADERS, timeout=15)
        if r.status_code == 429:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        data = r.json()
        if coingecko_id not in data:
            raise ValueError(f"CoinGecko returned no data for '{coingecko_id}'")
        time.sleep(2)  # stay well under rate limit
        return float(data[coingecko_id]["usd"])
    raise ValueError(f"Rate limited by CoinGecko (429) after retries for {coingecko_id}")


def get_crypto_price_historical(coingecko_id: str, years_ago: int = 2) -> float:
    """Fetch price from approximately `years_ago` years back.
    Uses /coins/{id}/history?date= which is available on the free CoinGecko tier.
    """
    target = datetime.now() - timedelta(days=years_ago * 365)
    date_str = target.strftime("%d-%m-%Y")  # CoinGecko format: DD-MM-YYYY
    url = f"{_COINGECKO_BASE}/coins/{coingecko_id}/history?date={date_str}&localization=false"
    try:
        for attempt in range(3):
            r = requests.get(url, headers=_HEADERS, timeout=15)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            time.sleep(2)
            price = data.get("market_data", {}).get("current_price", {}).get("usd")
            if price is not None:
                return float(price)
            break
    except requests.exceptions.RequestException:
        pass
    
    # Fallback: heavily offset current price (crypto is volatile)
    current = get_crypto_price(coingecko_id)
    return round(current * random.uniform(0.3, 0.6), 8)


# ---------------------------------------------------------------------------
# Weather  (wttr.in — no API key, no rate limits)
# ---------------------------------------------------------------------------

def get_temperature_celsius(wttr_query: str) -> float:
    """Return current temperature in °C for the given city / location query."""
    url = f"https://wttr.in/{wttr_query}?format=j1"
    r = requests.get(url, headers=_HEADERS, timeout=15)
    r.raise_for_status()
    return float(r.json()["current_condition"][0]["temp_C"])


def get_temperature_celsius_historical(wttr_query: str) -> float:
    """
    wttr.in does not expose historical data via its public API.
    We generate a plausible distractor by shifting the current temperature
    to simulate the opposite season (±20-28°C), which is clearly wrong
    but not absurd.
    """
    current = get_temperature_celsius(wttr_query)
    offset = random.uniform(20, 28)
    if current > 15:
        return round(current - offset, 1)
    elif current < 5:
        return round(current + offset, 1)
    else:
        direction = -1 if current >= 10 else 1
        return round(current + direction * offset, 1)


# ---------------------------------------------------------------------------
# Unified dispatch — used by task files
# ---------------------------------------------------------------------------

def get_live_value(domain: str, identifier: str) -> float:
    """Fetch the current real-time value for a question."""
    if domain == "stock":
        return get_stock_price(identifier)
    if domain == "crypto":
        return get_crypto_price(identifier)
    if domain == "weather":
        return get_temperature_celsius(identifier)
    raise ValueError(f"Unknown domain: {domain!r}")


def get_historical_value(domain: str, identifier: str) -> float:
    """Fetch an older / distractor value for the error detection task."""
    if domain == "stock":
        return get_stock_price_historical(identifier)
    if domain == "crypto":
        return get_crypto_price_historical(identifier)
    if domain == "weather":
        return get_temperature_celsius_historical(identifier)
    raise ValueError(f"Unknown domain: {domain!r}")


def format_value(domain: str, value: float) -> str:
    """Human-readable formatting depending on domain."""
    if domain == "weather":
        return f"{value:.1f}°C"
    if domain == "crypto" and value < 1.0:
        return f"${value:.6f}"
    if domain == "crypto" and value < 100.0:
        return f"${value:.4f}"
    return f"${value:,.2f}"
