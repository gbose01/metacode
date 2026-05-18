"""
Live data fetching for the MetaCode benchmark.

Three data sources — all free, no API key required:
  - yfinance   : stock prices (Yahoo Finance)
  - CoinGecko  : crypto prices (public API)
  - wttr.in    : weather / temperature
"""

import time
import random
import json
import os
import tempfile
from datetime import datetime, timedelta

import requests
import yfinance as yf


_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")
_CACHE_FILE = os.path.join(_CACHE_DIR, "fetcher_cache.json")

def _get_cache():
    if not os.path.exists(_CACHE_FILE):
        return {}
    try:
        with open(_CACHE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _set_cache(cache_data):
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(dir=_CACHE_DIR)
        with os.fdopen(fd, "w") as f:
            json.dump(cache_data, f, indent=2)
        os.replace(temp_path, _CACHE_FILE)
    except Exception:
        pass

def _cached_fetch(key: str, ttl_seconds: float, fetch_fn):
    cache = _get_cache()
    now = time.time()
    if key in cache:
        val, timestamp = cache[key]
        if now - timestamp < ttl_seconds:
            return val
    
    val = fetch_fn()
    
    cache = _get_cache()
    cache[key] = (val, now)
    _set_cache(cache)
    return val


# ---------------------------------------------------------------------------
# Stocks
# ---------------------------------------------------------------------------

def get_stock_price(symbol: str) -> float:
    def fetch():
        ticker = yf.Ticker(symbol)
        try:
            price = ticker.fast_info.last_price
            if price is not None:
                return float(price)
        except (AttributeError, KeyError):
            pass
        hist = ticker.history(period="2d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
        raise ValueError(f"Could not fetch price for {symbol}")
    
    return _cached_fetch(f"stock_live_{symbol}", 60, fetch)


def get_stock_price_historical(symbol: str, years_ago: int = 2) -> float:
    """Fetch closing price from approximately `years_ago` years back."""
    def fetch():
        target = datetime.now() - timedelta(days=years_ago * 365)
        start = (target - timedelta(days=5)).strftime("%Y-%m-%d")
        end = (target + timedelta(days=5)).strftime("%Y-%m-%d")
        hist = yf.Ticker(symbol).history(start=start, end=end)
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
        current = get_stock_price(symbol)
        return round(current * random.uniform(0.55, 0.75), 2)
    
    return _cached_fetch(f"stock_hist_{symbol}_{years_ago}", 86400, fetch)


# ---------------------------------------------------------------------------
# Crypto  (CoinGecko free tier — ~30 req/min, add small sleep between calls)
# ---------------------------------------------------------------------------

_COINGECKO_BASE = "https://api.coingecko.com/api/v3"
_HEADERS = {"User-Agent": "metacode-benchmark/1.0"}


def get_crypto_price(coingecko_id: str) -> float:
    def fetch():
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
    
    return _cached_fetch(f"crypto_live_{coingecko_id}", 60, fetch)


def get_crypto_price_historical(coingecko_id: str, years_ago: int = 2) -> float:
    """Fetch price from approximately `years_ago` years back.
    Uses /coins/{id}/history?date= which is available on the free CoinGecko tier.
    """
    def fetch():
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
        
        current = get_crypto_price(coingecko_id)
        return round(current * random.uniform(0.3, 0.6), 8)
    
    return _cached_fetch(f"crypto_hist_{coingecko_id}_{years_ago}", 86400, fetch)


# ---------------------------------------------------------------------------
# Weather  (wttr.in — no API key, no rate limits)
# ---------------------------------------------------------------------------

def get_temperature_celsius(wttr_query: str) -> float:
    """Return current temperature in °C for the given city / location query."""
    def fetch():
        url = f"https://wttr.in/{wttr_query}?format=j1"
        r = requests.get(url, headers=_HEADERS, timeout=15)
        r.raise_for_status()
        return float(r.json()["current_condition"][0]["temp_C"])
    
    return _cached_fetch(f"weather_live_{wttr_query}", 600, fetch)


def get_temperature_celsius_historical(wttr_query: str) -> float:
    """
    wttr.in does not expose historical data via its public API.
    We generate a plausible distractor by shifting the current temperature
    to simulate the opposite season (±20-28°C), which is clearly wrong
    but not absurd.
    """
    def fetch():
        current = get_temperature_celsius(wttr_query)
        offset = random.uniform(20, 28)
        if current > 15:
            return round(current - offset, 1)
        elif current < 5:
            return round(current + offset, 1)
        else:
            direction = -1 if current >= 10 else 1
            return round(current + direction * offset, 1)
    
    return _cached_fetch(f"weather_hist_{wttr_query}", 600, fetch)


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
