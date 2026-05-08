"""
Quick smoke test for all three data sources.
Run: python test_fetchers.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.fetcher import (
    get_stock_price,
    get_stock_price_historical,
    get_crypto_price,
    get_crypto_price_historical,
    get_temperature_celsius,
    get_temperature_celsius_historical,
    format_value,
)

passed_count = 0
failed_count = 0


def check(label, fn, domain, *args):
    global passed_count, failed_count
    try:
        result = fn(*args)
        assert isinstance(result, float), f"Expected float, got {type(result)}"
        assert result != 0.0, "Got zero -- likely a fetch failure"
        print(f"  PASS  {label}: {format_value(domain, result)}")
        passed_count += 1
        return True
    except Exception as e:
        print(f"  FAIL  {label}: {e}")
        failed_count += 1
        return False


print("\n--- STOCKS ---")
check("GOOGL live",       get_stock_price,           "stock", "GOOGL")
check("AAPL live",        get_stock_price,           "stock", "AAPL")
check("TSLA live",        get_stock_price,           "stock", "TSLA")
check("GOOGL historical", get_stock_price_historical, "stock", "GOOGL")

print("\n--- CRYPTO ---")
check("bitcoin live",       get_crypto_price,           "crypto", "bitcoin")
check("ethereum live",      get_crypto_price,           "crypto", "ethereum")
check("solana live",        get_crypto_price,           "crypto", "solana")
check("shiba-inu live",     get_crypto_price,           "crypto", "shiba-inu")
check("bitcoin historical", get_crypto_price_historical, "crypto", "bitcoin")

print("\n--- WEATHER ---")
check("London live",       get_temperature_celsius,           "weather", "London")
check("Tokyo live",        get_temperature_celsius,           "weather", "Tokyo")
check("Dubai live",        get_temperature_celsius,           "weather", "Dubai")
check("London distractor", get_temperature_celsius_historical, "weather", "London")

total = passed_count + failed_count
print(f"\n{'='*40}")
print(f"  {passed_count}/{total} passed", "-- All good!" if failed_count == 0 else f"-- {failed_count} FAILED")
print()
