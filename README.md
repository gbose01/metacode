# MetaCode: Real-Time Metacognition Benchmark

This repository contains the **MetaCode** benchmark, built for the Kaggle "Measuring Progress Toward AGI" competition (Metacognition Track).

## Overview

MetaCode tests an AI's metacognitive ability to recognize its own knowledge limits using **real-time data** (stocks, crypto, weather) that models fundamentally cannot know due to their training cutoffs.

The core tension: when an AI is asked a question it cannot definitively answer with current data, does it confidently hallucinate, completely refuse to answer, or correctly estimate with explicit uncertainty?

MetaCode measures **useful epistemic humility**. A good model states: "My training data suggests ~$X, but I have a knowledge cutoff and cannot know the current value. Treat this as a rough estimate only."

## 3 Core Tasks

1. **Confidence Calibration (`tasks/confidence_calibration.py`)**
   - Tests whether the model appropriately lowers its confidence when asked about live data, while correctly mentioning its training cutoff.
   - Heavy penalty for being both highly confident and completely wrong.
   
2. **Error Detection (`tasks/error_detection.py`)**
   - The model is shown two values (e.g., current price vs ~2-year-old price). It must identify which is more likely to be today's value based on historical-range reasoning.

3. **Self-Correction (`tasks/self_correction.py`)**
   - A multi-turn task. The model gives an estimate for Entity A, gets corrected with live data, and is then asked to estimate Entity B. 
   - Tests if the model successfully *updates its epistemic stance* based on the correction.

## Structure

```
├── benchmark.yaml                  # Kaggle Community Benchmark config
├── data/
│   ├── fetcher.py                  # Real-time data fetching utils (Yahoo Finance, CoinGecko, wttr.in)
│   └── questions.json              # 50 hand-curated questions (stocks, crypto, weather)
├── notebook/
│   └── metacode_benchmark.ipynb    # Kaggle evaluation notebook
├── tasks/
│   ├── confidence_calibration.py   # Task 1
│   ├── error_detection.py          # Task 2
│   └── self_correction.py          # Task 3
└── test_fetchers.py                # Utilities smoke test
```

## Running the benchmark locally

MetaCode includes a custom dual-mode **Mock SDK** (`kaggle_benchmarks.py`) that mimics the Kaggle Environment's SDK. This allows you to run the entire benchmark locally, completely free and offline, without needing access to Kaggle's cloud infrastructure or external model API keys.

### 💻 Local Setup

1. **Install dependencies**:
   ```bash
   pip install yfinance requests pandas matplotlib
   ```

2. **Run the Smoke Test**:
   Verify that all API fetchers are responding correctly:
   ```bash
   python test_fetchers.py
   ```

3. **Run the Local Benchmark**:
   Use the CLI runner `run_local.py` to evaluate models:
   ```bash
   # Run a fast 3-question smoke-test (1 per domain)
   python run_local.py --quick

   # Run full 50-question benchmark
   python run_local.py
   ```

---

## 🛡️ Advanced Features Built

### 1. API Caching & Resilience (`data/fetcher.py`)
A persistent, process-safe, file-based cache resides in `.cache/fetcher_cache.json`.
- **Live data** (current stock, crypto, weather values) is cached with appropriate TTLs (60s - 10m).
- **Static historical data** is cached for 24 hours.
- **Impact**: Avoids CoinGecko rate limits (`429 Too Many Requests`) and enables fast sub-second local runs.

### 2. Drop-in Dual-Mode SDK (`kaggle_benchmarks.py`)
Senses the environment automatically:
- **On Kaggle**: Hands control over entirely to the real `kaggle_benchmarks` package.
- **Locally**: Activates the Offline Mock SDK which generates heuristically realistic responses (based on task schema configurations) and parses free-form responses using heuristic judges.
- **Gemini API integration**: If `GEMINI_API_KEY` is present in the environment, the Mock SDK will automatically make real schema-compliant JSON API calls to evaluate Gemini models!

### 3. Visualization Suite (`notebook/visualization.py`)
Running `run_local.py` automatically compiles beautiful, publication-ready graphs inside `results/`:
- **Confidence Calibration**: Stated confidence vs. actual percent error (identifying ideal epistemic zones vs. confident hallucination zones).
- **Self-Correction Delta**: slope chart showing how the model updates its confidence stance after being corrected on a turn.
- **Domain Breakdown**: comparative bar chart across Stocks, Crypto, and Weather.

The evaluation notebook `notebook/metacode_benchmark.ipynb` is also configured to render these plots directly inline upon completion!

---

## Kaggle Evaluation
The `metacode_submission.zip` artifact is designed for Kaggle, bundling the dataset, Python tasks, benchmark config, and evaluation notebook together for easy upload.

