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

```bash
pip install -r requirements.txt
# Requires installing the Kaggle benchmarks SDK
python test_fetchers.py
```

## Kaggle Evaluation
The `metacode_submission.zip` artifact is designed for Kaggle, bundling the dataset, Python tasks, benchmark config, and evaluation notebook together for easy upload.
