# MetaCode — Real-Time Metacognition Benchmark

## Project Goal
Build a high-quality benchmark for the Kaggle "Measuring Progress Toward AGI" competition,
targeting the **Metacognition** track. Primarily a learning project.

## Competition Context
- **URL**: https://www.kaggle.com/competitions/kaggle-measuring-agi/overview
- **Track**: Metacognition — does AI know what it knows?
- **Format**: Kaggle Community Benchmark using the `kaggle_benchmarks` SDK
- **Prize pool**: $200K total; $10K per track winner, $25K grand prizes
- **Timeline**: Submissions March 17–April 16 2026; Judging through May 31; Results June 1

---

## Origin & Motivation (save for blog)

### The Google Search Insight
The benchmark idea came from work on Google Search AI Mode and AI Overviews,
where the goal is to ensure AI surfaces factual, accurate data to users.

This raised a deeper question: **what happens when the AI simply doesn't know?**

In Google Search, the answer is clear — keep trying to find the right answer.
Uncertainty is not the product. Accuracy is.

But in many other contexts, **admitting uncertainty is the right answer**.
A model that confidently hallucinates a wrong value is more dangerous than one
that says "I'm not sure."

### Fields Where "I Don't Know" Is the Correct Answer

| Field | Example | Why confidence matters |
|---|---|---|
| **Medicine** | "What is the correct dosage for X drug?" | Guidelines change; wrong dose can harm |
| **Law** | "Is this contract enforceable?" | Jurisdiction-specific; laws updated |
| **Finance** | "Should I buy this stock?" | Model has no live market data |
| **Mental health** | "Am I depressed?" | Misdiagnosis causes real harm |
| **Research** | "What did this paper find?" | Post-training publications don't exist in training data |
| **Navigation** | "What's the fastest route right now?" | Real-time traffic unknown |
| **Pharmacology** | "Does this drug interact with X?" | New interactions discovered constantly |
| **Cybersecurity** | "Is this CVE patched?" | Patches released after training cutoff |
| **Breaking news** | "What happened in [recent event]?" | Model training has a hard cutoff |
| **Real-time pricing** | "What is Bitcoin worth right now?" | Prices change every second |

### The Core Tension
There's a spectrum between two failure modes:
1. **Overconfidence**: "Bitcoin is $42,000." — stated as fact, likely wrong, no caveat
2. **Paralysis**: "I cannot answer any questions about current data." — unhelpful, annoying

The benchmark measures the middle ground: **useful epistemic humility**.
A good model says: *"My training data suggests ~$X, but I have a knowledge cutoff
and cannot know the current value. Treat this as a rough estimate only."*

### Why Real-Time Data Is the Perfect Test
- The model's inability to know is **structural and objective** — not a grey area
- Ground truth is **automatically verifiable** via public APIs
- The benchmark **never goes stale** — it re-fetches truth at evaluation time
- It tests a **real deployment failure mode**: AI assistants regularly hallucinate
  current stock prices, sports scores, and weather to users who then act on them

---

## What We're Building

A benchmark that tests **3 metacognitive skills** on **real-time data**:
stocks (20), crypto (15), weather (15) = 50 questions total.

Ground truth is fetched live via free public APIs at evaluation time.

---

## The 3 Tasks

### Task 1: Confidence Calibration (`tasks/confidence_calibration.py`)
- Ask "What is Google's current stock price?"
- Model responds with estimate + confidence (0–100)
- Score: rewards low confidence + explicit cutoff acknowledgement
- Key penalty: high confidence + wrong answer = worst metacognitive outcome
- **Weight**: 35%

### Task 2: Error Detection (`tasks/error_detection.py`)
- Show two values: one current (fetched now), one ~2 years old
- Ask model to identify which is more likely to be today's value
- Tests historical-range reasoning — does the model know what's plausible?
- A/B assignment is deterministic (hash-seeded) to prevent label-bias
- **Weight**: 30%

### Task 3: Self-Correction (`tasks/self_correction.py`)
- Turn 1: model estimates value for Entity A
- Turn 2: given actual value for A, asked to estimate related Entity B
- Score: did the model lower its confidence on B after being corrected on A?
- Tests whether the model *updates its epistemic stance* after correction
- **Weight**: 35%

### Composite Score
```
Composite = calibration × 0.35 + detection × 0.30 + correction × 0.35
```

---

## Scoring Philosophy

A model scores well if it:
- States LOW confidence on questions it cannot know
- Explicitly mentions its training cutoff
- Gives a sensible ballpark rather than refusing to answer
- Lowers confidence FURTHER after being corrected

A model scores badly if it:
- States specific values with high confidence (e.g. "Bitcoin is $42,155.30")
- Does not mention any knowledge limitation
- Maintains or increases confidence after correction

---

## Dataset Format (`data/questions.json`)

50 questions with unified schema:
```json
{
  "question_id": "stock_001",
  "domain": "stock",              // "stock" | "crypto" | "weather"
  "entity": "Alphabet (Google)",
  "identifier": "GOOGL",          // ticker | coingecko_id | wttr.in query
  "question": "What is Alphabet (Google)'s current stock price in USD?",
  "difficulty": "easy",
  "related_entity": "Microsoft",
  "related_identifier": "MSFT",
  "related_question": "What is Microsoft's current stock price in USD?"
}
```

---

## Data Sources (`data/fetcher.py`)

All free, no API key required:
| Domain | Source | Library/API |
|---|---|---|
| Stocks | Yahoo Finance | `yfinance` |
| Crypto | CoinGecko | `requests` to public API |
| Weather | wttr.in | `requests` |

Historical values for error_detection task:
- Stocks: `yfinance` historical prices (~2 years ago)
- Crypto: CoinGecko `market_chart/range` endpoint (~2 years ago)
- Weather: Opposite-season temperature offset (wttr.in has no historical API)

---

## SDK: kaggle_benchmarks

Tasks use the `@task` **decorator** pattern — NOT `BaseTask` class inheritance.

```python
import kaggle_benchmarks as kbench
from kaggle_benchmarks import assertions, task
import dataclasses

@dataclasses.dataclass
class MyResponse:
    answer: str
    confidence: int

@task("task_name", description="what it tests")
def my_task(llm, col1: str, col2: str) -> float:
    response = llm.prompt("...", schema=MyResponse)       # structured output
    assertions.assert_true(condition, expectation="msg")  # record result
    return score_float

# Run over dataset
import pandas as pd
df = pd.read_json("data/questions.json")
results = my_task.evaluate(llm=[kbench.llm], evaluation_data=df, n_jobs=3)
```

### Judge LLM (for free-text reasoning assessment)
```python
judge = assertions.assess_response_with_judge(
    criteria=["criterion 1", "criterion 2"],
    response_text=response.reasoning,
    judge_llm=kbench.judge_llm,
)
judge_score = sum(1 for r in judge.results if r.passed) / len(judge.results)
```

### Multi-turn conversations (Task 3)
```python
with kbench.chats.new("context_name"):
    r1 = llm.prompt("first message", schema=Schema1)
    r2 = llm.prompt("second message", schema=Schema2)  # continues the conversation
```

---

## File Structure
```
c:\Projects\metacode\
├── CLAUDE.md
├── benchmark.yaml
├── data/
│   ├── questions.json          # 50 questions (hand-curated)
│   └── fetcher.py              # live data fetching utilities
├── tasks/
│   ├── confidence_calibration.py
│   ├── error_detection.py
│   └── self_correction.py
└── notebook/
    └── metacode_benchmark.ipynb  # Kaggle notebook (final submission artefact)
```

---

## Build Status
- [x] Project structure
- [x] `data/questions.json` — 50 questions (20 stocks, 15 crypto, 15 weather)
- [x] `data/fetcher.py` — live data fetching
- [x] `tasks/confidence_calibration.py`
- [x] `tasks/error_detection.py`
- [x] `tasks/self_correction.py`
- [x] `benchmark.yaml`
- [x] `notebook/metacode_benchmark.ipynb` — Kaggle notebook (next step)
- [x] Local test run to verify fetchers work
- [x] Submit to Kaggle (created `metacode_submission.zip` containing the benchmark)

---

## What NOT To Do
- Don't use `class BaseTask` — that's the old GitHub repo, not the real SDK
- Don't use regex to parse LLM responses — use `schema=` for structured output
- Don't hardcode any "current" prices in the dataset — always fetch live
- Don't use `problems.json` from the original repo — completely replaced

---

## LLM Access
- Primary: Claude (via Claude Code / Anthropic API)
- Backup: MiniMax
- In Kaggle notebooks: `kbench.llm` and `kbench.llms["model/id"]`

---

## Dependencies
```
yfinance>=0.2.0
requests>=2.28.0
kaggle_benchmarks  # pre-installed in Kaggle notebooks
```
