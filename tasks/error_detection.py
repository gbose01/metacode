"""
Task 2 — Error Detection (Temporal Reasoning)

The model is shown two values for the same entity:
  - Source A: either the live value (fetched right now) or a ~2-year-old value
  - Source B: the other one

It must identify which source is reporting TODAY's value.

This tests whether the model has coherent priors about the historical trajectory
of an asset or metric — e.g., knowing that Bitcoin was much cheaper 2 years ago,
or that a stable retailer's stock hasn't moved dramatically.

Scoring: returns bool (correct / incorrect).
The A/B assignment is deterministic per question_id (hash-seeded) to prevent
a model from winning by always guessing the same label.
"""

import dataclasses
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kaggle_benchmarks as kbench
from kaggle_benchmarks import assertions, task

from data.fetcher import format_value, get_historical_value, get_live_value


@dataclasses.dataclass
class DetectionResponse:
    choice: str      # "A" or "B"
    reasoning: str
    confidence: int  # 0–100


def _deterministic_swap(question_id: str) -> bool:
    """Return True if current value should be placed in slot A (else slot B)."""
    digest = int(hashlib.md5(question_id.encode()).hexdigest()[:8], 16)
    return digest % 2 == 0


@task(
    "error_detection",
    description=(
        "Tests whether the model can reason about which of two values is more "
        "likely to be current vs ~2 years old, using its knowledge of historical "
        "value ranges for stocks, crypto, and weather."
    ),
)
def error_detection(
    llm,
    question_id: str,
    domain: str,
    entity: str,
    question: str,
    identifier: str,
    difficulty: str,
    related_entity: str,
    related_identifier: str,
    related_question: str,
) -> bool:

    current_value = get_live_value(domain, identifier)
    historical_value = get_historical_value(domain, identifier)

    current_is_a = _deterministic_swap(question_id)
    if current_is_a:
        value_a, value_b = current_value, historical_value
        correct_choice = "A"
    else:
        value_a, value_b = historical_value, current_value
        correct_choice = "B"

    a_str = format_value(domain, value_a)
    b_str = format_value(domain, value_b)

    if domain == "weather":
        metric = "temperature"
        unit_note = "in degrees Celsius"
    else:
        metric = "price"
        unit_note = "in USD"

    prompt = f"""Two independent data sources report different {metric}s for {entity}.

One source pulled its data RIGHT NOW from a live feed.
The other source's data is from approximately 2 years ago.

{entity} {metric} ({unit_note}):
  Source A: {a_str}
  Source B: {b_str}

Based on what you know about {entity}'s historical value range, which source is
more likely reporting TODAY's value?

Respond with valid JSON containing exactly these keys:
  "choice": "A" or "B",
  "reasoning": <explain which historical range clues led to your decision>,
  "confidence": <integer 0–100>"""

    response = llm.prompt(prompt, schema=DetectionResponse)

    is_correct = response.choice.strip().upper() == correct_choice

    assertions.assert_true(
        is_correct,
        expectation=(
            f"Expected {correct_choice} (current={format_value(domain, current_value)}, "
            f"historical={format_value(domain, historical_value)}). "
            f"Model chose {response.choice} with {response.confidence}% confidence."
        ),
    )

    return is_correct
