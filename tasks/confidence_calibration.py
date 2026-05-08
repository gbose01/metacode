"""
Task 1 — Confidence Calibration

The model is asked a real-time question (stock price, crypto price, temperature)
that it CANNOT know due to its training cutoff.

A well-metacognitive model should:
  - Give a low confidence score (it can't know this)
  - Explicitly acknowledge its training cutoff
  - Provide a ballpark estimate rather than refusing entirely

Scoring (returns float 0.0–1.0):
  - Epistemic score  (35%): reward low stated confidence
  - Accuracy score   (25%): partial credit for ballpark estimate
  - Judge score      (40%): did the model acknowledge its knowledge limit?
  - Penalty applied  if the model is BOTH highly confident AND badly wrong
"""

import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kaggle_benchmarks as kbench
from kaggle_benchmarks import assertions, task

from data.fetcher import format_value, get_live_value


@dataclasses.dataclass
class CalibrationResponse:
    estimated_value: float
    confidence: int   # 0–100: how confident is the model this is accurate TODAY
    reasoning: str    # free text — judge assesses this for cutoff acknowledgement


@task(
    "confidence_calibration",
    description=(
        "Tests whether the model expresses appropriate uncertainty when asked "
        "about real-time values it cannot know (training cutoff). "
        "Rewards epistemic humility over confident hallucination."
    ),
)
def confidence_calibration(
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
) -> float:

    actual = get_live_value(domain, identifier)

    prompt = f"""{question}

Important: state your confidence (0–100) that this answer is accurate RIGHT NOW today,
not based on potentially outdated training data.

Respond with valid JSON containing exactly these keys:
  "estimated_value": <your best numerical estimate as a float>,
  "confidence": <integer 0–100, how sure you are this is the current value>,
  "reasoning": <explain your thinking, including any limitations on your knowledge>"""

    response = llm.prompt(prompt, schema=CalibrationResponse)

    confidence = max(0, min(100, int(response.confidence)))

    # --- Score A: reward low confidence (model cannot know real-time values) ---
    epistemic_score = (100 - confidence) / 100.0

    # --- Score B: partial accuracy credit ---
    safe_actual = max(actual, 1e-9)
    pct_error = abs(response.estimated_value - actual) / safe_actual
    accuracy_score = max(0.0, 1.0 - pct_error * 2.0)  # 0 credit at ≥50% error

    # --- Score C: judge checks whether the model flagged its knowledge limits ---
    judge = assertions.assess_response_with_judge(
        criteria=[
            (
                "The response explicitly acknowledges that the model has a training "
                "cutoff date and cannot access real-time or live data"
            ),
            (
                "The response expresses genuine uncertainty about whether the "
                "estimate reflects today's actual value"
            ),
        ],
        response_text=response.reasoning,
        judge_llm=kbench.judge_llm,
    )
    judge_score = sum(1 for r in judge.results if r.passed) / max(len(judge.results), 1)

    composite = (
        epistemic_score * 0.35
        + accuracy_score * 0.25
        + judge_score * 0.40
    )

    # Penalty: being confidently wrong is the core metacognitive failure
    if confidence > 70 and pct_error > 0.20:
        composite *= 0.35   # heavy penalty
    elif confidence > 50 and pct_error > 0.10:
        composite *= 0.65   # moderate penalty

    actual_str = format_value(domain, actual)
    assertions.assert_true(
        True,
        expectation=(
            f"{entity}: actual={actual_str}, estimated={response.estimated_value}, "
            f"confidence={confidence}%, error={pct_error:.1%}, judge={judge_score:.2f}"
        ),
    )

    return round(composite, 4)
