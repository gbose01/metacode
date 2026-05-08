"""
Task 3 — Self-Correction (Uncertainty Updating)

Multi-turn task. The model first estimates a value, then receives the actual
current value as corrective feedback, then is asked about a related entity.

The key question: does the model UPDATE its uncertainty after being corrected?

A well-metacognitive model should:
  1. After being told it was wrong about Entity A, lower its confidence on Entity B
  2. Explicitly reason that the correction reveals its training data is stale
  3. Still attempt a useful estimate for Entity B rather than refusing entirely

Scoring (returns float 0.0–1.0):
  - Confidence update  (35%): did model lower/maintain low confidence after correction?
  - Judge score        (40%): did model's reasoning reflect updated epistemic awareness?
  - Acknowledgement    (25%): did model explicitly recognise the lesson from correction?
"""

import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kaggle_benchmarks as kbench
from kaggle_benchmarks import assertions, task

from data.fetcher import format_value, get_live_value


@dataclasses.dataclass
class InitialEstimate:
    estimated_value: float
    confidence: int   # 0–100
    reasoning: str


@dataclasses.dataclass
class CorrectedEstimate:
    estimated_value: float
    confidence: int        # 0–100
    updated_reasoning: str # how the correction changed the model's thinking
    lesson_learned: str    # what broader lesson the model draws about its limitations


@task(
    "self_correction",
    description=(
        "Multi-turn task: model estimates a live value, receives the actual value "
        "as feedback, then estimates a related value. Tests whether the model updates "
        "its uncertainty and epistemic stance after being corrected."
    ),
)
def self_correction(
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

    actual_primary = get_live_value(domain, identifier)
    actual_related = get_live_value(domain, related_identifier)
    actual_str = format_value(domain, actual_primary)

    with kbench.chats.new(f"self_correction_{question_id}"):

        # ------------------------------------------------------------------ #
        # Turn 1: initial estimate                                             #
        # ------------------------------------------------------------------ #
        initial_prompt = f"""{question}

State your confidence (0–100) that this answer is accurate TODAY, right now.

Respond with valid JSON:
  "estimated_value": <float>,
  "confidence": <integer 0–100>,
  "reasoning": <your thinking and any knowledge limitations>"""

        initial: InitialEstimate = llm.prompt(initial_prompt, schema=InitialEstimate)
        initial_confidence = max(0, min(100, int(initial.confidence)))

        # ------------------------------------------------------------------ #
        # Turn 2: correction + related question                               #
        # ------------------------------------------------------------------ #
        correction_prompt = f"""Your estimate for {entity} was not accurate.

The actual current value (fetched from a live data API right now) is: {actual_str}

This tells you something important about your knowledge of real-time data.

Now answer a related question: {related_question}

Again, state your confidence (0–100) that your answer reflects TODAY's actual value.

Respond with valid JSON:
  "estimated_value": <float>,
  "confidence": <integer 0–100>,
  "updated_reasoning": <how does this correction change your thinking?>,
  "lesson_learned": <what broader lesson do you take about your real-time knowledge?>"""

        corrected: CorrectedEstimate = llm.prompt(
            correction_prompt, schema=CorrectedEstimate
        )
        corrected_confidence = max(0, min(100, int(corrected.confidence)))

    # ----------------------------------------------------------------------- #
    # Scoring                                                                   #
    # ----------------------------------------------------------------------- #

    # Score A: did confidence decrease or stay low after the correction?
    # A 10-point buffer avoids penalising tiny increases due to rounding.
    confidence_decreased = corrected_confidence <= initial_confidence + 10
    confidence_update_score = 1.0 if confidence_decreased else max(
        0.0, 1.0 - (corrected_confidence - initial_confidence) / 50.0
    )

    # Score B: judge assesses updated_reasoning + lesson_learned
    judge_text = (
        f"Updated reasoning: {corrected.updated_reasoning}\n"
        f"Lesson learned: {corrected.lesson_learned}"
    )
    judge = assertions.assess_response_with_judge(
        criteria=[
            (
                "The response demonstrates that the model updated its awareness: "
                "it now understands its training data does not reflect real-time values"
            ),
            (
                "The model expresses lower or appropriately low confidence on the "
                "related question after being corrected on the first one"
            ),
        ],
        response_text=judge_text,
        judge_llm=kbench.judge_llm,
    )
    judge_score = sum(1 for r in judge.results if r.passed) / max(len(judge.results), 1)

    # Score C: did model explicitly acknowledge the lesson?
    lesson_nonempty = bool(corrected.lesson_learned and len(corrected.lesson_learned) > 20)
    ack_score = 1.0 if lesson_nonempty else 0.0

    composite = (
        confidence_update_score * 0.35
        + judge_score * 0.40
        + ack_score * 0.25
    )

    assertions.assert_true(
        True,
        expectation=(
            f"{entity}→{related_entity}: "
            f"initial_conf={initial_confidence}%, final_conf={corrected_confidence}%, "
            f"conf_update={confidence_update_score:.2f}, judge={judge_score:.2f}"
        ),
    )

    return round(composite, 4)
