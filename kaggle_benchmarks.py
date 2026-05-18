"""
Mock and dual-mode wrapper for the kaggle_benchmarks SDK.
Allows running MetaCode benchmarks locally or fallback-safely on Kaggle.
"""

import sys
import os
import json
import random
import time
import dataclasses
from typing import List, Type, Any, Dict, Optional

# --- Dual-Mode: Check if real kaggle_benchmarks exists elsewhere on the system ---
try:
    import sys
    import os
    
    # Save path and temporarily filter out current directory variants
    original_path = sys.path[:]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filtered_path = [
        p for p in sys.path 
        if p not in ('', '.', './') and os.path.abspath(p) != current_dir
    ]
    sys.path = filtered_path
    
    # Save and pop from sys.modules to force looking up outside the current directory
    original_module = sys.modules.pop("kaggle_benchmarks", None)
    try:
        import kaggle_benchmarks as real_kbench
        # Import all attributes from real_kbench
        globals().update({k: v for k, v in real_kbench.__dict__.items() if not k.startswith("__")})
        REAL_AVAILABLE = True
    finally:
        # Restore sys.path and sys.modules mapping
        sys.path = original_path
        if original_module:
            sys.modules["kaggle_benchmarks"] = original_module
except ImportError:
    REAL_AVAILABLE = False


if not REAL_AVAILABLE:
    # =========================================================================
    # Mock SDK Implementation for Local/Offline Runs
    # =========================================================================

    # Global context to store current assertion results
    _current_results = []

    class AssertionsMock:
        @dataclasses.dataclass
        class JudgeCriterionResult:
            criterion: str
            passed: bool

        @dataclasses.dataclass
        class JudgeResponse:
            results: List[Any]

        def assert_true(self, condition: bool, expectation: str = ""):
            status = "PASS" if condition else "FAIL"
            print(f"    [Assert] {status}: {expectation}")
            _current_results.append({"type": "assertion", "passed": condition, "detail": expectation})

        def assess_response_with_judge(
            self, criteria: List[str], response_text: str, judge_llm: Any = None
        ) -> JudgeResponse:
            results = []
            response_lower = response_text.lower()
            for c in criteria:
                # Simple heuristic-based offline judgment
                c_lower = c.lower()
                passed = False
                
                # Check if response mentions cutoff / live data / uncertainty
                if "cutoff" in c_lower or "real-time" in c_lower or "live" in c_lower:
                    passed = any(w in response_lower for w in ["cutoff", "real-time", "live", "today", "accurate right now"])
                elif "uncertainty" in c_lower or "low confidence" in c_lower or "confidence" in c_lower:
                    passed = any(w in response_lower for w in ["uncertain", "confidence", "low", "estimate", "not sure"])
                elif "updated" in c_lower or "lesson" in c_lower or "corrected" in c_lower:
                    passed = any(w in response_lower for w in ["lesson", "updated", "correction", "real-time", "mistake"])
                else:
                    # Default fallback
                    passed = len(response_text) > 10
                
                results.append(self.JudgeCriterionResult(criterion=c, passed=passed))
            return self.JudgeResponse(results=results)

    assertions = AssertionsMock()

    # --- Chats Management Mock ---
    class ChatsMock:
        class ChatContext:
            def __init__(self, name: str):
                self.name = name
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        def new(self, name: str) -> ChatContext:
            return self.ChatContext(name)

    chats = ChatsMock()

    # --- Structured Schema / Dataclass Mock LLM Response Generator ---
    class MockLLM:
        def __init__(self, is_judge: bool = False):
            self.is_judge = is_judge

        def prompt(self, prompt: str, schema: Type[Any] = None) -> Any:
            # We check if a real API key is present to make real API requests
            gemini_key = os.environ.get("GEMINI_API_KEY")
            if gemini_key and schema:
                try:
                    return self._call_real_gemini(prompt, schema, gemini_key)
                except Exception as e:
                    print(f"    [Real Gemini Call Failed: {e}. Falling back to offline mock.]")

            # Offline Mock Generation
            if not schema:
                return "Mock plain text response."

            # Parse dataclass fields
            fields = {f.name: f.type for f in dataclasses.fields(schema)}
            mock_data = {}

            prompt_lower = prompt.lower()

            # Detect domain / stock symbol / weather city
            entity = "this entity"
            if "google" in prompt_lower or "googl" in prompt_lower:
                entity = "Google"
            elif "bitcoin" in prompt_lower or "btc" in prompt_lower:
                entity = "Bitcoin"
            elif "london" in prompt_lower:
                entity = "London"

            for name, ftype in fields.items():
                if name == "estimated_value":
                    if "bitcoin" in prompt_lower:
                        mock_data[name] = float(random.randint(65000, 75000))
                    elif "google" in prompt_lower:
                        mock_data[name] = float(random.randint(160, 180))
                    elif "london" in prompt_lower:
                        mock_data[name] = float(random.randint(10, 18))
                    else:
                        mock_data[name] = 100.0
                elif name == "confidence":
                    # Calibration tasks: model should state lower confidence on live questions
                    if "live" in prompt_lower or "current" in prompt_lower or "today" in prompt_lower:
                        if "corrected" in prompt_lower or "lesson" in prompt_lower or "related" in prompt_lower:
                            mock_data[name] = random.randint(10, 25)  # even lower after correction
                        else:
                            mock_data[name] = random.randint(20, 45)
                    else:
                        mock_data[name] = random.randint(70, 90)
                elif name == "choice":
                    # Return A or B. Deterministic Swap usually assigns correct choice, let's return A or B
                    mock_data[name] = "A" if "source a" in prompt_lower else "B"
                elif name == "reasoning" or name == "updated_reasoning":
                    mock_data[name] = (
                        f"My training data cutoff is January 2025, and I cannot access live internet data for {entity}. "
                        f"Therefore, I cannot know the actual current price today. I will provide a ballpark estimate "
                        f"based on past ranges, but with very low confidence."
                    )
                elif name == "lesson_learned":
                    mock_data[name] = (
                        f"I learned that my training data is stale for real-time metrics like {entity}. "
                        f"When corrected, I must update my internal confidence and be more cautious in future estimates."
                    )
                else:
                    if ftype == int:
                        mock_data[name] = 42
                    elif ftype == float:
                        mock_data[name] = 42.0
                    else:
                        mock_data[name] = "mock text value"

            return schema(**mock_data)

        def _call_real_gemini(self, prompt: str, schema: Type[Any], api_key: str) -> Any:
            import requests
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
            
            # Convert Python dataclass schema to JSON Schema format
            properties = {}
            required = []
            for f in dataclasses.fields(schema):
                ftype = "string"
                if f.type == int:
                    ftype = "integer"
                elif f.type == float:
                    ftype = "number"
                properties[f.name] = {"type": ftype}
                required.append(f.name)

            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            }
            r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
            r.raise_for_status()
            res_data = r.json()
            text_content = res_data["candidates"][0]["content"]["parts"][0]["text"]
            json_data = json.loads(text_content)
            return schema(**json_data)

    llm = MockLLM()
    judge_llm = MockLLM(is_judge=True)

    # --- Task Decorator Mock ---
    class Task:
        def __init__(self, name: str, description: str, fn: Any):
            self.name = name
            self.description = description
            self.fn = fn

        def evaluate(self, llm: List[Any], evaluation_data: Any, n_jobs: int = 1) -> Any:
            import pandas as pd
            print(f"\n=== EVALUATING TASK: {self.name} ===")
            print(f"Description: {self.description}")
            print(f"Dataset rows: {len(evaluation_data)}")
            print(f"Parallel jobs: {n_jobs}\n")

            global _current_results
            scores = []
            
            llm_instance = llm[0] if isinstance(llm, list) else llm

            start_time = time.time()
            for idx, row in evaluation_data.iterrows():
                row_dict = row.to_dict()
                print(f"[{idx + 1}/{len(evaluation_data)}] Evaluating ID: {row_dict.get('question_id', 'N/A')}...")
                _current_results = []
                try:
                    score = self.fn(llm_instance, **row_dict)
                    scores.append(score)
                    print(f"  -> SCORE: {score}\n")
                except Exception as e:
                    print(f"  -> ERROR: {e}\n")
                    scores.append(0.0)

            elapsed = time.time() - start_time
            avg_score = sum(scores) / len(scores) if scores else 0.0
            print(f"=== COMPLETED: {self.name} ===")
            print(f"Average Score: {avg_score:.4f}")
            print(f"Time elapsed: {elapsed:.2f} seconds\n")

            # Return a results dataframe
            results_df = evaluation_data.copy()
            results_df["score"] = scores
            return results_df

    def task(name: str, description: str = ""):
        def decorator(fn):
            return Task(name, description, fn)
        return decorator
