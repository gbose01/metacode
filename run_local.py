"""
Local execution script for the MetaCode benchmark.
Runs all 3 tasks on the curated dataset and outputs performance reports.
"""

import os
import sys
import argparse
import json
import pandas as pd

# Add current directory to path so local kaggle_benchmarks is used
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import kaggle_benchmarks as kbench
from tasks.confidence_calibration import confidence_calibration
from tasks.error_detection import error_detection
from tasks.self_correction import self_correction


def parse_args():
    parser = argparse.ArgumentParser(description="MetaCode Local Benchmark Runner")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run in quick smoke-test mode (only 3 questions instead of all 50)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of jobs / workers to show in evaluation report",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save JSON/CSV results",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        default=True,
        help="Automatically generate visualization plots after run",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("==========================================")
    print("       MetaCode Local Benchmark Runner     ")
    print("==========================================\n")

    # Load questions dataset
    questions_path = os.path.join("data", "questions.json")
    if not os.path.exists(questions_path):
        print(f"Error: Could not find questions file at {questions_path}")
        sys.exit(1)

    with open(questions_path, "r") as f:
        questions = json.load(f)

    df = pd.DataFrame(questions)

    # If quick mode, sample 1 question per domain
    if args.quick:
        print("⚡ Running in QUICK smoke-test mode (1 question per domain)...")
        sampled_rows = []
        for domain in ["stock", "crypto", "weather"]:
            domain_df = df[df["domain"] == domain]
            if not domain_df.empty:
                sampled_rows.append(domain_df.iloc[0])
        df = pd.DataFrame(sampled_rows).reset_index(drop=True)
    else:
        print(f"📦 Running full benchmark on all {len(df)} curated questions...")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Run Task 1: Confidence Calibration ---
    print("\n--- [Task 1] Evaluating Confidence Calibration ---")
    calib_results = confidence_calibration.evaluate(
        llm=kbench.llm, evaluation_data=df, n_jobs=args.n_jobs
    )
    calib_results.to_csv(os.path.join(args.output_dir, "task1_calibration.csv"), index=False)

    # --- Run Task 2: Error Detection ---
    print("\n--- [Task 2] Evaluating Error Detection ---")
    detect_results = error_detection.evaluate(
        llm=kbench.llm, evaluation_data=df, n_jobs=args.n_jobs
    )
    detect_results.to_csv(os.path.join(args.output_dir, "task2_detection.csv"), index=False)

    # --- Run Task 3: Self-Correction ---
    print("\n--- [Task 3] Evaluating Self-Correction ---")
    correct_results = self_correction.evaluate(
        llm=kbench.llm, evaluation_data=df, n_jobs=args.n_jobs
    )
    correct_results.to_csv(os.path.join(args.output_dir, "task3_correction.csv"), index=False)

    # Calculate composite report
    print("\n==========================================")
    print("             BENCHMARK SUMMARY            ")
    print("==========================================")
    t1_score = calib_results["score"].mean()
    t2_score = detect_results["score"].mean()  # True/False maps to 1.0/0.0
    t3_score = correct_results["score"].mean()

    # MetaCode scoring weights: Calibration (35%), Detection (30%), Correction (35%)
    composite_score = t1_score * 0.35 + t2_score * 0.30 + t3_score * 0.35

    print(f"Task 1 (Confidence Calibration) Average: {t1_score:.4f}")
    print(f"Task 2 (Error Detection) Accuracy       : {t2_score:.4%}")
    print(f"Task 3 (Self-Correction) Average        : {t3_score:.4f}")
    print(f"------------------------------------------")
    print(f"COMPOSITE METACODE SCORE                : {composite_score:.4f}")
    print("==========================================\n")

    # Save composite summary to JSON
    summary = {
        "task1_calibration_avg": t1_score,
        "task2_detection_accuracy": t2_score,
        "task3_correction_avg": t3_score,
        "composite_metacode_score": composite_score,
        "timestamp": pd.Timestamp.now().isoformat(),
        "quick_mode": args.quick,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"💾 Results saved successfully to directory: {args.output_dir}")

    # If viz is enabled, run visualization generator
    if args.viz:
        try:
            from notebook.visualization import plot_benchmark_results
            print("\n📊 Generating visualization plots...")
            plot_benchmark_results(calib_results, detect_results, correct_results, args.output_dir)
            print("🎨 Plots saved to output directory.")
        except ImportError as e:
            print(f"⚠️ Could not generate plots: {e}. Build the visualization suite first.")
        except Exception as e:
            print(f"⚠️ Error generating plots: {e}")


if __name__ == "__main__":
    main()
