"""
run_pipeline.py
=============================================================================
Master Execution Script for Mental Health Twin Deep Learning Research Project.
Executes the full pipeline in sequence:
1. Data Pipeline Validation
2. Multi-Task Model Training & Checkpoint Saving
3. Test Set Evaluation & Novelty Benchmarking
4. Explainable AI (XAI) Plot Generation
5. Terminal Digital Twin Live Simulation
=============================================================================
"""

import subprocess
import sys


def run_step(step_name: str, script_name: str):
    print("\n" + "=" * 75)
    print(f">> STEP: {step_name} ({script_name})")
    print("=" * 75)
    ret = subprocess.run([sys.executable, script_name])
    if ret.returncode != 0:
        print(f"\n[FAIL] Step '{step_name}' failed with return code {ret.returncode}.")
        sys.exit(ret.returncode)


if __name__ == "__main__":
    print("[PIPELINE] STARTING COMPLETE MENTAL HEALTH TWIN ML PIPELINE")
    run_step("1. Data Pipeline", "data_pipeline.py")
    run_step("2. Multi-Task Training", "train.py")
    run_step("3. Evaluation & Novelty Benchmark", "evaluate.py")
    run_step("4. Explainable AI Plot Generation", "explainability.py")
    run_step("5. Live Digital Twin Simulation", "simulate_twin.py")
    print("\n" + "=" * 75)
    print("[SUCCESS] FULL MACHINE LEARNING RESEARCH PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 75)
