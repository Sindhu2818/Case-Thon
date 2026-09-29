"""One-command pipeline for the case-a-thon solution.

Run:  .venv/Scripts/python run_pipeline.py
Executes every stage in order and prints progress. Never touches EVALUATION
outcomes (they are blank in the pack and are never read).
"""
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

STEPS = [
    # ("linkage", "scripts/linkage.py"), # Requires raw data
    # ("features (temporal, leakage-checked)", "scripts/features.py"), # Requires raw data
    ("training + comparison + ablations + SHAP", "scripts/train.py"),
    ("priority engine + submissions", "scripts/action_queue.py"),
    ("EDA report + figures", "scripts/figures.py"),
    # ("data dictionary", "scripts/make_data_dictionary.py"), # Requires raw data
    ("example explanations", "scripts/explain_examples.py"),
]

if __name__ == "__main__":
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for name, script in STEPS:
        print(f"\n========== {name} ==========", flush=True)
        r = subprocess.run([PY, script], cwd=HERE, env=env)
        if r.returncode != 0:
            sys.exit(f"step failed: {name}")
    print("\nPIPELINE COMPLETE")
    print("submissions -> submission/  artifacts -> artifacts/  docs -> docs/")
