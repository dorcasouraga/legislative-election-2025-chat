import json
import sys
from pathlib import Path

# ============================================================
# 🔧 FIX PYTHON PATH (OBLIGATOIRE)
# ============================================================
ROOT = Path(__file__).resolve().parents[1]  # election-chat/
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# ============================================================
# IMPORTS
# ============================================================
from eval.metrics import compute_metrics

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    report_path = Path("eval/report.json")

    if not report_path.exists():
        raise FileNotFoundError("❌ eval/report.json not found. Run run_eval.py first.")

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    metrics = compute_metrics(report)

    print("\n📊 METRICS SUMMARY")
    print("-" * 40)

    for k, v in metrics.items():
        print(f"{k:25s}: {v}")

    print("\n✅ Metrics computed successfully.")
