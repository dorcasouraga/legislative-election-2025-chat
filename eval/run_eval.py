import json
import os
import time
import sys
from pathlib import Path
from typing import Dict, Any, List

# ============================================================
# 🔧 PYTHON PATH FIX (ABSOLUTELY REQUIRED)
# ============================================================
ROOT = Path(__file__).resolve().parents[1]  # election-chat/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ============================================================
# IMPORTS (SAFE)
# ============================================================
from src.agent.unified_agent import ask
from eval.assertions import (
    assert_engine,
    assert_status,
    assert_contains_text,
    assert_has_sources,
)

# ============================================================
# OPTIONAL LANGFUSE (NEVER BREAK EVAL)
# ============================================================
try:
    from langfuse import Langfuse

    lf = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_BASE_URL"),
    )
except Exception:
    lf = None


# ============================================================
# 🔒 JSON SAFE CONVERTER (int64, numpy, etc.)
# ============================================================
def json_safe(obj):
    try:
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
    except Exception:
        pass

    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]

    return obj


# ============================================================
# MAIN EVAL RUNNER
# ============================================================
def run_eval(dataset_path: str) -> Dict[str, Any]:
    results: List[Dict] = []
    failures: List[Dict] = []

    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"❌ Dataset not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()

            # 🛡️ ignore empty lines
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"❌ Invalid JSONL at line {lineno}: {e}"
                ) from e

            qid = row.get("id", f"line_{lineno}")
            question = row["question"]
            expect = row.get("expect", {})

            os.environ["EVAL_MODE"] = "true"

            # ----------------------------
            # 🔍 OBSERVABILITY
            # ----------------------------
            trace = lf.start_span(name="eval_case") if lf else None
            if trace:
                trace.input = {"id": qid, "question": question}

            start = time.time()
            result = ask(question)
            latency_ms = int((time.time() - start) * 1000)

            if trace:
                trace.output = {
                    "engine": result.get("engine"),
                    "status": result.get("status"),
                    "latency_ms": latency_ms,
                }
                trace.end()

            # ----------------------------
            # ✅ ASSERTIONS
            # ----------------------------
            checks = {
                "engine": assert_engine(result, expect),
                "status": assert_status(result, expect),
                "contains": assert_contains_text(result, expect),
                "sources": assert_has_sources(result, expect),
            }

            ok = all(checks.values())

            entry = {
                "id": qid,
                "question": question,
                "ok": ok,
                "checks": checks,
                "engine": result.get("engine"),
                "status": result.get("status"),
                "latency_ms": latency_ms,
            }

            results.append(entry)

            if not ok:
                failures.append(
                    {
                        **entry,
                        "answer": result.get("text"),
                        "sources": result.get("sources"),
                        "sql": result.get("sql"),
                    }
                )

    report = {
        "total": len(results),
        "passed": sum(r["ok"] for r in results),
        "failed": len(failures),
        "results": results,
        "failures": failures,
    }

    return json_safe(report)


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    report = run_eval("eval/datasets/eval_questions.jsonl")

    os.makedirs("eval", exist_ok=True)
    with open("eval/report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"✅ Passed: {report['passed']}/{report['total']}")
    print(f"❌ Failed: {report['failed']}")
