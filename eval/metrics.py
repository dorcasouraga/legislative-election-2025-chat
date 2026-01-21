def compute_metrics(report: dict) -> dict:
    total = report.get("total", 0)
    passed = report.get("passed", 0)
    failed = report.get("failed", 0)

    engines = {}
    for r in report.get("results", []):
        e = r.get("engine") or "none"
        engines[e] = engines.get(e, 0) + 1

    avg_latency = round(
        sum(r.get("latency_ms", 0) for r in report.get("results", [])) / max(total, 1),
        2
    )

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / max(total, 1), 4),
        "engine_distribution": engines,
        "avg_latency_ms": avg_latency,
    }
