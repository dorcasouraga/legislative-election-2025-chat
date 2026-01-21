def assert_engine(result, expect):
    if "engine" not in expect:
        return True
    return result.get("engine") == expect["engine"]


def assert_status(result, expect):
    if "status" not in expect:
        return True
    return result.get("status") == expect["status"]


def assert_contains_text(result, expect):
    if "contains" not in expect:
        return True
    text = (result.get("text") or "").lower()
    return expect["contains"].lower() in text


def assert_has_sources(result, expect):
    if not expect.get("sources"):
        return True

    engine = result.get("engine")

    if engine == "sql":
        return result.get("sql") is not None

    if engine == "rag":
        return bool(result.get("sources"))

    return False
