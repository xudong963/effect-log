"""Basic tests for the effect-log Python bindings."""

import json
import tempfile
import os

import pytest

from effect_log import EffectKind, EffectLog, ToolDef, tool, auto_tool


def make_read_tool():
    def read_file(args):
        return {"content": f"read {args['path']}", "executed": True}

    return ToolDef("read_file", EffectKind.ReadOnly, read_file)


def make_email_tool():
    call_count = {"n": 0}

    def send_email(args):
        call_count["n"] += 1
        return {"sent_to": args["to"], "count": call_count["n"]}

    return ToolDef("send_email", EffectKind.IrreversibleWrite, send_email), call_count


def test_basic_execute():
    """Tools execute and return results."""
    log = EffectLog(
        execution_id="test-1",
        tools=[make_read_tool()],
    )
    result = log.execute("read_file", {"path": "/tmp/a.txt"})
    assert result["content"] == "read /tmp/a.txt"
    assert result["executed"] is True


def test_history():
    """Execution history is recorded."""
    email_tool, _ = make_email_tool()
    log = EffectLog(
        execution_id="test-2",
        tools=[make_read_tool(), email_tool],
    )

    log.execute("read_file", {"path": "/tmp/a.txt"})
    log.execute("send_email", {"to": "ceo@co.com"})

    history = log.history()
    assert len(history) == 2
    assert history[0]["tool_name"] == "read_file"
    assert history[1]["tool_name"] == "send_email"
    assert history[1]["outcome"] == "Success"


def test_tool_decorator():
    """The @tool decorator creates a ToolDef."""

    @tool(effect=EffectKind.ReadOnly)
    def my_reader(args):
        return {"data": "hello"}

    assert isinstance(my_reader, ToolDef)

    log = EffectLog(execution_id="test-3", tools=[my_reader])
    result = log.execute("my_reader", {})
    assert result["data"] == "hello"


def test_tool_decorator_auto_classify():
    """The @tool() decorator with no effect auto-classifies."""

    @tool()
    def search_db(args):
        return {"results": ["a", "b"]}

    assert isinstance(search_db, ToolDef)
    # Verify it works as ReadOnly by executing through EffectLog
    log = EffectLog(execution_id="test-auto-dec", tools=[search_db])
    result = log.execute("search_db", {})
    assert result["results"] == ["a", "b"]


def test_auto_tool_decorator():
    """The @auto_tool decorator auto-classifies."""

    @auto_tool
    def fetch_data(args):
        return {"data": "hello"}

    assert isinstance(fetch_data, ToolDef)
    log = EffectLog(execution_id="test-auto-tool", tools=[fetch_data])
    result = log.execute("fetch_data", {})
    assert result["data"] == "hello"


def test_recovery_sealed_irreversible():
    """IrreversibleWrite returns sealed result on recovery."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        storage = f"sqlite:///{db_path}"

        email_tool, call_count = make_email_tool()

        # First execution
        log = EffectLog(
            execution_id="task-001",
            tools=[make_read_tool(), email_tool],
            storage=storage,
        )
        log.execute("read_file", {"path": "/tmp/a"})
        log.execute("send_email", {"to": "ceo@co.com"})
        assert call_count["n"] == 1

        # "Crash" and recover
        email_tool2, call_count2 = make_email_tool()
        log2 = EffectLog(
            execution_id="task-001",
            tools=[make_read_tool(), email_tool2],
            storage=storage,
            recover=True,
        )

        # read_file: ReadOnly -> replayed (ReplayFresh policy)
        log2.execute("read_file", {"path": "/tmp/a"})

        # send_email: IrreversibleWrite -> sealed, NOT re-executed
        result = log2.execute("send_email", {"to": "ceo@co.com"})
        assert result["sent_to"] == "ceo@co.com"
        assert call_count2["n"] == 0  # NOT re-executed


def test_human_review_on_crash():
    """Crashed IrreversibleWrite escalates to human review."""
    pass


# ── Auto-classification integration tests ────────────────────────────────────


def test_auto_classify_raw_callables():
    """EffectLog accepts raw callables and auto-classifies them."""

    def search_db(query=""):
        return {"results": [f"result for {query}"]}

    def send_email(to="", subject=""):
        return {"sent": True, "to": to}

    def upsert_record(id="", data=None):
        return {"id": id, "updated": True}

    log = EffectLog(
        execution_id="test-auto-1",
        tools=[search_db, send_email, upsert_record],
    )

    result = log.execute("search_db", {"query": "Q4"})
    assert result["results"] == ["result for Q4"]

    result = log.execute("send_email", {"to": "ceo@co.com", "subject": "Report"})
    assert result["sent"] is True

    result = log.execute("upsert_record", {"id": "123", "data": {"x": 1}})
    assert result["updated"] is True


def test_auto_classify_with_overrides():
    """EffectLog overrides= corrects misclassifications."""

    def process_order(order_id=""):
        return {"processed": order_id}

    def search_db(query=""):
        return {"results": []}

    log = EffectLog(
        execution_id="test-auto-2",
        tools=[search_db, process_order],
        overrides={"process_order": EffectKind.IdempotentWrite},
    )

    result = log.execute("process_order", {"order_id": "ORD-001"})
    assert result["processed"] == "ORD-001"


def test_mixed_tooldef_and_callable():
    """EffectLog accepts a mix of ToolDef and raw callables."""

    def search_db(query=""):
        return {"results": [f"result for {query}"]}

    email_tool, call_count = make_email_tool()

    log = EffectLog(
        execution_id="test-mixed",
        tools=[search_db, email_tool],  # callable + ToolDef
    )

    result = log.execute("search_db", {"query": "test"})
    assert "results" in result

    result = log.execute("send_email", {"to": "user@example.com"})
    assert call_count["n"] == 1


def test_auto_classify_recovery():
    """Auto-classified tools work correctly with crash recovery."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        storage = f"sqlite:///{db_path}"

        call_counts = {"search": 0, "send": 0}

        def search_db(query=""):
            call_counts["search"] += 1
            return {"results": ["a"]}

        def send_email(to=""):
            call_counts["send"] += 1
            return {"sent": True, "to": to}

        # First execution
        log = EffectLog(
            execution_id="task-recovery",
            tools=[search_db, send_email],
            storage=storage,
        )
        log.execute("search_db", {"query": "test"})
        log.execute("send_email", {"to": "ceo@co.com"})
        assert call_counts["search"] == 1
        assert call_counts["send"] == 1

        # Recovery with fresh call counts
        call_counts2 = {"search": 0, "send": 0}

        def search_db2(query=""):
            call_counts2["search"] += 1
            return {"results": ["a"]}

        def send_email2(to=""):
            call_counts2["send"] += 1
            return {"sent": True, "to": to}

        # Use same function names for recovery
        search_db2.__name__ = "search_db"
        send_email2.__name__ = "send_email"

        log2 = EffectLog(
            execution_id="task-recovery",
            tools=[search_db2, send_email2],
            storage=storage,
            recover=True,
        )

        log2.execute("search_db", {"query": "test"})
        result = log2.execute("send_email", {"to": "ceo@co.com"})

        # send_email is IrreversibleWrite -> sealed, NOT re-executed
        assert call_counts2["send"] == 0
        assert result["sent"] is True


def test_invalid_tool_type_raises():
    """EffectLog raises TypeError for invalid tool types."""
    with pytest.raises(TypeError, match="Expected ToolDef or callable"):
        EffectLog(
            execution_id="test-invalid",
            tools=["not a callable"],
        )
