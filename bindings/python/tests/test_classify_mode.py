"""Tests for ClassifyMode enum and EffectLog.auto() / EffectLog.manual()."""

import pytest

from effect_log import ClassifyMode, EffectKind, EffectLog, ToolDef


# -- helpers ------------------------------------------------------------------


def search_db(query=""):
    return {"results": [f"result for {query}"]}


def send_email(to=""):
    return {"sent": True, "to": to}


def _make_tooldef():
    def read_file(args):
        return {"content": "data"}

    return ToolDef("read_file", EffectKind.ReadOnly, read_file)


# -- EffectLog.auto() --------------------------------------------------------


def test_auto_with_callables():
    """EffectLog.auto() succeeds with raw callables."""
    log = EffectLog.auto("test-auto-ok", tools=[search_db, send_email])
    result = log.execute("search_db", {"query": "test"})
    assert result["results"] == ["result for test"]


def test_auto_with_tooldef_raises():
    """EffectLog.auto() rejects ToolDef instances."""
    with pytest.raises(TypeError, match="AUTO mode"):
        EffectLog.auto("test-auto-fail", tools=[_make_tooldef()])


def test_auto_with_overrides():
    """EffectLog.auto() allows overrides."""
    log = EffectLog.auto(
        "test-auto-override",
        tools=[search_db],
        overrides={"search_db": EffectKind.IdempotentWrite},
    )
    result = log.execute("search_db", {"query": "x"})
    assert "results" in result


# -- EffectLog.manual() ------------------------------------------------------


def test_manual_with_tooldefs():
    """EffectLog.manual() succeeds with ToolDef instances."""
    td = _make_tooldef()
    log = EffectLog.manual("test-manual-ok", tools=[td])
    result = log.execute("read_file", {})
    assert result["content"] == "data"


def test_manual_with_callable_raises():
    """EffectLog.manual() rejects raw callables."""
    with pytest.raises(TypeError, match="MANUAL mode"):
        EffectLog.manual("test-manual-fail", tools=[search_db])


def test_manual_with_overrides_raises():
    """MANUAL mode + overrides raises ValueError."""
    with pytest.raises(ValueError, match="overrides.*MANUAL mode"):
        EffectLog(
            "test-manual-overrides",
            tools=[_make_tooldef()],
            mode=ClassifyMode.MANUAL,
            overrides={"read_file": EffectKind.IdempotentWrite},
        )


# -- HYBRID (default) --------------------------------------------------------


def test_hybrid_mixed_input():
    """Default HYBRID mode accepts both callables and ToolDefs."""
    td = _make_tooldef()
    log = EffectLog("test-hybrid", tools=[search_db, td])
    assert log.execute("search_db", {"query": "q"})["results"] == ["result for q"]
    assert log.execute("read_file", {})["content"] == "data"


def test_default_mode_is_hybrid():
    """No mode= argument behaves as HYBRID (backward compat)."""
    td = _make_tooldef()
    log = EffectLog("test-default", tools=[search_db, td])
    result = log.execute("search_db", {"query": "q"})
    assert "results" in result


def test_explicit_hybrid_mode():
    """Explicitly passing mode=HYBRID works the same."""
    log = EffectLog(
        "test-explicit-hybrid",
        tools=[search_db],
        mode=ClassifyMode.HYBRID,
    )
    result = log.execute("search_db", {"query": "q"})
    assert "results" in result


# -- Edge cases ---------------------------------------------------------------


def test_auto_empty_tools():
    """EffectLog.auto() with empty tools list succeeds."""
    log = EffectLog.auto("test-auto-empty", tools=[])
    assert log.history() == []


def test_manual_empty_tools():
    """EffectLog.manual() with empty tools list succeeds."""
    log = EffectLog.manual("test-manual-empty", tools=[])
    assert log.history() == []


def test_classify_mode_enum_values():
    """ClassifyMode enum has expected values."""
    assert ClassifyMode.AUTO.value == "auto"
    assert ClassifyMode.MANUAL.value == "manual"
    assert ClassifyMode.HYBRID.value == "hybrid"
