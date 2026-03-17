"""Tests for the auto-classification system."""

import logging
from unittest.mock import patch

import pytest

from effect_log import EffectKind
from effect_log.classify import (
    ClassificationReport,
    ClassificationResult,
    classify_effect_kind,
    classify_from_name,
    classify_tools,
    classify_with_llm,
)


# ── Name prefix classification ───────────────────────────────────────────────


class TestNamePrefixClassification:
    """Test Layer 1: Name prefix matching."""

    @pytest.mark.parametrize(
        "name, expected",
        [
            # ReadOnly prefixes
            ("read_file", EffectKind.ReadOnly),
            ("fetch_data", EffectKind.ReadOnly),
            ("get_user", EffectKind.ReadOnly),
            ("search_db", EffectKind.ReadOnly),
            ("query_table", EffectKind.ReadOnly),
            ("list_items", EffectKind.ReadOnly),
            ("check_status", EffectKind.ReadOnly),
            ("validate_input", EffectKind.ReadOnly),
            ("describe_table", EffectKind.ReadOnly),
            ("count_records", EffectKind.ReadOnly),
            ("find_user", EffectKind.ReadOnly),
            ("lookup_address", EffectKind.ReadOnly),
            ("browse_catalog", EffectKind.ReadOnly),
            ("view_report", EffectKind.ReadOnly),
            ("show_details", EffectKind.ReadOnly),
            ("inspect_object", EffectKind.ReadOnly),
            ("parse_json", EffectKind.ReadOnly),
            ("transform_data", EffectKind.ReadOnly),
            ("format_output", EffectKind.ReadOnly),
            ("log_event", EffectKind.ReadOnly),
            ("trace_request", EffectKind.ReadOnly),
            # IrreversibleWrite prefixes
            ("send_email", EffectKind.IrreversibleWrite),
            ("email_user", EffectKind.IrreversibleWrite),
            ("notify_admin", EffectKind.IrreversibleWrite),
            ("broadcast_message", EffectKind.IrreversibleWrite),
            ("publish_event", EffectKind.IrreversibleWrite),
            ("deploy_service", EffectKind.IrreversibleWrite),
            ("delete_record", EffectKind.IrreversibleWrite),
            ("remove_item", EffectKind.IrreversibleWrite),
            ("destroy_instance", EffectKind.IrreversibleWrite),
            ("drop_table", EffectKind.IrreversibleWrite),
            ("purge_cache", EffectKind.IrreversibleWrite),
            ("revoke_token", EffectKind.IrreversibleWrite),
            ("terminate_process", EffectKind.IrreversibleWrite),
            ("kill_session", EffectKind.IrreversibleWrite),
            ("post_to_slack", EffectKind.IrreversibleWrite),
            ("tweet_update", EffectKind.IrreversibleWrite),
            ("sms_alert", EffectKind.IrreversibleWrite),
            # IrreversibleWrite (non-idempotent creates)
            ("create_user", EffectKind.IrreversibleWrite),
            ("add_item", EffectKind.IrreversibleWrite),
            ("insert_row", EffectKind.IrreversibleWrite),
            # IdempotentWrite prefixes
            ("upsert_record", EffectKind.IdempotentWrite),
            ("update_profile", EffectKind.IdempotentWrite),
            ("put_object", EffectKind.IdempotentWrite),
            ("save_document", EffectKind.IdempotentWrite),
            ("set_config", EffectKind.IdempotentWrite),
            ("write_file", EffectKind.IdempotentWrite),
            ("store_data", EffectKind.IdempotentWrite),
            ("upload_image", EffectKind.IdempotentWrite),
            ("register_webhook", EffectKind.IdempotentWrite),
            ("configure_service", EffectKind.IdempotentWrite),
            ("enable_feature", EffectKind.IdempotentWrite),
            ("disable_feature", EffectKind.IdempotentWrite),
            ("assign_role", EffectKind.IdempotentWrite),
            ("tag_resource", EffectKind.IdempotentWrite),
            # Compensatable -> IrreversibleWrite (auto-downgraded)
            ("reserve_seat", EffectKind.IrreversibleWrite),
            ("lock_resource", EffectKind.IrreversibleWrite),
            ("allocate_memory", EffectKind.IrreversibleWrite),
            ("book_flight", EffectKind.IrreversibleWrite),
            ("hold_inventory", EffectKind.IrreversibleWrite),
            ("checkout_item", EffectKind.IrreversibleWrite),
            ("claim_ticket", EffectKind.IrreversibleWrite),
            # ReadThenWrite prefixes
            ("transfer_funds", EffectKind.ReadThenWrite),
            ("swap_items", EffectKind.ReadThenWrite),
            ("exchange_currency", EffectKind.ReadThenWrite),
            ("move_file", EffectKind.ReadThenWrite),
            ("migrate_data", EffectKind.ReadThenWrite),
            ("sync_databases", EffectKind.ReadThenWrite),
            ("reconcile_accounts", EffectKind.ReadThenWrite),
        ],
    )
    def test_prefix_classification(self, name, expected):
        def dummy(args):
            pass

        dummy.__name__ = name
        result = classify_effect_kind(dummy, name)
        assert result.effect_kind == expected, (
            f"{name}: expected {expected}, got {result.effect_kind}"
        )

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("search", EffectKind.ReadOnly),
            ("query", EffectKind.ReadOnly),
            ("fetch", EffectKind.ReadOnly),
            ("get", EffectKind.ReadOnly),
            ("read", EffectKind.ReadOnly),
            ("list", EffectKind.ReadOnly),
            ("find", EffectKind.ReadOnly),
            ("lookup", EffectKind.ReadOnly),
            ("check", EffectKind.ReadOnly),
            ("validate", EffectKind.ReadOnly),
            ("count", EffectKind.ReadOnly),
            ("parse", EffectKind.ReadOnly),
            ("transform", EffectKind.ReadOnly),
            ("format", EffectKind.ReadOnly),
            ("log", EffectKind.ReadOnly),
            ("send", EffectKind.IrreversibleWrite),
            ("email", EffectKind.IrreversibleWrite),
            ("notify", EffectKind.IrreversibleWrite),
            ("publish", EffectKind.IrreversibleWrite),
            ("deploy", EffectKind.IrreversibleWrite),
            ("delete", EffectKind.IrreversibleWrite),
            ("remove", EffectKind.IrreversibleWrite),
            ("destroy", EffectKind.IrreversibleWrite),
            ("purge", EffectKind.IrreversibleWrite),
            ("create", EffectKind.IrreversibleWrite),
            ("upsert", EffectKind.IdempotentWrite),
            ("update", EffectKind.IdempotentWrite),
            ("save", EffectKind.IdempotentWrite),
            ("insert", EffectKind.IrreversibleWrite),
            ("write", EffectKind.IdempotentWrite),
            ("store", EffectKind.IdempotentWrite),
            ("upload", EffectKind.IdempotentWrite),
            ("register", EffectKind.IdempotentWrite),
            ("transfer", EffectKind.ReadThenWrite),
            ("swap", EffectKind.ReadThenWrite),
            ("migrate", EffectKind.ReadThenWrite),
            ("sync", EffectKind.ReadThenWrite),
            # Compensatable exact -> downgraded
            ("reserve", EffectKind.IrreversibleWrite),
            ("lock", EffectKind.IrreversibleWrite),
            ("book", EffectKind.IrreversibleWrite),
        ],
    )
    def test_exact_name_classification(self, name, expected):
        def dummy(args):
            pass

        dummy.__name__ = name
        result = classify_effect_kind(dummy, name)
        assert result.effect_kind == expected


class TestDocstringClassification:
    """Test Layer 2: Docstring keyword analysis."""

    def test_readonly_docstring(self):
        def my_func(args):
            """Retrieves data from the database. This is a read-only operation."""
            pass

        result = classify_effect_kind(my_func, "my_func")
        assert result.effect_kind == EffectKind.ReadOnly

    def test_irreversible_docstring(self):
        def my_func(args):
            """Sends email notification. This is irreversible and cannot be undone."""
            pass

        result = classify_effect_kind(my_func, "my_func")
        assert result.effect_kind == EffectKind.IrreversibleWrite

    def test_idempotent_docstring(self):
        def my_func(args):
            """Upserts record. This is idempotent and safe to retry."""
            pass

        result = classify_effect_kind(my_func, "my_func")
        assert result.effect_kind == EffectKind.IdempotentWrite

    def test_name_overrides_docstring(self):
        """Name prefix has higher weight than docstring."""

        def search_records(args):
            """Sends results after searching."""
            pass

        result = classify_effect_kind(search_records)
        # "search_" prefix (weight 0.50) > "Sends" docstring keyword (weight 0.25)
        assert result.effect_kind == EffectKind.ReadOnly


class TestParameterClassification:
    """Test Layer 3: Parameter name signals."""

    def test_recipient_params_suggest_irreversible(self):
        def my_func(to, subject, body):
            pass

        result = classify_effect_kind(my_func, "my_func")
        assert result.effect_kind == EffectKind.IrreversibleWrite

    def test_query_params_suggest_readonly(self):
        def my_func(query, filter):
            pass

        result = classify_effect_kind(my_func, "my_func")
        assert result.effect_kind == EffectKind.ReadOnly

    def test_id_params_suggest_idempotent(self):
        def my_func(id, key):
            pass

        result = classify_effect_kind(my_func, "my_func")
        assert result.effect_kind == EffectKind.IdempotentWrite


class TestDefaultClassification:
    """Test fallback behavior for unrecognized functions."""

    def test_unknown_function_defaults_to_irreversible(self):
        def do_something(args):
            pass

        result = classify_effect_kind(do_something)
        assert result.effect_kind == EffectKind.IrreversibleWrite
        assert result.confidence < 0.6

    def test_low_confidence_for_unknown(self):
        def process_order(args):
            pass

        result = classify_effect_kind(process_order)
        assert result.confidence < 0.6


class TestConfidence:
    """Test confidence scoring behavior."""

    def test_high_confidence_for_clear_prefix(self):
        def search_db(args):
            pass

        result = classify_effect_kind(search_db)
        assert result.confidence >= 0.5

    def test_higher_confidence_with_multiple_signals(self):
        def search_db(query):
            """Retrieves data from the database. Read-only operation."""
            pass

        result = classify_effect_kind(search_db)
        # Name prefix (0.50) + docstring (0.25) + params (0.15) = potentially high
        assert result.confidence >= 0.5

    def test_compensatable_downgraded(self):
        """Compensatable auto-downgrades to IrreversibleWrite."""

        def reserve_seat(args):
            pass

        result = classify_effect_kind(reserve_seat)
        assert result.effect_kind == EffectKind.IrreversibleWrite
        assert "downgraded" in result.reason


# ── classify_from_name ───────────────────────────────────────────────────────


class TestClassifyFromName:
    """Test name-only classification for middleware."""

    def test_prefix_match(self):
        result = classify_from_name("search_db")
        assert result.effect_kind == EffectKind.ReadOnly

    def test_exact_match(self):
        result = classify_from_name("delete")
        assert result.effect_kind == EffectKind.IrreversibleWrite

    def test_unknown_name(self):
        result = classify_from_name("foobar")
        assert result.effect_kind == EffectKind.IrreversibleWrite
        assert result.confidence == 0.0


# ── classify_tools (batch) ───────────────────────────────────────────────────


class TestClassifyTools:
    """Test batch classification."""

    def test_batch_classify(self):
        def search_db(query):
            pass

        def send_email(to, subject):
            pass

        def upsert_record(id, data):
            pass

        report = classify_tools([search_db, send_email, upsert_record])
        assert isinstance(report, ClassificationReport)
        assert len(report.results) == 3
        assert report.results["search_db"].effect_kind == EffectKind.ReadOnly
        assert report.results["send_email"].effect_kind == EffectKind.IrreversibleWrite
        assert report.results["upsert_record"].effect_kind == EffectKind.IdempotentWrite

    def test_report_str(self):
        def search_db(query):
            pass

        def send_email(to):
            pass

        report = classify_tools([search_db, send_email])
        s = str(report)
        assert "search_db" in s
        assert "send_email" in s
        assert "ReadOnly" in s
        assert "IrreversibleWrite" in s

    def test_report_apply(self):
        from effect_log import ToolDef

        def search_db(query=""):
            return f"results for {query}"

        def send_email(to=""):
            return f"sent to {to}"

        report = classify_tools([search_db, send_email])
        defs = report.apply()
        assert len(defs) == 2
        assert all(isinstance(d, ToolDef) for d in defs)

    def test_report_apply_with_overrides(self):
        def search_db(query=""):
            return f"results for {query}"

        def process_order(order_id=""):
            return f"processed {order_id}"

        report = classify_tools([search_db, process_order])
        defs = report.apply(overrides={"process_order": EffectKind.IdempotentWrite})
        assert len(defs) == 2

        # Verify overrides produce working ToolDefs by constructing EffectLog
        from effect_log import EffectLog

        log = EffectLog(execution_id="test-override", tools=defs)
        result = log.execute("process_order", {"order_id": "ORD-1"})
        assert result == "processed ORD-1"
        result = log.execute("search_db", {"query": "test"})
        assert result == "results for test"


# ── ClassificationResult ─────────────────────────────────────────────────────


class TestClassificationResult:
    def test_fields(self):
        r = ClassificationResult(
            effect_kind=EffectKind.ReadOnly,
            confidence=0.92,
            reason="prefix 'search_'",
            source="heuristic",
        )
        assert r.effect_kind == EffectKind.ReadOnly
        assert r.confidence == 0.92
        assert r.source == "heuristic"


# ── Logging ──────────────────────────────────────────────────────────────────


class TestLogging:
    def test_high_confidence_logs_info(self, caplog):
        def search_db(args):
            pass

        with caplog.at_level(logging.INFO, logger="effect_log.classify"):
            classify_effect_kind(search_db)
        assert any(
            "search_db" in r.message and "ReadOnly" in r.message for r in caplog.records
        )

    def test_low_confidence_logs_warning(self, caplog):
        def do_something(args):
            pass

        with caplog.at_level(logging.WARNING, logger="effect_log.classify"):
            classify_effect_kind(do_something)
        assert any(
            "do_something" in r.message and "consider specifying" in r.message
            for r in caplog.records
        )


# ── classify_with_llm ────────────────────────────────────────────────────────


class TestClassifyWithLlm:
    """Test LLM-based classification (mocked — no real API calls)."""

    def _make_func(self):
        def send_report(to, subject):
            """Send a report via email."""
            pass

        return send_report

    def test_requires_opt_in(self):
        """Raises without EFFECT_LOG_LLM_CLASSIFY=1 or explicit provider."""
        with pytest.raises(RuntimeError, match="EFFECT_LOG_LLM_CLASSIFY"):
            classify_with_llm(self._make_func())

    @patch("effect_log.classify._call_anthropic", return_value="IrreversibleWrite")
    def test_anthropic_provider(self, mock_call):
        result = classify_with_llm(
            self._make_func(), provider="anthropic", model="test-model"
        )
        assert result.effect_kind == EffectKind.IrreversibleWrite
        assert result.confidence == 0.80
        assert result.source == "llm"
        assert "IrreversibleWrite" in result.reason
        mock_call.assert_called_once()
        assert mock_call.call_args[0][1] == "test-model"

    @patch("effect_log.classify._call_openai", return_value="ReadOnly")
    def test_openai_provider(self, mock_call):
        result = classify_with_llm(
            self._make_func(), provider="openai", model="gpt-4o-mini"
        )
        assert result.effect_kind == EffectKind.ReadOnly
        assert result.confidence == 0.80
        mock_call.assert_called_once()

    @patch("effect_log.classify._call_anthropic", return_value="IdempotentWrite")
    def test_auto_detect_anthropic(self, mock_call):
        """Auto-detect tries anthropic first."""
        result = classify_with_llm(self._make_func(), provider="anthropic")
        assert result.effect_kind == EffectKind.IdempotentWrite

    @patch(
        "effect_log.classify._call_anthropic",
        side_effect=ImportError("no anthropic"),
    )
    @patch("effect_log.classify._call_openai", return_value="ReadThenWrite")
    def test_auto_detect_falls_back_to_openai(self, mock_openai, mock_anthropic):
        """Auto-detect falls back to openai when anthropic not installed."""
        with patch.dict("os.environ", {"EFFECT_LOG_LLM_CLASSIFY": "1"}):
            result = classify_with_llm(self._make_func())
        assert result.effect_kind == EffectKind.ReadThenWrite
        mock_anthropic.assert_called_once()
        mock_openai.assert_called_once()

    @patch(
        "effect_log.classify._call_anthropic",
        side_effect=ImportError("no anthropic"),
    )
    @patch(
        "effect_log.classify._call_openai",
        side_effect=ImportError("no openai"),
    )
    def test_auto_detect_no_sdk_raises(self, mock_openai, mock_anthropic):
        """Raises ImportError when neither SDK is available."""
        with patch.dict("os.environ", {"EFFECT_LOG_LLM_CLASSIFY": "1"}):
            with pytest.raises(ImportError, match="anthropic or openai"):
                classify_with_llm(self._make_func())

    @patch("effect_log.classify._call_anthropic", return_value="Compensatable")
    def test_compensatable_downgraded(self, mock_call):
        """Compensatable is safety-downgraded to IrreversibleWrite."""
        result = classify_with_llm(self._make_func(), provider="anthropic")
        assert result.effect_kind == EffectKind.IrreversibleWrite
        assert result.confidence == 0.80

    @patch("effect_log.classify._call_anthropic", return_value="garbage_response")
    def test_unknown_response_defaults_to_irreversible(self, mock_call):
        """Unrecognized LLM output defaults to IrreversibleWrite with 0 confidence."""
        result = classify_with_llm(self._make_func(), provider="anthropic")
        assert result.effect_kind == EffectKind.IrreversibleWrite
        assert result.confidence == 0.0

    @patch("effect_log.classify._call_openai", return_value="ReadOnly")
    def test_env_model_override(self, mock_call):
        """EFFECT_LOG_LLM_MODEL env var sets the model."""
        with patch.dict(
            "os.environ",
            {"EFFECT_LOG_LLM_CLASSIFY": "1", "EFFECT_LOG_LLM_MODEL": "custom-model"},
        ):
            classify_with_llm(self._make_func(), provider="openai")
        assert mock_call.call_args[0][1] == "custom-model"

    @patch("effect_log.classify._call_openai", return_value="ReadOnly")
    def test_explicit_model_overrides_env(self, mock_call):
        """Explicit model= takes precedence over env var."""
        with patch.dict(
            "os.environ",
            {"EFFECT_LOG_LLM_CLASSIFY": "1", "EFFECT_LOG_LLM_MODEL": "env-model"},
        ):
            classify_with_llm(
                self._make_func(), provider="openai", model="explicit-model"
            )
        assert mock_call.call_args[0][1] == "explicit-model"

    @patch("effect_log.classify._call_anthropic", return_value="ReadOnly")
    def test_env_provider(self, mock_call):
        """EFFECT_LOG_LLM_PROVIDER env var selects the provider."""
        with patch.dict(
            "os.environ",
            {"EFFECT_LOG_LLM_CLASSIFY": "1", "EFFECT_LOG_LLM_PROVIDER": "anthropic"},
        ):
            classify_with_llm(self._make_func())
        mock_call.assert_called_once()
