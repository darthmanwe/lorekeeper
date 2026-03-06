"""Tests for src/tracing.py — OpenTelemetry instrumentation layer."""

import pytest
from unittest.mock import MagicMock, patch

from opentelemetry import trace

from src.tracing import (
    get_tracer,
    init_tracing,
    record_extraction_metrics,
    record_generation_metrics,
    record_guard_metrics,
    record_retrieval_metrics,
    trace_node,
    trace_segment_cycle,
)


class TestInitTracing:
    """Tests for tracer initialization."""

    def test_init_returns_tracer(self):
        import src.tracing as t
        t._initialized = False
        tracer = init_tracing(service_name="test-svc", exporter_type="console")
        assert tracer is not None
        t._initialized = False  # Reset for other tests

    def test_get_tracer_auto_initializes(self):
        import src.tracing as t
        t._initialized = False
        tracer = get_tracer()
        assert tracer is not None
        t._initialized = False

    def test_double_init_is_safe(self):
        import src.tracing as t
        t._initialized = False
        t1 = init_tracing(service_name="test1", exporter_type="console")
        t2 = init_tracing(service_name="test2", exporter_type="console")
        assert t1 is not None
        assert t2 is not None
        t._initialized = False


class TestTraceSegmentCycle:
    """Tests for the segment cycle context manager."""

    def test_context_manager_yields_span(self):
        import src.tracing as t
        t._initialized = False
        init_tracing(service_name="test", exporter_type="console")
        with trace_segment_cycle(
            seq_id=5, branch_id="main", mode="nkge", player_action="attack"
        ) as span:
            assert span is not None
        t._initialized = False


class TestTraceNode:
    """Tests for the trace_node decorator."""

    def test_decorator_preserves_return_value(self):
        import src.tracing as t
        t._initialized = False
        init_tracing(service_name="test", exporter_type="console")

        @trace_node("test_fn")
        def my_fn(x: int) -> dict:
            return {"value": x * 2}

        result = my_fn(5)
        assert result == {"value": 10}
        t._initialized = False

    def test_decorator_propagates_exceptions(self):
        import src.tracing as t
        t._initialized = False
        init_tracing(service_name="test", exporter_type="console")

        @trace_node("failing_fn")
        def failing_fn():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            failing_fn()
        t._initialized = False


class TestRecordMetrics:
    """Tests for metric recording functions."""

    def test_record_retrieval_metrics(self):
        span = MagicMock(spec=trace.Span)
        record_retrieval_metrics(span, graph_tokens=500, vector_tokens=200,
                                tier_counts={"T1": 3, "T2": 5})
        span.set_attribute.assert_any_call("retrieval.graph_tokens", 500)
        span.set_attribute.assert_any_call("retrieval.vector_tokens", 200)
        span.set_attribute.assert_any_call("retrieval.tier.T1", 3)

    def test_record_guard_metrics(self):
        span = MagicMock(spec=trace.Span)
        record_guard_metrics(span, violation_count=2,
                            severities={"critical": 1, "minor": 1}, blocking=True)
        span.set_attribute.assert_any_call("guard.violation_count", 2)
        span.set_attribute.assert_any_call("guard.blocking", True)
        span.set_attribute.assert_any_call("guard.severity.critical", 1)

    def test_record_generation_metrics(self):
        span = MagicMock(spec=trace.Span)
        record_generation_metrics(span, output_tokens=300, retry_count=1)
        span.set_attribute.assert_any_call("generation.output_tokens", 300)
        span.set_attribute.assert_any_call("generation.retry_count", 1)

    def test_record_extraction_metrics(self):
        span = MagicMock(spec=trace.Span)
        record_extraction_metrics(span, proposed=10, approved=8, flagged=2, committed=7)
        span.set_attribute.assert_any_call("extraction.proposed", 10)
        span.set_attribute.assert_any_call("extraction.approved", 8)
        span.set_attribute.assert_any_call("extraction.flagged", 2)
        span.set_attribute.assert_any_call("extraction.committed", 7)

    def test_record_retrieval_without_tier_counts(self):
        span = MagicMock(spec=trace.Span)
        record_retrieval_metrics(span, graph_tokens=100, vector_tokens=50)
        assert span.set_attribute.call_count == 2

    def test_record_guard_without_severities(self):
        span = MagicMock(spec=trace.Span)
        record_guard_metrics(span, violation_count=0)
        span.set_attribute.assert_any_call("guard.violation_count", 0)
        span.set_attribute.assert_any_call("guard.blocking", False)
