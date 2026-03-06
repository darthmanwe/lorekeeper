"""OpenTelemetry tracing for the Lorekeeper pipeline.

Provides a thin wrapper around OpenTelemetry to instrument the segment
generation cycle. Traces capture: retrieval latency, guard check results,
LLM generation time, extraction commit counts, and token budgets.

Two exporters are supported (configured via OTEL_EXPORTER env var):
- "console": prints spans to stdout (development)
- "otlp":    exports to an OTLP-compatible collector (production)
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

logger = logging.getLogger(__name__)

_TRACER_NAME = "lorekeeper"
_initialized = False


def init_tracing(
    service_name: str | None = None,
    exporter_type: str | None = None,
) -> trace.Tracer:
    """Initialize the OpenTelemetry tracer provider and return a tracer.

    Safe to call multiple times — subsequent calls return the existing tracer.

    Args:
        service_name: Override OTEL_SERVICE_NAME env var.
        exporter_type: "console" or "otlp". Override OTEL_EXPORTER env var.

    Returns:
        Configured Tracer instance.
    """
    global _initialized  # noqa: PLW0603
    if _initialized:
        return trace.get_tracer(_TRACER_NAME)

    svc = service_name or os.getenv("OTEL_SERVICE_NAME", "lorekeeper")
    exp = exporter_type or os.getenv("OTEL_EXPORTER", "console")

    resource = Resource.create({"service.name": svc})
    provider = TracerProvider(resource=resource)

    if exp == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            processor = BatchSpanProcessor(OTLPSpanExporter())
        except ImportError:
            logger.warning(
                "OTLP exporter not available, falling back to console"
            )
            processor = SimpleSpanProcessor(ConsoleSpanExporter())
    else:
        processor = SimpleSpanProcessor(ConsoleSpanExporter())

    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    _initialized = True
    logger.info("OpenTelemetry tracing initialized: service=%s, exporter=%s", svc, exp)
    return trace.get_tracer(_TRACER_NAME)


def get_tracer() -> trace.Tracer:
    """Return the lorekeeper tracer, initializing if needed."""
    if not _initialized:
        return init_tracing()
    return trace.get_tracer(_TRACER_NAME)


@contextmanager
def trace_segment_cycle(
    seq_id: int,
    branch_id: str,
    mode: str,
    player_action: str,
) -> Generator[trace.Span, None, None]:
    """Context manager that wraps a full segment generation cycle in a span.

    Records key attributes and yields the span for child span creation.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(
        "segment_generation_cycle",
        attributes={
            "lorekeeper.seq_id": seq_id,
            "lorekeeper.branch_id": branch_id,
            "lorekeeper.mode": mode,
            "lorekeeper.player_action": player_action[:200],
        },
    ) as span:
        yield span


def trace_node(name: str) -> Callable:
    """Decorator that wraps a pipeline node function in a child span.

    Captures function arguments as span attributes and records exceptions.
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            with tracer.start_as_current_span(f"node.{name}") as span:
                try:
                    result = fn(*args, **kwargs)
                    if isinstance(result, dict):
                        _record_result_attrs(span, result)
                    return result
                except Exception as exc:
                    span.set_status(
                        trace.StatusCode.ERROR, str(exc)
                    )
                    span.record_exception(exc)
                    raise
        return wrapper
    return decorator


def record_retrieval_metrics(
    span: trace.Span,
    graph_tokens: int,
    vector_tokens: int,
    tier_counts: dict[str, int] | None = None,
) -> None:
    """Record retrieval metrics on the current span."""
    span.set_attribute("retrieval.graph_tokens", graph_tokens)
    span.set_attribute("retrieval.vector_tokens", vector_tokens)
    if tier_counts:
        for tier, count in tier_counts.items():
            span.set_attribute(f"retrieval.tier.{tier}", count)


def record_guard_metrics(
    span: trace.Span,
    violation_count: int,
    severities: dict[str, int] | None = None,
    blocking: bool = False,
) -> None:
    """Record guard check metrics on the current span."""
    span.set_attribute("guard.violation_count", violation_count)
    span.set_attribute("guard.blocking", blocking)
    if severities:
        for sev, count in severities.items():
            span.set_attribute(f"guard.severity.{sev}", count)


def record_generation_metrics(
    span: trace.Span,
    output_tokens: int,
    retry_count: int = 0,
) -> None:
    """Record LLM generation metrics on the current span."""
    span.set_attribute("generation.output_tokens", output_tokens)
    span.set_attribute("generation.retry_count", retry_count)


def record_extraction_metrics(
    span: trace.Span,
    proposed: int,
    approved: int,
    flagged: int,
    committed: int,
) -> None:
    """Record extraction pipeline metrics on the current span."""
    span.set_attribute("extraction.proposed", proposed)
    span.set_attribute("extraction.approved", approved)
    span.set_attribute("extraction.flagged", flagged)
    span.set_attribute("extraction.committed", committed)


def _record_result_attrs(span: trace.Span, result: dict[str, Any]) -> None:
    """Extract telemetry-safe attributes from a node result dict."""
    for key, value in result.items():
        if isinstance(value, (int, float, str, bool)):
            span.set_attribute(f"result.{key}", value)
