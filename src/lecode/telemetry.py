"""Telemetry: Sentry error reporting + OpenTelemetry metrics (opt-in).

Off by default; enabled via ``[telemetry]`` in the config. Both backends
require the ``telemetry`` extra (``uv tool install 'lecode[telemetry]'``) —
imports are guarded and every failure mode degrades to no-op (telemetry
must never break the agent).

Metrics exported over OTLP/HTTP:

- ``lecode.turns`` (counter, by model/stop_reason)
- ``lecode.turn.duration_s`` (histogram)
- ``lecode.tokens.input`` / ``lecode.tokens.output`` (counters, by model)
- ``lecode.cost_usd`` (counter, by model)
- ``lecode.tool_calls`` (counter, by tool/is_error)
- ``lecode.tool.duration_s`` (histogram, by tool)

Sentry captures uncaught provider/tool errors and unhandled exceptions.
"""

from __future__ import annotations

import logging
from typing import Any

from lecode.config.models import TelemetryConfig

log = logging.getLogger(__name__)

#: Module-level handles; ``None`` until :func:`init_telemetry` succeeds.
_meter: Any | None = None
_provider: Any | None = None
_sentry_on = False

_counters: dict[str, Any] = {}
_histograms: dict[str, Any] = {}


def _counter(name: str, description: str, unit: str) -> Any | None:
    if _meter is None:
        return None
    if name not in _counters:
        _counters[name] = _meter.create_counter(name, description=description, unit=unit)
    return _counters[name]


def _histogram(name: str, description: str, unit: str) -> Any | None:
    if _meter is None:
        return None
    if name not in _histograms:
        _histograms[name] = _meter.create_histogram(name, description=description, unit=unit)
    return _histograms[name]


def init_telemetry(config: TelemetryConfig, *, version: str = "") -> list[str]:
    """Set up Sentry and OTel per config; returns warnings (fail-open).

    Safe to call once at startup; later calls are no-ops. Missing extras or
    backend errors produce warnings, never exceptions.
    """
    global _meter, _provider, _sentry_on
    warnings: list[str] = []
    if not config.enabled:
        return warnings

    if config.sentry_dsn:
        try:
            import sentry_sdk

            sentry_sdk.init(
                dsn=config.sentry_dsn,
                release=f"lecode@{version}" if version else None,
                environment=config.environment,
                send_default_pii=False,
                enable_tracing=False,
            )
            _sentry_on = True
        except ImportError:
            warnings.append("telemetry: sentry-sdk not installed (pip extra: telemetry)")
        except Exception as e:  # bad DSN etc. — never block startup
            warnings.append(f"telemetry: sentry init failed: {e}")

    if config.otlp_endpoint:
        try:
            from opentelemetry import metrics
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                OTLPMetricExporter,
            )
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.sdk.resources import Resource

            exporter = OTLPMetricExporter(endpoint=f"{config.otlp_endpoint}/v1/metrics")
            reader = PeriodicExportingMetricReader(
                exporter, export_interval_millis=int(config.export_interval_s * 1000)
            )
            _provider = MeterProvider(
                resource=Resource.create(
                    {
                        "service.name": config.service_name,
                        "deployment.environment": config.environment,
                    }
                ),
                metric_readers=[reader],
            )
            metrics.set_meter_provider(_provider)
            _meter = metrics.get_meter("lecode", version=version or None)
        except ImportError:
            warnings.append("telemetry: opentelemetry-sdk not installed (pip extra: telemetry)")
        except Exception as e:
            warnings.append(f"telemetry: otel init failed: {e}")

    return warnings


def shutdown_telemetry() -> None:
    """Flush pending metrics and Sentry events; safe when never initialized."""
    global _meter, _provider, _sentry_on
    if _provider is not None:
        try:
            _provider.force_flush()
            _provider.shutdown()
        except Exception:
            log.debug("otel shutdown failed", exc_info=True)
        _provider = None
        _meter = None
    if _sentry_on:
        try:
            import sentry_sdk

            sentry_sdk.flush(timeout=2.0)
        except Exception:
            log.debug("sentry flush failed", exc_info=True)
        _sentry_on = False


# -- recording API (all no-ops when telemetry is off) ---------------------------


def record_turn(
    *,
    model: str,
    stop_reason: str,
    turns: int,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    elapsed_s: float,
) -> None:
    """One completed agent run (a user prompt → final answer)."""
    attrs = {"model": model, "stop_reason": stop_reason}
    if (c := _counter("lecode.turns", "Completed agent runs", "{turn}")) is not None:
        c.add(1, attrs)
    if (h := _histogram("lecode.turn.duration_s", "Agent run duration", "s")) is not None:
        h.record(elapsed_s, {"model": model})
    if input_tokens and (c := _counter("lecode.tokens.input", "Input tokens", "{token}")):
        c.add(input_tokens, {"model": model})
    if output_tokens and (c := _counter("lecode.tokens.output", "Output tokens", "{token}")):
        c.add(output_tokens, {"model": model})
    if cost_usd and (c := _counter("lecode.cost_usd", "LLM cost", "USD")):
        c.add(cost_usd, {"model": model})


def record_tool_call(name: str, *, is_error: bool, duration_s: float) -> None:
    """One executed tool call."""
    attrs = {"tool": name, "is_error": str(is_error).lower()}
    if (c := _counter("lecode.tool_calls", "Tool calls", "{call}")) is not None:
        c.add(1, attrs)
    if (h := _histogram("lecode.tool.duration_s", "Tool call duration", "s")) is not None:
        h.record(duration_s, {"tool": name})


def capture_exception(exc: BaseException, *, context: str = "") -> None:
    """Send an error to Sentry (no-op when Sentry is off)."""
    if not _sentry_on:
        return
    try:
        import sentry_sdk

        with sentry_sdk.isolation_scope() as scope:
            if context:
                scope.set_tag("lecode.context", context)
            sentry_sdk.capture_exception(exc)
    except Exception:
        log.debug("sentry capture failed", exc_info=True)
