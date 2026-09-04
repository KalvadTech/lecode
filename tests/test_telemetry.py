"""Tests for the opt-in Sentry + OpenTelemetry telemetry layer."""

from __future__ import annotations

import sys

import pytest
from tests.fakes import FakeProvider

import lecode.telemetry as telemetry
from lecode.agent.runner import AgentRunner
from lecode.agent.tools.base import Tool, ToolRegistry
from lecode.agent.tools.base import ToolResult as ToolExecResult
from lecode.config.models import TelemetryConfig
from lecode.providers.openai_compat import ProviderError


@pytest.fixture(autouse=True)
def reset_telemetry(monkeypatch):
    """Isolate module-level telemetry state between tests."""
    monkeypatch.setattr(telemetry, "_meter", None)
    monkeypatch.setattr(telemetry, "_provider", None)
    monkeypatch.setattr(telemetry, "_sentry_on", False)
    monkeypatch.setattr(telemetry, "_counters", {})
    monkeypatch.setattr(telemetry, "_histograms", {})


@pytest.fixture
def in_memory_meter(monkeypatch):
    """A meter backed by OTel's InMemoryMetricReader (no network)."""
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    monkeypatch.setattr(telemetry, "_meter", provider.get_meter("test"))
    return reader


def _metric_values(reader) -> dict[str, float]:
    data = reader.get_metrics_data()
    values: dict[str, float] = {}
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                total = 0.0
                for dp in metric.data.data_points:
                    total += dp.value if hasattr(dp, "value") else dp.sum
                values[metric.name] = total
    return values


def test_disabled_by_default_is_noop():
    warnings = telemetry.init_telemetry(TelemetryConfig())
    assert warnings == []
    # recording APIs must not raise when telemetry is off
    telemetry.record_turn(
        model="m",
        stop_reason="done",
        turns=1,
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.01,
        elapsed_s=1.0,
    )
    telemetry.record_tool_call("bash", is_error=False, duration_s=0.1)
    telemetry.capture_exception(ValueError("boom"), context="test")
    telemetry.shutdown_telemetry()


def test_enabled_without_backends_configured():
    warnings = telemetry.init_telemetry(TelemetryConfig(enabled=True))
    assert warnings == []
    assert telemetry._meter is None
    assert telemetry._sentry_on is False


def test_sentry_missing_extra_warns(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentry_sdk", None)
    warnings = telemetry.init_telemetry(
        TelemetryConfig(enabled=True, sentry_dsn="https://key@example.com/1")
    )
    assert any("sentry" in w for w in warnings)
    assert telemetry._sentry_on is False


def test_sentry_init_and_capture(monkeypatch):
    import sentry_sdk

    calls: dict = {}
    monkeypatch.setattr(sentry_sdk, "init", lambda **kw: calls.update(kw))
    warnings = telemetry.init_telemetry(
        TelemetryConfig(enabled=True, sentry_dsn="https://key@example.com/1"),
        version="1.2.3",
    )
    assert warnings == []
    assert telemetry._sentry_on is True
    assert calls["dsn"] == "https://key@example.com/1"
    assert calls["release"] == "lecode@1.2.3"
    assert calls["send_default_pii"] is False

    captured: list = []
    monkeypatch.setattr(sentry_sdk, "capture_exception", captured.append)
    err = ValueError("boom")
    telemetry.capture_exception(err, context="tool:bash")
    assert captured == [err]


def test_capture_exception_noop_when_sentry_off():
    telemetry.capture_exception(ValueError("boom"))  # must not raise


def test_otlp_missing_extra_warns(monkeypatch):
    monkeypatch.setitem(sys.modules, "opentelemetry", None)
    warnings = telemetry.init_telemetry(
        TelemetryConfig(enabled=True, otlp_endpoint="http://localhost:4318")
    )
    assert any("opentelemetry" in w for w in warnings)
    assert telemetry._meter is None


def test_otlp_bogus_endpoint_does_not_raise():
    warnings = telemetry.init_telemetry(
        TelemetryConfig(enabled=True, otlp_endpoint="http://127.0.0.1:9")
    )
    assert warnings == []
    assert telemetry._meter is not None
    telemetry.shutdown_telemetry()  # flush failure is swallowed
    assert telemetry._meter is None


def test_record_turn_metrics(in_memory_meter):
    telemetry.record_turn(
        model="gpt-x",
        stop_reason="done",
        turns=2,
        input_tokens=100,
        output_tokens=40,
        cost_usd=0.002,
        elapsed_s=1.5,
    )
    values = _metric_values(in_memory_meter)
    assert values["lecode.turns"] == 1
    assert values["lecode.tokens.input"] == 100
    assert values["lecode.tokens.output"] == 40
    assert values["lecode.cost_usd"] == pytest.approx(0.002)
    assert values["lecode.turn.duration_s"] == pytest.approx(1.5)


def test_record_tool_call_metrics(in_memory_meter):
    telemetry.record_tool_call("bash", is_error=False, duration_s=0.25)
    telemetry.record_tool_call("bash", is_error=True, duration_s=0.5)
    values = _metric_values(in_memory_meter)
    assert values["lecode.tool_calls"] == 2
    assert values["lecode.tool.duration_s"] == pytest.approx(0.75)


class EchoTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="echo",
            description="Echo text.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
            },
        )

    async def run(self, args, ctx) -> ToolExecResult:
        return ToolExecResult(content=str(args.get("text", "")))


class BoomTool(Tool):
    def __init__(self) -> None:
        super().__init__(name="boom", description="Always raises.", parameters={})

    async def run(self, args, ctx) -> ToolExecResult:
        raise RuntimeError("kaboom")


async def test_tool_dispatch_records_metrics(tool_ctx, in_memory_meter):
    registry = ToolRegistry([EchoTool()])
    _, result = await registry.dispatch_result("c1", "echo", '{"text": "hi"}', tool_ctx)
    assert result.content == "hi"
    assert _metric_values(in_memory_meter)["lecode.tool_calls"] == 1


async def test_tool_error_records_and_captures(tool_ctx, in_memory_meter, monkeypatch):
    captured: list = []
    monkeypatch.setattr(telemetry, "_sentry_on", True)

    def fake_capture(e, *, context=""):
        captured.append((e, context))

    monkeypatch.setattr(telemetry, "capture_exception", fake_capture)
    # re-bind the reference used inside base.py
    import lecode.agent.tools.base as base

    monkeypatch.setattr(base, "capture_exception", telemetry.capture_exception)

    registry = ToolRegistry([BoomTool()])
    _, result = await registry.dispatch_result("c1", "boom", "{}", tool_ctx)
    assert result.is_error
    assert captured and captured[0][1] == "tool:boom"
    assert _metric_values(in_memory_meter)["lecode.tool_calls"] == 1


async def test_runner_records_turn(tool_ctx, in_memory_meter):
    usage = {"prompt_tokens": 5, "completion_tokens": 3}
    provider = FakeProvider([{"text": ["done"], "usage": usage}])
    runner = AgentRunner(provider, ToolRegistry([EchoTool()]), tool_ctx)
    result = await runner.run([{"role": "user", "content": "hi"}])
    assert result.stop_reason == "done"
    values = _metric_values(in_memory_meter)
    assert values["lecode.turns"] == 1
    assert values["lecode.tokens.input"] == 5
    assert values["lecode.tokens.output"] == 3


async def test_runner_provider_error_captured(tool_ctx, monkeypatch):
    captured: list = []
    import lecode.agent.runner as runner_mod

    monkeypatch.setattr(
        runner_mod, "capture_exception", lambda e, *, context="": captured.append((e, context))
    )
    provider = FakeProvider([{"error": ProviderError("upstream down", retryable=False)}])
    runner = AgentRunner(provider, ToolRegistry([]), tool_ctx)
    with pytest.raises(ProviderError):
        await runner.run([{"role": "user", "content": "hi"}])
    assert captured and captured[0][1] == "provider"
