"""可观测：可选 Langfuse trace；未配置时 no-op。

.env 配置 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST 即启用。
"""

import logging
import time
from typing import Any, Protocol

from .config import Settings

log = logging.getLogger(__name__)


class TraceSink(Protocol):
    def trace(
        self, name: str, input: Any, output: Any, metadata: dict[str, Any], duration_ms: float
    ) -> None: ...


class NoopSink:
    """无配置时的空实现。"""

    def trace(
        self, name: str, input: Any, output: Any, metadata: dict[str, Any], duration_ms: float
    ) -> None:
        return None


class RecordingSink:
    """测试/本地调试用：把事件收进内存。"""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def trace(
        self, name: str, input: Any, output: Any, metadata: dict[str, Any], duration_ms: float
    ) -> None:
        self.events.append(
            {
                "name": name,
                "input": input,
                "output": output,
                "metadata": metadata,
                "duration_ms": duration_ms,
            }
        )


class LangfuseSink:
    """Langfuse 云端 / 自托管。"""

    def __init__(self) -> None:
        from langfuse import Langfuse

        self._client = Langfuse()

    def trace(
        self, name: str, input: Any, output: Any, metadata: dict[str, Any], duration_ms: float
    ) -> None:
        self._client.trace(
            name=name,
            input=input,
            output=output,
            metadata={**metadata, "duration_ms": duration_ms},
        )
        self._client.flush()


class TimedSink:
    """包装：自动计时。"""

    def __init__(self, sink: TraceSink) -> None:
        self._sink = sink
        self._start = time.monotonic()

    def trace(self, name: str, input: Any, output: Any, metadata: dict[str, Any]) -> None:
        duration_ms = (time.monotonic() - self._start) * 1000
        self._sink.trace(
            name=name, input=input, output=output, metadata=metadata, duration_ms=duration_ms
        )


def get_trace_sink(settings: Settings) -> TraceSink:
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        try:
            return LangfuseSink()
        except Exception as e:  # noqa: BLE001
            log.warning("Langfuse 初始化失败，降级为 no-op: %s", e)
            return NoopSink()
    return NoopSink()
