"""OOB timing instrumentation for Python-visible blocking stages."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any, Generic, Literal, TypeVar

from lm_resiliency.detection.op_tracker import DiagnosticStage, OpTracker, StageToken

T = TypeVar("T")
CheckpointOperation = Literal["read", "write"]


def stage_key(value: str) -> int:
    """Return a stable signed int64 event key."""
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=True) or 1


class DiagnosticStageMonitor:
    """Publish stage timing without holding a lock across the measured operation."""

    def __init__(self, tracker: OpTracker, lock: threading.Lock) -> None:
        self._tracker = tracker
        self._lock = lock

    def start(self, kind: DiagnosticStage, key: str) -> StageToken:
        with self._lock:
            return self._tracker.stage_started(kind, stage_key(key))

    def finish(self, token: StageToken) -> float:
        with self._lock:
            duration_ns = self._tracker.stage_finished(token)
        return duration_ns / 1_000_000.0

    def should_measure_next_step(self, detection_interval: int) -> bool:
        """Whether the next training step is on the configured detection cadence."""
        return detection_interval > 0 and (self._tracker.step + 1) % detection_interval == 0

    @contextmanager
    def measure(self, kind: DiagnosticStage, key: str) -> Iterator[None]:
        token = self.start(kind, key)
        try:
            yield
        finally:
            self.finish(token)


class InstrumentedDataLoader(Generic[T]):
    """Transparent iterable proxy that samples ``next()`` at the configured cadence."""

    def __init__(
        self,
        dataloader: Iterable[T],
        monitor: DiagnosticStageMonitor | None,
        *,
        name: str = "train",
        detection_interval: int = 1,
    ) -> None:
        self._scout_dataloader = dataloader
        self._scout_monitor = monitor
        self._scout_key = f"dataloader:{name}"
        self._scout_detection_interval = detection_interval

    def __iter__(self) -> Iterator[T]:
        return _InstrumentedIterator(
            iter(self._scout_dataloader),
            self._scout_monitor,
            self._scout_key,
            self._scout_detection_interval,
        )

    def __len__(self) -> int:
        return len(self._scout_dataloader)  # type: ignore[arg-type]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._scout_dataloader, name)


class _InstrumentedIterator(Generic[T]):
    def __init__(
        self,
        iterator: Iterator[T],
        monitor: DiagnosticStageMonitor | None,
        key: str,
        detection_interval: int,
    ) -> None:
        self._iterator = iterator
        self._monitor = monitor
        self._key = key
        self._detection_interval = detection_interval

    def __iter__(self) -> _InstrumentedIterator[T]:
        return self

    def __next__(self) -> T:
        if self._monitor is None or not self._monitor.should_measure_next_step(
            self._detection_interval
        ):
            return next(self._iterator)
        with self._monitor.measure(DiagnosticStage.DATA_LOADING, self._key):
            return next(self._iterator)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._iterator, name)


def instrument_dataloader(
    dataloader: Iterable[T],
    monitor: DiagnosticStageMonitor | None,
    *,
    name: str = "train",
    detection_interval: int = 1,
) -> InstrumentedDataLoader[T]:
    """Return an iterable that publishes DataLoader wait timing to SCOUT."""
    if isinstance(dataloader, InstrumentedDataLoader):
        return dataloader
    return InstrumentedDataLoader(
        dataloader,
        monitor,
        name=name,
        detection_interval=detection_interval,
    )


@contextmanager
def checkpoint_io(
    monitor: DiagnosticStageMonitor | None,
    operation: CheckpointOperation,
    *,
    name: str = "framework",
) -> Iterator[None]:
    """Publish a checkpoint read or write boundary to the OOB daemon."""
    if operation not in ("read", "write"):
        raise ValueError("checkpoint operation must be 'read' or 'write'")
    if monitor is None:
        yield
        return
    kind = (
        DiagnosticStage.CHECKPOINT_READ if operation == "read" else DiagnosticStage.CHECKPOINT_WRITE
    )
    with monitor.measure(kind, f"checkpoint:{operation}:{name}"):
        yield
