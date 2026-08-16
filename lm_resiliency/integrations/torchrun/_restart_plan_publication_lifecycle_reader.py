"""Read-only preparation of restart-plan publication lifecycle fencing."""

from __future__ import annotations

from lm_resiliency.integrations.torchrun._control_store import ControlStore
from lm_resiliency.integrations.torchrun._restart_intent_lifecycle_reader import (
    InitialRestartIntentLifecycleReader,
    RestartIntentLifecycleReadCorrupt,
    RestartIntentLifecycleReadError,
)
from lm_resiliency.integrations.torchrun._restart_plan_publication_lifecycle import (
    RestartPlanPublicationLifecycleFence,
)


class RestartPlanPublicationLifecycleReadError(RuntimeError):
    """Base error for reading restart-plan publication lifecycle fencing."""


class RestartPlanPublicationLifecycleConflict(RestartPlanPublicationLifecycleReadError):
    """Raised when the intent is not closed or its successor already exists."""


class RestartPlanPublicationLifecycleCorrupt(RestartPlanPublicationLifecycleReadError):
    """Raised when persisted restart-intent lifecycle state is contradictory."""


class RestartPlanPublicationLifecycleReader:
    """Read one authenticated closed-intent publication fence."""

    def __init__(self, store: ControlStore, *, run_id: str) -> None:
        self._lifecycle_reader = InitialRestartIntentLifecycleReader(
            store,
            run_id=run_id,
        )

    def read(self) -> RestartPlanPublicationLifecycleFence:
        """Return the exact lifecycle revisions required for plan publication."""

        try:
            closure = self._lifecycle_reader.read()
        except RestartIntentLifecycleReadCorrupt as error:
            raise RestartPlanPublicationLifecycleCorrupt(
                "persisted restart-intent lifecycle is corrupt"
            ) from error
        except RestartIntentLifecycleReadError as error:
            raise RestartPlanPublicationLifecycleConflict(
                "restart-intent lifecycle changed repeatedly during read"
            ) from error
        if closure is None:
            raise RestartPlanPublicationLifecycleConflict(
                "restart intent is not closed for plan publication"
            )
        if closure.immediate_successor is not None:
            raise RestartPlanPublicationLifecycleConflict(
                "restart plan successor generation is already committed"
            )
        try:
            return RestartPlanPublicationLifecycleFence(closure)
        except (TypeError, ValueError) as error:
            raise RestartPlanPublicationLifecycleCorrupt(
                "authenticated restart-intent closure cannot fence plan publication"
            ) from error


__all__ = [
    "RestartPlanPublicationLifecycleConflict",
    "RestartPlanPublicationLifecycleCorrupt",
    "RestartPlanPublicationLifecycleReadError",
    "RestartPlanPublicationLifecycleReader",
]
