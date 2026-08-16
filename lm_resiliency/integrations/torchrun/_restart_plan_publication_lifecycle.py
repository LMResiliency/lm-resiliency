"""Closed-lifecycle revision fencing for restart-plan publication."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from lm_resiliency.integrations.torchrun._control_store import ControlStore
from lm_resiliency.integrations.torchrun._coordinator_lease_history import (
    CoordinatorLeaseHistoryCorrupt,
    CoordinatorLeaseHistoryError,
)
from lm_resiliency.integrations.torchrun._generation_reader import (
    GenerationStateCorrupt,
    GenerationStateError,
)
from lm_resiliency.integrations.torchrun._restart_intent_lifecycle_auth import (
    AuthenticatedInitialRestartIntentClosure,
)
from lm_resiliency.integrations.torchrun._restart_intent_lifecycle_reader import (
    InitialRestartIntentLifecycleReader,
    RestartIntentLifecycleReadCorrupt,
    RestartIntentLifecycleReadError,
)

_CONTROL_PREFIX = "lm_resiliency/torchrun/v1"


@dataclass(frozen=True, slots=True)
class RestartPlanPublicationLifecycleFence:
    """Exact closed-intent revisions that must survive plan publication."""

    closure: AuthenticatedInitialRestartIntentClosure

    def __post_init__(self) -> None:
        if not isinstance(self.closure, AuthenticatedInitialRestartIntentClosure):
            raise TypeError(
                "RestartPlanPublicationLifecycleFence.closure must be "
                "AuthenticatedInitialRestartIntentClosure"
            )
        if self.closure.immediate_successor is not None:
            raise ValueError(
                "RestartPlanPublicationLifecycleFence successor generation is already committed"
            )

    @property
    def run_prefix(self) -> str:
        run_digest = hashlib.sha256(self.closure.intent.intent.run_id.encode("utf-8")).hexdigest()
        return f"{_CONTROL_PREFIX}/runs/{run_digest}"

    @property
    def intent_key(self) -> str:
        intent_digest = hashlib.sha256(
            self.closure.intent.intent.intent_id.encode("utf-8")
        ).hexdigest()
        return f"{self.run_prefix}/restart-intents/{intent_digest}"

    @property
    def intent_head_key(self) -> str:
        return f"{self.run_prefix}/restart-intent-head"

    @property
    def closure_key(self) -> str:
        return (
            f"{self.run_prefix}/restart-intent-closures/{self.closure.lifecycle_head.closure_index}"
        )

    @property
    def lifecycle_head_key(self) -> str:
        return f"{self.run_prefix}/restart-intent-lifecycle-head"

    @property
    def conditions(self) -> Mapping[str, int]:
        state = self.closure.state
        return MappingProxyType(
            {
                self.intent_key: state.intent_entry.revision,
                self.intent_head_key: state.closed_head_entry.revision,
                self.closure_key: state.lifecycle_entry.revision,
                self.lifecycle_head_key: state.lifecycle_head_entry.revision,
            }
        )

    @property
    def closed_at_unix_ms(self) -> int:
        return self.closure.closed_at_unix_ms

    @property
    def transaction_sequence(self) -> int:
        return self.closure.transaction_sequence


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
        except (CoordinatorLeaseHistoryCorrupt, GenerationStateCorrupt) as error:
            raise RestartPlanPublicationLifecycleCorrupt(
                "persisted restart-intent lifecycle dependencies are corrupt"
            ) from error
        except RestartIntentLifecycleReadError as error:
            raise RestartPlanPublicationLifecycleConflict(
                "restart-intent lifecycle changed repeatedly during read"
            ) from error
        except (CoordinatorLeaseHistoryError, GenerationStateError) as error:
            raise RestartPlanPublicationLifecycleConflict(
                "restart-intent lifecycle dependencies changed repeatedly during read"
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
    "RestartPlanPublicationLifecycleFence",
    "RestartPlanPublicationLifecycleReadError",
    "RestartPlanPublicationLifecycleReader",
]
