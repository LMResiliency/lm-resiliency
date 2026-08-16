"""Closed-lifecycle revision fencing for restart-plan publication."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from lm_resiliency.integrations.torchrun._restart_intent_lifecycle_auth import (
    AuthenticatedInitialRestartIntentClosure,
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


__all__ = ["RestartPlanPublicationLifecycleFence"]
