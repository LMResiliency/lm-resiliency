"""Stable, fail-closed reads of agent-registration authority history."""

from __future__ import annotations

from dataclasses import dataclass

from lm_resiliency.integrations.torchrun._agent_registration import (
    agent_registration_key,
)
from lm_resiliency.integrations.torchrun._agent_registration_history import (
    AgentRegistrationAuthority,
    AgentRegistrationAuthorityCorrupt,
)
from lm_resiliency.integrations.torchrun._agent_registration_records import (
    HeldAgentRegistration,
)
from lm_resiliency.integrations.torchrun._control_store import ControlStore

_MAX_READ_ATTEMPTS = 8


class AgentRegistrationHistoryError(RuntimeError):
    """Base error for agent-registration history reads."""


class AgentRegistrationHistoryCorrupt(AgentRegistrationHistoryError):
    """Raised when persisted registration history is contradictory."""


@dataclass(frozen=True, slots=True)
class AgentRegistrationHistory:
    """One complete authority history and its optional current registration."""

    authorities: tuple[AgentRegistrationAuthority, ...]
    current: HeldAgentRegistration | None

    def __post_init__(self) -> None:
        if not isinstance(self.authorities, tuple) or not all(
            isinstance(authority, AgentRegistrationAuthority) for authority in self.authorities
        ):
            raise TypeError(
                "AgentRegistrationHistory.authorities must be a tuple of AgentRegistrationAuthority"
            )
        if self.current is not None and not isinstance(
            self.current,
            HeldAgentRegistration,
        ):
            raise TypeError(
                "AgentRegistrationHistory.current must be HeldAgentRegistration or None"
            )
        _validate_history(self.authorities)
        if self.current is not None and (
            not self.authorities or self.authorities[-1].registration != self.current
        ):
            raise ValueError(
                "AgentRegistrationHistory current registration is not its history tail"
            )


class AgentRegistrationHistoryReader:
    """Read and verify one run/node registration's retained value history."""

    def __init__(
        self,
        store: ControlStore,
        *,
        run_id: str,
        node_id: str,
    ) -> None:
        self._store = store
        self._run_id = _nonempty_string(run_id, "run_id")
        self._node_id = _nonempty_string(node_id, "node_id")
        self._registration_key = agent_registration_key(
            self._run_id,
            self._node_id,
        )

    @property
    def registration_key(self) -> str:
        return self._registration_key

    def read(self) -> AgentRegistrationHistory:
        """Return one stable, verified registration-history snapshot."""

        for _ in range(_MAX_READ_ATTEMPTS):
            history = self._store.get_history(self._registration_key)
            current = self._store.get(self._registration_key)
            has_history = self._store.has_history(self._registration_key)
            if (
                history != self._store.get_history(self._registration_key)
                or current != self._store.get(self._registration_key)
                or has_history != self._store.has_history(self._registration_key)
            ):
                continue
            if bool(history) != has_history:
                raise AgentRegistrationHistoryCorrupt(
                    "agent-registration value history contradicts its durable marker"
                )
            if current is not None and (not history or history[-1] != current):
                raise AgentRegistrationHistoryCorrupt(
                    "current agent registration is absent from its value history"
                )
            try:
                authorities = tuple(
                    AgentRegistrationAuthority.from_entry(
                        entry,
                        run_id=self._run_id,
                        node_id=self._node_id,
                    )
                    for entry in history
                )
                current_registration = (
                    None
                    if current is None
                    else AgentRegistrationAuthority.from_entry(
                        current,
                        run_id=self._run_id,
                        node_id=self._node_id,
                    ).registration
                )
                return AgentRegistrationHistory(
                    authorities=authorities,
                    current=current_registration,
                )
            except AgentRegistrationAuthorityCorrupt as error:
                raise AgentRegistrationHistoryCorrupt(
                    "agent-registration history contains an invalid authority"
                ) from error
            except (TypeError, ValueError) as error:
                raise AgentRegistrationHistoryCorrupt(
                    "agent-registration history contains invalid state"
                ) from error
        raise AgentRegistrationHistoryError(
            "agent-registration history changed repeatedly during read"
        )


def _validate_history(
    authorities: tuple[AgentRegistrationAuthority, ...],
) -> None:
    if not authorities:
        return
    first = authorities[0]
    if first.mutation_sequence != 1 or first.value_sequence != 1 or first.lifetime_sequence != 1:
        raise AgentRegistrationHistoryCorrupt(
            "agent-registration history does not begin at initial store sequences"
        )
    seen_registration_ids = {first.registration.record.registration_id}
    seen_fencing_tokens = {first.registration.fencing_token}
    for previous, current in zip(authorities, authorities[1:], strict=False):
        _validate_transition(previous, current)
        fencing_token = current.registration.fencing_token
        if fencing_token in seen_fencing_tokens:
            raise AgentRegistrationHistoryCorrupt(
                "agent-registration fencing token reappears in history"
            )
        seen_fencing_tokens.add(fencing_token)
        previous_id = previous.registration.record.registration_id
        current_id = current.registration.record.registration_id
        if current_id != previous_id:
            if current_id in seen_registration_ids:
                raise AgentRegistrationHistoryCorrupt(
                    "agent-registration identity reappears after replacement"
                )
            seen_registration_ids.add(current_id)


def _validate_transition(
    previous: AgentRegistrationAuthority,
    current: AgentRegistrationAuthority,
) -> None:
    if current.transaction_sequence <= previous.transaction_sequence:
        raise AgentRegistrationHistoryCorrupt(
            "agent-registration transaction sequences do not advance"
        )
    previous_registration = previous.registration
    current_registration = current.registration
    if current_registration.granted_at_unix_ms < previous_registration.granted_at_unix_ms:
        raise AgentRegistrationHistoryCorrupt("agent-registration grant times move backward")
    mutation_delta = current.mutation_sequence - previous.mutation_sequence
    value_delta = current.value_sequence - previous.value_sequence
    lifetime_delta = current.lifetime_sequence - previous.lifetime_sequence
    transaction_delta = current.transaction_sequence - previous.transaction_sequence
    if transaction_delta < mutation_delta:
        raise AgentRegistrationHistoryCorrupt(
            "agent-registration mutation count exceeds transaction ordering"
        )
    if lifetime_delta not in (0, 1):
        raise AgentRegistrationHistoryCorrupt("agent-registration history omits a key lifetime")
    if mutation_delta != (1 if lifetime_delta == 0 else 2):
        raise AgentRegistrationHistoryCorrupt("agent-registration history omits a key mutation")
    same_record = current_registration.record == previous_registration.record
    expected_value_delta = 0 if lifetime_delta == 0 and same_record else 1
    if value_delta != expected_value_delta:
        raise AgentRegistrationHistoryCorrupt(
            "agent-registration value sequence contradicts its records"
        )
    if same_record:
        if lifetime_delta != 0:
            raise AgentRegistrationHistoryCorrupt(
                "one agent registration crosses a recreated key lifetime"
            )
        if current_registration.granted_at_unix_ms >= previous_registration.expires_at_unix_ms:
            raise AgentRegistrationHistoryCorrupt(
                "agent-registration history renews an expired registration"
            )
        return
    if current_registration.record.registration_id == previous_registration.record.registration_id:
        raise AgentRegistrationHistoryCorrupt(
            "one agent-registration identity changes its persisted record"
        )
    if (
        lifetime_delta == 0
        and current_registration.granted_at_unix_ms < previous_registration.expires_at_unix_ms
    ):
        raise AgentRegistrationHistoryCorrupt("agent-registration replacements overlap")


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


__all__ = [
    "AgentRegistrationHistory",
    "AgentRegistrationHistoryCorrupt",
    "AgentRegistrationHistoryError",
    "AgentRegistrationHistoryReader",
]
