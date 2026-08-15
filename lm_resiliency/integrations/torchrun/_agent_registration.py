"""Lease-backed registration for torchrun agents."""

from __future__ import annotations

import hashlib
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from lm_resiliency.integrations.torchrun._agent_registration_records import (
    AgentRegistrationRecord,
    HeldAgentRegistration,
)
from lm_resiliency.integrations.torchrun._control_store import (
    ControlStore,
    ControlStoreClockError,
    ControlStoreConflict,
    ControlStoreDeadlineExceeded,
    ControlStoreEntry,
    ControlStoreTooEarly,
)
from lm_resiliency.integrations.torchrun._protocol import AgentIdentity

_CONTROL_PREFIX = "lm_resiliency/torchrun/v1"
_MAX_CAS_ATTEMPTS = 16


class AgentRegistrationError(RuntimeError):
    """Base error for agent-registration operations."""


class AgentRegistrationUnavailable(AgentRegistrationError):
    """Raised when another live agent owns the node registration."""


class AgentRegistrationLost(AgentRegistrationError):
    """Raised when a held registration expired or was fenced."""


class AgentRegistrationCorrupt(AgentRegistrationError):
    """Raised when the persisted registration is malformed or contradictory."""


class AgentRegistrationClockError(AgentRegistrationError):
    """Raised when client or store time contradicts the registration timeline."""


@dataclass(frozen=True, slots=True)
class AgentRegistrationObservation:
    """One conservative observation of trusted node registrations."""

    observed_at_unix_ms: int
    live: Mapping[str, HeldAgentRegistration]
    expired: Mapping[str, HeldAgentRegistration]
    missing_node_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _positive_integer(
            self.observed_at_unix_ms,
            "AgentRegistrationObservation.observed_at_unix_ms",
        )
        live = _registration_mapping(
            self.live,
            "AgentRegistrationObservation.live",
        )
        expired = _registration_mapping(
            self.expired,
            "AgentRegistrationObservation.expired",
        )
        missing = _node_ids(
            self.missing_node_ids,
            "AgentRegistrationObservation.missing_node_ids",
            require_nonempty=False,
        )
        overlap = (set(live) & set(expired)) | ((set(live) | set(expired)) & set(missing))
        if overlap:
            raise ValueError("AgentRegistrationObservation node classifications must be disjoint")
        object.__setattr__(self, "live", MappingProxyType(live))
        object.__setattr__(self, "expired", MappingProxyType(expired))
        object.__setattr__(self, "missing_node_ids", missing)


class AgentRegistrationReader:
    """Read registrations for an explicit trusted scheduler node set."""

    def __init__(
        self,
        store: ControlStore,
        *,
        run_id: str,
        clock: Callable[[], int],
    ) -> None:
        self._store = store
        self._run_id = _nonempty_string(run_id, "run_id")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock
        self._clock_lock = threading.Lock()
        self._last_now_unix_ms = 0

    def get(self, node_id: str) -> HeldAgentRegistration | None:
        normalized_node_id = _nonempty_string(node_id, "node_id")
        entry = self._store.get(agent_registration_key(self._run_id, normalized_node_id))
        if entry is None:
            return None
        return _decode_registration_entry(
            entry,
            run_id=self._run_id,
            node_id=normalized_node_id,
        )

    def observe(self, node_ids: Sequence[str]) -> AgentRegistrationObservation:
        normalized_node_ids = _node_ids(
            node_ids,
            "node_ids",
            require_nonempty=True,
        )
        registrations = {
            node_id: registration
            for node_id in normalized_node_ids
            if (registration := self.get(node_id)) is not None
        }
        observed_at_unix_ms = self._now_unix_ms()
        live: dict[str, HeldAgentRegistration] = {}
        expired: dict[str, HeldAgentRegistration] = {}
        for node_id, registration in registrations.items():
            if observed_at_unix_ms < registration.granted_at_unix_ms:
                raise AgentRegistrationClockError(
                    "observer clock precedes an authoritative registration commit"
                )
            target = live if registration.expires_at_unix_ms > observed_at_unix_ms else expired
            target[node_id] = registration
        return AgentRegistrationObservation(
            observed_at_unix_ms=observed_at_unix_ms,
            live=live,
            expired=expired,
            missing_node_ids=tuple(
                node_id for node_id in normalized_node_ids if node_id not in registrations
            ),
        )

    def _now_unix_ms(self) -> int:
        with self._clock_lock:
            now_unix_ms = _positive_integer(self._clock(), "clock")
            if now_unix_ms < self._last_now_unix_ms:
                raise AgentRegistrationClockError(
                    "agent registration observer clock moved backward"
                )
            self._last_now_unix_ms = now_unix_ms
        return now_unix_ms


class AgentRegistrationManager:
    """Maintain one run- and node-scoped agent registration."""

    def __init__(
        self,
        store: ControlStore,
        *,
        agent_identity: AgentIdentity,
        lease_duration_ms: int,
        clock: Callable[[], int],
    ) -> None:
        if not isinstance(agent_identity, AgentIdentity):
            raise TypeError("agent_identity must be AgentIdentity")
        self._store = store
        self._agent_identity = agent_identity
        self._lease_duration_ms = _positive_integer(
            lease_duration_ms,
            "lease_duration_ms",
        )
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock
        self._clock_lock = threading.Lock()
        self._last_now_unix_ms = 0
        self._registration_key = agent_registration_key(
            agent_identity.run_id,
            agent_identity.node_id,
        )

    @property
    def registration_key(self) -> str:
        return self._registration_key

    def current(self) -> HeldAgentRegistration | None:
        entry = self._store.get(self._registration_key)
        if entry is None:
            return None
        return _decode_registration_entry(
            entry,
            run_id=self._agent_identity.run_id,
            node_id=self._agent_identity.node_id,
        )

    def register(self) -> HeldAgentRegistration:
        now_unix_ms = self._now_unix_ms()
        for _ in range(_MAX_CAS_ATTEMPTS):
            current = self.current()
            if current is not None and current.expires_at_unix_ms > now_unix_ms:
                if current.record.agent_identity != self._agent_identity:
                    raise AgentRegistrationUnavailable(
                        f"node {self._agent_identity.node_id!r} is registered by "
                        f"agent {current.record.agent_identity.agent_id!r} until "
                        f"{current.expires_at_unix_ms}"
                    )
                if current.record.lease_duration_ms != self._lease_duration_ms:
                    raise AgentRegistrationUnavailable(
                        "active agent registration uses a different lease duration"
                    )
                try:
                    entry = self._store.compare_set_in_window(
                        self._registration_key,
                        expected_revision=current.fencing_token,
                        not_before_unix_ms=now_unix_ms,
                        deadline_unix_ms=current.expires_at_unix_ms,
                        value=current.record.to_json(),
                    )
                except ControlStoreDeadlineExceeded as error:
                    raise AgentRegistrationUnavailable(
                        "existing agent registration expired at the control store "
                        "during registration"
                    ) from error
                except ControlStoreTooEarly as error:
                    raise AgentRegistrationClockError(
                        "control-store time precedes the agent registration observation"
                    ) from error
                except ControlStoreClockError as error:
                    raise AgentRegistrationClockError(
                        "authoritative control-store clock moved backward"
                    ) from error
                except ControlStoreConflict:
                    continue
                return self._require_live_response(
                    _decode_registration_entry(
                        entry,
                        run_id=self._agent_identity.run_id,
                        node_id=self._agent_identity.node_id,
                    )
                )
            record = AgentRegistrationRecord(
                agent_identity=self._agent_identity,
                registration_id=uuid.uuid4().hex,
                lease_duration_ms=self._lease_duration_ms,
            )
            try:
                entry = self._store.compare_set_in_window(
                    self._registration_key,
                    expected_revision=None if current is None else current.fencing_token,
                    not_before_unix_ms=max(
                        now_unix_ms,
                        0 if current is None else current.expires_at_unix_ms,
                    ),
                    deadline_unix_ms=None,
                    value=record.to_json(),
                )
            except ControlStoreTooEarly as error:
                raise AgentRegistrationClockError(
                    "control-store time precedes the agent registration observation "
                    "or existing registration expiry"
                ) from error
            except ControlStoreClockError as error:
                raise AgentRegistrationClockError(
                    "authoritative control-store clock moved backward"
                ) from error
            except ControlStoreConflict:
                continue
            return self._require_live_response(
                _decode_registration_entry(
                    entry,
                    run_id=self._agent_identity.run_id,
                    node_id=self._agent_identity.node_id,
                )
            )
        raise AgentRegistrationUnavailable(
            "agent registration changed repeatedly during registration"
        )

    def renew(self, registration: HeldAgentRegistration) -> HeldAgentRegistration:
        self._validate_owned_handle(registration)
        now_unix_ms = self._now_unix_ms()
        if registration.expires_at_unix_ms <= now_unix_ms:
            raise AgentRegistrationLost("agent registration expired before renewal")
        try:
            entry = self._store.compare_set_in_window(
                self._registration_key,
                expected_revision=registration.fencing_token,
                not_before_unix_ms=now_unix_ms,
                deadline_unix_ms=registration.expires_at_unix_ms,
                value=registration.record.to_json(),
            )
        except ControlStoreDeadlineExceeded as error:
            raise AgentRegistrationLost(
                "agent registration expired at the control store before renewal"
            ) from error
        except ControlStoreTooEarly as error:
            raise AgentRegistrationClockError(
                "control-store time precedes the agent renewal observation"
            ) from error
        except ControlStoreClockError as error:
            raise AgentRegistrationClockError(
                "authoritative control-store clock moved backward"
            ) from error
        except ControlStoreConflict as error:
            raise AgentRegistrationLost("agent registration changed before renewal") from error
        try:
            return self._require_live_response(
                _decode_registration_entry(
                    entry,
                    run_id=self._agent_identity.run_id,
                    node_id=self._agent_identity.node_id,
                )
            )
        except AgentRegistrationUnavailable as error:
            raise AgentRegistrationLost(
                "renewed agent registration expired before the response arrived"
            ) from error

    def release(self, registration: HeldAgentRegistration) -> int:
        self._validate_owned_handle(registration)
        now_unix_ms = self._now_unix_ms()
        if registration.expires_at_unix_ms <= now_unix_ms:
            raise AgentRegistrationLost("agent registration expired before release")
        try:
            return self._store.compare_delete_in_window(
                self._registration_key,
                expected_revision=registration.fencing_token,
                not_before_unix_ms=now_unix_ms,
                deadline_unix_ms=registration.expires_at_unix_ms,
            )
        except ControlStoreDeadlineExceeded as error:
            raise AgentRegistrationLost(
                "agent registration expired at the control store before release"
            ) from error
        except ControlStoreTooEarly as error:
            raise AgentRegistrationClockError(
                "control-store time precedes the agent release observation"
            ) from error
        except ControlStoreClockError as error:
            raise AgentRegistrationClockError(
                "authoritative control-store clock moved backward"
            ) from error
        except ControlStoreConflict as error:
            raise AgentRegistrationLost("agent registration changed before release") from error

    def _require_live_response(
        self,
        registration: HeldAgentRegistration,
    ) -> HeldAgentRegistration:
        now_unix_ms = self._now_unix_ms()
        if now_unix_ms < registration.granted_at_unix_ms:
            raise AgentRegistrationClockError(
                "agent clock precedes the authoritative registration commit time"
            )
        if registration.expires_at_unix_ms <= now_unix_ms:
            raise AgentRegistrationUnavailable(
                "agent registration expired before the store response arrived"
            )
        return registration

    def _validate_owned_handle(self, registration: HeldAgentRegistration) -> None:
        if not isinstance(registration, HeldAgentRegistration):
            raise TypeError("registration must be HeldAgentRegistration")
        if (
            registration.record.agent_identity != self._agent_identity
            or registration.record.lease_duration_ms != self._lease_duration_ms
        ):
            raise AgentRegistrationLost("agent registration handle belongs to another manager")
        current = self.current()
        if current != registration:
            raise AgentRegistrationLost(
                "agent registration handle does not match persisted ownership"
            )

    def _now_unix_ms(self) -> int:
        with self._clock_lock:
            now_unix_ms = _positive_integer(self._clock(), "clock")
            if now_unix_ms < self._last_now_unix_ms:
                raise AgentRegistrationClockError("agent registration clock moved backward")
            self._last_now_unix_ms = now_unix_ms
        return now_unix_ms


def _positive_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


def agent_registration_key(run_id: str, node_id: str) -> str:
    """Derive one run/node-scoped registration key without exposing identities."""
    normalized_run_id = _nonempty_string(run_id, "run_id")
    normalized_node_id = _nonempty_string(node_id, "node_id")
    run_digest = hashlib.sha256(normalized_run_id.encode("utf-8")).hexdigest()
    node_digest = hashlib.sha256(normalized_node_id.encode("utf-8")).hexdigest()
    return f"{_CONTROL_PREFIX}/runs/{run_digest}/agent-registrations/{node_digest}"


def _decode_registration_entry(
    entry: ControlStoreEntry,
    *,
    run_id: str,
    node_id: str,
) -> HeldAgentRegistration:
    try:
        record = AgentRegistrationRecord.from_json(entry.value)
    except (TypeError, ValueError) as error:
        raise AgentRegistrationCorrupt("persisted agent registration is malformed") from error
    identity = record.agent_identity
    if identity.run_id != run_id or identity.node_id != node_id:
        raise AgentRegistrationCorrupt(
            "persisted agent registration belongs to another run or node"
        )
    if entry.committed_at_unix_ms is None:
        raise AgentRegistrationCorrupt(
            "persisted agent registration has no authoritative commit time"
        )
    return HeldAgentRegistration(
        record=record,
        fencing_token=entry.revision,
        granted_at_unix_ms=entry.committed_at_unix_ms,
    )


def _registration_mapping(
    value: object,
    path: str,
) -> dict[str, HeldAgentRegistration]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    result: dict[str, HeldAgentRegistration] = {}
    for node_id, registration in value.items():
        normalized_node_id = _nonempty_string(node_id, f"{path}.key")
        if not isinstance(registration, HeldAgentRegistration):
            raise TypeError(f"{path}[{node_id!r}] must be HeldAgentRegistration")
        if registration.record.agent_identity.node_id != normalized_node_id:
            raise ValueError(f"{path}[{node_id!r}] registration node does not match key")
        result[normalized_node_id] = registration
    return result


def _node_ids(
    value: object,
    path: str,
    *,
    require_nonempty: bool,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{path} must be a sequence")
    result = tuple(
        _nonempty_string(node_id, f"{path}[{index}]") for index, node_id in enumerate(value)
    )
    if require_nonempty and not result:
        raise ValueError(f"{path} must contain at least one node ID")
    if len(result) != len(set(result)):
        raise ValueError(f"{path} must contain unique node IDs")
    return result


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


__all__ = [
    "AgentRegistrationClockError",
    "AgentRegistrationCorrupt",
    "AgentRegistrationError",
    "AgentRegistrationLost",
    "AgentRegistrationManager",
    "AgentRegistrationObservation",
    "AgentRegistrationReader",
    "AgentRegistrationUnavailable",
    "agent_registration_key",
]
