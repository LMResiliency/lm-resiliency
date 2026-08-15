"""Lease-backed registration for torchrun agents."""

from __future__ import annotations

import hashlib
import threading
import uuid
from collections.abc import Callable

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
        run_digest = hashlib.sha256(agent_identity.run_id.encode("utf-8")).hexdigest()
        node_digest = hashlib.sha256(agent_identity.node_id.encode("utf-8")).hexdigest()
        self._registration_key = (
            f"{_CONTROL_PREFIX}/runs/{run_digest}/agent-registrations/{node_digest}"
        )

    @property
    def registration_key(self) -> str:
        return self._registration_key

    def current(self) -> HeldAgentRegistration | None:
        entry = self._store.get(self._registration_key)
        if entry is None:
            return None
        return self._decode_entry(entry)

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
                return self._require_live_response(self._decode_entry(entry))
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
            return self._require_live_response(self._decode_entry(entry))
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
            return self._require_live_response(self._decode_entry(entry))
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

    def _decode_entry(self, entry: ControlStoreEntry) -> HeldAgentRegistration:
        try:
            record = AgentRegistrationRecord.from_json(entry.value)
        except (TypeError, ValueError) as error:
            raise AgentRegistrationCorrupt("persisted agent registration is malformed") from error
        identity = record.agent_identity
        if (
            identity.run_id != self._agent_identity.run_id
            or identity.node_id != self._agent_identity.node_id
        ):
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


__all__ = [
    "AgentRegistrationClockError",
    "AgentRegistrationCorrupt",
    "AgentRegistrationError",
    "AgentRegistrationLost",
    "AgentRegistrationManager",
    "AgentRegistrationUnavailable",
]
