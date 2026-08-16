"""Strict persisted restart-acknowledgement receipt records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, TypeVar

from lm_resiliency.integrations.torchrun._agent_registration import (
    agent_registration_key,
)
from lm_resiliency.integrations.torchrun._agent_registration_history import (
    AgentRegistrationAuthority,
)
from lm_resiliency.integrations.torchrun._agent_registration_records import (
    AgentRegistrationRecord,
    HeldAgentRegistration,
)
from lm_resiliency.integrations.torchrun._control_store import ControlStoreWrite
from lm_resiliency.integrations.torchrun._coordinator_lease import HeldCoordinatorLease
from lm_resiliency.integrations.torchrun._coordinator_lease_history import (
    CoordinatorLeaseAuthority,
)
from lm_resiliency.integrations.torchrun._protocol import (
    AgentIdentity,
    ProtocolValidationError,
    RestartAck,
)
from lm_resiliency.integrations.torchrun._restart_intent_open_execution import (
    CommittedInitialRestartIntentOpen,
)
from lm_resiliency.integrations.torchrun._restart_intent_records import (
    RestartIntentRecord,
)

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class RestartAckReceiptRecord:
    """One worker acknowledgement and its authenticated receipt provenance."""

    SCHEMA_VERSION: ClassVar[int] = 1

    acknowledgement: RestartAck
    intent_record: RestartIntentRecord
    agent_registration: AgentRegistrationRecord
    registration_fencing_token: int
    registration_granted_at_unix_ms: int
    received_at_unix_ms: int

    def __post_init__(self) -> None:
        if not isinstance(self.acknowledgement, RestartAck):
            raise TypeError("RestartAckReceiptRecord.acknowledgement must be RestartAck")
        if not isinstance(self.intent_record, RestartIntentRecord):
            raise TypeError("RestartAckReceiptRecord.intent_record must be RestartIntentRecord")
        if not isinstance(self.agent_registration, AgentRegistrationRecord):
            raise TypeError(
                "RestartAckReceiptRecord.agent_registration must be AgentRegistrationRecord"
            )
        _positive_integer(
            self.registration_fencing_token,
            "RestartAckReceiptRecord.registration_fencing_token",
        )
        _positive_integer(
            self.registration_granted_at_unix_ms,
            "RestartAckReceiptRecord.registration_granted_at_unix_ms",
        )
        _positive_integer(
            self.received_at_unix_ms,
            "RestartAckReceiptRecord.received_at_unix_ms",
        )
        intent = self.intent_record.intent
        acknowledgement = self.acknowledgement
        if (
            acknowledgement.intent_id != intent.intent_id
            or acknowledgement.run_id != intent.run_id
            or acknowledgement.generation != intent.generation
        ):
            raise ValueError(
                "RestartAckReceiptRecord acknowledgement does not identify its restart intent"
            )
        identity = self.agent_registration.agent_identity
        if (
            acknowledgement.run_id != identity.run_id
            or acknowledgement.node_id != identity.node_id
            or acknowledgement.agent_id != identity.agent_id
        ):
            raise ValueError(
                "RestartAckReceiptRecord acknowledgement does not match its authenticated "
                "agent registration"
            )
        if self.received_at_unix_ms < self.registration_granted_at_unix_ms:
            raise ValueError(
                "RestartAckReceiptRecord receipt precedes its agent registration grant"
            )
        if self.received_at_unix_ms >= self.registration_expires_at_unix_ms:
            raise ValueError(
                "RestartAckReceiptRecord receipt is outside its agent registration lifetime"
            )
        if self.received_at_unix_ms >= intent.prepare_deadline_unix_ms:
            raise ValueError(
                "RestartAckReceiptRecord receipt is at or after the restart intent deadline"
            )

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json()).hexdigest()

    @property
    def authenticated_registration(self) -> HeldAgentRegistration:
        return HeldAgentRegistration(
            record=self.agent_registration,
            fencing_token=self.registration_fencing_token,
            granted_at_unix_ms=self.registration_granted_at_unix_ms,
        )

    @property
    def registration_expires_at_unix_ms(self) -> int:
        return self.registration_granted_at_unix_ms + self.agent_registration.lease_duration_ms

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "acknowledgement": self.acknowledgement.to_dict(),
            "intent_record": self.intent_record.to_dict(),
            "agent_registration": self.agent_registration.to_dict(),
            "registration_fencing_token": self.registration_fencing_token,
            "registration_granted_at_unix_ms": self.registration_granted_at_unix_ms,
            "received_at_unix_ms": self.received_at_unix_ms,
        }

    def to_json(self) -> bytes:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, encoded: bytes) -> RestartAckReceiptRecord:
        value = _json_object(encoded, cls.__name__)
        _fields(
            value,
            path=cls.__name__,
            required={
                "schema_version",
                "acknowledgement",
                "intent_record",
                "agent_registration",
                "registration_fencing_token",
                "registration_granted_at_unix_ms",
                "received_at_unix_ms",
            },
        )
        _schema_version(
            value["schema_version"],
            cls.__name__,
            expected=cls.SCHEMA_VERSION,
        )
        acknowledgement_value = _mapping(
            value["acknowledgement"],
            "RestartAckReceiptRecord.acknowledgement",
        )
        _schema_version(
            acknowledgement_value.get("schema_version"),
            "RestartAckReceiptRecord.acknowledgement",
            expected=RestartAck.SCHEMA_VERSION,
        )
        try:
            acknowledgement = RestartAck.from_dict(acknowledgement_value)
        except ProtocolValidationError as error:
            raise ValueError("RestartAckReceiptRecord.acknowledgement is invalid") from error
        intent_record = _decode_nested_record(
            value["intent_record"],
            path="RestartAckReceiptRecord.intent_record",
            decoder=RestartIntentRecord.from_json,
        )
        registration_value = _mapping(
            value["agent_registration"],
            "RestartAckReceiptRecord.agent_registration",
        )
        identity_value = _mapping(
            registration_value.get("agent_identity"),
            "RestartAckReceiptRecord.agent_registration.agent_identity",
        )
        _schema_version(
            identity_value.get("schema_version"),
            "RestartAckReceiptRecord.agent_registration.agent_identity",
            expected=AgentIdentity.SCHEMA_VERSION,
        )
        agent_registration = _decode_nested_record(
            registration_value,
            path="RestartAckReceiptRecord.agent_registration",
            decoder=AgentRegistrationRecord.from_json,
        )
        return cls(
            acknowledgement=acknowledgement,
            intent_record=intent_record,
            agent_registration=agent_registration,
            registration_fencing_token=_positive_integer(
                value["registration_fencing_token"],
                "RestartAckReceiptRecord.registration_fencing_token",
            ),
            registration_granted_at_unix_ms=_positive_integer(
                value["registration_granted_at_unix_ms"],
                "RestartAckReceiptRecord.registration_granted_at_unix_ms",
            ),
            received_at_unix_ms=_positive_integer(
                value["received_at_unix_ms"],
                "RestartAckReceiptRecord.received_at_unix_ms",
            ),
        )


@dataclass(frozen=True, slots=True)
class RestartAckWriteRecords:
    """Immutable receipt write and read conditions for one active node."""

    receipt: RestartAckReceiptRecord
    opened: CommittedInitialRestartIntentOpen
    acknowledgement_key: str
    agent_registration_key: str

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, RestartAckReceiptRecord):
            raise TypeError("RestartAckWriteRecords.receipt must be RestartAckReceiptRecord")
        if not isinstance(self.opened, CommittedInitialRestartIntentOpen):
            raise TypeError(
                "RestartAckWriteRecords.opened must be CommittedInitialRestartIntentOpen"
            )
        if self.receipt.intent_record != self.opened.prepared.record:
            raise ValueError(
                "RestartAckWriteRecords receipt does not answer its committed restart intent"
            )
        if self.receipt.received_at_unix_ms < self.opened.committed_at_unix_ms:
            raise ValueError("RestartAckWriteRecords receipt predates its committed restart intent")
        acknowledgement = self.receipt.acknowledgement
        active_nodes = set(
            self.opened.prepared.current.snapshot.record.assignment.slot_to_node_id.values()
        )
        if acknowledgement.node_id not in active_nodes:
            raise ValueError(
                "RestartAckWriteRecords acknowledgement node is not active in its generation"
            )
        expected_registration_key = agent_registration_key(
            acknowledgement.run_id,
            acknowledgement.node_id,
        )
        if self.agent_registration_key != expected_registration_key:
            raise ValueError("RestartAckWriteRecords.agent_registration_key is not canonical")
        node_digest = hashlib.sha256(acknowledgement.node_id.encode("utf-8")).hexdigest()
        expected_acknowledgement_key = (
            f"{self.opened.prepared.intent_key}/acknowledgements/{node_digest}"
        )
        if self.acknowledgement_key != expected_acknowledgement_key:
            raise ValueError("RestartAckWriteRecords.acknowledgement_key is not canonical")
        keys = {
            self.acknowledgement_key,
            self.agent_registration_key,
            self.opened.prepared.intent_key,
            self.opened.prepared.intent_head_key,
        }
        if len(keys) != 4:
            raise ValueError("RestartAckWriteRecords transaction key roles must be distinct")

    @property
    def writes(self) -> Mapping[str, ControlStoreWrite]:
        return MappingProxyType(
            {
                self.acknowledgement_key: ControlStoreWrite(
                    expected_revision=None,
                    value=self.receipt.to_json(),
                    require_never_created=True,
                )
            }
        )

    @property
    def conditions(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                self.opened.prepared.intent_key: self.opened.intent_entry.revision,
                self.opened.prepared.intent_head_key: self.opened.head_entry.revision,
                self.agent_registration_key: self.receipt.registration_fencing_token,
            }
        )


@dataclass(frozen=True, slots=True)
class AuthenticatedRestartAckState:
    """One receipt bound to current intent, registration, and lease state."""

    receipt: RestartAckReceiptRecord
    opened: CommittedInitialRestartIntentOpen
    registration_authority: AgentRegistrationAuthority
    coordinator_authority: CoordinatorLeaseAuthority

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, RestartAckReceiptRecord):
            raise TypeError("AuthenticatedRestartAckState.receipt must be RestartAckReceiptRecord")
        if not isinstance(self.opened, CommittedInitialRestartIntentOpen):
            raise TypeError(
                "AuthenticatedRestartAckState.opened must be CommittedInitialRestartIntentOpen"
            )
        if not isinstance(self.registration_authority, AgentRegistrationAuthority):
            raise TypeError(
                "AuthenticatedRestartAckState.registration_authority must be "
                "AgentRegistrationAuthority"
            )
        if not isinstance(self.coordinator_authority, CoordinatorLeaseAuthority):
            raise TypeError(
                "AuthenticatedRestartAckState.coordinator_authority must be "
                "CoordinatorLeaseAuthority"
            )
        if self.receipt.intent_record != self.opened.prepared.record:
            raise ValueError(
                "AuthenticatedRestartAckState receipt does not answer its current intent"
            )
        if self.receipt.received_at_unix_ms < self.opened.committed_at_unix_ms:
            raise ValueError("AuthenticatedRestartAckState receipt predates its current intent")
        if self.registration != self.receipt.authenticated_registration:
            raise ValueError(
                "AuthenticatedRestartAckState receipt does not match its current registration"
            )
        run_id = self.receipt.acknowledgement.run_id
        if self.coordinator_authority.lease.record.run_id != run_id:
            raise ValueError(
                "AuthenticatedRestartAckState coordinator lease belongs to another run"
            )
        active_node_ids = set(
            self.opened.prepared.current.snapshot.record.assignment.slot_to_node_id.values()
        )
        if self.receipt.acknowledgement.node_id not in active_node_ids:
            raise ValueError("AuthenticatedRestartAckState acknowledgement node is not active")

    @property
    def registration(self) -> HeldAgentRegistration:
        """Return the authenticated held registration."""

        return self.registration_authority.registration


@dataclass(frozen=True, slots=True)
class PreparedRestartAckWrite:
    """One immutable acknowledgement write with coordinator authority."""

    records: RestartAckWriteRecords
    registration_authority: AgentRegistrationAuthority
    coordinator_authority: CoordinatorLeaseAuthority
    not_before_unix_ms: int
    deadline_unix_ms: int

    def __post_init__(self) -> None:
        if not isinstance(self.records, RestartAckWriteRecords):
            raise TypeError("PreparedRestartAckWrite.records must be RestartAckWriteRecords")
        if not isinstance(self.registration_authority, AgentRegistrationAuthority):
            raise TypeError(
                "PreparedRestartAckWrite.registration_authority must be AgentRegistrationAuthority"
            )
        if not isinstance(self.coordinator_authority, CoordinatorLeaseAuthority):
            raise TypeError(
                "PreparedRestartAckWrite.coordinator_authority must be CoordinatorLeaseAuthority"
            )
        _positive_integer(
            self.not_before_unix_ms,
            "PreparedRestartAckWrite.not_before_unix_ms",
        )
        _positive_integer(
            self.deadline_unix_ms,
            "PreparedRestartAckWrite.deadline_unix_ms",
        )
        receipt = self.records.receipt
        if self.registration != receipt.authenticated_registration:
            raise ValueError(
                "PreparedRestartAckWrite receipt does not match its registration authority"
            )
        expected_registration_revision = self.records.conditions[
            self.records.agent_registration_key
        ]
        if expected_registration_revision != self.registration.fencing_token:
            raise ValueError(
                "PreparedRestartAckWrite registration condition does not match its authority"
            )
        lease = self.coordinator_authority.lease
        if lease.record.run_id != receipt.acknowledgement.run_id:
            raise ValueError("PreparedRestartAckWrite coordinator lease belongs to another run")
        if self.not_before_unix_ms < receipt.received_at_unix_ms:
            raise ValueError("PreparedRestartAckWrite cannot precede acknowledgement receipt")
        if self.not_before_unix_ms < lease.granted_at_unix_ms:
            raise ValueError("PreparedRestartAckWrite cannot precede coordinator lease grant")
        if self.not_before_unix_ms >= self.deadline_unix_ms:
            raise ValueError("PreparedRestartAckWrite.not_before_unix_ms must precede its deadline")
        if self.deadline_unix_ms > lease.expires_at_unix_ms:
            raise ValueError("PreparedRestartAckWrite deadline exceeds coordinator lease")
        if self.deadline_unix_ms > receipt.intent_record.intent.prepare_deadline_unix_ms:
            raise ValueError("PreparedRestartAckWrite deadline exceeds restart intent")
        if self.deadline_unix_ms > receipt.registration_expires_at_unix_ms:
            raise ValueError("PreparedRestartAckWrite deadline exceeds agent registration")

    @property
    def lease(self) -> HeldCoordinatorLease:
        return self.coordinator_authority.lease

    @property
    def registration(self) -> HeldAgentRegistration:
        return self.registration_authority.registration

    @property
    def coordinator_lease_key(self) -> str:
        return self.records.opened.prepared.coordinator_lease_key

    @property
    def expected_guard_revision(self) -> int:
        return self.lease.fencing_token

    @property
    def writes(self) -> Mapping[str, ControlStoreWrite]:
        return self.records.writes

    @property
    def conditions(self) -> Mapping[str, int]:
        return self.records.conditions


def _decode_nested_record(
    value: object,
    *,
    path: str,
    decoder: Callable[[bytes], _T],
) -> _T:
    mapping = _mapping(value, path)
    try:
        return decoder(_canonical_json(mapping))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} is invalid") from error


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _json_object(encoded: bytes, path: str) -> Mapping[str, object]:
    if not isinstance(encoded, bytes):
        raise ValueError(f"{path}: expected encoded bytes")
    try:
        value = json.loads(
            encoded,
            object_pairs_hook=_reject_duplicate_object_fields,
        )
    except _DuplicateRestartAckField as error:
        raise ValueError(f"{path}: {error}") from error
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: invalid JSON") from error
    return _mapping(value, path)


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _fields(
    value: Mapping[str, object],
    *,
    path: str,
    required: set[str],
) -> None:
    missing = required - set(value)
    unknown = set(value) - required
    if missing:
        raise ValueError(f"{path}: missing fields {sorted(missing)!r}")
    if unknown:
        raise ValueError(f"{path}: unknown fields {sorted(unknown)!r}")


def _schema_version(value: object, path: str, *, expected: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"{path}.schema_version: unsupported value {value!r}")
    return value


def _positive_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


class _DuplicateRestartAckField(ValueError):
    pass


def _reject_duplicate_object_fields(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateRestartAckField(f"duplicate field {key!r}")
        value[key] = item
    return value


__all__ = [
    "AuthenticatedRestartAckState",
    "PreparedRestartAckWrite",
    "RestartAckReceiptRecord",
    "RestartAckWriteRecords",
]
