"""Read-only preparation of one complete restart-plan publication."""

from __future__ import annotations

from collections.abc import Callable

from lm_resiliency.integrations.torchrun._control_store import ControlStore
from lm_resiliency.integrations.torchrun._restart_plan_publication_authority import (
    RestartPlanPublicationAuthority,
)
from lm_resiliency.integrations.torchrun._restart_plan_publication_lifecycle import (
    RestartPlanPublicationLifecycleFence,
)
from lm_resiliency.integrations.torchrun._restart_plan_publication_lifecycle_reader import (
    RestartPlanPublicationLifecycleConflict,
    RestartPlanPublicationLifecycleCorrupt,
    RestartPlanPublicationLifecycleReader,
)
from lm_resiliency.integrations.torchrun._restart_plan_publication_preparation import (
    RestartPlanPublicationAuthorityPreparer,
    RestartPlanPublicationPreparationClockError,
    RestartPlanPublicationPreparationConflict,
    RestartPlanPublicationPreparationCorrupt,
    RestartPlanPublicationPreparationLeaseLost,
)
from lm_resiliency.integrations.torchrun._restart_plan_publication_records import (
    RestartPlanPublicationRecords,
)
from lm_resiliency.integrations.torchrun._restart_plan_publication_state import (
    PreparedRestartPlanPublication,
)


class RestartPlanPublicationError(RuntimeError):
    """Base error for preparing one complete restart-plan publication."""


class RestartPlanPublicationConflict(RestartPlanPublicationError):
    """Raised when publication inputs change or no closed intent is available."""


class RestartPlanPublicationLeaseLost(RestartPlanPublicationError):
    """Raised when the plan's coordinator authority is no longer live."""


class RestartPlanPublicationClockError(RestartPlanPublicationError):
    """Raised when the publication preparation clock is unsafe."""


class RestartPlanPublicationCorrupt(RestartPlanPublicationError):
    """Raised when durable publication dependencies are contradictory."""


class RestartPlanPublicationPreparer:
    """Compose authenticated authority and lifecycle state without mutation."""

    def __init__(
        self,
        store: ControlStore,
        *,
        run_id: str,
        clock: Callable[[], int],
    ) -> None:
        self._authority_preparer = RestartPlanPublicationAuthorityPreparer(
            store,
            run_id=run_id,
            clock=clock,
        )
        self._lifecycle_reader = RestartPlanPublicationLifecycleReader(
            store,
            run_id=run_id,
        )

    def prepare(
        self,
        records: RestartPlanPublicationRecords,
    ) -> PreparedRestartPlanPublication:
        """Return one authenticated, lifecycle-fenced publication value."""

        authority = self._prepare_authority(records)
        lifecycle_fence = self._read_lifecycle()
        try:
            return PreparedRestartPlanPublication(
                authority=authority,
                lifecycle_fence=lifecycle_fence,
            )
        except TypeError as error:
            raise RestartPlanPublicationCorrupt(
                "authenticated publication inputs have invalid types"
            ) from error
        except ValueError as error:
            raise RestartPlanPublicationConflict(
                "restart-plan publication inputs changed during preparation"
            ) from error

    def _prepare_authority(
        self,
        records: RestartPlanPublicationRecords,
    ) -> RestartPlanPublicationAuthority:
        try:
            return self._authority_preparer.prepare(records)
        except RestartPlanPublicationPreparationClockError as error:
            raise RestartPlanPublicationClockError(str(error)) from error
        except RestartPlanPublicationPreparationLeaseLost as error:
            raise RestartPlanPublicationLeaseLost(str(error)) from error
        except RestartPlanPublicationPreparationConflict as error:
            raise RestartPlanPublicationConflict(str(error)) from error
        except RestartPlanPublicationPreparationCorrupt as error:
            raise RestartPlanPublicationCorrupt(str(error)) from error

    def _read_lifecycle(self) -> RestartPlanPublicationLifecycleFence:
        try:
            return self._lifecycle_reader.read()
        except RestartPlanPublicationLifecycleConflict as error:
            raise RestartPlanPublicationConflict(str(error)) from error
        except RestartPlanPublicationLifecycleCorrupt as error:
            raise RestartPlanPublicationCorrupt(str(error)) from error


__all__ = [
    "RestartPlanPublicationClockError",
    "RestartPlanPublicationConflict",
    "RestartPlanPublicationCorrupt",
    "RestartPlanPublicationError",
    "RestartPlanPublicationLeaseLost",
    "RestartPlanPublicationPreparer",
]
