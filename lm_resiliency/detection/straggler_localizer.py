"""Deprecated — straggler logic inlined into LayerReplayDetector."""

from lm_resiliency.detection.layer_replay import OpTiming, StragglerDetail

_MSG = "StragglerLocalizer has been inlined into LayerReplayDetector."


class StragglerLocalizer:
    """Deprecated. Use LayerReplayDetector.localize_straggler() directly."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_MSG)


__all__ = ["OpTiming", "StragglerDetail", "StragglerLocalizer"]
