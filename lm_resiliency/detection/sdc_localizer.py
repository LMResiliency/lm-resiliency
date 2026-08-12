"""Deprecated — SDC logic inlined into LayerReplayDetector."""

_MSG = "SDCLocalizer has been inlined into LayerReplayDetector."


class SDCLocalizer:
    """Deprecated. Use LayerReplayDetector directly."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_MSG)
