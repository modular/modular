# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides debugging tools and visualizations."""

from .callbacks import ServerCallbacks
from ._kserve_impl import ModelInferRequest, ModelInferResponse
from ._serve_rt import Batch


struct ColoredTextCodes:
    """Utilities for printing colored ASCII text."""

    @staticmethod
    fn code(idx: Int) -> String:
        return chr(27) + "[37;4" + min(max(0, idx), 7) + "m"

    @staticmethod
    fn reset() -> String:
        return chr(27) + "[0m"


@value
struct BatchHeatMap(ServerCallbacks):
    """Prints a colored ASCII heat map of completed batch sizes."""

    fn on_server_start(inout self):
        pass

    fn on_server_stop(inout self):
        pass

    fn on_batch_receive(inout self, batch: Batch):
        pass

    fn on_batch_complete(inout self, start_ns: Int, batch: Batch):
        var size = len(batch)
        # TODO: Grab out from batch  and fix.
        var capacity = 8
        var idx = size - 1
        if capacity > 8:
            # Rescale
            idx = size * 8 // capacity
        print(
            ColoredTextCodes.code(idx)
            + "["
            + str(size)
            + "]"
            + ColoredTextCodes.reset(),
            end="",
        )

    fn on_request_receive(inout self, request: ModelInferRequest):
        pass

    fn on_request_ok(inout self, start_ns: Int, request: ModelInferRequest):
        pass

    fn on_request_fail(inout self, request: ModelInferRequest):
        pass
