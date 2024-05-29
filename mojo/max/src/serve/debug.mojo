# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides debugging tools and visualizations."""

from .callbacks import ServerCallbacks
from ._serve_rt import (
    InferenceRequestImpl,
    InferenceResponseImpl,
    InferenceBatch,
)


struct ColoredTextCodes:
    """Utilities for printing colored ASCII text."""

    @staticmethod
    fn code(idx: Int) -> String:
        """Returns a colored ASCII string based on the index.

        This is used to display the different batch sizes processed by the
        server in different colors. Each batch size is used as the `idx` value
        to generate the color.

        Args:
            idx: The index of the color to be used.

        Returns:
            The ASCII color coded string.
        """
        return chr(27) + "[37;4" + str(min(max(0, idx), 7)) + "m"

    @staticmethod
    fn reset() -> String:
        """Resets the ASCII color code.

        This resets the output to stop using the color codes.

        Returns:
            The ASCII string that terminates the color coded region.
        """
        return chr(27) + "[0m"


@value
struct BatchHeatMap(ServerCallbacks):
    """Prints a colored ASCII heat map of completed batch sizes."""

    fn on_server_start(inout self):
        pass

    fn on_server_stop(inout self):
        pass

    fn on_batch_receive(inout self, batch: InferenceBatch):
        pass

    fn on_batch_complete(inout self, start_ns: Int, batch: InferenceBatch):
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

    fn on_request_receive(inout self, request: InferenceRequestImpl):
        pass

    fn on_request_ok(
        inout self,
        start_ns: Int,
        request: InferenceRequestImpl,
        response: InferenceResponseImpl,
    ):
        pass

    fn on_request_fail(
        inout self, request: InferenceRequestImpl, error: String
    ):
        pass
