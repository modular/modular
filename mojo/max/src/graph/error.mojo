# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Error helpers."""

from builtin._location import __call_location


@always_inline
fn error[T: Stringable](message: T) -> Error:
    """Creates an error to raise that includes call information.

    This should be called internally at every point that can raise inside
    Graph API. Currently this only includes the specific call site of the raise.
    We hope to improve this in the future.

    Parameters:
        T: The message type.

    Args:
        message: An error message to raise.

    Returns:
        The error message augmented with call context information.
    """
    return str(message) + "\n\tat " + str(__call_location())
