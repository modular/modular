# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Error helpers."""

from builtin._location import __call_location
from memory import stack_allocation
from sys.ffi import external_call


@always_inline
def error[T: Stringable](graph: Optional[Graph], message: T) -> Error:
    """Creates an error to raise that includes call information.

    This should be called internally at every point that can raise inside
    Graph API. Currently this only includes the specific call site of the raise.
    We hope to improve this in the future.

    Parameters:
        T: The message type.

    Args:
        graph: The graph for context information.
        message: An error message to raise.

    Returns:
        The error message augmented with call context information.
    """
    var message_string: String
    if graph:
        message_string = graph.value()[].current_layer() + " - " + message
    else:
        message_string = str(message)
    return (
        message_string
        + "\n\tat"
        + str(__call_location())
        + "\n\n"
        + format_system_stack()
    )


def format_system_stack[MAX_STACK_SIZE: Int = 128]() -> String:
    """Formats a stack trace using the system's `backtrace` call.

    Parameters:
        MAX_STACK_SIZE: The maximum number of function calls to report in
            the stack trace.

    Returns:
        The system stack trace as a formatted string, with one indented
        call per line.
    """
    call_stack = stack_allocation[MAX_STACK_SIZE, Pointer[NoneType]]()
    frames = external_call["backtrace", Int](call_stack, MAX_STACK_SIZE)
    frame_strs = external_call[
        "backtrace_symbols", Pointer[DTypePointer[DType.int8]]
    ](call_stack, frames)
    formatted = str("System stack:\n")
    for i in range(frames):
        formatted += "\t" + str(StringRef(frame_strs[i])) + "\n"
    return formatted
