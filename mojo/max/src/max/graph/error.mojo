# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Error helpers."""

from builtin._location import __call_location, _SourceLocation
from builtin.breakpoint import breakpoint
from collections import Optional
from memory import stack_allocation
from sys.ffi import external_call, C_char
from sys import param_env
from utils import StringRef


@always_inline
fn error[
    T: Stringable
](
    graph: Optional[Graph],
    message: T,
    location: Optional[_SourceLocation] = None,
) -> Error:
    """Creates an error to raise that includes call information.

    This should be called internally at every point that can raise inside
    Graph API. By default, this only includes the specific call site of the raise.
    We hope to improve this in the future.

    Parameters:
        T: The message type.

    Args:
        graph: The graph for context information.
        message: An error message to raise.
        location: An optional location for a more specific error message.

    Returns:
        The error message augmented with call context information.
    """
    return _error_impl(graph, message, location, __call_location())


fn _error_impl[
    T: Stringable
](
    graph: Optional[Graph],
    message: T,
    location: Optional[_SourceLocation],
    call_loc: _SourceLocation,
) -> Error:
    @parameter
    if param_env.is_defined["MODULAR_DEBUG_GRAPH"]():
        breakpoint()

    return format_error(graph, message, location or call_loc)


@always_inline
fn format_error[
    T: Stringable
](
    graph: Optional[Graph],
    message: T,
    location: Optional[_SourceLocation] = None,
) -> String:
    """Formats an error string that includes call information.

    Parameters:
        T: The message type.

    Args:
        graph: The graph for context information.
        message: An error message to raise.
        location: An optional location for a more specific error message.

    Returns:
        The string for an error message augmented with call context information.
    """
    return _format_error_impl(graph, message, location, __call_location())


fn _format_error_impl[
    T: Stringable
](
    graph: Optional[Graph],
    message: T,
    location: Optional[_SourceLocation],
    call_loc: _SourceLocation,
) -> String:
    var layer_string = str("")
    if graph:
        layer_string = str(graph.value().current_layer())

    var message_string: String
    if layer_string:
        message_string = layer_string + " - " + str(message)
    else:
        message_string = str(message)

    return (
        "\n\n"
        + message_string
        + "\n\tat "
        + str((location or call_loc).value())
        + "\n\n"
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
    call_stack = stack_allocation[MAX_STACK_SIZE, UnsafePointer[NoneType]]()
    frames = external_call["backtrace", Int](call_stack, MAX_STACK_SIZE)
    frame_strs = external_call[
        "backtrace_symbols", UnsafePointer[UnsafePointer[C_char]]
    ](call_stack, frames)
    formatted = str("System stack:\n")
    for i in range(frames):
        formatted += "\t" + str(StringRef(frame_strs[i])) + "\n"
    return formatted
