# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Error helpers."""

from collections import Optional, InlineArray
from collections.string import StaticString
from sys import param_env
from sys.ffi import c_char, external_call

from builtin._location import __call_location, _SourceLocation
from builtin.breakpoint import breakpoint
from memory import UnsafePointer

from utils.write import _WriteBufferStack, write_args


@always_inline
fn error[
    *Ts: Writable,
](
    graph: Optional[Graph],
    *messages: *Ts,
    location: Optional[_SourceLocation] = None,
) -> Error:
    """Creates an error to raise that includes call information.

    This should be called internally at every point that can raise inside
    Graph API. By default, this only includes the specific call site of the raise.
    We hope to improve this in the future.

    Parameters:
        Ts: The Writeable message types.

    Args:
        graph: The graph for context information.
        messages: An error message to raise.
        location: An optional location for a more specific error message.

    Returns:
        The error message augmented with call context information.
    """
    return _error_impl(graph, messages, location, __call_location())


fn _error_impl[
    *Ts: Writable
](
    graph: Optional[Graph],
    messages: VariadicPack[_, _, Writable, *Ts],
    location: Optional[_SourceLocation],
    call_loc: _SourceLocation,
) -> Error:
    @parameter
    if param_env.is_defined["MODULAR_DEBUG_GRAPH"]():
        breakpoint()

    return format_error(graph, messages, location or call_loc)


@always_inline
fn format_error[
    *Ts: Writable
](
    graph: Optional[Graph],
    *messages: *Ts,
    location: Optional[_SourceLocation] = None,
) -> String:
    """Formats an error string that includes call information.

    Parameters:
        Ts: The message types.

    Args:
        graph: The graph for context information.
        messages: Error messages to raise.
        location: An optional location for a more specific error message.

    Returns:
        The string for an error message augmented with call context information.
    """
    return _format_error_impl(graph, messages, location, __call_location())


@always_inline
fn format_error[
    *Ts: Writable
](
    graph: Optional[Graph],
    messages: VariadicPack[_, _, Writable, *Ts],
    location: Optional[_SourceLocation] = None,
) -> String:
    """Formats an error string that includes call information.

    Parameters:
        Ts: The message types.

    Args:
        graph: The graph for context information.
        messages: Error messages to raise.
        location: An optional location for a more specific error message.

    Returns:
        The string for an error message augmented with call context information.
    """
    return _format_error_impl(graph, messages, location, __call_location())


fn _format_error_impl[
    *Ts: Writable
](
    graph: Optional[Graph],
    messages: VariadicPack[_, _, Writable, *Ts],
    location: Optional[_SourceLocation],
    call_loc: _SourceLocation,
) -> String:
    var layer_string = String()
    var buffer = _WriteBufferStack(layer_string)
    buffer.write("\n\n")

    if graph:
        buffer.write(graph.value().current_layer(), " - ")

    write_args(buffer, messages)
    buffer.write("\n\nat ", (location or call_loc).value(), "\n\n")
    buffer.flush()

    return layer_string


def format_system_stack[MAX_STACK_SIZE: Int = 128]() -> String:
    """Formats a stack trace using the system's `backtrace` call.

    Parameters:
        MAX_STACK_SIZE: The maximum number of function calls to report in
            the stack trace.

    Returns:
        The system stack trace as a formatted string, with one indented
        call per line.
    """
    var call_stack = InlineArray[UnsafePointer[NoneType], MAX_STACK_SIZE](
        uninitialized=True
    )
    var num_frames = external_call["backtrace", Int32](
        call_stack.unsafe_ptr(), Int(len(call_stack)), MAX_STACK_SIZE
    )
    var frame_strs = external_call[
        "backtrace_symbols", UnsafePointer[UnsafePointer[c_char]]
    ](call_stack.unsafe_ptr(), num_frames)

    var formatted = String()
    var buffer = _WriteBufferStack(formatted)
    buffer.write("System stack:\n")
    for i in range(num_frames):
        formatted.write(
            "\t",
            StaticString(unsafe_from_utf8_ptr=frame_strs[i]),
            "\n",
        )

    buffer.flush()
    return formatted^
