# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from runtime.tracing import Trace, TraceLevel


def test_trace_context_with_dynamic_name():
    # Exercise the Trace context manager to ensure no crashes and proper begin/end
    # against the Tracy bridge. We cannot assert on UI-visible names here, but
    # this covers the dynamic path used by __enter__/__exit__.
    with Trace[level = TraceLevel.OP](name="trace_context_dynamic"):
        pass

    # Also exercise start()/end() explicitly.
    tr = Trace[level = TraceLevel.OP](name="trace_start_end_dynamic")
    tr.start()
    tr.end()


def main():
    test_trace_context_with_dynamic_name()
