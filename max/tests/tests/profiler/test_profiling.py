# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from max.profiler import Trace, traced


def test_profiling() -> None:
    """Tests that profiling functions do not error."""
    with Trace("foo"):
        pass

    @traced(message="baz", color="red")
    def foo() -> None:
        # The span is named "baz".
        pass

    @traced
    def bar() -> None:
        # The span is named "bar".
        pass

    foo()
    bar()

    Trace("I'm here").mark()
