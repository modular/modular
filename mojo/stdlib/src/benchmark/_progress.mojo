# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements benchmark progress bar."""

from os import getenv
from builtin.range import _StridedRange

alias HIDE_CURSOR = "\x1b[?25l"
alias SHOW_CURSOR = "\x1b[?25h"
alias PHASES = (" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█")
alias WIDTH = 80


fn _get_terminal_size(fallback: Tuple[Int, Int] = (80, 24)) -> Tuple[Int, Int]:
    """Gets the size of the terminal.

    Args:
      fallback: The size of the terminal if it cannot be queried.

    Returns:
      The width and height of the terminal.
    """
    try:
        var columns = int(getenv("COLUMNS", str(fallback[0])))
        var rows = int(getenv("LINES", str(fallback[1])))

        return (columns, rows)
    except:
        return (80, 24)


struct Progress:
    var _range: _StridedRange
    var _term_dims: Tuple[Int, Int]

    @always_inline("nodebug")
    fn __init__(inout self, end: Int):
        self = Self(0, end)

    @always_inline("nodebug")
    fn __init__(inout self, start: Int, end: Int, step: Int = 1):
        self._range = _StridedRange(start, end, step)
        self._term_dims = _get_terminal_size()

    fn __copyinit__(inout self, other: Self):
        self._range = other._range
        self._term_dims = (other._term_dims[0], other._term_dims[1])

    fn advance(inout self):
        if len(self._range) <= 0:
            return

        var i = self._range.__next__()
        var adj = int(self._term_dims[0] * (i / self._range.end))
        print("\0337\0338", end="")
        print(String(".") * adj)
        print("\033[1A", end="")

    fn __enter__(self) -> Self:
        return self

    fn __exit__(self):
        print("")
        print(HIDE_CURSOR, end="")
        print(SHOW_CURSOR, end="")
