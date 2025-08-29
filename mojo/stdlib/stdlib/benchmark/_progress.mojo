# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Implements benchmark progress bar."""

from os import getenv

from builtin.range import _StridedRange


fn _get_terminal_size(fallback: Tuple[Int, Int] = (80, 24)) -> Tuple[Int, Int]:
    """Gets the size of the terminal.

    Args:
      fallback: The size of the terminal if it cannot be queried.

    Returns:
      The width and height of the terminal.
    """
    try:
        var columns = Int(getenv("COLUMNS", String(fallback[0])))
        var rows = Int(getenv("LINES", String(fallback[1])))

        return (columns, rows)
    except:
        return (80, 24)


fn _hide_cursor():
    print("\x1b[?25l", end="")


fn _show_cursor():
    print("\x1b[?25h", end="")


fn _clear_line():
    print("\033[A\33[2K")


fn _del():
    print("\r\33[2K", end="")


fn _end():
    print("\033[0m\r", end="")


struct Progress:
    """
    Implements a basic progress bar with the following usage.

    ```mojo
    import time


    fn main():
        with Progress(10) as p:
            for i in range(10):
                p.advance()
                time.sleep(0.1)

    ```
    """

    var _range: _StridedRange
    var _percentage: Float64
    var _term_dims: Tuple[Int, Int]

    @always_inline("nodebug")
    fn __init__(out self, end: Int):
        self = Self(0, end)

    @always_inline("nodebug")
    fn __init__(out self, start: Int, end: Int, step: Int = 1):
        self._range = _StridedRange(start, end, step)
        self._percentage = Float64(1) / len(self._range)
        self._term_dims = _get_terminal_size()
        print("")

    fn __copyinit__(out self, other: Self):
        self._range = other._range
        self._percentage = other._percentage
        self._term_dims = (other._term_dims[0], other._term_dims[1])

    fn advance(mut self, steps: Int = 1):
        alias BLOCK = "▇"
        alias PLACE_HOLDER = " "

        if len(self._range) <= 0 or steps <= 0:
            return

        var i = self._range.start
        for _ in range(steps):
            i = self._range.__next__()

        var width = self._term_dims[0]
        var blocks_to_print = Int(i * width * self._percentage) + 1
        var placeholders_to_print = max(width - blocks_to_print, 0)

        _del()
        print(BLOCK * blocks_to_print, end="")
        print(PLACE_HOLDER * placeholders_to_print, end="")
        _end()

    fn __enter__(self) -> Self:
        return self

    fn __exit__(self):
        print("")
        _hide_cursor()
        _show_cursor()

    fn __exit__(self, err: Error) -> Bool:
        self.__exit__()
        return Bool(err)
