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

from math import ceil, floor
from os import sep
from time import perf_counter_ns

from builtin._location import __call_location
from compile.reflection import get_linkage_name


fn _get_test_func_name[
    func_type: AnyTrivialRegType, //,
    func: func_type,
]() -> String:
    """Get the name of a function."""

    var name = get_linkage_name[func]()
    return name.split("::")[-1].split("(", maxsplit=1)[0]


@fieldwise_init
struct _Color(ImplicitlyCopyable, Movable, Writable):
    """ANSI colors for terminal output."""

    var color: StaticString

    alias RED = Self("\033[91m")
    alias GREEN = Self("\033[92m")
    alias YELLOW = Self("\033[93m")
    alias BLUE = Self("\033[94m")
    alias MAGENTA = Self("\033[95m")
    alias CYAN = Self("\033[96m")
    alias BOLD_WHITE = Self("\033[1;97m")
    alias END = Self("\033[0m")

    fn write_to(self, mut writer: Some[Writer]):
        writer.write(self.color)


@fieldwise_init
struct _ColorText[W: Writable, origin: ImmutableOrigin](Writable):
    """Colors the given writable with the given `_Color`."""

    var writable: Pointer[W, origin]
    var color: _Color

    fn __init__(out self, ref [origin]w: W, color: _Color):
        self.writable = Pointer(to=w)
        self.color = color

    fn write_to(self, mut writer: Some[Writer]):
        writer.write(self.color)
        self.writable[].write_to(writer)
        writer.write(_Color.END)


struct _Indent[W: Writable, origin: ImmutableOrigin](Writable):
    """Indents the given writable by the given level."""

    alias IndentStr = "  "

    var writable: Pointer[W, origin]
    var level: Int

    fn __init__(out self, ref [origin]w: W, *, level: Int):
        self.writable = Pointer(to=w)
        self.level = level

    fn write_to(self, mut writer: Some[Writer]):
        for _ in range(self.level):
            writer.write(Self.IndentStr)
        writer.write(self.writable[])


fn _format_nsec(nanoseconds: UInt) -> String:
    """Formats the given number of nanoseconds as milliseconds.

    The returned string is in the format of "NNN.NNN"
    """
    var ms_total = nanoseconds // 1_000_000
    var ns_remainder = nanoseconds % 1_000_000

    var fractional_ms = (ns_remainder * 1000) // 1_000_000

    if ms_total == 0 and fractional_ms == 0:
        return String("0.001")

    var result = String(ms_total, ".")

    if fractional_ms < 10:
        result.write("00")
    elif fractional_ms < 100:
        result.write("0")

    result.write(fractional_ms)
    return result


@fieldwise_init
struct _Test(Copyable & Movable):
    """A single test to run."""

    var test_fn: fn () raises
    var name: String


@explicit_destroy("TestSuite must be ran via `TestSuite.run`")
struct TestSuite(Movable):
    """A suite of tests to run.

    You can enqueue tests by calling the `test` method, and then running the
    entire suite by calling the `run` method.

    Example:

    ```mojo
    from testing import assert_equal, TestSuite

    def some_test():
        assert_equal(1 + 1, 2)

    def main():
        var suite = TestSuite()

        suite.test[some_test]()

        suite^.run()
    ```
    """

    alias _ErrorIndent = 3

    var tests: List[_Test]
    var name: StaticString

    @always_inline
    fn __init__(out self):
        """Create a new test suite."""
        self.tests = List[_Test]()

        var file_name = __call_location().file_name
        self.name = file_name.split(sep)[-1]

    fn __del__(deinit self):
        pass

    fn test[f: fn () raises](mut self):
        """Enqueues a test to run.

        Parameters:
            f: The function to run.
        """
        self.tests.append(_Test(f, _get_test_func_name[f]()))

    @staticmethod
    fn _format_error(e: Error) -> String:
        var replacement = String("\n", _Indent("", level=Self._ErrorIndent))
        return e.__str__().replace("\n", replacement)

    def run(deinit self):
        """Runs the test suite and prints the results to the console.

        Raises:
            An error if a test in the test suite fails.
        """
        var n_tests = len(self.tests)
        var failures = 0
        var runtime = 0
        print(
            _ColorText("Running", _Color.GREEN),
            _ColorText(n_tests, _Color.BOLD_WHITE),
            "tests for",
            _ColorText(self.name, _Color.CYAN),
        )

        var passed = _ColorText("PASS", _Color.GREEN)
        var failed = _ColorText("FAIL", _Color.RED)

        for test in self.tests:
            var name = _ColorText(test.name, _Color.CYAN)
            var start = perf_counter_ns()
            try:
                test.test_fn()
                var duration = perf_counter_ns() - start
                runtime += duration
                print(
                    _Indent(passed, level=2),
                    "[",
                    _format_nsec(duration),
                    "]",
                    name,
                )
            except e:
                failures += 1
                var duration = perf_counter_ns() - start
                runtime += duration
                print(
                    _Indent(failed, level=2),
                    "[",
                    _format_nsec(duration),
                    "]",
                    name,
                )
                print(_Indent(Self._format_error(e), level=Self._ErrorIndent))

        print("--------")
        print(
            " ",
            _ColorText("Summary", _Color.MAGENTA),
            " [ ",
            _format_nsec(runtime),
            " ] ",
            _ColorText(n_tests, _Color.BOLD_WHITE),
            " tests run: ",
            _ColorText(n_tests - failures, _Color.BOLD_WHITE),
            _ColorText(" passed", _Color.GREEN),
            ", ",
            _ColorText(failures, _Color.BOLD_WHITE),
            _ColorText(" failed", _Color.RED),
            sep="",
        )

        if failures > 0:
            raise Error(
                "Test suite '", _ColorText(self.name, _Color.CYAN), "' failed!"
            )
