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

from builtin._location import __call_location
from compile.reflection import get_linkage_name
from math import ceil, floor
from os import sep
from time import perf_counter_ns


fn _get_test_func_name[
    func_type: AnyTrivialRegType, //,
    func: func_type,
]() -> String:
    """Get the name of a function."""

    var name = get_linkage_name[func]()
    return name.split("::")[-1].split("(", maxsplit=1)[0]


@fieldwise_init
struct Color(ImplicitlyCopyable, Movable, Writable):
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
struct ColorText[W: Writable, origin: ImmutableOrigin](Writable):
    """Colors the given writable with the given `Color`."""

    var writable: Pointer[W, origin]
    var color: Color

    fn __init__(out self, ref [origin]w: W, color: Color):
        self.writable = Pointer(to=w)
        self.color = color

    fn write_to(self, mut writer: Some[Writer]):
        writer.write(self.color)
        self.writable[].write_to(writer)
        writer.write(Color.END)


struct Indent[W: Writable, origin: ImmutableOrigin](Writable):
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
    """Formats the given number of nanoseconds.

    The returned string is in the format of "NNN.NNN"
    """
    var ms_total = nanoseconds // 1_000_000
    var ns_remainder = nanoseconds % 1_000_000

    var fractional_ms = (ns_remainder * 1000) // 1_000_000

    if ms_total == 0 and fractional_ms == 0:
        return String("0.001")

    var result = String(ms_total, ".")

    if fractional_ms < 10:
        result += "00"
    elif fractional_ms < 100:
        result += "0"

    result += String(fractional_ms)
    return result


@fieldwise_init
struct Test(Copyable & Movable):
    """A single test to run."""

    var test_fn: fn () raises
    var name: String


@explicit_destroy("TestSuite must be ran via `TestSuite.run`")
struct TestSuite:
    """A suite of tests to run.

    You can enqueue tests by calling the `test` method, and then running the
    entire suite by calling the `run` method.
    """

    alias _ErrorIndent = 3

    var tests: List[Test]
    var name: StaticString

    @always_inline
    fn __init__(out self):
        """Create a new test suite."""
        self.tests = List[Test]()

        var file_name = __call_location().file_name
        self.name = file_name.split(sep)[-1]

    fn __del__(deinit self):
        pass

    fn test[f: fn () raises](mut self):
        """Enqueues a test to run.

        Parameters:
            f: The function to run.
        """
        self.tests.append(Test(f, _get_test_func_name[f]()))

    @staticmethod
    fn _format_error(e: Error) -> String:
        var replacement = String("\n", Indent("", level=Self._ErrorIndent))
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
            ColorText("Running", Color.GREEN),
            ColorText(n_tests, Color.BOLD_WHITE),
            "tests for",
            ColorText(self.name, Color.CYAN),
        )

        var passed = ColorText("PASS", Color.GREEN)
        var failed = ColorText("FAIL", Color.RED)

        for test in self.tests:
            var name = ColorText(test.name, Color.CYAN)
            var start = perf_counter_ns()
            try:
                test.test_fn()
                var duration = perf_counter_ns() - start
                runtime += duration
                print(
                    Indent(passed, level=2),
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
                    Indent(failed, level=2),
                    "[",
                    _format_nsec(duration),
                    "]",
                    name,
                )
                print(Indent(Self._format_error(e), level=Self._ErrorIndent))

        print("--------")
        print(
            " ",
            ColorText("Summary", Color.MAGENTA),
            " [ ",
            _format_nsec(runtime),
            " ] ",
            ColorText(n_tests, Color.BOLD_WHITE),
            " tests run: ",
            ColorText(n_tests - failures, Color.BOLD_WHITE),
            ColorText(" passed", Color.GREEN),
            ", ",
            ColorText(failures, Color.BOLD_WHITE),
            ColorText(" failed", Color.RED),
            sep="",
        )

        if failures > 0:
            raise Error(
                "Test suite '", ColorText(self.name, Color.CYAN), "' failed!"
            )
