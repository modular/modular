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
from utils._ansi import Color, Text

from builtin._location import __call_location, _SourceLocation
from compile.reflection import get_linkage_name
from sys.intrinsics import _type_is_eq
from utils import Variant


fn _get_test_func_name[
    func_type: AnyType, //,
    func: func_type,
]() -> String:
    """Get the name of a function."""

    var name = get_linkage_name[func]()
    return name.split("::")[-1].split("(", maxsplit=1)[0]


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


# TODO: (MOCO-2450) - Add defaulted `writeln` to `Writer` trait.
fn _writeln[
    *Ts: Writable
](mut writer: Some[Writer], *args: *Ts, sep: StaticString = StaticString("")):
    @parameter
    for i in range(args.__len__()):
        args[i].write_to(writer)
        sep.write_to(writer)
    writer.write("\n")


@fieldwise_init
struct TestResult(EqualityComparable, ImplicitlyCopyable, Movable, Writable):
    """A test result code."""

    var _value: Int

    alias PASS = Self(0)
    """The test passed."""

    alias FAIL = Self(1)
    """The test failed."""

    alias SKIP = Self(2)
    """The test was skipped."""

    fn __eq__(self, rhs: Self) -> Bool:
        return self._value == rhs._value

    fn write_to(self, mut writer: Some[Writer]):
        """Write the result code to the writer."""
        if self == Self.PASS:
            writer.write(Text[Color.GREEN]("PASS"))
        elif self == Self.FAIL:
            writer.write(Text[Color.RED]("FAIL"))
        elif self == Self.SKIP:
            writer.write(Text[Color.YELLOW]("SKIP"))


struct TestReport(Copyable, Movable, Writable):
    """A report for a single unit test."""

    alias _ErrorIndent = 3

    var name: String
    """The name of the test."""

    var duration_ns: UInt
    """The duration of the test in nanoseconds."""

    var result: TestResult
    """The result code of the test."""

    var error: Error
    """The error associated with a failing test."""

    @staticmethod
    fn passed(*, var name: String, duration_ns: UInt) -> Self:
        """Create a passing test report.

        Args:
            name: The name of the test.
            duration_ns: The duration of the test in nanoseconds.

        Returns:
            A new passing test report.
        """
        return {
            name = name^,
            duration_ns = duration_ns,
            result = TestResult.PASS,
        }

    @staticmethod
    fn failed(*, var name: String, duration_ns: UInt, var error: Error) -> Self:
        """Create a failing test report.

        Args:
            name: The name of the test.
            duration_ns: The duration of the test in nanoseconds.
            error: The error raised by the failing test.

        Returns:
            A new failing test report.
        """
        return {
            name = name^,
            duration_ns = duration_ns,
            result = TestResult.FAIL,
            error = error^,
        }

    @staticmethod
    fn skipped(*, var name: String) -> Self:
        """Create a skipped test report.

        Args:
            name: The name of the test.

        Returns:
            A new skipped test report.
        """
        return {name = name^, duration_ns = 0, result = TestResult.SKIP}

    @doc_private
    fn __init__(
        out self,
        *,
        var name: String,
        duration_ns: UInt,
        result: TestResult,
        var error: Error = {},
    ):
        self.name = name^
        self.duration_ns = duration_ns
        self.result = result
        self.error = error^

    @doc_private
    @staticmethod
    fn _format_error(e: Error) -> String:
        var replacement = String("\n", _Indent("", level=Self._ErrorIndent))
        return e.__str__().replace("\n", replacement)

    fn write_to(self, mut writer: Some[Writer]):
        """Write the formatted test report to the writer."""
        writer.write(_Indent(self.result, level=2))

        writer.write(" [ ", _format_nsec(self.duration_ns), " ] ")
        writer.write(Text[Color.CYAN](self.name))

        if self.result == TestResult.FAIL:
            writer.write(
                "\n",
                _Indent(
                    Self._format_error(self.error),
                    level=Self._ErrorIndent,
                ),
            )


struct TestSuiteReport(Copyable, Movable, Writable):
    """A report for an entire test suite."""

    var reports: List[TestReport]
    """The reports for each test in the suite."""

    var total_duration_ns: UInt
    """The total duration of the suite in nanoseconds."""

    var failures: Int
    """The number of tests that failed."""

    var skipped: Int
    """The number of tests skipped."""

    var location: _SourceLocation
    """The source location of the test suite."""

    fn __init__(
        out self, *, var reports: List[TestReport], location: _SourceLocation
    ):
        self.reports = reports^
        self.total_duration_ns = 0
        self.failures = 0
        self.skipped = 0
        self.location = location

        for ref report in self.reports:
            self.total_duration_ns += report.duration_ns
            if report.result == TestResult.FAIL:
                self.failures += 1
            elif report.result == TestResult.SKIP:
                self.skipped += 1

    fn write_to(self, mut writer: Some[Writer]):
        _writeln(writer)
        _writeln(
            writer,
            Text[Color.GREEN]("Running"),
            Text[Color.BOLD_WHITE](len(self.reports)),
            "tests for",
            Text[Color.CYAN](self.location.file_name),
            sep=" ",
        )
        for ref report in self.reports:
            _writeln(writer, report)
        _writeln(writer, "--------")
        _writeln(
            writer,
            Text[Color.MAGENTA]("Summary"),
            "[",
            _format_nsec(self.total_duration_ns),
            "]",
            Text[Color.BOLD_WHITE](len(self.reports)),
            "tests run:",
            Text[Color.BOLD_WHITE](
                len(self.reports) - self.failures - self.skipped
            ),
            Text[Color.GREEN]("passed"),
            ",",
            Text[Color.BOLD_WHITE](self.failures),
            Text[Color.RED]("failed"),
            ",",
            Text[Color.BOLD_WHITE](self.skipped),
            Text[Color.YELLOW]("skipped"),
            sep=" ",
        )

        if self.failures > 0:
            _writeln(
                writer,
                "Test suite'",
                Text[Color.CYAN](self.location.file_name),
                "'failed!",
                sep=" ",
            )


@fieldwise_init
struct _Test(Copyable & Movable):
    """A single test to run."""

    alias fn_type = fn () raises
    var test_fn: Self.fn_type
    var name: String


@explicit_destroy("TestSuite must be destroyed via `run()` or `disable()`")
struct TestSuite(Movable):
    """A suite of tests to run.

    You can automatically collect and register test functions starting with
    `test_` by calling the `discover_tests` static method, and then running the
    entire suite by calling the `run` method.

    Example:

    ```mojo
    from testing import assert_equal, TestSuite

    def test_something():
        assert_equal(1 + 1, 2)

    def test_some_other_thing():
        assert_equal(2 + 2, 4)

    def main():
        TestSuite.discover_tests[__functions_in_module()]().run()
    ```

    Alternatively, you can manually register tests by calling the `test` method.

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

    var tests: List[_Test]
    var location: _SourceLocation

    @always_inline
    fn __init__(out self, *, location: Optional[_SourceLocation] = None):
        """Create a new test suite.

        Arguments:
            location: The location of the test suite (defaults to
                `__call_location`).
        """
        self.tests = List[_Test]()
        self.location = location.or_else(__call_location())

    fn _register_tests[test_funcs: Tuple, /](mut self):
        """Internal function to prevent all registrations from being inlined."""

        @parameter
        for idx in range(len(test_funcs)):
            alias test_func = test_funcs[idx]

            @parameter
            if _type_is_eq[type_of(test_func), _Test.fn_type]():

                @parameter
                if _get_test_func_name[test_func]().startswith("test_"):
                    self.test[rebind[_Test.fn_type](test_func)]()

            # TODO: raise or notify the user if `test_*` function has
            # nonconforming signature. This will need some other reflection,
            # since `_get_test_func_name` currently cannot work on parametric
            # functions.

    @always_inline
    @staticmethod
    fn discover_tests[
        test_funcs: Tuple, /
    ](*, location: Optional[_SourceLocation] = None) -> Self:
        """Discover tests from the given list of functions, and register them.

        Parameters:
            test_funcs: The pack of functions to discover tests from. In most
                cases, callers should pass `__functions_in_module()`.

        Arguments:
            location: The location of the test suite (defaults to
                `__call_location`).
        """

        var suite = Self(location=location.or_else(__call_location()))
        suite._register_tests[test_funcs]()
        return suite^

    fn __del__(deinit self):
        pass

    fn test[f: _Test.fn_type](mut self):
        """Registers a test to be run.

        Parameters:
            f: The function to run.
        """
        self.tests.append(_Test(f, _get_test_func_name[f]()))

    fn generate_report(mut self) -> TestSuiteReport:
        """Runs the test suite and generates a report."""
        var reports = List[TestReport](capacity=len(self.tests))

        for test in self.tests:
            var error: Optional[Error] = None
            var start = perf_counter_ns()
            try:
                test.test_fn()
            except e:
                error = {e^}
            var duration = perf_counter_ns() - start

            var name = test.name.copy()
            if error:
                reports.append(
                    TestReport.failed(
                        name=name^, duration_ns=duration, error=error.take()
                    )
                )
            else:
                reports.append(
                    TestReport.passed(name=name^, duration_ns=duration)
                )

            # TODO: Check for skipped tests `append(TestReport.skipped(...))`

        return TestSuiteReport(reports=reports^, location=self.location)

    fn run(deinit self) raises:
        """Runs the test suite and prints the results to the console.

        Raises:
            An error if a test in the test suite fails.
        """
        var report = self.generate_report()
        if report.failures > 0:
            raise Error(report)

    fn disable(deinit self):
        """Disables the test suite, not running any of the tests."""
        pass
