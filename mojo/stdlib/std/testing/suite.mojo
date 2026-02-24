# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Implements test suite infrastructure for organizing and running tests.

This module provides the `TestSuite` type for collecting, filtering, and
executing unit tests with automatic discovery, command-line filtering, and
formatted reporting. It includes support for colored output, timing statistics,
and flexible test selection via CLI arguments.
"""

from math import ceil, floor
from os import sep
from time import perf_counter_ns
from utils._ansi import Color, Text
from collections import Set

from reflection import get_function_name, call_location, SourceLocation
from sys.intrinsics import _type_is_eq
from sys import argv


struct _Indent[W: Writable, origin: ImmutOrigin](Writable):
    """Indents the given writable by the given level."""

    comptime IndentStr = "  "

    var writable: Pointer[Self.W, Self.origin]
    var level: Int

    fn __init__(out self, ref[Self.origin] w: Self.W, *, level: Int):
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
    comptime for i in range(args.__len__()):
        args[i].write_to(writer)
        sep.write_to(writer)
    writer.write("\n")


@fieldwise_init
struct TestResult(Equatable, ImplicitlyCopyable, Writable):
    """A test result code."""

    var _value: Int

    comptime PASS = Self(0)
    """The test passed."""

    comptime FAIL = Self(1)
    """The test failed."""

    comptime SKIP = Self(2)
    """The test was skipped."""

    fn __eq__(self, rhs: Self) -> Bool:
        """Compare two test result codes for equality.

        Args:
            rhs: The other test result code to compare against.

        Returns:
            True if the result codes are equal, False otherwise.
        """
        return self._value == rhs._value

    fn write_to(self, mut writer: Some[Writer]):
        """Write the result code to the writer.

        Args:
            writer: The writer to output the result code to.
        """
        if self == Self.PASS:
            writer.write(Text[Color.GREEN]("PASS"))
        elif self == Self.FAIL:
            writer.write(Text[Color.RED]("FAIL"))
        elif self == Self.SKIP:
            writer.write(Text[Color.YELLOW]("SKIP"))


struct TestReport(Copyable, Writable):
    """A report for a single unit test."""

    comptime _ErrorIndent = 3

    var name: String
    """The name of the test."""

    var duration_ns: UInt
    """The duration of the test in nanoseconds."""

    var result: TestResult
    """The result code of the test."""

    var error: Optional[Error]
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
        var error: Optional[Error] = {},
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
        """Write the formatted test report to the writer.

        Args:
            writer: The writer to output the formatted report to.
        """
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


struct TestSuiteReport(Copyable, Writable):
    """A report for an entire test suite."""

    var reports: List[TestReport]
    """The reports for each test in the suite."""

    var total_duration_ns: UInt
    """The total duration of the suite in nanoseconds."""

    var failures: Int
    """The number of tests that failed."""

    var skipped: Int
    """The number of tests skipped."""

    var passed: Int
    """The number of tests that passed."""

    var location: SourceLocation
    """The source location of the test suite."""

    fn __init__(
        out self, *, var reports: List[TestReport], location: SourceLocation
    ):
        """Initialize a test suite report.

        Args:
            reports: The list of individual test reports.
            location: The source location where the test suite is defined.
        """
        self.reports = reports^
        self.total_duration_ns = 0
        self.failures = 0
        self.skipped = 0
        self.passed = 0
        self.location = location

        for ref report in self.reports:
            self.total_duration_ns += report.duration_ns
            if report.result == TestResult.FAIL:
                self.failures += 1
            elif report.result == TestResult.SKIP:
                self.skipped += 1
        self.passed = len(self.reports) - self.failures - self.skipped

    fn write_to(self, mut writer: Some[Writer]):
        """Write the formatted test suite report to the writer.

        Args:
            writer: The writer to output the formatted suite report to.
        """
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
struct _Test(Copyable):
    """A single test to run."""

    comptime fn_type = fn() raises
    var test_fn: Self.fn_type
    var name: StaticString


@explicit_destroy("TestSuite must be destroyed via `run()`")
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
    """The list of tests registered in this suite."""

    var location: SourceLocation
    """The source location where the test suite was created."""

    var skip_list: Set[String]
    """The list of tests to skip in this suite."""

    var allow_list: Optional[Set[String]]
    """The list of tests to allow in this suite."""

    var cli_args: List[StaticString]
    """The raw command line arguments passed to the test suite."""

    @always_inline
    fn __init__(
        out self,
        *,
        location: Optional[SourceLocation] = None,
        var cli_args: Optional[List[StaticString]] = None,
    ):
        """Create a new test suite.

        Args:
            location: The location of the test suite (defaults to
                `call_location`).
            cli_args: The command line arguments to pass to the test suite
                (defaults to `sys.argv()`).
        """
        self.tests = List[_Test]()
        self.location = location.or_else(call_location())
        self.skip_list = {}
        self.allow_list = None  # None means no allow list specified.
        self.cli_args = cli_args^.or_else(List[StaticString](argv()))

    fn _register_tests[test_funcs: Tuple, /](mut self) raises:
        """Internal function to prevent all registrations from being inlined."""

        comptime for idx in range(len(test_funcs)):
            comptime test_func = test_funcs[idx]

            comptime if get_function_name[test_func]().startswith("test_"):
                comptime if _type_is_eq[type_of(test_func), _Test.fn_type]():
                    self.test[rebind[_Test.fn_type](test_func)]()
                else:
                    raise Error(
                        "test function '",
                        get_function_name[test_func](),
                        "' has nonconforming signature",
                    )

    @always_inline
    @staticmethod
    fn discover_tests[
        test_funcs: Tuple, /
    ](
        *,
        location: Optional[SourceLocation] = None,
        var cli_args: Optional[List[StaticString]] = None,
    ) raises -> Self:
        """Discover tests from the given list of functions, and register them.

        Parameters:
            test_funcs: The pack of functions to discover tests from. In most
                cases, callers should pass `__functions_in_module()`.

        Args:
            location: The location of the test suite (defaults to
                `call_location`).
            cli_args: The command line arguments to pass to the test suite
                (defaults to `sys.argv()`).

        Raises:
            If test discovery fails (e.g. because of a nonconforming test
            function signature).

        Returns:
            A new TestSuite with all discovered tests registered.
        """

        var suite = Self(
            location=location.or_else(call_location()), cli_args=cli_args^
        )
        try:
            suite._register_tests[test_funcs]()
        except e:
            suite^.abandon()
            raise e^
        return suite^

    fn test[f: _Test.fn_type](mut self):
        """Registers a test to be run.

        Parameters:
            f: The function to run.
        """
        self.tests.append(_Test(f, get_function_name[f]()))

    fn skip[f: _Test.fn_type](mut self):
        """Registers a test to be skipped.

        If attempting to skip a test that is not registered in the suite (either
        explicitly or via automatic discovery), an error will be raised when the
        suite is run.

        Parameters:
            f: The function to skip.
        """
        comptime skipped_name = get_function_name[f]()
        self.skip_list.add(skipped_name)

    fn _parse_filter_lists(mut self) raises:
        # TODO: We need a proper argument parsing library to do this right.
        ref args = self.cli_args
        var num_args = len(args)
        if num_args <= 1:
            return

        if args[1] == "--only":
            self.allow_list = Set[String]()
        elif args[1] == "--skip-all":
            if num_args > 2:
                raise Error("'--skip-all' does not take any arguments")
            # --skip-all implies an empty allow list.
            self.allow_list = Set[String]()
            return
        elif args[1] != "--skip":
            raise Error(
                "invalid argument: ",
                args[1],
                " (expected '--only' or '--skip')",
            )

        if num_args == 2:
            raise Error("expected test name(s) after '--only' or '--skip'")

        var discovered_tests = Set[String]()
        for test in self.tests:
            discovered_tests.add(test.name)

        for idx in range(2, num_args):
            var arg = args[idx]
            if arg not in discovered_tests:
                raise Error(
                    "explicitly ",
                    "allowed" if self.allow_list else "skipped",
                    " test not found in suite: ",
                    arg,
                )
            if self.allow_list:
                self.allow_list[].add(arg)
            else:
                self.skip_list.add(arg)

    fn _should_skip(self, test: _Test) -> Bool:
        if test.name in self.skip_list:
            return True
        if not self.allow_list:
            return False
        # SAFETY: We know that `self.allow_list` is not `None` here.
        return test.name not in self.allow_list.unsafe_value()

    fn _validate_skip_list(self) raises:
        # TODO: _Test doesn't conform to Equatable, so we can't use
        # `in` here. Also, we might wanna do this in O(1) time.
        for test_name in self.skip_list:
            var found = False
            for test in self.tests:
                if test.name == test_name:
                    found = True
                    break
            if not found:
                raise Error(
                    (
                        "trying to skip a test that is not registered in the"
                        " suite: "
                    ),
                    test_name,
                )

    fn generate_report(
        mut self, skip_all: Bool = False
    ) raises -> TestSuiteReport:
        """Runs the test suite and generates a report.

        Args:
            skip_all: Only collect tests, but don't execute them (defaults to
                `False`).

        Raises:
            If an error occurs during test collection.

        Returns:
            A report containing the results of all tests.
        """
        self._validate_skip_list()

        # We call `_parse_filter_lists` even if `skip_all` is true to make sure
        # CLI arguments are parsed and checked. We should probably refactor this
        # when we have a proper argument parsing library.
        self._parse_filter_lists()
        if skip_all:
            self.allow_list = Set[String]()

        var reports = List[TestReport](capacity=len(self.tests))
        for test in self.tests:
            if self._should_skip(test):
                reports.append(TestReport.skipped(name=test.name))
                continue

            var error: Optional[Error] = None
            var start = perf_counter_ns()
            try:
                test.test_fn()
            except e:
                error = {e^}
            var duration = perf_counter_ns() - start

            if error:
                reports.append(
                    TestReport.failed(
                        name=test.name, duration_ns=duration, error=error.take()
                    )
                )
            else:
                reports.append(
                    TestReport.passed(name=test.name, duration_ns=duration)
                )

        return TestSuiteReport(reports=reports^, location=self.location)

    fn run(deinit self, *, quiet: Bool = False, skip_all: Bool = False) raises:
        """Runs the test suite and prints the results to the console.

        Args:
            quiet: Suppresses printing the report when the suite does not fail
                (defaults to `False`).
            skip_all: Only collect tests, but don't execute them (defaults to
                `False`).

        Raises:
            If a test in the test suite fails or if an error occurs during test
            collection.
        """
        var report = self.generate_report(skip_all=skip_all)

        if report.failures > 0:
            raise Error(report^)
        if not quiet:
            print(report)

    fn abandon(deinit self):
        """Destroy a test suite without running any tests."""
        pass
