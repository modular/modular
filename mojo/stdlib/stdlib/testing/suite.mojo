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


@fieldwise_init
struct _Test(Copyable & Movable):
    """A single test to run."""

    alias fn_type = fn () raises
    var test_fn: Self.fn_type
    var name: String


@explicit_destroy("TestSuite must be ran via `TestSuite.run`")
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

    alias _ErrorIndent = 3

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

    def _register_tests[test_funcs: Tuple, /](mut self):
        """Internal function to prevent all registrations from being inlined."""

        @parameter
        for idx in range(len(test_funcs)):
            alias test_func = test_funcs[idx]

            @parameter
            if _type_is_eq[__type_of(test_func), _Test.fn_type]():

                @parameter
                if _get_test_func_name[test_func]().startswith("test_"):
                    self.test[rebind[_Test.fn_type](test_func)]()

            # TODO: raise or notify the user if `test_*` function has
            # nonconforming signature. This will need some other reflection,
            # since `_get_test_func_name` currently cannot work on parametric
            # functions.

    @always_inline
    @staticmethod
    def discover_tests[
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
            Text[Color.GREEN]("Running"),
            Text[Color.BOLD_WHITE](n_tests),
            "tests for",
            Text[Color.CYAN](self.location.file_name),
        )

        var passed = Text[Color.GREEN]("PASS")
        var failed = Text[Color.RED]("FAIL")

        for test in self.tests:
            var name = Text[Color.CYAN](test.name)
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
            Text[Color.MAGENTA]("Summary"),
            " [ ",
            _format_nsec(UInt(runtime)),
            " ] ",
            Text[Color.BOLD_WHITE](n_tests),
            " tests run: ",
            Text[Color.BOLD_WHITE](n_tests - failures),
            Text[Color.GREEN](" passed"),
            ", ",
            Text[Color.BOLD_WHITE](failures),
            Text[Color.RED](" failed"),
            sep="",
        )

        if failures > 0:
            raise Error(
                "Test suite '",
                Text[Color.CYAN](self.location.file_name),
                "' failed!",
            )
