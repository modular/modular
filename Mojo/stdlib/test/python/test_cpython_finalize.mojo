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

from std.pathlib import Path
from std.python import Python
from std.subprocess import run
from std.sys import stderr
from std.sys.arg import argv
from std.tempfile import NamedTemporaryFile

from std.testing import assert_equal, assert_true, TestSuite


def test_python_stdout_pipe() raises:
    var exe = argv()[0]
    var output = run(exe + " child_stdout")
    assert_true("hello from Python" in output)
    assert_true("hello from Mojo" in output)


def test_python_stdout_file() raises:
    var exe = argv()[0]
    with NamedTemporaryFile() as temp_file:
        var path = temp_file.name
        _ = run(exe + " child_stdout > " + path)
        with open(path, "r") as f:
            var content = f.read()
            assert_true("hello from Python" in content)
            assert_true("hello from Mojo" in content)


def test_python_atexit_handler() raises:
    var exe = argv()[0]
    with NamedTemporaryFile() as temp_file:
        var path = temp_file.name
        _ = run(exe + " child_atexit " + path)
        with open(path, "r") as f:
            var content = f.read()
            assert_equal(content, "atexit_ran")


def test_python_stderr_flushed() raises:
    var exe = argv()[0]
    var output = run(exe + " child_stderr 2>&1")
    assert_true("stderr from Python" in output)
    assert_true("stderr from Mojo" in output)


def test_python_never_initialized() raises:
    var exe = argv()[0]
    var output = run(exe + " child_no_python")
    assert_equal(output, "no python used")


def test_python_no_output() raises:
    var exe = argv()[0]
    var output = run(exe + " child_no_output")
    assert_equal(output, "mojo finished")


def test_python_multiple_evaluates() raises:
    var exe = argv()[0]
    var output = run(exe + " child_multiple_evaluates")
    assert_equal(output, "first\nsecond\nthird")


def test_python_raising_main() raises:
    var exe = argv()[0]
    with NamedTemporaryFile() as temp_file:
        var path = temp_file.name
        _ = run(exe + " child_raising " + path + " 2>/dev/null || true")
        with open(path, "r") as f:
            var content = f.read()
            assert_equal(content, "atexit_on_raise")


def run_child(mode: String) raises:
    if mode == "child_stdout":
        _ = Python.evaluate("print('hello from Python')")
        print("hello from Mojo")
    elif mode == "child_atexit":
        var marker_file = argv()[2]
        var code = (
            "import atexit\natexit.register(lambda: open('"
            + marker_file
            + "', 'w').write('atexit_ran'))\n"
        )
        _ = Python.evaluate(code, file=True)
    elif mode == "child_stderr":
        var py_sys = Python.import_module("sys")
        _ = py_sys.stderr.write("stderr from Python\n")
        print("stderr from Mojo", file=stderr)
    elif mode == "child_no_python":
        print("no python used")
    elif mode == "child_no_output":
        _ = Python.evaluate("1 + 1")
        print("mojo finished")
    elif mode == "child_multiple_evaluates":
        _ = Python.evaluate("print('first')")
        _ = Python.evaluate("print('second')")
        _ = Python.evaluate("print('third')")
    elif mode == "child_raising":
        var marker_file = argv()[2]
        var code = (
            "import atexit\natexit.register(lambda: open('"
            + marker_file
            + "', 'w').write('atexit_on_raise'))\n"
        )
        _ = Python.evaluate(code, file=True)
        raise Error("intentional error for testing raising main")


def main() raises:
    var args = argv()
    if len(args) > 1:
        run_child(args[1])
        return

    TestSuite.discover_tests[__functions_in_module()]().run()
