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

from collections import List
from os.path import exists
from os import Process
from os.process import Pipe, ProcessStatus

from testing import (
    assert_false,
    assert_raises,
    assert_true,
    assert_equal,
)


def test_pipe():
    var p = Pipe()
    var s = InlineArray[UInt8, 5](fill=0)
    p.write_bytes("hello".as_bytes())
    assert_equal(p.read_bytes(Span(s)), 5)
    assert_true(Span(s) == "hello".as_bytes())
    p.set_output_only()
    with assert_raises():
        _ = p.read_bytes(Span(s))


def test_process_run():
    print("== test_process_run")
    # CHECK-LABEL: == test_process_run
    # CHECK-NEXT: == TEST_ECHO
    var command = "echo"
    _ = Process.run(command, ["== TEST_ECHO"])


def test_process_wait():
    print("== test_process_wait")
    var command = "echo"
    var p = Process.run(command, ["== TEST_WAIT"])
    assert_true(p.wait().has_exited())
    assert_equal(p.wait().exit_code.value(), 0)
    assert_false(p.kill())


def test_process_kill():
    print("== test_process_kill")
    var command = "sleep"
    var p = Process.run(command, ["60"])
    assert_true(p.kill())
    assert_true(p.wait().has_exited())


def test_process_run_missing():
    print("== test_process_run_missing")
    # CHECK-LABEL: == test_process_run_missing
    # CHECK-NEXT: Failed to execute ThIsFiLeCoUlDNoTPoSsIbLlYExIsT.NoTAnExTeNsIoN, EINT error code: 2
    missing_executable_file = "ThIsFiLeCoUlDNoTPoSsIbLlYExIsT.NoTAnExTeNsIoN"

    # verify that the test file does not exist before starting the test
    assert_false(
        exists(missing_executable_file),
        "Unexpected file '" + missing_executable_file + "' it should not exist",
    )

    try:
        _ = Process.run(missing_executable_file, List[String]())
    except e:
        print(e)

    with assert_raises():
        _ = Process.run(missing_executable_file, List[String]())


def test_processstatus_str():
    assert_equal(
        String(ProcessStatus(exit_code=0)), "ProcessStatus(exit_code: 0)"
    )
    assert_equal(
        String(ProcessStatus(exit_code=1)), "ProcessStatus(exit_code: 1)"
    )
    assert_equal(
        String(ProcessStatus(term_signal=15)), "ProcessStatus(term_signal: 15)"
    )
    assert_equal(String(ProcessStatus.running()), "ProcessStatus(running)")


def test_processstatus_repr():
    assert_equal(
        ProcessStatus(exit_code=0).__repr__(), "ProcessStatus(exit_code: 0)"
    )
    assert_equal(
        ProcessStatus(term_signal=15).__repr__(), "ProcessStatus(term_signal: 15)"
    )
    assert_equal(ProcessStatus.running().__repr__(), "ProcessStatus(running)")


def main():
    test_process_run()
    test_process_run_missing()
    test_process_wait()
    test_process_kill()
    test_pipe()
    test_processstatus_str()
    test_processstatus_repr()
