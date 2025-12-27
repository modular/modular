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

from collections import List
from os import Process
from sys._libc import SignalCodes
from time import sleep

from testing import assert_equal, assert_false, assert_true


def test_exit_code_verification():
    var process = Process.run("sh", ["-c", "exit 42"])
    var status = process.wait()
    assert_true(status.exit_code, "Should have exit code")
    assert_equal(status.exit_code.value(), 42)
    assert_false(status.term_signal, "Should not have term signal")


def test_process_exits_immediately():
    var process = Process.run("true", List[String]())
    var status = process.wait()
    assert_true(status.exit_code, "Should have exit code")
    assert_equal(status.exit_code.value(), 0)
    assert_false(status.term_signal, "Should not have term signal")


def test_multiple_wait_calls():
    var process = Process.run("echo", ["hello"])
    var status1 = process.wait()
    var status2 = process.wait()
    assert_true(status1.exit_code, "Should have exit code")
    assert_equal(status1.exit_code.value(), 0)
    assert_true(status2.exit_code, "Should also have exit code")
    assert_equal(status2.exit_code.value(), 0)

    assert_false(status1.term_signal, "Should not have term signal")
    assert_false(status2.term_signal, "Should also not have term signal")


def test_signal_hangup():
    var process = Process.run("sleep", ["3"])
    sleep(0.5)
    _ = process.hangup()
    var status = process.wait()
    assert_false(status.exit_code, "Should not have exit code")
    assert_true(status.term_signal, "Should have term signal")
    assert_equal(status.term_signal.value(), SignalCodes.HUP)


def test_signal_interrupt():
    var process = Process.run("sleep", ["3"])
    sleep(0.5)
    _ = process.interrupt()
    var status = process.wait()
    assert_false(status.exit_code, "Should not have exit code")
    assert_true(status.term_signal, "Should have term signal")
    assert_equal(status.term_signal.value(), SignalCodes.INT)


def test_signal_kill():
    var process = Process.run("sleep", ["3"])
    sleep(0.5)
    _ = process.kill()
    var status = process.wait()
    assert_false(status.exit_code, "Should not have exit code")
    assert_true(status.term_signal, "Should have term signal")
    assert_equal(status.term_signal.value(), SignalCodes.KILL)


def main():
    test_exit_code_verification()
    test_process_exits_immediately()
    test_multiple_wait_calls()
    test_signal_hangup()
    test_signal_interrupt()
    test_signal_kill()
