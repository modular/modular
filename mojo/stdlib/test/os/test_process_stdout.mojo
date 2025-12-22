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
# RUN: %mojo-no-debug %s | FileCheck %s

from collections import Dict, List
from os.path import exists
from os import Process

from testing import assert_false, assert_raises


def test_process_run():
    print("== test_process_run")
    # CHECK-LABEL: == test_process_run
    # CHECK-NEXT: == TEST_ECHO
    var command = "echo"
    _ = Process.run(command, ["== TEST_ECHO"])


def test_process_run_with_env():
    print("== test_process_run_with_env")
    # CHECK-LABEL: == test_process_run_with_env
    # CHECK-NEXT: HELLO_ENV
    var command = "sh"
    var args = List[String](["-c", "echo $MY_VAR"])
    var env = Dict[String, String]()
    env["MY_VAR"] = "HELLO_ENV"
    _ = Process.run(command, args^, env^)


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


def main():
    test_process_run()
    test_process_run_with_env()
    test_process_run_missing()
