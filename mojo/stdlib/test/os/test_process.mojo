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
# RUN: %mojo %s | FileCheck %s

from collections import List
from os.path import exists
from os import Process

from testing import assert_false, assert_raises


def test_process_run():
    var command = "echo"
    _ = Process.run(command, List[String]("== TEST_ECHO"))


def test_process_run_missing():
    missing_executable_file = "ThIsFiLeCoUlDNoTPoSsIbLlYExIsT.NoTAnExTeNsIoN"

    # verify that the test file does not exist before starting the test
    assert_false(
        exists(missing_executable_file),
        "Unexpected file '" + missing_executable_file + "' it should not exist",
    )

    with assert_raises():
        _ = Process.run(missing_executable_file, List[String]())


# CHECK-LABEL: TEST_ECHO
def main():
    test_process_run()
    test_process_run_missing()
