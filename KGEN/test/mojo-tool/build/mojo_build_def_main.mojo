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

# RUN: %mojo-build %s -o %t
# RUN: not %t --arg1 | FileCheck %s

from std.sys import argv


def main() raises:
    # CHECK: This was called inside of `def` main
    print("This was called inside of `def` main")

    # CHECK: --arg1
    print(argv()[1])

    # CHECK: Unhandled exception caught during execution: main raised an error
    raise Error("main raised an error")
