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

# RUN: %mojo %s | FileCheck %s
#
# A statically dead arm of a multi-arm `if` (one carrying an `elif`) must not
# make SCCP treat the enclosing loop body as early-exiting.

from std.os import abort
from std.sys import argv


def run(n: Int) -> Int:
    var i = 0
    while i < n:
        if i < 0:
            if i >= 0:
                abort()
        elif i < 0:
            pass
        i += 1
    return i


def main():
    # CHECK: True
    var n = len(argv()) + 3
    print(run(n) == n)
