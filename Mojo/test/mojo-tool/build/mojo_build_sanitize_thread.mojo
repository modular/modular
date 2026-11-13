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

# Note: Don't run with pre-existing sanitizers to ensure sanitizers work in a
#       clean environment.
# UNSUPPORTED: asan,msan,tsan
# UNSUPPORTED: system-darwin

# A race-free `parallelize` runs clean under ThreadSanitizer, and a racy one is
# reported.

# RUN: %mojo-build %s --sanitize thread -o %t
# RUN: %t 2>&1 | FileCheck %s --check-prefix=NO-RACE --implicit-check-not=ThreadSanitizer

# RUN: %mojo-build %s --sanitize thread -D RACE=1 -o %t.race
# RUN: not %t.race 2>&1 | FileCheck %s --check-prefix=RACE

# NO-RACE: sum=4950
# RACE: WARNING: ThreadSanitizer: data race

from max.algorithm import parallelize
from std.sys import is_defined


def main():
    comptime n = 100
    var values = List[Int](length=n, fill=0)

    def fill(i: Int) {mut values}:
        comptime if is_defined["RACE"]():
            values[0] += i
        else:
            values[i] = i

    parallelize(fill, n)

    var total = 0
    for v in values:
        total += v
    print("sum=" + String(total))
