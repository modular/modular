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

# Verify `raise-for-loops` turns `for i in range(n)` (an exception-based
# `hlcf.loop`/`lit.try`) into an ascending, unit-step, `slt`-bounded `hlcf.for`.

# RUN: kgen --emit=llvm --mlir-print-ir-after=raise-for-loops %s -o %t.ll 2>&1 \
# RUN:   | FileCheck %s


@export
def range_sum(n: Int) -> Int:
    var total = 0
    for i in range(n):
        total = total + i
    return total


# CHECK-LABEL: @range_sum
# CHECK: hlcf.for [{{.*}} to {{.*}} step {{.*}} slt add]
# CHECK: hlcf.for.yield
# CHECK-NOT: hlcf.loop
# CHECK-NOT: lit.try
