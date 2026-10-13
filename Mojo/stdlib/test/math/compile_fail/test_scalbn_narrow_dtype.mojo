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

# RUN: not %mojo -D test=1 %s 2>&1 | FileCheck %s
# RUN: not %mojo -D test=2 %s 2>&1 | FileCheck %s

from std.math.math import _scalbn_impl
from std.sys import get_defined_int


# CHECK: constraint failed: _scalbn_impl requires float32 or float64
def main():
    comptime if get_defined_int["test"]() == 1:
        print(_scalbn_impl(Float16(1), Int32(40000)))
    else:
        print(_scalbn_impl(BFloat16(1), Int32(40000)))
