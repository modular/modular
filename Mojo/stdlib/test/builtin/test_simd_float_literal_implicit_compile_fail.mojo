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
# RUN: not %mojo -D test=3 %s 2>&1 | FileCheck %s
# RUN: not %mojo -D test=4 %s 2>&1 | FileCheck %s

# Test that float literals do not implicitly convert to integral SIMD types.
# Explicit construction (`UInt(400.)`) is supported, but implicit narrowing
# from float to int stays rejected. Bool is not an integral dtype and rejects
# explicit construction from a float literal too.

from std.sys import get_defined_int


def takes_uint(x: UInt):
    pass


def returns_uint() -> UInt:
    comptime if get_defined_int["test"]() == 3:
        return 400.0
    else:
        return UInt(0)


# CHECK: constraint failed: the SIMD type must be floating point
def main():
    comptime if get_defined_int["test"]() == 1:
        takes_uint(400.0)
    elif get_defined_int["test"]() == 2:
        var value: UInt = 400.0
        _ = value
    elif get_defined_int["test"]() == 3:
        _ = returns_uint()
    elif get_defined_int["test"]() == 4:
        _ = Scalar[DType.bool](1.0)
