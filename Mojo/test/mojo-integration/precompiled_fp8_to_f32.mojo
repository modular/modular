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

# Runtime numerical-correctness check for the software fp8 -> f32 conversion,
# which lowers `pop.cast` to a precompiled MLIR snippet (see
# `PrecompiledMLIRRegistry`). Each input is wrapped in
# `black_box` so the cast cannot be constant-folded and must actually execute
# the snippet at runtime.

# RUN: %mojo %s | FileCheck %s

from std.benchmark import black_box


def cvt[dt: DType](bits: UInt8) -> Float32:
    return SIMD[dt, 1](from_bits=black_box(bits)).cast[DType.float32]()


def cvt4[dt: DType](bits: SIMD[DType.uint8, 4]) -> SIMD[DType.float32, 4]:
    return SIMD[dt, 4](from_bits=black_box(bits)).cast[DType.float32]()


def main():
    # CHECK: e4m3fn 1.0 -2.0 0.5
    print(
        "e4m3fn",
        cvt[DType.float8_e4m3fn](0x38),
        cvt[DType.float8_e4m3fn](0xC0),
        cvt[DType.float8_e4m3fn](0x30),
    )
    # CHECK: e5m2 1.0 -2.0
    print(
        "e5m2",
        cvt[DType.float8_e5m2](0x3C),
        cvt[DType.float8_e5m2](0xC0),
    )
    # CHECK: e4m3fnuz 1.0 2.0
    print(
        "e4m3fnuz",
        cvt[DType.float8_e4m3fnuz](0x40),
        cvt[DType.float8_e4m3fnuz](0x48),
    )
    # CHECK: e5m2fnuz 1.0 2.0
    print(
        "e5m2fnuz",
        cvt[DType.float8_e5m2fnuz](0x40),
        cvt[DType.float8_e5m2fnuz](0x44),
    )
    # A SIMD cast becomes one snippet per lane.
    # CHECK: e4m3fn-x4 [1.0, -2.0, 0.5, 0.0]
    print(
        "e4m3fn-x4",
        cvt4[DType.float8_e4m3fn](SIMD[DType.uint8, 4](0x38, 0xC0, 0x30, 0x00)),
    )
