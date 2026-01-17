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
#
# This file only tests that `sin` and `cos` on float64 are unsupported.
#
# ===----------------------------------------------------------------------=== #
# RUN: not %bare-mojo %s 2>&1 | FileCheck %s

from compile import compile_info
from math.math import sin, cos
from gpu.host.info import _get_h100_target


def sin_func(x: Float64) -> Float64:
    # CHECK: constraint failed: DType.float64 is not supported on NVIDIA GPU
    return sin(x)


def cos_func(x: Float64) -> Float64:
    # CHECK: constraint failed: DType.float64 is not supported on NVIDIA GPU
    return cos(x)


def main():
    print(
        compile_info[
            sin_func, emission_kind="llvm", target = _get_h100_target()
        ]()
    )
    print(
        compile_info[
            cos_func, emission_kind="llvm", target = _get_h100_target()
        ]()
    )
