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

# `mma()` lowers to the 16x16 simdgroup MMA on every Apple target that has the
# neural accelerator (M5 and newer), and is rejected on older generations.

# RUN: %mojo -D TARGET=apple-m5-metal4 %s | FileCheck %s --check-prefix=CHECK-MMA
# RUN: %mojo -D TARGET=apple-m6-metal4 %s | FileCheck %s --check-prefix=CHECK-MMA
# RUN: not %mojo -D TARGET=apple-m4-metal4 %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-MMA

# CHECK-MMA: call {{.*}} @air.simdgroup_matrix_16x16x16_multiply_accumulate
# CHECK-NO-MMA: Current compilation target does not support operation: mma.

from std.compile import compile_info
from std.sys import get_defined_string

from max.gpu.compute.mma import mma
from max.gpu.host.info import get_gpu_target


def mma_kernel(dst: Pointer[Float32, MutAnyOrigin]):
    var a = SIMD[.float16, 8](1)
    var b = SIMD[.float16, 8](1)
    var c = SIMD[.float32, 8](0)
    var d = SIMD[.float32, 8](0)
    mma(d, a, b, c)
    dst[] = d[0]


def main():
    print(
        compile_info[
            mma_kernel,
            emission_kind="llvm",
            target=get_gpu_target[get_defined_string["TARGET"]()](),
        ]()
    )
