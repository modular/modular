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
"""AMD RDNA3/4 WMMA implementation for matrix multiply-accumulate operations.

This module provides MMA implementations for AMD RDNA3 and RDNA4 consumer GPUs
using the WMMA (Wave Matrix Multiply Accumulate) instructions.

Reference: https://gpuopen.com/learn/wmma_on_rdna3/
"""

from sys import llvm_intrinsic

# Import helper functions from parent module
from ..mma import _has_type, _has_shape, _unsupported_mma_op


@always_inline
fn _mma_wmma_rdna(mut d: SIMD, a: SIMD, b: SIMD, c: SIMD):
    """AMD RDNA3+ WMMA implementation for matrix multiplication.

    RDNA3/4 GPUs use WMMA instructions.
    Per https://gpuopen.com/learn/wmma_on_rdna3/
    the following intrinsics are supported:
    - llvm.amdgcn.wmma.f32.16x16x16.f16
    - llvm.amdgcn.wmma.f32.16x16x16.bf16
    - llvm.amdgcn.wmma.f16.16x16x16.f16
    - llvm.amdgcn.wmma.bf16.16x16x16.bf16
    - llvm.amdgcn.wmma.i32.16x16x16.iu8
    - llvm.amdgcn.wmma.i32.16x16x16.iu4
    """

    @parameter
    fn get_intrinsic_name() -> String:
        # ===------------------------------------------------------------------===#
        # F32 = F16 * F16 + F32 (16x16x16)
        # Or
        # F32 = BF16 * BF16 + F32 (16x16x16)
        # ===------------------------------------------------------------------===#
        @parameter
        if (
            _has_type[
                (DType.float16, DType.float16, DType.float32, DType.float32)
            ](a.dtype, b.dtype, c.dtype, d.dtype)
            or _has_type[
                (DType.bfloat16, DType.bfloat16, DType.float32, DType.float32)
            ](a.dtype, b.dtype, c.dtype, d.dtype)
        ) and _has_shape[4](a.size, b.size, c.size, d.size):
            alias type_name = "f16" if a.dtype is DType.float16 else "bf16"
            return "llvm.amdgcn.wmma.f32.16x16x16." + type_name
        else:
            _unsupported_mma_op(d, a, b, c)
            return ""

    var r = llvm_intrinsic[get_intrinsic_name(), SIMD[c.dtype, c.size]](a, b, c)
    d = rebind[type_of(d)](r)
