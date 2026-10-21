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

# Compiles kernels that evaluate `std.math` functions as `comptime` constants
# for NVIDIA and AMD targets. The GPU lowerings of these functions use PTX and
# `llvm.nvvm.*`/`llvm.amdgcn.*` intrinsics, which the compile-time interpreter
# cannot run, so the test fails to compile if one of them is reached.

from std.math import cos, exp, exp2, log2, recip, rsqrt, sin, sqrt, tanh
from std.sys.info import CompilationTarget

from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code

comptime _UnaryFn = def[dtype: DType, width: SIMDLength, //](
    SIMD[dtype, width]
) thin -> SIMD[dtype, width] where dtype.is_floating_point()


def _kernel[
    func: _UnaryFn, dtype: DType
](
    output: MutPointer[Scalar[dtype], MutAnyOrigin]
) where dtype.is_floating_point():
    comptime result = func(Scalar[dtype](2))
    output[0] = result


def _compile[func: _UnaryFn, target: CompilationTarget]():
    _ = _compile_code[_kernel[func, .float32], target=target]()
    _ = _compile_code[_kernel[func, .float16], target=target]()
    _ = _compile_code[_kernel[func, .bfloat16], target=target]()


def _compile_all[target: CompilationTarget]():
    _compile[sqrt, target]()
    _compile[rsqrt, target]()
    _compile[recip, target]()
    _compile[exp2, target]()
    _compile[exp, target]()
    _compile[tanh, target]()
    _compile[log2, target]()
    _compile[sin, target]()
    _compile[cos, target]()


def main():
    _compile_all[get_gpu_target["sm_90"]()]()
    _compile_all[get_gpu_target["mi355x"]()]()
