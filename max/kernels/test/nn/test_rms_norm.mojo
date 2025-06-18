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

from math import ceildiv, sqrt
from sys import simdwidthof

from buffer import NDBuffer
from buffer.dimlist import DimList
from nn.normalization import *
from testing import assert_almost_equal
from sys import has_neon

from utils.index import Index, IndexList


fn compute_rms[
    type: DType
](data: NDBuffer[type, 1], size: Int, eps: Scalar[type]) -> Scalar[
    DType.float32
]:
    var sum_of_squares = Scalar[DType.float32]()
    for i in range(size):
        var d = data[i].cast[DType.float32]()
        sum_of_squares += d * d
    return sqrt((sum_of_squares / len(data)) + eps.cast[DType.float32]())


fn run_rms_norm_cpu[
    type: DType, rank: Int
](shape: IndexList[rank], rtol: Float64 = 0.001) raises:
    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    var input_ptr = UnsafePointer[Scalar[type]].alloc(rows * cols)
    var output_ptr = UnsafePointer[Scalar[type]].alloc(rows * cols)
    var gamma_ptr = UnsafePointer[Scalar[type]].alloc(cols)

    for i in range(rows * cols):
        input_ptr[i] = Scalar[type](i)

    for i in range(cols):
        gamma_ptr[i] = ((i + cols) / cols).cast[type]()

    var param_shape = Index(cols)

    var input_buf = NDBuffer[type, rank](input_ptr, shape)
    var output_buf = NDBuffer[type, rank](output_ptr, shape)
    var gamma = NDBuffer[type, 1](gamma_ptr, param_shape)
    var epsilon = Scalar[type](0.0001)
    var weight_offset = Scalar[type](0.0)

    @__copy_capture(input_buf)
    @always_inline
    @parameter
    fn input_fn[
        width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[type, width]:
        return input_buf.load[width=width](rebind[IndexList[rank]](idx))

    @always_inline
    @__copy_capture(output_buf)
    @parameter
    fn identity_output_fn[
        width: Int
    ](idx: IndexList[rank], val: SIMD[type, width]) -> None:
        output_buf.store(idx, val)

    rms_norm_cpu[input_fn, identity_output_fn, multiply_before_cast=True](
        shape,
        gamma,
        epsilon,
        weight_offset,
    )

    for r in range(rows):
        var vec = NDBuffer[type, 1](input_ptr + r * cols, cols)
        var rms_ref = compute_rms(vec, cols, epsilon)
        for c in range(cols):
            var idx = r * cols + c
            # PyTorch converts the input to float32 before computing the RMS norm
            # https://github.com/meta-llama/llama/blob/689c7f261b9c5514636ecc3c5fefefcbb3e6eed7/llama/model.py#L76
            var val = (input_ptr[idx].cast[DType.float32]() / rms_ref).cast[
                type
            ]() * (gamma_ptr[c] + weight_offset)
            assert_almost_equal(val, output_ptr[idx], rtol=rtol)

    input_ptr.free()
    output_ptr.free()
    gamma_ptr.free()


fn run_rms_norm_tests[type: DType](rtol: Float64 = 0.001) raises:
    run_rms_norm_cpu[type](Index(2, 5), rtol)
    run_rms_norm_cpu[type](Index(2, 55), rtol)
    run_rms_norm_cpu[type](Index(7, 557), rtol)
    run_rms_norm_cpu[type](Index(2, 8191), rtol)
    run_rms_norm_cpu[type](Index(2, 8192), rtol)
    run_rms_norm_cpu[type](Index(2, 16384), rtol)
    run_rms_norm_cpu[type](Index(2, 16385), rtol)

    # variable rank
    run_rms_norm_cpu[type](Index(0), rtol)
    run_rms_norm_cpu[type](Index(5), rtol)
    run_rms_norm_cpu[type](Index(3, 4, 10, 20, 8), rtol)
    run_rms_norm_cpu[type](Index(1, 5, 6, 10, 128), rtol)


def main():
    run_rms_norm_tests[DType.float32]()

    @parameter
    if not has_neon():
        run_rms_norm_tests[DType.bfloat16](rtol=1e-2)
