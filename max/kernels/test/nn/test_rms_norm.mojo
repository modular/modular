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

from math import sqrt

from layout import (
    LayoutTensor,
    Layout,
    RuntimeLayout,
    RuntimeTuple,
    UNKNOWN_VALUE,
)
from layout.int_tuple import fill_like
from nn.normalization import *
from testing import assert_almost_equal
from sys.info import CompilationTarget

from utils.index import Index, IndexList


fn compute_rms[
    dtype: DType
](data: LayoutTensor[dtype, **_], size: Int, eps: Scalar[dtype]) -> Scalar[
    DType.float32
]:
    constrained[data.rank == 1, "data.rank must be 1"]()
    var sum_of_squares = Scalar[DType.float32]()
    for i in range(size):
        var d = data.ptr[i].cast[DType.float32]()
        sum_of_squares += d * d
    return sqrt((sum_of_squares / data.size()) + eps.cast[DType.float32]())


fn run_rms_norm_cpu[
    dtype: DType, rank: Int
](shape: IndexList[rank], rtol: Float64 = 0.001) raises:
    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    var input_ptr = UnsafePointer[Scalar[dtype]].alloc(rows * cols)
    var output_ptr = UnsafePointer[Scalar[dtype]].alloc(rows * cols)
    var gamma_ptr = UnsafePointer[Scalar[dtype]].alloc(cols)

    for i in range(rows * cols):
        input_ptr[i] = Scalar[dtype](i)

    for i in range(cols):
        gamma_ptr[i] = ((i + cols) / cols).cast[dtype]()

    var param_shape = Index(cols)

    alias layout = Layout.row_major[rank]()
    var input_buf = LayoutTensor[dtype, layout](
        input_ptr, RuntimeLayout[layout].row_major(shape)
    )
    var output_buf = LayoutTensor[dtype, layout](
        output_ptr, RuntimeLayout[layout].row_major(shape)
    )
    var gamma = LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE)](
        gamma_ptr,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(param_shape),
    )
    var epsilon = Scalar[dtype](0.0001)
    var weight_offset = Scalar[dtype](0.0)

    @__copy_capture(input_buf)
    @always_inline
    @parameter
    fn input_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var idx = input_buf.runtime_layout(
            RuntimeTuple[fill_like(input_buf.layout.shape, UNKNOWN_VALUE)](
                coords
            )
        )
        return input_buf.ptr.load[width=width](idx)

    @always_inline
    @__copy_capture(output_buf)
    @parameter
    fn identity_output_fn[
        width: Int, alignment: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]) -> None:
        var idx = output_buf.runtime_layout(
            RuntimeTuple[fill_like(output_buf.layout.shape, UNKNOWN_VALUE)](
                coords
            )
        )
        output_buf.ptr.store[width=width, alignment=alignment](idx, val)

    rms_norm_cpu[input_fn, identity_output_fn, multiply_before_cast=True](
        shape,
        gamma,
        epsilon,
        weight_offset,
    )

    for r in range(rows):
        var vec = LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE)](
            input_ptr + r * cols,
            RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                IndexList[1](cols)
            ),
        )
        var rms_ref = compute_rms(vec, cols, epsilon)
        for c in range(cols):
            var idx = r * cols + c
            # PyTorch converts the input to float32 before computing the RMS norm
            # https://github.com/meta-llama/llama/blob/689c7f261b9c5514636ecc3c5fefefcbb3e6eed7/llama/model.py#L76
            var val = (input_ptr[idx].cast[DType.float32]() / rms_ref).cast[
                dtype
            ]() * (gamma_ptr[c] + weight_offset)
            assert_almost_equal(val, output_ptr[idx], rtol=rtol)

    input_ptr.free()
    output_ptr.free()
    gamma_ptr.free()


fn run_rms_norm_tests[dtype: DType](rtol: Float64 = 0.001) raises:
    run_rms_norm_cpu[dtype](Index(15, 11), rtol)
    # run_rms_norm_cpu[dtype](Index(2, 5), rtol)
    # run_rms_norm_cpu[dtype](Index(2, 55), rtol)
    # run_rms_norm_cpu[dtype](Index(7, 557), rtol)
    # run_rms_norm_cpu[dtype](Index(2, 8191), rtol)
    # run_rms_norm_cpu[dtype](Index(2, 8192), rtol)
    # run_rms_norm_cpu[dtype](Index(2, 16384), rtol)
    # run_rms_norm_cpu[dtype](Index(2, 16385), rtol)

    # # variable rank
    # run_rms_norm_cpu[dtype](Index(0), rtol)
    # run_rms_norm_cpu[dtype](Index(5), rtol)
    # run_rms_norm_cpu[dtype](Index(3, 4, 10, 20, 8), rtol)
    # run_rms_norm_cpu[dtype](Index(1, 5, 6, 10, 128), rtol)


def main():
    run_rms_norm_tests[DType.float32]()

    @parameter
    if not CompilationTarget.has_neon():
        run_rms_norm_tests[DType.bfloat16](rtol=1e-2)
