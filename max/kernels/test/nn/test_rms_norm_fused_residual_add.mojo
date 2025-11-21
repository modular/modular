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
from sys.info import simd_width_of

from algorithm.functional import elementwise
from layout import (
    UNKNOWN_VALUE,
    Layout,
    LayoutTensor,
    RuntimeTuple,
    RuntimeLayout,
)
from layout._fillers import random
from layout.int_tuple import fill_like
from memory import alloc
from nn.normalization import rms_norm_cpu, rms_norm_fused_residual_add_cpu
from testing import assert_almost_equal

from utils.index import Index, IndexList


fn run_rms_norm_fused_residual_add_gpu[
    rank: Int, //,
    dtype: DType,
](shape: IndexList[rank], rtol: Float64 = 0.01) raises:
    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    # Allocate host memory
    alias layout = Layout.row_major[rank]()
    var data_heap = alloc[Scalar[dtype]](rows * cols)
    var data_h = LayoutTensor[dtype, layout](
        data_heap, RuntimeLayout[layout].row_major(shape)
    )
    var unfused_intermediate_heap = alloc[Scalar[dtype]](rows * cols)
    var unfused_intermediate_h = LayoutTensor[dtype, layout](
        unfused_intermediate_heap, RuntimeLayout[layout].row_major(shape)
    ).fill(0)
    var result_unfused_heap = alloc[Scalar[dtype]](rows * cols)
    var result_unfused_h = LayoutTensor[dtype, layout](
        result_unfused_heap, RuntimeLayout[layout].row_major(shape)
    ).fill(0)
    var result_fused_heap = alloc[Scalar[dtype]](rows * cols)
    var result_fused_h = LayoutTensor[dtype, layout](
        result_fused_heap, RuntimeLayout[layout].row_major(shape)
    ).fill(0)
    var residual_fused_output_heap = alloc[Scalar[dtype]](rows * cols)
    var residual_fused_output_h = LayoutTensor[dtype, layout](
        residual_fused_output_heap, RuntimeLayout[layout].row_major(shape)
    ).fill(0)
    alias layout_1d = Layout(UNKNOWN_VALUE)
    var gamma1_heap = alloc[Scalar[dtype]](cols)
    var gamma1_h = LayoutTensor[dtype, layout_1d](
        gamma1_heap, RuntimeLayout[layout_1d].row_major(Index(cols))
    )
    var gamma2_heap = alloc[Scalar[dtype]](cols)
    var gamma2_h = LayoutTensor[dtype, layout_1d](
        gamma2_heap, RuntimeLayout[layout_1d].row_major(Index(cols))
    )

    # Initialize input data
    random(data_h)
    random(gamma1_h)
    random(gamma2_h)

    var _ = Index(cols)

    var data_buf = data_h
    var gamma1 = gamma1_h
    var gamma2 = gamma2_h
    var result_fused_buf = result_fused_h
    var result_unfused_buf = result_unfused_h
    var unfused_intermediate_buf = unfused_intermediate_h
    var residual_fused_output_buf = residual_fused_output_h
    var epsilon1 = Scalar[dtype](0.001)
    var epsilon2 = Scalar[dtype](0.002)
    var weight_offset1 = Scalar[dtype](0.0)
    var weight_offset2 = Scalar[dtype](0.0)

    # Test fused operation
    @__copy_capture(data_buf)
    @always_inline
    @parameter
    fn input_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var idx = data_buf.runtime_layout(
            RuntimeTuple[fill_like(data_buf.layout.shape, UNKNOWN_VALUE)](
                coords
            )
        )
        return data_buf.ptr.load[width=width](idx)

    @parameter
    @always_inline
    fn residual_input_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var idx = data_buf.runtime_layout(
            RuntimeTuple[fill_like(data_buf.layout.shape, UNKNOWN_VALUE)](
                coords
            )
        )
        return data_buf.ptr.load[width=width](idx)

    @always_inline
    @__copy_capture(result_fused_buf)
    @parameter
    fn fused_output_fn[
        width: Int, alignment: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]) -> None:
        var idx = result_fused_buf.runtime_layout(
            RuntimeTuple[
                fill_like(result_fused_buf.layout.shape, UNKNOWN_VALUE)
            ](coords)
        )
        result_fused_buf.ptr.store[width=width, alignment=alignment](idx, val)

    @always_inline
    @__copy_capture(residual_fused_output_buf)
    @parameter
    fn fused_residual_output_fn[
        width: Int, alignment: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]) -> None:
        var idx = residual_fused_output_buf.runtime_layout(
            RuntimeTuple[
                fill_like(residual_fused_output_buf.layout.shape, UNKNOWN_VALUE)
            ](coords)
        )
        residual_fused_output_buf.ptr.store[width=width, alignment=alignment](
            idx, val
        )

    # Call fused kernel
    rms_norm_fused_residual_add_cpu[
        input_fn,
        residual_input_fn,
        fused_output_fn,
        fused_residual_output_fn,
        multiply_before_cast=True,
    ](
        shape,
        gamma1,
        epsilon1,
        weight_offset1,
        gamma2,
        epsilon2,
        weight_offset2,
    )

    # Test unfused operations for comparison
    @always_inline
    @__copy_capture(unfused_intermediate_buf)
    @parameter
    fn unfused_output_fn[
        width: Int, alignment: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]) -> None:
        var idx = unfused_intermediate_buf.runtime_layout(
            RuntimeTuple[
                fill_like(unfused_intermediate_buf.layout.shape, UNKNOWN_VALUE)
            ](coords)
        )
        unfused_intermediate_buf.ptr.store[width=width, alignment=alignment](
            idx, val
        )

    # Step 1: First RMS norm
    rms_norm_cpu[input_fn, unfused_output_fn, multiply_before_cast=True](
        shape, gamma1, epsilon1, weight_offset1
    )

    @parameter
    @always_inline
    @__copy_capture(unfused_intermediate_buf, data_buf)
    fn sum_fn[
        width: Int, rank_: Int, alignment: Int = 1
    ](coords: IndexList[rank_]):
        var data_buf_idx = data_buf.runtime_layout(
            RuntimeTuple[fill_like(data_buf.layout.shape, UNKNOWN_VALUE)](
                coords
            )
        )
        var residual_val = data_buf.ptr.load[width=width](data_buf_idx)
        var unfused_intermediate_buf_idx = (
            unfused_intermediate_buf.runtime_layout(
                RuntimeTuple[
                    fill_like(
                        unfused_intermediate_buf.layout.shape, UNKNOWN_VALUE
                    )
                ](coords)
            )
        )
        var result_val = unfused_intermediate_buf.ptr.load[width=width](
            unfused_intermediate_buf_idx
        )

        var residual_add_val = residual_val + result_val
        unfused_intermediate_buf.ptr.store[width=width](
            unfused_intermediate_buf_idx, residual_add_val
        )

    elementwise[sum_fn, simd_width_of[dtype](), target="cpu"](
        unfused_intermediate_buf.runtime_layout.shape.value.canonicalize(),
    )

    @parameter
    @always_inline
    @__copy_capture(unfused_intermediate_buf)
    fn unfused_input2_fn[
        width: Int, rank: Int
    ](coords: IndexList[rank]) -> SIMD[dtype, width]:
        var idx = unfused_intermediate_buf.runtime_layout(
            RuntimeTuple[
                fill_like(unfused_intermediate_buf.layout.shape, UNKNOWN_VALUE)
            ](coords)
        )

        return unfused_intermediate_buf.ptr.load[width=width](idx)

    # Test unfused operations for comparison
    @always_inline
    @__copy_capture(result_unfused_buf)
    @parameter
    fn unfused_output2_fn[
        width: Int, alignment: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]) -> None:
        var idx = result_unfused_buf.runtime_layout(
            RuntimeTuple[
                fill_like(result_unfused_buf.layout.shape, UNKNOWN_VALUE)
            ](coords)
        )

        result_unfused_buf.ptr.store[width=width, alignment=alignment](idx, val)

    rms_norm_cpu[
        unfused_input2_fn,
        unfused_output2_fn,
        multiply_before_cast=True,
    ](shape, gamma2, epsilon2, weight_offset2)

    var flattened_size = rows * cols
    for i in range(flattened_size):
        assert_almost_equal(
            result_fused_h.ptr[i],
            result_unfused_h.ptr[i],
            rtol=rtol,
        )
        assert_almost_equal(
            residual_fused_output_h.ptr[i],
            unfused_intermediate_h.ptr[i],
            rtol=rtol,
        )
    data_heap.free()
    unfused_intermediate_heap.free()
    result_unfused_heap.free()
    result_fused_heap.free()
    residual_fused_output_heap.free()
    gamma1_heap.free()
    gamma2_heap.free()


def main():
    # Test various shapes similar to test_rms_norm.mojo
    run_rms_norm_fused_residual_add_gpu[DType.float32](Index(5))
    run_rms_norm_fused_residual_add_gpu[DType.float32](Index(3, 4, 10, 20, 8))
    run_rms_norm_fused_residual_add_gpu[DType.float32](Index(1, 5, 6, 10, 128))
    run_rms_norm_fused_residual_add_gpu[DType.float32](Index(2, 5))
    run_rms_norm_fused_residual_add_gpu[DType.float32](Index(2, 55))
    run_rms_norm_fused_residual_add_gpu[DType.float32](Index(7, 557))
    run_rms_norm_fused_residual_add_gpu[DType.float32](Index(2, 8191))
    run_rms_norm_fused_residual_add_gpu[DType.float32](Index(2, 8192))
    run_rms_norm_fused_residual_add_gpu[DType.float32](Index(2, 16384))
    run_rms_norm_fused_residual_add_gpu[DType.float32](Index(2, 16385))

    run_rms_norm_fused_residual_add_gpu[DType.float32](Index(2, 16384))
    run_rms_norm_fused_residual_add_gpu[DType.float32](Index(2, 16385))
