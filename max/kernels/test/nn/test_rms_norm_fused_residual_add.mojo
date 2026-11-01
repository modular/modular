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

from std.sys.info import align_of, simd_width_of

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from layout._fillers import random
from nn.normalization import rms_norm_cpu
from builtin_kernels.rowwise_normalization import rms_norm_fused_residual_add
from std.testing import assert_almost_equal

from std.utils.index import Index, IndexList


def run_rms_norm_fused_residual_add_gpu[
    rank: Int,
    //,
    dtype: DType,
](shape: IndexList[rank], rtol: Float64 = 0.01) raises:
    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    # Allocate host memory
    var data_heap = List(length=rows * cols, fill=Scalar[dtype](0))
    var data_h = TileTensor(data_heap, row_major(Coord(shape)))
    var unfused_intermediate_heap = List(
        length=rows * cols, fill=Scalar[dtype](0)
    )
    var unfused_intermediate_h = TileTensor(
        unfused_intermediate_heap, row_major(Coord(shape))
    )
    var result_unfused_heap = List(length=rows * cols, fill=Scalar[dtype](0))
    var result_unfused_h = TileTensor(
        result_unfused_heap, row_major(Coord(shape))
    )
    var result_fused_heap = List(length=rows * cols, fill=Scalar[dtype](0))
    var result_fused_h = TileTensor(result_fused_heap, row_major(Coord(shape)))
    var residual_fused_output_heap = List(
        length=rows * cols, fill=Scalar[dtype](0)
    )
    var residual_fused_output_h = TileTensor(
        residual_fused_output_heap, row_major(Coord(shape))
    )
    var gamma1_heap = List(length=cols, fill=Scalar[dtype](0))
    var gamma1_h = TileTensor(gamma1_heap, row_major(cols))
    var gamma2_heap = List(length=cols, fill=Scalar[dtype](0))
    var gamma2_h = TileTensor(gamma2_heap, row_major(cols))

    # Initialize input data
    random(data_h)
    random(gamma1_h)
    random(gamma2_h)

    var _ = Index(cols)

    var data_buf = data_h
    # Separate residual buffer (a copy of the input): the fused kernel's two
    # input loaders are distinct value-closure args and must reference distinct
    # buffer origins. Copying keeps residual == input numerically.
    var residual_heap = List(length=rows * cols, fill=Scalar[dtype](0))
    var data_buf_res = TileTensor(residual_heap, row_major(Coord(shape)))
    for i in range(rows * cols):
        residual_heap[i] = data_heap[i]
    var gamma1 = gamma1_h
    var gamma2 = gamma2_h
    var result_fused_buf = result_fused_h
    var result_unfused_buf = result_unfused_h
    var unfused_intermediate_buf = unfused_intermediate_h
    var residual_fused_output_buf = residual_fused_output_h
    comptime epsilon1 = Float32(0.001)
    comptime epsilon2 = Float32(0.002)
    comptime weight_offset1 = Float32(0.0)
    comptime weight_offset2 = Float32(0.0)

    # Test fused operation
    @inline(.always)
    def input_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var data_buf} -> SIMD[dtype, width]:
        # `alignment` counts elements; `load` takes bytes.
        return data_buf.load[
            width=width, alignment=alignment * align_of[dtype]()
        ](coords)

    @inline(.always)
    def residual_input_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var data_buf_res} -> SIMD[dtype, width]:
        # `alignment` counts elements; `load` takes bytes.
        return data_buf_res.load[
            width=width, alignment=alignment * align_of[dtype]()
        ](coords)

    @inline(.always)
    def fused_output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) {var result_fused_buf} -> None:
        result_fused_buf.store[alignment=alignment](coords, val)

    @inline(.always)
    def fused_residual_output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) {
        var residual_fused_output_buf
    } -> None:
        residual_fused_output_buf.store[alignment=alignment](coords, val)

    # Call fused kernel
    rms_norm_fused_residual_add[
        dtype,
        rank,
        target="cpu",
        epsilon1=epsilon1,
        weight_offset1=weight_offset1,
        epsilon2=epsilon2,
        weight_offset2=weight_offset2,
        multiply_before_cast=True,
    ](
        input_fn,
        residual_input_fn,
        fused_output_fn,
        fused_residual_output_fn,
        Coord(shape),
        Int(cols),
        gamma1,
        gamma2,
    )

    # Test unfused operations for comparison
    @inline(.always)
    def unfused_output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) {
        var unfused_intermediate_buf
    } -> None:
        unfused_intermediate_buf.store[alignment=alignment](coords, val)

    @inline(.always)
    def rms_input_fn[
        width: Int
    ](coords: Coord) {var data_buf} -> SIMD[dtype, width]:
        return data_buf.load[width=width](coords)

    # Step 1: First RMS norm
    rms_norm_cpu[
        epsilon=epsilon1,
        weight_offset=weight_offset1,
        multiply_before_cast=True,
    ](
        rms_input_fn,
        unfused_output_fn,
        Coord(shape),
        gamma1,
    )

    @inline(.always)
    def sum_fn[width: Int, alignment: Int = 1](coords: Coord) {var}:
        var residual_val = data_buf.load[width=width](coords)
        var result_val = unfused_intermediate_buf.load[width=width](coords)

        var residual_add_val = residual_val + result_val
        unfused_intermediate_buf.store(coords, residual_add_val)

    elementwise[simd_width_of[dtype](), target="cpu"](
        sum_fn,
        unfused_intermediate_buf.layout.shape_coord(),
        DeviceContext(api="cpu"),
    )

    @inline(.always)
    def unfused_input2_fn[
        width: Int
    ](coords: Coord) {var unfused_intermediate_buf} -> SIMD[dtype, width]:
        return unfused_intermediate_buf.load[width=width](coords)

    # Test unfused operations for comparison
    @inline(.always)
    def unfused_output2_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) {var result_unfused_buf} -> None:
        result_unfused_buf.store[alignment=alignment](coords, val)

    rms_norm_cpu[
        epsilon=epsilon2,
        weight_offset=weight_offset2,
        multiply_before_cast=True,
    ](
        unfused_input2_fn,
        unfused_output2_fn,
        Coord(shape),
        gamma2,
    )

    var flattened_size = rows * cols
    var flat_layout = row_major(flattened_size)
    var result_fused_flat = result_fused_h.reshape(flat_layout)
    var result_unfused_flat = result_unfused_h.reshape(flat_layout)
    var residual_fused_output_flat = residual_fused_output_h.reshape(
        flat_layout
    )
    var unfused_intermediate_flat = unfused_intermediate_h.reshape(flat_layout)
    for i in range(flattened_size):
        assert_almost_equal(
            result_fused_flat[i],
            result_unfused_flat[i],
            rtol=rtol,
        )
        assert_almost_equal(
            residual_fused_output_flat[i],
            unfused_intermediate_flat[i],
            rtol=rtol,
        )
    _ = gamma2_heap^
    _ = gamma1_heap^
    _ = residual_fused_output_heap^
    _ = result_fused_heap^
    _ = result_unfused_heap^
    _ = unfused_intermediate_heap^
    _ = data_heap^


def main() raises:
    # Test various shapes similar to test_rms_norm.mojo
    run_rms_norm_fused_residual_add_gpu[.float32](Index(5))
    run_rms_norm_fused_residual_add_gpu[.float32](Index(3, 4, 10, 20, 8))
    run_rms_norm_fused_residual_add_gpu[.float32](Index(1, 5, 6, 10, 128))
    run_rms_norm_fused_residual_add_gpu[.float32](Index(2, 5))
    run_rms_norm_fused_residual_add_gpu[.float32](Index(2, 55))
    run_rms_norm_fused_residual_add_gpu[.float32](Index(7, 557))
    run_rms_norm_fused_residual_add_gpu[.float32](Index(2, 8191))
    run_rms_norm_fused_residual_add_gpu[.float32](Index(2, 8192))
    run_rms_norm_fused_residual_add_gpu[.float32](Index(2, 16384))
    run_rms_norm_fused_residual_add_gpu[.float32](Index(2, 16385))

    run_rms_norm_fused_residual_add_gpu[.float32](Index(2, 16384))
    run_rms_norm_fused_residual_add_gpu[.float32](Index(2, 16385))
