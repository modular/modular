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

import extensibility as compiler

from std.gpu import MAX_THREADS_PER_BLOCK_METADATA, global_idx, grid_dim
from std.gpu.host import DeviceContext
from std.gpu.host.info import is_gpu
from std.math import ceildiv
from std.utils import StaticTuple

from extensibility import InputTensor, OutputTensor
from layout import TensorLayout, TileTensor
from quantization.w4a16.common import BM, BN, PRODUCTION_TOTAL_THREADS
from quantization.w4a16.kernels.ring_b_staged import (
    gemm_w4a16_ring_b_staged_kernel,
)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
@__name(t"gptq_to_w4a16_repack_qweight")
def _repack_qweight_kernel[
    qweight_layout: TensorLayout,
    qweight_bytes_layout: TensorLayout,
](
    qweight: TileTensor[DType.int32, qweight_layout, MutAnyOrigin],
    qweight_bytes: TileTensor[DType.uint8, qweight_bytes_layout, MutAnyOrigin],
    k: Int,
    n: Int,
):
    var tid = global_idx.x
    comptime block_size = 256
    var stride = grid_dim.x * block_size
    var n_packs = n // 8
    var total = k * n_packs
    var raw_qweight = qweight_bytes.ptr.bitcast[UInt32]()
    var packed_qweight = qweight.ptr.bitcast[UInt32]()

    for idx in range(tid, total, stride):
        var global_k = idx // n_packs
        var n_pack = idx - global_k * n_packs
        var raw_row = global_k // 8
        var k_lane = global_k - raw_row * 8
        var k_shift = UInt32(k_lane * 4)
        var packed = UInt32(0)

        comptime for lane in range(8):
            var n_col = n_pack * 8 + lane
            var raw = raw_qweight.load(raw_row * n + n_col)
            var nibble = (raw >> k_shift) & UInt32(0xF)
            packed = packed | (nibble << UInt32(lane * 4))

        packed_qweight.store(idx, packed)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
@__name(t"gptq_to_w4a16_repack_qweight_perm")
def _repack_qweight_perm_kernel[
    qweight_layout: TensorLayout,
    qweight_bytes_layout: TensorLayout,
    perm_idx_layout: TensorLayout,
](
    qweight: TileTensor[DType.int32, qweight_layout, MutAnyOrigin],
    qweight_bytes: TileTensor[DType.uint8, qweight_bytes_layout, MutAnyOrigin],
    perm_idx: TileTensor[DType.int32, perm_idx_layout, MutAnyOrigin],
    k: Int,
    n: Int,
):
    var tid = global_idx.x
    comptime block_size = 256
    var stride = grid_dim.x * block_size
    var n_packs = n // 8
    var total = k * n_packs
    var raw_qweight = qweight_bytes.ptr.bitcast[UInt32]()
    var packed_qweight = qweight.ptr.bitcast[UInt32]()

    for idx in range(tid, total, stride):
        var global_k = idx // n_packs
        var raw_global_k = Int(perm_idx[global_k])
        var n_pack = idx - global_k * n_packs
        var raw_row = raw_global_k // 8
        var k_lane = raw_global_k - raw_row * 8
        var k_shift = UInt32(k_lane * 4)
        var packed = UInt32(0)

        comptime for lane in range(8):
            var n_col = n_pack * 8 + lane
            var raw = raw_qweight.load(raw_row * n + n_col)
            var nibble = (raw >> k_shift) & UInt32(0xF)
            packed = packed | (nibble << UInt32(lane * 4))

        packed_qweight.store(idx, packed)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
@__name(t"gptq_to_w4a16_fill_qzeros")
def _fill_qzeros_kernel[
    qzeros_layout: TensorLayout,
](qzeros: TileTensor[DType.int32, qzeros_layout, MutAnyOrigin], total: Int,):
    var tid = global_idx.x
    comptime block_size = 256
    var stride = grid_dim.x * block_size
    var qzeros_bits = qzeros.ptr.bitcast[UInt32]()

    for idx in range(tid, total, stride):
        qzeros_bits.store(idx, UInt32(0x88888888))


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
@__name(t"gptq_to_w4a16_copy_scales")
def _copy_scales_kernel[
    scales_layout: TensorLayout,
    scales_bytes_layout: TensorLayout,
](
    scales: TileTensor[DType.float16, scales_layout, MutAnyOrigin],
    scales_bytes: TileTensor[DType.uint8, scales_bytes_layout, MutAnyOrigin],
    total: Int,
):
    var tid = global_idx.x
    comptime block_size = 256
    var stride = grid_dim.x * block_size
    var src = scales_bytes.ptr.bitcast[UInt16]()
    var dst = scales.ptr.bitcast[UInt16]()

    for idx in range(tid, total, stride):
        dst.store(idx, src.load(idx))


@compiler.register("gptq_to_w4a16")
struct GPTQToW4A16:
    @staticmethod
    @always_inline
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        qweight: OutputTensor[dtype=DType.int32, rank=2, ...],
        qzeros: OutputTensor[dtype=DType.int32, rank=2, ...],
        scales: OutputTensor[dtype=DType.float16, rank=2, ...],
        qweight_bytes: InputTensor[dtype=DType.uint8, rank=2, ...],
        scales_bytes: InputTensor[dtype=DType.uint8, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "gptq_to_w4a16 requires a GPU target"

        var k = qweight.dim_size(0)
        var n = scales.dim_size(1)
        var groups = scales.dim_size(0)
        var qweight_elems = k * (n // 8)
        var qzeros_elems = groups * (n // 8)
        var scales_elems = groups * n

        comptime repack_qweight = _repack_qweight_kernel[
            type_of(qweight.to_tile_tensor[DType.int64]()).LayoutType,
            type_of(qweight_bytes.to_tile_tensor[DType.int64]()).LayoutType,
        ]
        comptime fill_qzeros = _fill_qzeros_kernel[
            type_of(qzeros.to_tile_tensor[DType.int64]()).LayoutType,
        ]
        comptime copy_scales = _copy_scales_kernel[
            type_of(scales.to_tile_tensor[DType.int64]()).LayoutType,
            type_of(scales_bytes.to_tile_tensor[DType.int64]()).LayoutType,
        ]

        ctx.enqueue_function[repack_qweight](
            qweight.to_tile_tensor[DType.int64](),
            qweight_bytes.to_tile_tensor[DType.int64](),
            k,
            n,
            grid_dim=ceildiv(qweight_elems, 256),
            block_dim=256,
        )
        ctx.enqueue_function[fill_qzeros](
            qzeros.to_tile_tensor[DType.int64](),
            qzeros_elems,
            grid_dim=ceildiv(qzeros_elems, 256),
            block_dim=256,
        )
        ctx.enqueue_function[copy_scales](
            scales.to_tile_tensor[DType.int64](),
            scales_bytes.to_tile_tensor[DType.int64](),
            scales_elems,
            grid_dim=ceildiv(scales_elems, 256),
            block_dim=256,
        )


@compiler.register("gemm_w4a16_fp16")
struct GemmW4A16FP16:
    @staticmethod
    @always_inline
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        c: OutputTensor[dtype=DType.float16, rank=2, ...],
        a: InputTensor[dtype=DType.float16, rank=2, ...],
        qweight: InputTensor[dtype=DType.int32, rank=2, ...],
        qzeros: InputTensor[dtype=DType.int32, rank=2, ...],
        scales: InputTensor[dtype=DType.float16, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[
            target
        ](), "gemm_w4a16_fp16 requires a GPU target"

        var m = a.dim_size(0)
        var k = a.dim_size(1)
        var n = scales.dim_size(1)

        comptime kernel = gemm_w4a16_ring_b_staged_kernel[
            type_of(a.to_tile_tensor[DType.int64]()).LayoutType,
            type_of(qweight.to_tile_tensor[DType.int64]()).LayoutType,
            type_of(qzeros.to_tile_tensor[DType.int64]()).LayoutType,
            type_of(scales.to_tile_tensor[DType.int64]()).LayoutType,
            type_of(c.to_tile_tensor[DType.int64]()).LayoutType,
            type_of(a.to_tile_tensor[DType.int64]()).origin,
            type_of(qweight.to_tile_tensor[DType.int64]()).origin,
            type_of(qzeros.to_tile_tensor[DType.int64]()).origin,
            type_of(scales.to_tile_tensor[DType.int64]()).origin,
        ]

        ctx.enqueue_function[kernel](
            c.to_tile_tensor[DType.int64](),
            a.to_tile_tensor[DType.int64](),
            qweight.to_tile_tensor[DType.int64](),
            qzeros.to_tile_tensor[DType.int64](),
            scales.to_tile_tensor[DType.int64](),
            m,
            n,
            k,
            grid_dim=(ceildiv(n, BN), ceildiv(m, BM)),
            block_dim=(PRODUCTION_TOTAL_THREADS, 1),
        )


@compiler.register("gptq_to_w4a16_perm")
struct GPTQToW4A16Perm:
    @staticmethod
    @always_inline
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        qweight: OutputTensor[dtype=DType.int32, rank=2, ...],
        qzeros: OutputTensor[dtype=DType.int32, rank=2, ...],
        scales: OutputTensor[dtype=DType.float16, rank=2, ...],
        qweight_bytes: InputTensor[dtype=DType.uint8, rank=2, ...],
        scales_bytes: InputTensor[dtype=DType.uint8, rank=2, ...],
        perm_idx: InputTensor[dtype=DType.int32, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[
            target
        ](), "gptq_to_w4a16_perm requires a GPU target"

        var k = qweight.dim_size(0)
        var n = scales.dim_size(1)
        var groups = scales.dim_size(0)
        var qweight_elems = k * (n // 8)
        var qzeros_elems = groups * (n // 8)
        var scales_elems = groups * n

        comptime repack_qweight = _repack_qweight_perm_kernel[
            type_of(qweight.to_tile_tensor[DType.int64]()).LayoutType,
            type_of(qweight_bytes.to_tile_tensor[DType.int64]()).LayoutType,
            type_of(perm_idx.to_tile_tensor[DType.int64]()).LayoutType,
        ]
        comptime fill_qzeros = _fill_qzeros_kernel[
            type_of(qzeros.to_tile_tensor[DType.int64]()).LayoutType,
        ]
        comptime copy_scales = _copy_scales_kernel[
            type_of(scales.to_tile_tensor[DType.int64]()).LayoutType,
            type_of(scales_bytes.to_tile_tensor[DType.int64]()).LayoutType,
        ]

        ctx.enqueue_function[repack_qweight](
            qweight.to_tile_tensor[DType.int64](),
            qweight_bytes.to_tile_tensor[DType.int64](),
            perm_idx.to_tile_tensor[DType.int64](),
            k,
            n,
            grid_dim=ceildiv(qweight_elems, 256),
            block_dim=256,
        )
        ctx.enqueue_function[fill_qzeros](
            qzeros.to_tile_tensor[DType.int64](),
            qzeros_elems,
            grid_dim=ceildiv(qzeros_elems, 256),
            block_dim=256,
        )
        ctx.enqueue_function[copy_scales](
            scales.to_tile_tensor[DType.int64](),
            scales_bytes.to_tile_tensor[DType.int64](),
            scales_elems,
            grid_dim=ceildiv(scales_elems, 256),
            block_dim=256,
        )
