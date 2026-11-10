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
"""The packed NVFP4 MLA latent cache and its exact FP8 mirror.

A cache row is the 576-wide MLA row stored as a 512-wide e2m1 latent, two
elements per byte, followed by the 64 rope dims as raw FP8 bytes. One E4M3
scale per 16 latent elements lives in a separate scale pool with its own page
table: block `b`'s scales are in block `(b * mult + offset) % num_blocks`, a
permutation, as a scales leaf with block ids of its own has. An index row is
`2 * top_k` wide: the picks' value rows, then their scale rows.

The FP8 mirror holds `e4m3(e2m1 * scale)` for the latent and the rope bytes
unchanged, which is exactly what the SM100 kernels compute from the packed
row, so a kernel over the mirror is the reference for one over the packed
cache.
"""

from std.math import ceildiv
from std.memory import bitcast
from std.sys import size_of

from bench_sweep import uniform_bits
from linalg.fp4_utils import E2M1_TO_FLOAT32
from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

comptime _LATENT = 512
comptime _ROPE = 64
comptime _ROW = _LATENT + _ROPE
comptime PACKED_ROW = _LATENT // 2 + _ROPE
comptime SF_PER_ROW = _LATENT // 16
# `quantization_granularity` counts packed bytes: 16 elements are 8 bytes.
comptime QUANT_GRANULARITY = 8
comptime _BLOCK = 256


@inline(.always)
def _start() -> Int:
    return Int(block_idx.x * block_dim.x + thread_idx.x)


@inline(.always)
def _stride() -> Int:
    return Int(grid_dim.x * block_dim.x)


@inline(.always)
def _grid(n: Int) -> Int:
    return max(1, min(ceildiv(n, _BLOCK), 1 << 16))


@inline(.always)
def _scale_block(block: Int, mult: Int, offset: Int, num_blocks: Int) -> Int:
    return (block * mult + offset) % num_blocks


def _fill_packed_kernel(
    packed: Pointer[UInt8, MutAnyOrigin],
    scales: Pointer[Scalar[DType.float8_e4m3fn], MutAnyOrigin],
    rows: Int64,
    seed: UInt32,
):
    var packed_n = Int(rows) * PACKED_ROW
    var scales_n = Int(rows) * SF_PER_ROW
    var i = _start()
    while i < max(packed_n, scales_n):
        var bits = uniform_bits(seed, i)
        if i < packed_n:
            var byte = UInt8(bits & 0xFF)
            # Clearing bit 0 keeps the rope bytes clear of the E4M3 NaNs,
            # 0x7F and 0xFF; any byte is a valid pair of e2m1 codes.
            if i % PACKED_ROW >= _LATENT // 2:
                byte &= 0xFE
            packed[unsafe_offset=i] = byte
        if i < scales_n:
            # A small-exponent band: finite, positive and nonzero.
            scales[unsafe_offset=i] = bitcast[DType.float8_e4m3fn, 1](
                UInt8(0x30) | UInt8((bits >> 8) & 7)
            )
        i += _stride()


def _dequant_kernel(
    packed: Pointer[UInt8, ImmutAnyOrigin],
    scales: Pointer[Scalar[DType.float8_e4m3fn], ImmutAnyOrigin],
    mirror: Pointer[Scalar[DType.float8_e4m3fn], MutAnyOrigin],
    rows: Int64,
    page_size: Int64,
    mult: Int64,
    offset: Int64,
    num_blocks: Int64,
):
    var i = _start()
    while i < Int(rows) * _ROW:
        var row, col = divmod(i, _ROW)
        if col >= _LATENT:
            mirror[unsafe_offset=i] = bitcast[DType.float8_e4m3fn, 1](
                packed[
                    unsafe_offset=row * PACKED_ROW
                    + _LATENT // 2
                    + col
                    - _LATENT
                ]
            )
        else:
            var code = Int(packed[unsafe_offset=row * PACKED_ROW + col // 2])
            code = (code >> (4 * (col % 2))) & 15
            var block, in_page = divmod(row, Int(page_size))
            var scale_row = (
                _scale_block(block, Int(mult), Int(offset), Int(num_blocks))
                * Int(page_size)
                + in_page
            )
            var sf = scales[unsafe_offset=scale_row * SF_PER_ROW + col // 16]
            mirror[unsafe_offset=i] = (
                E2M1_TO_FLOAT32[code] * sf.cast[DType.float32]()
            ).cast[DType.float8_e4m3fn]()
        i += _stride()


def _permute_blocks_kernel(
    dst: Pointer[UInt32, MutAnyOrigin],
    src: Pointer[UInt32, ImmutAnyOrigin],
    n: Int64,
    mult: Int64,
    offset: Int64,
    num_blocks: Int64,
):
    var i = _start()
    while i < Int(n):
        dst[unsafe_offset=i] = UInt32(
            _scale_block(
                Int(src[unsafe_offset=i]),
                Int(mult),
                Int(offset),
                Int(num_blocks),
            )
        )
        i += _stride()


def _interleave_kernel(
    dst: Pointer[Int32, MutAnyOrigin],
    values: Pointer[Int32, ImmutAnyOrigin],
    scales: Pointer[Int32, ImmutAnyOrigin],
    tokens: Int64,
    top_k: Int64,
):
    var k = Int(top_k)
    var i = _start()
    while i < Int(tokens) * 2 * k:
        var g, j = divmod(i, 2 * k)
        dst[unsafe_offset=i] = (
            values[unsafe_offset=g * k + j] if j
            < k else scales[unsafe_offset=g * k + j - k]
        )
        i += _stride()


struct PackedLatent(Movable):
    """A packed NVFP4 latent pool, its scale pool and its FP8 mirror.

    All three span `rows` KV rows, the rows of a `PageTable`'s blocks, so the
    value page table addresses the packed pool and the mirror alike.
    """

    var rows: Int
    var page_size: Int
    var mult: Int
    var offset: Int
    var packed: DeviceBuffer[DType.uint8]
    var scales: DeviceBuffer[DType.float8_e4m3fn]
    var mirror: DeviceBuffer[DType.float8_e4m3fn]

    def __init__(
        out self,
        ctx: DeviceContext,
        rows: Int,
        page_size: Int,
        seed: UInt32,
        permute_scales: Bool = True,
    ) raises:
        """Allocates the pools and fills them with random valid rows.

        `permute_scales` pages the scales apart from the values; off, a
        row's scales share its block, as when one page table serves both.
        """
        self.rows = rows
        self.page_size = page_size
        var num_blocks = rows // page_size
        self.mult = 1
        self.offset = 0
        if permute_scales and num_blocks > 1:
            self.offset = 1
            self.mult = 3
            while _gcd(self.mult, num_blocks) != 1:
                self.mult += 2
        self.packed = ctx.enqueue_create_buffer[DType.uint8](rows * PACKED_ROW)
        self.scales = ctx.enqueue_create_buffer[DType.float8_e4m3fn](
            rows * SF_PER_ROW
        )
        self.mirror = ctx.enqueue_create_buffer[DType.float8_e4m3fn](
            rows * _ROW
        )
        var n = rows * PACKED_ROW
        ctx.enqueue_function[_fill_packed_kernel](
            self.packed.unsafe_ptr().as_unsafe_any_origin(),
            self.scales.unsafe_ptr().as_unsafe_any_origin(),
            Int64(rows),
            seed,
            grid_dim=_grid(n),
            block_dim=_BLOCK,
        )
        self.rebuild_mirror(ctx)

    @staticmethod
    def bytes_for(rows: Int) -> Int:
        return rows * (PACKED_ROW + SF_PER_ROW + _ROW)

    def bytes(self) -> Int:
        return Self.bytes_for(self.rows)

    def num_blocks(self) -> Int:
        return self.rows // self.page_size

    def rebuild_mirror(self, ctx: DeviceContext) raises:
        """Rewrites the mirror from the packed pool, after anything that
        changed the packed rows, such as a write through the cache."""
        var n = self.rows * _ROW
        ctx.enqueue_function[_dequant_kernel](
            self.packed.unsafe_ptr().as_imm().as_unsafe_any_origin(),
            self.scales.unsafe_ptr().as_imm().as_unsafe_any_origin(),
            self.mirror.unsafe_ptr().as_unsafe_any_origin(),
            Int64(self.rows),
            Int64(self.page_size),
            Int64(self.mult),
            Int64(self.offset),
            Int64(self.num_blocks()),
            grid_dim=_grid(n),
            block_dim=_BLOCK,
        )

    def scale_lut(
        self,
        ctx: DeviceContext,
        dst: DeviceBuffer[DType.uint32, ...],
        value_lut: DeviceBuffer[DType.uint32, ...],
        n: Int,
    ) raises:
        """Writes the scale page table matching `n` entries of `value_lut`."""
        permute_blocks(
            ctx, dst, value_lut, n, self.mult, self.offset, self.num_blocks()
        )


def _gcd(a: Int, b: Int) -> Int:
    var x = a
    var y = b
    while y != 0:
        var t = y
        y = x % y
        x = t
    return x


def permute_blocks(
    ctx: DeviceContext,
    dst: DeviceBuffer[DType.uint32, ...],
    src: DeviceBuffer[DType.uint32, ...],
    n: Int,
    mult: Int,
    offset: Int,
    num_blocks: Int,
) raises:
    """Writes `(src[i] * mult + offset) % num_blocks` for the first `n`
    entries."""
    ctx.enqueue_function[_permute_blocks_kernel](
        dst.unsafe_ptr().as_unsafe_any_origin(),
        src.unsafe_ptr().as_imm().as_unsafe_any_origin(),
        Int64(n),
        Int64(mult),
        Int64(offset),
        Int64(num_blocks),
        grid_dim=_grid(n),
        block_dim=_BLOCK,
    )


def interleave_index_rows(
    ctx: DeviceContext,
    dst: DeviceBuffer[DType.int32, ...],
    values: DeviceBuffer[DType.int32, ...],
    scales: DeviceBuffer[DType.int32, ...],
    tokens: Int,
    top_k: Int,
) raises:
    """Writes `2 * top_k`-wide NVFP4 index rows from `top_k`-wide value and
    scale rows."""
    var n = tokens * 2 * top_k
    ctx.enqueue_function[_interleave_kernel](
        dst.unsafe_ptr().as_unsafe_any_origin(),
        values.unsafe_ptr().as_imm().as_unsafe_any_origin(),
        scales.unsafe_ptr().as_imm().as_unsafe_any_origin(),
        Int64(tokens),
        Int64(top_k),
        grid_dim=_grid(n),
        block_dim=_BLOCK,
    )
