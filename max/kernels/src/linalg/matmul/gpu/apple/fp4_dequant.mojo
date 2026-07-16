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
"""Portable NVFP4 -> bf16 weight-only dequant for the Apple M5 matmul (W4A16).

SM100/Apple M5 (`compute_capability == 5`), but the math here is hardware-neutral
(plain LUT + scale, no PTX / MFMA intrinsics). FLUX.2-dev-NVFP4 stores Linear
weights as packed E2M1 values with per-16-element FP8-E4M3 block scales; the
goal is to keep them packed in DRAM and dequant to bf16 in-register at the
matmul loader seam, feeding the EXISTING bf16 Apple MMA (the M5 MMA has no FP4
input).

Layout contract (Apple = PLAIN rank-2, NOT the SM100 5D TCGEN interleave):

  - packed weight: `uint8 [N, K // 2]`, two E2M1 nibbles per byte, LOW nibble
    first (element `2*j` is `byte & 0xF`, element `2*j+1` is `byte >> 4`). This
    matches `cast_uint_to_fp4e2m1` in `linalg/fp4_utils.mojo`.
  - block scales: `float8_e4m3fn [N, K // NVFP4_SF_VECTOR_SIZE]` (block size 16
    along K), applied as `abs(scale)`.
  - per-element dequant: `w_bf16 = E2M1_TO_FLOAT32[nibble] * |scale_block|`.

The global / tensor scalar scale (`alpha`) folds in once at the matmul epilogue,
not here -- this routine produces the per-element dequantized weight only.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext

from layout import TileTensor, Idx
from layout.tile_layout import TensorLayout, row_major

from linalg.fp4_utils import E2M1_TO_FLOAT32, NVFP4_SF_VECTOR_SIZE


@always_inline
def dequant_fp4_nibble[
    out_type: DType
](packed_byte: UInt8, nibble_hi: Bool, scale_abs: Float32) -> Scalar[out_type]:
    """Dequantizes one E2M1 nibble of a packed byte to `out_type`.

    Low nibble (`nibble_hi=False`) is the even K element; high nibble is the odd
    one. The 4-bit value indexes `E2M1_TO_FLOAT32` (sign + magnitude already
    baked into the 16-entry LUT) and is scaled by the block's `|scale|`.

    Parameters:
        out_type: Output element dtype (bf16 for the Apple W4A16 path).

    Args:
        packed_byte: The two-nibble packed FP4 byte.
        nibble_hi: True selects the high nibble (`byte >> 4`), False the low.
        scale_abs: The block scale, already passed through `abs()`.

    Returns:
        The dequantized value `E2M1_TO_FLOAT32[nibble] * scale_abs` as
        `Scalar[out_type]`.
    """
    var shift = UInt8(4) if nibble_hi else UInt8(0)
    var nibble = Int((packed_byte >> shift) & UInt8(0xF))
    var v = E2M1_TO_FLOAT32[nibble] * scale_abs
    return v.cast[out_type]()


@__name(t"apple_fp4_materialize_{out_type}")
def fp4_materialize_kernel[
    out_type: DType,
    w_layout: TensorLayout,
    s_layout: TensorLayout,
    out_layout: TensorLayout,
](
    out_w: TileTensor[out_type, out_layout, MutAnyOrigin],
    packed: TileTensor[DType.uint8, w_layout, ImmutAnyOrigin],
    scales: TileTensor[DType.float8_e4m3fn, s_layout, ImmutAnyOrigin],
):
    """Materializes the packed-FP4 weight into a dense `[N, K]` `out_type` buffer.

    One thread per output element `(n, k)`. `packed` is `[N, K//2]` (lo-nibble
    first), `scales` is `[N, K//16]`. Used by the Stage-1 oracle: it dequants the
    weight to bf16 so the EXISTING `AppleM5MatMul` can consume it, proving the
    dequant math against a host reference before the fused loader is written.
    """
    comptime assert out_w.flat_rank == 2, "out_w must be 2D [N, K]"
    comptime assert packed.flat_rank == 2, "packed must be 2D [N, K//2]"
    comptime assert scales.flat_rank == 2, "scales must be 2D [N, K//16]"

    var N = Int(out_w.dim[0]())
    var K = Int(out_w.dim[1]())

    var n = Int(global_idx.y)
    var k = Int(global_idx.x)
    if n >= N or k >= K:
        return

    var byte = rebind[Scalar[DType.uint8]](packed[n, k // 2])
    var scale = rebind[Scalar[DType.float8_e4m3fn]](
        scales[n, k // NVFP4_SF_VECTOR_SIZE]
    )
    var scale_abs = abs(scale.cast[DType.float32]())
    out_w[n, k] = rebind[out_w.ElementType](
        dequant_fp4_nibble[out_type](byte, (k % 2) == 1, scale_abs)
    )


@always_inline
def enqueue_fp4_materialize[
    out_type: DType
](
    out_w: TileTensor[mut=True, out_type, ...],
    packed: TileTensor[DType.uint8, ...],
    scales: TileTensor[DType.float8_e4m3fn, ...],
    ctx: DeviceContext,
) raises:
    """Enqueues `fp4_materialize_kernel`: packed FP4 + scales -> dense `[N, K]`.

    `out_w` is `[N, K]`, `packed` is `[N, K//2]`, `scales` is `[N, K//16]`. The
    grid is `(ceil(K/16), ceil(N/16))` threadgroups of 16x16 threads; bounds are
    checked per thread so ragged K/N are fine.
    """
    var N = Int(out_w.dim[0]())
    var K = Int(out_w.dim[1]())

    comptime BLK = 16
    comptime kernel = fp4_materialize_kernel[
        out_type,
        type_of(packed).LayoutType,
        type_of(scales).LayoutType,
        type_of(out_w).LayoutType,
    ]
    ctx.enqueue_function[kernel](
        out_w,
        packed.as_immut(),
        scales.as_immut(),
        grid_dim=((K + BLK - 1) // BLK, (N + BLK - 1) // BLK),
        block_dim=(BLK, BLK),
    )
