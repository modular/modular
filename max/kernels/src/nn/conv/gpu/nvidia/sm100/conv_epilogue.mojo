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
"""Register epilogue for the dedicated SM100 conv2d kernels.

Target family: Blackwell sm_100a, 1-SM UMMA with an `M = 128` accumulator in
tensor memory (lane = output pixel, column = output channel).

The dedicated kernels stage a plain convolution through shared memory and a
TMA store. A caller that fuses work into the convolution (a bias and an
activation, a residual) needs the value of every output element in registers
with its coordinates, so this helper reads the accumulator with the `32x32b`
load, where a thread owns one output pixel and a run of consecutive output
channels, and applies the same contracts the general kernel
(`conv2d_fprop_kernel.mojo`, `output_writer.mojo`) applies:

- `has_compute_fn`: the accumulator is cast to the epilogue dtype (bf16 for a
  bf16 output, float32 otherwise), `compute_fn(coord, value)` returns the
  value to store, and it is stored in the output dtype.
- `has_epilogue_fn`: the value is cast to the output dtype and
  `epilogue_fn(coord, value)` stores it. The kernel writes nothing itself.
- `has_residual`: `value += beta * source[coord]` in the epilogue dtype before
  either of the above; with neither, the sum is stored.

The coordinate is `(pixel, channel)` in the `[M, N]` view of the output. A
kernel calls `apply_conv_epilogue` once per accumulator tile from its epilogue
warps, passing its own row and column offsets.

All of a thread's columns are read from tensor memory and every closure that
only returns a value (`compute_fn`, the residual) is evaluated before the first
store. The closures load per-channel operands from global memory and the
stores may alias them, so interleaving the two would serialize one global load
latency per vector; this order keeps them all in flight together.
"""

from std.collections import Array
from std.sys import size_of

from linalg.utils import ElementwiseComputeFn, ElementwiseEpilogueFn
from max.gpu.compute.arch.tcgen05 import tcgen05_ld, tcgen05_load_wait


comptime _VEC = 8
"""Output channels per closure call."""
comptime _LOAD = 32
"""Accumulator columns per `tcgen05.ld`."""


def conv_epilogue_dtype[out_type: DType]() -> DType:
    """The dtype the compute and residual arithmetic runs in: bf16 for a bf16
    output, float32 otherwise (the general kernel's rule).

    Parameters:
        out_type: Element type of the output.

    Returns:
        The epilogue dtype.
    """
    comptime if out_type == .bfloat16:
        return .bfloat16
    else:
        return .float32


@always_inline
def apply_conv_epilogue[
    out_type: DType,
    EpilogueFnType: ElementwiseEpilogueFn,
    ComputeFnType: ElementwiseComputeFn,
    //,
    *,
    cols: Int,
    has_epilogue_fn: Bool,
    has_compute_fn: Bool,
    has_residual: Bool,
    nhwc: Bool = False,
    ndhwc: Bool = False,
    round_first: Bool = False,
](
    taddr: UInt32,
    row: Int,
    col: Int,
    ldc: Int,
    out_ptr: MutPointer[Scalar[out_type], _],
    src_ptr: MutPointer[Scalar[out_type], _],
    beta: Float32,
    epilogue_fn: EpilogueFnType,
    compute_fn: ComputeFnType,
    active: Bool = True,
    out_hw: Int = 0,
    out_w: Int = 0,
    out_d: Int = 0,
):
    """Applies the epilogue to this thread's output pixel over `cols` channels.

    Every thread of the epilogue warps calls this together: the `tcgen05.ld`
    is warp-collective, and each warp reads its own quarter of the 128 lanes.

    Parameters:
        out_type: Element type of the output and the residual source.
        EpilogueFnType: Type of `epilogue_fn` (inferred).
        ComputeFnType: Type of `compute_fn` (inferred).
        cols: Accumulator columns (output channels) to process, a multiple
            of 32.
        has_epilogue_fn: `epilogue_fn` stores each value.
        has_compute_fn: `compute_fn` transforms each value, which is then
            stored to `out_ptr`; excludes `has_epilogue_fn`.
        has_residual: Add `beta * source` first.
        nhwc: Hand the closures the NHWC coordinate `(n, h, w, c)` of the
            element, the contract of `conv_gpu`'s callers, instead of the
            `[M, N]` coordinate `(pixel, c)` the matmul-shaped kernels use.
            This is the 4-D to 2-D wrapper the bf16 path builds around each
            closure in `conv.mojo`, done once per thread here.
        ndhwc: Hand the closures the NDHWC coordinate `(n, d, h, w, c)` instead,
            for a 3-D convolution; `row` is then the output voxel
            `((n * D + d) * H + h) * W + w`. Takes effect without `nhwc`.
        round_first: Run the residual and compute arithmetic in the output
            dtype, as cuDNN followed by an elementwise kernel does, instead of
            the general kernel's rule (float32 for fp16). The bf16 rule is the
            same either way.

    Args:
        taddr: Tensor-memory address of the first column, with the warp's
            lane field zero.
        row: This thread's pixel in the `[M, N]` output view.
        col: Output channel of the first column.
        ldc: Elements between consecutive output pixels (`N`).
        out_ptr: Output, written when no `epilogue_fn` stores it.
        src_ptr: Residual source `[M, N]`, read when `has_residual`.
        beta: Residual scale factor.
        epilogue_fn: Storing elementwise epilogue.
        compute_fn: Transforming elementwise epilogue.
        active: Whether this thread's lane holds an output pixel. A tile with
            fewer than 128 rows (`M = 64`) leaves lanes idle; they still take
            part in the warp-collective load but call no closure and store
            nothing.
        out_hw: `H_out * W_out`, used when `nhwc` or `ndhwc`.
        out_w: `W_out`, used when `nhwc` or `ndhwc`.
        out_d: `D_out`, used when `ndhwc`.
    """
    comptime assert not (
        has_epilogue_fn and has_compute_fn
    ), "a store epilogue excludes the compute epilogue"
    comptime assert cols % _LOAD == 0, "whole 32-column loads"
    comptime ep_type = out_type if round_first else conv_epilogue_dtype[
        out_type
    ]()
    comptime vectors = cols // _VEC
    # A vector access is only one 16-byte instruction when the compiler knows
    # it is aligned; the default element alignment splits it into eight 2-byte
    # stores, which L2 handles as partial sectors. The caller guarantees `col`
    # and `ldc` are multiples of the vector width and the pointers are 16-byte
    # aligned.
    comptime _ACCESS = _VEC * size_of[out_type]()

    var acc = Array[Scalar[.float32], cols](uninitialized=True)
    comptime for piece in range(cols // _LOAD):
        var part = tcgen05_ld[
            datapaths=32,
            bits=32,
            repeat=_LOAD,
            dtype=.float32,
            pack=False,
        ](taddr + UInt32(piece * _LOAD))
        comptime for i in range(_LOAD):
            acc[piece * _LOAD + i] = part[i]
    tcgen05_load_wait()

    if not active:
        return

    var beta_ep = Scalar[out_type](beta).cast[ep_type]()
    var base = row * ldc + col
    var image = 0
    var depth = 0
    var y = 0
    var x = 0
    comptime if ndhwc:
        var voxels: Int
        var in_plane: Int
        image, voxels = divmod(row, out_d * out_hw)
        depth, in_plane = divmod(voxels, out_hw)
        y, x = divmod(in_plane, out_w)
    elif nhwc:
        var in_image: Int
        image, in_image = divmod(row, out_hw)
        y, x = divmod(in_image, out_w)

    # Values first, stores second (see the module note).
    var results = Array[SIMD[out_type, _VEC], vectors](uninitialized=True)
    comptime for g in range(vectors):
        var v = SIMD[.float32, _VEC]()
        comptime for i in range(_VEC):
            v[i] = acc[g * _VEC + i]
        var c = col + g * _VEC

        # The residual and compute arithmetic is in the epilogue dtype; a
        # plain store epilogue takes the accumulator straight to the output
        # dtype.
        comptime if has_residual:
            # `beta == 0` is no residual, not `0 * source`: the general SM100
            # residual path bypasses the load for it, and a NaN or infinity in
            # an unused source must not contaminate a finite convolution.
            if beta_ep != 0:
                var s = src_ptr.unsafe_offset(base + g * _VEC).unsafe_load[
                    width=_VEC, alignment=_ACCESS
                ]()
                var sum = v.cast[ep_type]() + beta_ep * s.cast[ep_type]()
                results[g] = sum.cast[out_type]()
            else:
                results[g] = v.cast[out_type]()
        elif has_compute_fn:
            comptime if ndhwc:
                var r = compute_fn[alignment=size_of[out_type]()](
                    (image, depth, y, x, c), v.cast[ep_type]()
                )
                results[g] = r.cast[out_type]()
            elif nhwc:
                var r = compute_fn[alignment=size_of[out_type]()](
                    (image, y, x, c), v.cast[ep_type]()
                )
                results[g] = r.cast[out_type]()
            else:
                var r = compute_fn[alignment=size_of[out_type]()](
                    (row, c), v.cast[ep_type]()
                )
                results[g] = r.cast[out_type]()
        else:
            results[g] = v.cast[out_type]()

    comptime for g in range(vectors):
        comptime if has_epilogue_fn:
            # The storing closure writes the output, whose base the TMA path
            # already needs 16-byte aligned, so it may use full-width stores.
            comptime if ndhwc:
                epilogue_fn[alignment=_ACCESS](
                    (image, depth, y, x, col + g * _VEC), results[g]
                )
            elif nhwc:
                epilogue_fn[alignment=_ACCESS](
                    (image, y, x, col + g * _VEC), results[g]
                )
            else:
                epilogue_fn[alignment=_ACCESS](
                    (row, col + g * _VEC), results[g]
                )
        else:
            out_ptr.unsafe_offset(base + g * _VEC).unsafe_store[
                alignment=_ACCESS
            ](results[g])
