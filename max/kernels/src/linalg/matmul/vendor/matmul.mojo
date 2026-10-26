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

"""Vendor-backed matrix multiplication dispatch for TileTensor operands."""

from std.sys import simd_width_of, size_of

from max.algorithm import elementwise
from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu.host.info import B200
from layout import Coord, TileTensor, row_major


from ...utils import (
    ElementwiseComputeFn,
    ElementwiseEpilogueFn,
)
from .blas import matmul as vendor_matmul


def _epilogue_simd_size[c_type: DType]() -> Int:
    # We hardcode simd width to 16B for Nvidia GPUs but >= sm_100
    # arch support 32B load/store to global memory, see KERN-2037.
    comptime use_32b_simd = (
        DeviceContext.target.is_nvidia_gpu()
        and DeviceContext.default_device_info.compute >= B200.compute
    )
    return 32 // size_of[c_type]() if use_32b_simd else (
        simd_width_of[c_type, target=get_gpu_target()]()
    )


def matmul[
    transpose_b: Bool = False,
](
    c: TileTensor[mut=True, ...],
    a: TileTensor,
    b: TileTensor,
    ctx: DeviceContext,
) raises:
    """Vendor matmul dispatch for TileTensor operands. Callers are
    responsible for allocating `c`.

    Parameters:
        transpose_b: Whether to treat `b` as transposed, computing
            `a @ b.T` instead of `a @ b` (defaults to `False`).

    Args:
        c: Output matrix of shape `(m, n)` and rank 2 that receives the
            matmul result. Caller-allocated and mutable.
        a: Left-hand input matrix of rank 2.
        b: Right-hand input matrix of rank 2, transposed when
            `transpose_b` is `True`.
        ctx: Device context used to select the vendor dispatch path and
            query device capabilities.
    """
    comptime assert c.flat_rank == 2, "c must be of rank 2"
    comptime assert a.flat_rank == 2, "a must be of rank 2"
    comptime assert b.flat_rank == 2, "b must be of rank 2"

    vendor_matmul[use_tf32=True](
        ctx,
        c,
        a,
        b,
        c_row_major=True,
        transpose_b=transpose_b,
    )


def matmul[
    EpilogueFnType: ElementwiseEpilogueFn,
    //,
    transpose_b: Bool = False,
](
    c: TileTensor[mut=True, ...],
    a: TileTensor,
    b: TileTensor,
    epilogue_fn: EpilogueFnType,
    ctx: DeviceContext,
) raises:
    """Vendor matmul into a scratch buffer, followed by a pass that hands
    each output chunk to an epilogue closure value.

    `epilogue_fn` stores the final output, so `c` supplies only the output
    shape and is neither read nor written. Callers may pass a `c` over a
    null pointer.

    Parameters:
        EpilogueFnType: Type of `epilogue_fn` (inferred).
        transpose_b: Whether to treat `b` as transposed, computing
            `a @ b.T` instead of `a @ b` (defaults to `False`).

    Args:
        c: Rank-2 matrix giving the output shape `(m, n)`. Its storage is
            not accessed.
        a: Left-hand input matrix of rank 2.
        b: Right-hand input matrix of rank 2, transposed when
            `transpose_b` is `True`.
        epilogue_fn: Receives each output chunk after the matmul.
        ctx: Device context used to select the vendor dispatch path and
            query device capabilities.
    """
    comptime assert c.flat_rank == 2, "c must be of rank 2"
    comptime assert a.flat_rank == 2, "a must be of rank 2"
    comptime assert b.flat_rank == 2, "b must be of rank 2"

    comptime c_type = c.dtype
    comptime simd_size = _epilogue_simd_size[c_type]()

    var scratch = ctx.enqueue_create_buffer[c_type](c.num_elements())
    var c_tt = TileTensor(scratch, row_major(c.layout.shape_coord()))

    def epilogue_wrapper[
        simd_width: Int, alignment: Int = 1
    ](idx: Coord) {var c_tt, var epilogue_fn}:
        var c_val = c_tt.load[
            width=simd_width,
            # Load takes alignment in bytes, the epilogue takes elements.
            alignment=alignment * size_of[c_type](),
        ](idx)
        epilogue_fn[c_type, simd_width, alignment=alignment](idx, c_val)

    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())

    vendor_matmul[use_tf32=True](
        ctx,
        c_tt,
        a,
        b,
        c_row_major=True,
        transpose_b=transpose_b,
    )
    elementwise[
        simd_size,
        target="gpu",
        _trace_description="elementwise_vendor_matmul_epilogue",
    ](epilogue_wrapper, (m, n), ctx)

    _ = scratch^


def matmul[
    ComputeFnType: ElementwiseComputeFn,
    //,
    transpose_b: Bool = False,
](
    c: TileTensor[mut=True, ...],
    a: TileTensor,
    b: TileTensor,
    compute_fn: ComputeFnType,
    ctx: DeviceContext,
) raises:
    """Vendor matmul into `c`, followed by a pass that replaces each output
    chunk with `compute_fn` of it.

    Parameters:
        ComputeFnType: Type of `compute_fn` (inferred).
        transpose_b: Whether to treat `b` as transposed, computing
            `a @ b.T` instead of `a @ b` (defaults to `False`).

    Args:
        c: Output matrix of shape `(m, n)` and rank 2. Caller-allocated
            and mutable.
        a: Left-hand input matrix of rank 2.
        b: Right-hand input matrix of rank 2, transposed when
            `transpose_b` is `True`.
        compute_fn: Maps each output index and chunk to the value to store.
        ctx: Device context used to select the vendor dispatch path and
            query device capabilities.
    """
    comptime assert c.flat_rank == 2, "c must be of rank 2"
    comptime assert a.flat_rank == 2, "a must be of rank 2"
    comptime assert b.flat_rank == 2, "b must be of rank 2"

    vendor_matmul[use_tf32=True](
        ctx,
        c,
        a,
        b,
        c_row_major=True,
        transpose_b=transpose_b,
    )
    apply_compute_fn_in_place(c, compute_fn, ctx)


def apply_compute_fn_in_place[
    ComputeFnType: ElementwiseComputeFn, //
](
    c: TileTensor[mut=True, ...],
    compute_fn: ComputeFnType,
    ctx: DeviceContext,
) raises:
    """Replaces each chunk of the row-major matrix `c` with `compute_fn` of it.

    Matmul paths that cannot fuse `compute_fn` into their epilogue run this
    pass after writing the plain product into `c`.

    Parameters:
        ComputeFnType: Type of `compute_fn` (inferred).

    Args:
        c: Rank-2 row-major matrix, updated in place.
        compute_fn: Maps each output index and chunk to the value to store.
        ctx: Device context used to launch the pass.
    """
    comptime assert c.flat_rank == 2, "c must be of rank 2"
    comptime c_type = c.dtype
    comptime simd_size = _epilogue_simd_size[c_type]()

    var c_tt = c

    def compute_wrapper[
        simd_width: Int, alignment: Int = 1
    ](idx: Coord) {var c_tt, var compute_fn}:
        # Load and store take alignment in bytes, compute_fn takes elements.
        comptime byte_alignment = alignment * size_of[c_type]()
        var c_val = c_tt.load[width=simd_width, alignment=byte_alignment](idx)
        c_tt.store[alignment=byte_alignment](
            idx,
            compute_fn[c_type, simd_width, alignment=alignment](idx, c_val),
        )

    elementwise[
        simd_size,
        target="gpu",
        _trace_description="apply_compute_fn_in_place",
    ](compute_wrapper, (Int(c.dim[0]()), Int(c.dim[1]())), ctx)
