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
"""Provides the backend implementation for matmuls."""


from std.collections import OptionalReg
from std.collections.string.string_span import get_static_string
from std.math import align_up, ceildiv
from std.sys.info import align_of, simd_width_of

from max.gpu.host import DeviceContext
from max.gpu.host.info import is_cpu, is_valid_target
from layout import (
    Layout,
    RowMajorLayout,
    TileTensor,
    UNKNOWN_VALUE,
    coord_to_index_list,
)
from max.runtime.tracing import Trace, TraceLevel, trace_arg

from std.utils.index import Index, IndexList

from . import cpu
from ..gemv import gemv
from ..utils import (
    ElementwiseComputeFn,
    ElementwiseEpilogueFn,
    GemmShape,
    no_compute_fn,
    no_epilogue_fn,
)
from .gpu import _matmul_gpu


@inline(.always)
def matmul[
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    b_packed: Bool = False,
    saturated_vnni: Bool = False,
    _trace_description: StaticString = "",
    target: StaticString = "cpu",
    # Kept last so positional-parameter callers are unaffected; set it by
    # keyword.
    use_tf32: Bool = True,
](
    c: TileTensor[mut=True, address_space=.GENERIC, ...],
    a: TileTensor[address_space=.GENERIC, ...],
    b: TileTensor[address_space=.GENERIC, ...],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Primary TileTensor matmul implementation. Routes GPU directly, delegates
    CPU path to cpu.matmul.

    `use_tf32=False` (GPU only) requires IEEE-fp32 multiplies for fp32
    inputs instead of TF32 tensor-core truncation; see `_matmul_gpu`. The
    CPU path is always IEEE fp32.

    Parameters:
        transpose_a: Transpose `a` before the matmul (defaults to `False`);
            currently unsupported.
        transpose_b: Transpose `b` before the matmul (defaults to `False`).
        b_packed: `b` is already in a CPU-friendly packed layout (CPU only;
            defaults to `False`).
        saturated_vnni: Use the saturating VNNI variant on x86 (CPU only;
            defaults to `False`).
        _trace_description: Extra string folded into the op trace label
            (defaults to an empty string).
        target: Target platform string such as `"cpu"` or a GPU target
            (defaults to `"cpu"`).
        use_tf32: On GPU, allow TF32 tensor-core truncation for fp32 inputs
            (defaults to `True`).

    Args:
        c: Rank-2 output `TileTensor` accumulating the matmul result.
        a: Rank-2 LHS `TileTensor` of the matmul.
        b: Rank-2 RHS `TileTensor` of the matmul.
        ctx: Optional `DeviceContext` required for GPU targets and used for
            CPU tracing (defaults to `None`).
    """
    _matmul_impl[
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        b_packed=b_packed,
        saturated_vnni=saturated_vnni,
        _trace_description=_trace_description,
        target=target,
        use_tf32=use_tf32,
        has_epilogue_fn=False,
        has_compute_fn=False,
    ](c, a, b, no_epilogue_fn, no_compute_fn, ctx)


@inline(.always)
def matmul[
    EpilogueFnType: ElementwiseEpilogueFn,
    //,
    *,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    b_packed: Bool = False,
    saturated_vnni: Bool = False,
    _trace_description: StaticString = "",
    target: StaticString = "cpu",
    use_tf32: Bool = True,
](
    c: TileTensor[mut=True, address_space=.GENERIC, ...],
    a: TileTensor[address_space=.GENERIC, ...],
    b: TileTensor[address_space=.GENERIC, ...],
    epilogue_fn: EpilogueFnType,
    ctx: Optional[DeviceContext] = None,
) raises:
    """Matmul that stores each output tile through `epilogue_fn`.

    Parameters:
        EpilogueFnType: Type of `epilogue_fn` (inferred).
        transpose_a: Transpose `a` before the matmul (defaults to `False`);
            currently unsupported.
        transpose_b: Transpose `b` before the matmul (defaults to `False`).
        b_packed: `b` is already in a CPU-friendly packed layout (CPU only;
            defaults to `False`).
        saturated_vnni: Use the saturating VNNI variant on x86 (CPU only;
            defaults to `False`).
        _trace_description: Extra string folded into the op trace label
            (defaults to an empty string).
        target: Target platform string such as `"cpu"` or a GPU target
            (defaults to `"cpu"`).
        use_tf32: On GPU, allow TF32 tensor-core truncation for fp32 inputs
            (defaults to `True`).

    Args:
        c: Rank-2 output `TileTensor`. Some dispatch paths use it as scratch
            for the product.
        a: Rank-2 LHS `TileTensor` of the matmul.
        b: Rank-2 RHS `TileTensor` of the matmul.
        epilogue_fn: Stores each output index and tile.
        ctx: Optional `DeviceContext` required for GPU targets and used for
            CPU tracing (defaults to `None`).
    """
    _matmul_impl[
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        b_packed=b_packed,
        saturated_vnni=saturated_vnni,
        _trace_description=_trace_description,
        target=target,
        use_tf32=use_tf32,
        has_epilogue_fn=True,
        has_compute_fn=False,
    ](c, a, b, epilogue_fn, no_compute_fn, ctx)


@inline(.always)
def matmul[
    ComputeFnType: ElementwiseComputeFn,
    //,
    *,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    b_packed: Bool = False,
    saturated_vnni: Bool = False,
    _trace_description: StaticString = "",
    target: StaticString = "cpu",
    use_tf32: Bool = True,
](
    c: TileTensor[mut=True, address_space=.GENERIC, ...],
    a: TileTensor[address_space=.GENERIC, ...],
    b: TileTensor[address_space=.GENERIC, ...],
    compute_fn: ComputeFnType,
    ctx: Optional[DeviceContext] = None,
) raises:
    """Matmul that stores `compute_fn(idx, a @ b)` into `c`.

    Parameters:
        ComputeFnType: Type of `compute_fn` (inferred).
        transpose_a: Transpose `a` before the matmul (defaults to `False`);
            currently unsupported.
        transpose_b: Transpose `b` before the matmul (defaults to `False`).
        b_packed: `b` is already in a CPU-friendly packed layout (CPU only;
            defaults to `False`).
        saturated_vnni: Use the saturating VNNI variant on x86 (CPU only;
            defaults to `False`).
        _trace_description: Extra string folded into the op trace label
            (defaults to an empty string).
        target: Target platform string such as `"cpu"` or a GPU target
            (defaults to `"cpu"`).
        use_tf32: On GPU, allow TF32 tensor-core truncation for fp32 inputs
            (defaults to `True`).

    Args:
        c: Rank-2 output `TileTensor`.
        a: Rank-2 LHS `TileTensor` of the matmul.
        b: Rank-2 RHS `TileTensor` of the matmul.
        compute_fn: Maps each output index and tile to the value to store.
        ctx: Optional `DeviceContext` required for GPU targets and used for
            CPU tracing (defaults to `None`).
    """
    _matmul_impl[
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        b_packed=b_packed,
        saturated_vnni=saturated_vnni,
        _trace_description=_trace_description,
        target=target,
        use_tf32=use_tf32,
        has_epilogue_fn=False,
        has_compute_fn=True,
    ](c, a, b, no_epilogue_fn, compute_fn, ctx)


@inline(.always)
def _matmul_impl[
    EpilogueFnType: ElementwiseEpilogueFn,
    ComputeFnType: ElementwiseComputeFn,
    //,
    *,
    transpose_a: Bool,
    transpose_b: Bool,
    b_packed: Bool,
    saturated_vnni: Bool,
    _trace_description: StaticString,
    target: StaticString,
    use_tf32: Bool,
    has_epilogue_fn: Bool,
    has_compute_fn: Bool,
](
    c: TileTensor[mut=True, address_space=.GENERIC, ...],
    a: TileTensor[address_space=.GENERIC, ...],
    b: TileTensor[address_space=.GENERIC, ...],
    epilogue_fn: EpilogueFnType,
    compute_fn: ComputeFnType,
    ctx: Optional[DeviceContext],
) raises:
    comptime assert c.rank == 2, "c must be rank 2"
    comptime assert a.rank == 2, "a must be rank 2"
    comptime assert b.rank == 2, "b must be rank 2"
    comptime assert c.flat_rank == 2, "c must have a non-nested layout"
    comptime assert a.flat_rank == 2, "a must have a non-nested layout"
    comptime assert b.flat_rank == 2, "b must have a non-nested layout"

    comptime if not is_cpu[target]():
        # GPU path: call _matmul_gpu directly with tracing. CPU-only params
        # (b_packed, saturated_vnni) are intentionally not forwarded here.
        comptime assert not transpose_a, "transpose_a not yet supported"
        assert Bool(ctx), "expected DeviceContext for GPU target"

        if Int(c.dim[0]()) == 0 or Int(c.dim[1]()) == 0:
            return

        @inline(.always)
        def description_fn() {imm} -> String:
            var shape = GemmShape.get[transpose_b](c, a, b)
            # fmt: off
            return String(
                "(",
                target,
                ";", trace_arg("A", (shape.M, shape.K), a.dtype),
                ";", trace_arg("B", (shape.K, shape.N), b.dtype),
                ";", trace_arg("C", (shape.M, shape.N), c.dtype),
                ";transpose_a=", transpose_a,
                ";transpose_b=", transpose_b,
                ")"
            )
            # fmt: on

        with Trace[TraceLevel.OP, target=target](
            get_static_string[
                "matmul",
                _trace_description if _trace_description else "",
            ](),
            Trace[TraceLevel.OP]._get_detail_str(description_fn),
            task_id=OptionalReg(Int(ctx.value().id())),
        ):
            comptime if has_epilogue_fn:
                _matmul_gpu[
                    use_tensor_core=True,
                    transpose_b=transpose_b,
                    use_tf32=use_tf32,
                ](c, a, b, epilogue_fn, ctx.value())
            elif has_compute_fn:
                _matmul_gpu[
                    use_tensor_core=True,
                    transpose_b=transpose_b,
                    use_tf32=use_tf32,
                ](c, a, b, compute_fn, ctx.value())
            else:
                _matmul_gpu[
                    use_tensor_core=True,
                    transpose_b=transpose_b,
                    use_tf32=use_tf32,
                ](c, a, b, ctx.value())
    else:
        # CPU path: handle tracing and compute lambda wrapping, then
        # delegate to TileTensor cpu.matmul overload.
        comptime assert is_valid_target[target](), "unsupported target"
        comptime assert not transpose_a, "transpose_a not yet supported"

        if Int(c.dim[0]()) == 0 or Int(c.dim[1]()) == 0:
            return

        @inline(.always)
        def cpu_description_fn() {imm} -> String:
            var shape = GemmShape.get[transpose_b](c, a, b)
            # fmt: off
            return String(
                "(",
                target,
                ";", trace_arg("A", (shape.M, shape.K), a.dtype),
                ";", trace_arg("B", (shape.K, shape.N), b.dtype),
                ";", trace_arg("C", (shape.M, shape.N), c.dtype),
                ";transpose_a=", transpose_a,
                ";transpose_b=", transpose_b,
                ";b_packed=", b_packed,
                ")"
            )
            # fmt: on

        with Trace[TraceLevel.OP, target=target](
            get_static_string[
                "matmul",
                _trace_description if _trace_description else "",
            ](),
            Trace[TraceLevel.OP]._get_detail_str(cpu_description_fn),
            task_id=OptionalReg(Int(ctx.value().id())) if ctx else None,
        ):
            var kernel_type_m = (
                a.static_shape[0] if a.static_shape[0] > -1 else 0
            )

            var cpu_ctx = DeviceContext._map_assert_cpu_target(ctx)

            comptime if has_compute_fn:
                cpu.matmul[
                    transpose_b=transpose_b,
                    b_packed=b_packed,
                    saturated_vnni=saturated_vnni,
                ](c, a, b, compute_fn, kernel_type_m, ctx=cpu_ctx)
            elif has_epilogue_fn:
                cpu.matmul[
                    transpose_b=transpose_b,
                    b_packed=b_packed,
                    saturated_vnni=saturated_vnni,
                ](c, a, b, epilogue_fn, kernel_type_m, ctx=cpu_ctx)
            else:
                cpu.matmul[
                    transpose_b=transpose_b,
                    b_packed=b_packed,
                    saturated_vnni=saturated_vnni,
                ](c, a, b, kernel_type_m, ctx=cpu_ctx)
