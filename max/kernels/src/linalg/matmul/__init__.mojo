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
from std.collections.string.string_slice import get_static_string
from std.math import align_up, ceildiv
from std.sys.info import align_of, simd_width_of

from std.gpu.host import DeviceContext
from std.gpu.host.info import is_cpu, is_valid_target
from layout import (
    Layout,
    LayoutTensor,
    TileTensor,
    UNKNOWN_VALUE,
    coord_to_index_list,
)
from std.runtime.asyncrt import DeviceContextPtr, parallelism_level
from std.runtime.tracing import Trace, TraceLevel, trace_arg

from std.utils.index import Index, IndexList

import .cpu
from ..gemv import gemv
from ..utils import (
    GemmShape,
    elementwise_compute_lambda_type,
    elementwise_epilogue_type,
)
from .gpu import _matmul_gpu


@always_inline
def matmul[
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    b_packed: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    saturated_vnni: Bool = False,
    _trace_description: StaticString = "",
    target: StaticString = "cpu",
](
    c: TileTensor[mut=True, address_space=AddressSpace.GENERIC, ...],
    a: TileTensor[address_space=AddressSpace.GENERIC, ...],
    b: TileTensor[address_space=AddressSpace.GENERIC, ...],
    ctx: DeviceContextPtr = DeviceContextPtr(),
) raises:
    """TileTensor overload of `matmul` with DeviceContextPtr."""
    var device_ctx = ctx.get_optional_device_context()

    return matmul[
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        b_packed=b_packed,
        elementwise_lambda_fn=elementwise_lambda_fn,
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        saturated_vnni=saturated_vnni,
        _trace_description=_trace_description,
        target=target,
    ](c, a, b, device_ctx)


@always_inline
def matmul[
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    b_packed: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    saturated_vnni: Bool = False,
    _trace_description: StaticString = "",
    target: StaticString = "cpu",
](
    c: TileTensor[mut=True, address_space=AddressSpace.GENERIC, ...],
    a: TileTensor[address_space=AddressSpace.GENERIC, ...],
    b: TileTensor[address_space=AddressSpace.GENERIC, ...],
    ctx: Optional[DeviceContext],
) raises:
    """Primary TileTensor matmul implementation. Routes GPU directly, delegates
    CPU path to cpu.matmul."""
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

        @always_inline
        @parameter
        def description_fn() -> String:
            var shape = GemmShape.get[transpose_b](c, a, b)
            # fmt: off
            return String(
                "(",
                target,
                ";", trace_arg("A", IndexList[2](shape.M, shape.K), a.dtype),
                ";", trace_arg("B", IndexList[2](shape.K, shape.N), b.dtype),
                ";", trace_arg("C", IndexList[2](shape.M, shape.N), c.dtype),
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
            Trace[TraceLevel.OP]._get_detail_str[description_fn](),
            task_id=OptionalReg(Int(ctx.value().id())),
        ):
            _matmul_gpu[
                use_tensor_core=True,
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            ](c, a, b, ctx.value())
    else:
        # CPU path: handle tracing and compute lambda wrapping, then
        # delegate to TileTensor cpu.matmul overload.
        comptime assert is_valid_target[target](), "unsupported target"
        comptime assert not transpose_a, "transpose_a not yet supported"

        if Int(c.dim[0]()) == 0 or Int(c.dim[1]()) == 0:
            return

        @always_inline
        @parameter
        def cpu_description_fn() -> String:
            var shape = GemmShape.get[transpose_b](c, a, b)
            # fmt: off
            return String(
                "(",
                target,
                ";", trace_arg("A", IndexList[2](shape.M, shape.K), a.dtype),
                ";", trace_arg("B", IndexList[2](shape.K, shape.N), b.dtype),
                ";", trace_arg("C", IndexList[2](shape.M, shape.N), c.dtype),
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
            Trace[TraceLevel.OP]._get_detail_str[cpu_description_fn](),
            task_id=OptionalReg(Int(ctx.value().id())) if ctx else None,
        ):
            var kernel_type_m = (
                a.static_shape[0] if a.static_shape[0] > -1 else 0
            )

            # The CPU version of matmul doesn't support compute lambda.
            # Wrap it around an epilogue lambda instead.
            @parameter
            @always_inline
            def compute_lambda_wrapper[
                _type: DType, _width: SIMDSize, *, alignment: Int = 1
            ](coords: IndexList[2], val: SIMD[_type, _width]):
                comptime if elementwise_compute_lambda_fn:
                    comptime compute_lambda = elementwise_compute_lambda_fn.value()
                    var output = compute_lambda(coords, val)
                    c.store_linear[alignment=alignment](
                        coords, rebind[SIMD[c.dtype, _width]](output)
                    )

            comptime elementwise_lambda_wrapper = Optional[
                elementwise_epilogue_type
            ](
                compute_lambda_wrapper
            ) if elementwise_compute_lambda_fn else elementwise_lambda_fn

            cpu.matmul[
                transpose_b=transpose_b,
                b_packed=b_packed,
                elementwise_lambda_fn=elementwise_lambda_wrapper,
                saturated_vnni=saturated_vnni,
            ](c, a, b, kernel_type_m, ctx=ctx)
