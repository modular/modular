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
"""Provides the backend implementation for matmuls."""


from collections import OptionalReg
from collections.string.string_slice import get_static_string
from math import align_up, ceildiv
from sys.info import align_of, simd_width_of

from algorithm import sync_parallelize, tile, vectorize
from buffer.buffer import Dim, NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from gpu.host.info import is_cpu, is_valid_target
from layout import Layout, LayoutTensor, UNKNOWN_VALUE
from memory import memset_zero
from runtime.asyncrt import DeviceContextPtr, parallelism_level
from runtime.tracing import Trace, TraceLevel, trace_arg

from utils.index import Index, IndexList

import .cpu
from ..gemv import gemv
from ..utils import (
    GemmShape,
    elementwise_compute_lambda_type,
    elementwise_epilogue_type,
)
from .gpu import _matmul_gpu


@always_inline
fn matmul[
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    b_packed: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
    saturated_vnni: Bool = False,
    single_thread_blocking_override: Bool = False,
    _trace_description: StaticString = "",
    target: StaticString = "cpu",
](
    c: LayoutTensor[mut=True, address_space = AddressSpace.GENERIC, **_],
    a: LayoutTensor[address_space = AddressSpace.GENERIC, **_],
    b: LayoutTensor[address_space = AddressSpace.GENERIC, **_],
    ctx: Optional[DeviceContext],
) raises:
    return matmul[
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        b_packed=b_packed,
        elementwise_lambda_fn=elementwise_lambda_fn,
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        saturated_vnni=saturated_vnni,
        single_thread_blocking_override=single_thread_blocking_override,
        _trace_description=_trace_description,
        target=target,
    ](
        NDBuffer[
            c.dtype,
            2,
            c.origin,
            DimList(
                _to_value_or_dim(Int(c.layout.shape[0])),
                _to_value_or_dim(Int(c.layout.shape[1])),
            ),
        ](
            c.ptr,
            rebind[IndexList[2]](c.runtime_layout.shape.value.canonicalize()),
        ),
        NDBuffer[
            a.dtype,
            2,
            a.origin,
            DimList(
                _to_value_or_dim(Int(a.layout.shape[0])),
                _to_value_or_dim(Int(a.layout.shape[1])),
            ),
        ](
            a.ptr,
            rebind[IndexList[2]](a.runtime_layout.shape.value.canonicalize()),
        ),
        NDBuffer[
            b.dtype,
            2,
            b.origin,
            DimList(
                _to_value_or_dim(Int(b.layout.shape[0])),
                _to_value_or_dim(Int(b.layout.shape[1])),
            ),
        ](
            b.ptr,
            rebind[IndexList[2]](b.runtime_layout.shape.value.canonicalize()),
        ),
        ctx,
    )


@always_inline
fn matmul[
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    b_packed: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
    saturated_vnni: Bool = False,
    single_thread_blocking_override: Bool = False,
    _trace_description: StaticString = "",
    target: StaticString = "cpu",
](
    c: NDBuffer[mut=True, _, 2, _, _],
    a: NDBuffer[_, 2, _, _],
    b: NDBuffer[_, 2, _, _],
    ctx: DeviceContextPtr = DeviceContextPtr(),
) raises:
    var cuda_ctx = Optional[DeviceContext]() if is_cpu[
        target
    ]() else ctx.get_device_context()

    return matmul[
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        b_packed=b_packed,
        elementwise_lambda_fn=elementwise_lambda_fn,
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        saturated_vnni=saturated_vnni,
        single_thread_blocking_override=single_thread_blocking_override,
        _trace_description=_trace_description,
        target=target,
    ](c, a, b, cuda_ctx)


@always_inline
fn matmul[
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    b_packed: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
    saturated_vnni: Bool = False,
    single_thread_blocking_override: Bool = False,
    _trace_description: StaticString = "",
    target: StaticString = "cpu",
](
    c: NDBuffer[mut=True, _, 2, _, _],
    a: NDBuffer[_, 2, _, _],
    b: NDBuffer[_, 2, _, _],
    ctx: Optional[DeviceContext],
) raises:
    constrained[is_valid_target[target](), "unsupported target"]()
    constrained[not transpose_a, "transpose_a not yet supported"]()
    debug_assert(
        is_cpu[target]() or Bool(ctx),
        "expected DeviceContext to be provided if target != cpu",
    )

    # If any of the dimensions are 0, we can skip the kernel.
    if c.dim[0]() == 0 or c.dim[1]() == 0:
        return

    @always_inline
    @parameter
    fn description_fn() -> String:
        var shape = GemmShape.get[transpose_b](c, a, b)
        # fmt: off
        return String(
            "(",
            target,
            ";", trace_arg("A", IndexList[2](shape.M, shape.K), a.type),
            ";", trace_arg("B", IndexList[2](shape.K, shape.N), b.type),
            ";", trace_arg("C", IndexList[2](shape.M, shape.N), c.type),
            ";transpose_a=", transpose_a,
            ";transpose_b=", transpose_b,
            ";b_packed=", b_packed,
            ")"
        )
        # fmt: on

    # TODO(#23049): Pipe info on whether using faster, saturated_vnni is ok
    with Trace[TraceLevel.OP, target=target](
        # Create a string literal so that the event label works with the
        # AsyncRT profiler, whose event labels must be `StaticString`s.
        get_static_string[
            "matmul",
            _trace_description if _trace_description else "",
        ](),
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=OptionalReg(Int(ctx.value().id())) if ctx else None,
    ):

        @parameter
        if is_cpu[target]():
            var kernel_type_m = a.shape.at[0]().or_else(0)

            # The CPU version of matmul doesn't support compute lambda
            # We wrap it around an epilogue lambda instead.
            @parameter
            @always_inline
            fn compute_lambda_wrapper[
                _type: DType, _width: Int, *, alignment: Int = 1
            ](coords: IndexList[2], val: SIMD[_type, _width]):
                @parameter
                if elementwise_compute_lambda_fn:
                    alias compute_lambda = elementwise_compute_lambda_fn.value()
                    var output = compute_lambda(coords, val)
                    c.store[alignment=alignment](
                        coords, rebind[SIMD[c.type, _width]](output)
                    )

            alias elementwise_lambda_wrapper = OptionalReg[
                elementwise_epilogue_type
            ](
                compute_lambda_wrapper
            ) if elementwise_compute_lambda_fn else elementwise_lambda_fn

            cpu.matmul[
                transpose_b=transpose_b,
                b_packed=b_packed,
                elementwise_lambda_fn=elementwise_lambda_wrapper,
                saturated_vnni=saturated_vnni,
                single_thread_blocking_override=single_thread_blocking_override,
            ](c, a, b, kernel_type_m)

        else:
            _matmul_gpu[
                use_tensor_core=True,
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            ](c, a, b, ctx.value())


fn _to_value_or_dim(value: Int) -> Dim:
    if value != UNKNOWN_VALUE:
        return Dim(value)
    else:
        return Dim()
