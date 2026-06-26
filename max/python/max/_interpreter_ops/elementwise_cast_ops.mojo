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

"""Mojo kernel wrappers for the Cast MO interpreter operation.

Cast has a large number of kernel specializations (13x13 input/output dtype
combinations), so it is split into its own compilation unit to reduce peak
memory usage during compilation.
"""

from std.os import abort
from std.gpu.host import DeviceContext
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys.info import has_accelerator, simd_width_of
from std.utils.coord import Coord

from std.algorithm.functional import elementwise, IndexList
from std.reflection import reflect

from std.sys.info import has_apple_gpu_accelerator
from extensibility import ElementwiseUnaryMixedOp
from builtin_kernels import Cast

from op_utils import _get_dtype, _get_buffer_ptr, _get_size, _get_ctx


# =============================================================================
# GPU Support Configuration
# =============================================================================


def _is_gpu_allowed_mixed_unary_op[op: ElementwiseUnaryMixedOp]() -> Bool:
    """Check if a mixed-type unary op is allowed on GPU at compile time."""
    comptime name = reflect[op].base_name()
    return name == "Cast"


# =============================================================================
# Python bindings
# =============================================================================


@export
def PyInit_elementwise_cast_ops() abi("C") -> PythonObject:
    """Create a Python module with cast kernel function bindings."""
    try:
        var b = PythonModuleBuilder("elementwise_cast_ops")

        # Cast operation (mixed input/output dtypes)
        b.def_function[cast_dispatcher](
            "Cast", docstring="Elementwise Cast with dtype dispatch"
        )

        return b.finalize()
    except e:
        abort(t"failed to create elementwise cast op bindings module: {e}")


# =============================================================================
# Dispatchers
# =============================================================================


def cast_dispatcher(
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Cast operation dispatcher that handles double dtype dispatch.

    Args:
        out_buffer: The output buffer object.
        in_buffer: The input buffer object.
        device_context_ptr: Device context pointer.
    """
    var in_dtype = _get_dtype(in_buffer)
    var out_dtype = _get_dtype(out_buffer)
    var size = _get_size(out_buffer)
    var ctx = _get_ctx(device_context_ptr)

    if in_dtype == DType.float16:
        _cast_dispatch_out[DType.float16](
            out_buffer, in_buffer, out_dtype, size, ctx
        )
    elif in_dtype == DType.float32:
        _cast_dispatch_out[DType.float32](
            out_buffer, in_buffer, out_dtype, size, ctx
        )
    elif in_dtype == DType.float64:
        comptime if not has_apple_gpu_accelerator():
            _cast_dispatch_out[DType.float64](
                out_buffer, in_buffer, out_dtype, size, ctx
            )
    elif in_dtype == DType.bfloat16:
        _cast_dispatch_out[DType.bfloat16](
            out_buffer, in_buffer, out_dtype, size, ctx
        )
    elif in_dtype == DType.int8:
        _cast_dispatch_out[DType.int8](
            out_buffer, in_buffer, out_dtype, size, ctx
        )
    elif in_dtype == DType.int16:
        _cast_dispatch_out[DType.int16](
            out_buffer, in_buffer, out_dtype, size, ctx
        )
    elif in_dtype == DType.int32:
        _cast_dispatch_out[DType.int32](
            out_buffer, in_buffer, out_dtype, size, ctx
        )
    elif in_dtype == DType.int64:
        _cast_dispatch_out[DType.int64](
            out_buffer, in_buffer, out_dtype, size, ctx
        )
    elif in_dtype == DType.uint8:
        _cast_dispatch_out[DType.uint8](
            out_buffer, in_buffer, out_dtype, size, ctx
        )
    elif in_dtype == DType.uint16:
        _cast_dispatch_out[DType.uint16](
            out_buffer, in_buffer, out_dtype, size, ctx
        )
    elif in_dtype == DType.uint32:
        _cast_dispatch_out[DType.uint32](
            out_buffer, in_buffer, out_dtype, size, ctx
        )
    elif in_dtype == DType.uint64:
        _cast_dispatch_out[DType.uint64](
            out_buffer, in_buffer, out_dtype, size, ctx
        )
    elif in_dtype == DType.bool:
        _cast_dispatch_out[DType.bool](
            out_buffer, in_buffer, out_dtype, size, ctx
        )
    else:
        raise Error("Unsupported input dtype for cast: " + String(in_dtype))


def _cast_dispatch_out[
    in_dtype: DType
](
    out_buffer: PythonObject,
    in_buffer: PythonObject,
    out_dtype: DType,
    size: Int,
    ctx: DeviceContext,
) raises:
    """Second level dispatch for cast: dispatches on output dtype.

    Parameters:
        in_dtype: The input data type (already resolved).

    Args:
        out_buffer: The output buffer object.
        in_buffer: The input buffer object.
        out_dtype: The output data type to dispatch on.
        size: Number of elements.
        ctx: Device context pointer.
    """
    var in_ptr = _get_buffer_ptr[in_dtype](in_buffer)

    if out_dtype == DType.float16:
        unary_mixed_op[Cast, in_dtype, DType.float16](
            _get_buffer_ptr[DType.float16](out_buffer), in_ptr, size, ctx
        )
    elif out_dtype == DType.float32:
        unary_mixed_op[Cast, in_dtype, DType.float32](
            _get_buffer_ptr[DType.float32](out_buffer), in_ptr, size, ctx
        )
    elif out_dtype == DType.float64:
        comptime if not has_apple_gpu_accelerator():
            unary_mixed_op[Cast, in_dtype, DType.float64](
                _get_buffer_ptr[DType.float64](out_buffer), in_ptr, size, ctx
            )
    elif out_dtype == DType.bfloat16:
        unary_mixed_op[Cast, in_dtype, DType.bfloat16](
            _get_buffer_ptr[DType.bfloat16](out_buffer), in_ptr, size, ctx
        )
    elif out_dtype == DType.int8:
        unary_mixed_op[Cast, in_dtype, DType.int8](
            _get_buffer_ptr[DType.int8](out_buffer), in_ptr, size, ctx
        )
    elif out_dtype == DType.int16:
        unary_mixed_op[Cast, in_dtype, DType.int16](
            _get_buffer_ptr[DType.int16](out_buffer), in_ptr, size, ctx
        )
    elif out_dtype == DType.int32:
        unary_mixed_op[Cast, in_dtype, DType.int32](
            _get_buffer_ptr[DType.int32](out_buffer), in_ptr, size, ctx
        )
    elif out_dtype == DType.int64:
        unary_mixed_op[Cast, in_dtype, DType.int64](
            _get_buffer_ptr[DType.int64](out_buffer), in_ptr, size, ctx
        )
    elif out_dtype == DType.uint8:
        unary_mixed_op[Cast, in_dtype, DType.uint8](
            _get_buffer_ptr[DType.uint8](out_buffer), in_ptr, size, ctx
        )
    elif out_dtype == DType.uint16:
        unary_mixed_op[Cast, in_dtype, DType.uint16](
            _get_buffer_ptr[DType.uint16](out_buffer), in_ptr, size, ctx
        )
    elif out_dtype == DType.uint32:
        unary_mixed_op[Cast, in_dtype, DType.uint32](
            _get_buffer_ptr[DType.uint32](out_buffer), in_ptr, size, ctx
        )
    elif out_dtype == DType.uint64:
        unary_mixed_op[Cast, in_dtype, DType.uint64](
            _get_buffer_ptr[DType.uint64](out_buffer), in_ptr, size, ctx
        )
    elif out_dtype == DType.bool:
        unary_mixed_op[Cast, in_dtype, DType.bool](
            _get_buffer_ptr[DType.bool](out_buffer), in_ptr, size, ctx
        )
    else:
        raise Error("Unsupported output dtype for cast: " + String(out_dtype))


# =============================================================================
# Kernel implementations
# =============================================================================


@always_inline
def unary_mixed_op[
    op: ElementwiseUnaryMixedOp, dtype: DType, out_dtype: DType
](
    out_ptr: UnsafePointer[Scalar[out_dtype], MutUntrackedOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    size: Int,
    ctx: DeviceContext,
) raises:
    """Elementwise unary mixed-type operation: out = op(input).

    Parameters:
        op: The unary mixed-type elementwise operation to perform.
        dtype: The input data type.
        out_dtype: The output data type.

    Args:
        out_ptr: Pointer to the output buffer data.
        in_ptr: Pointer to the input buffer data.
        size: Number of elements to process.
        ctx: Device context.
    """

    @always_inline
    def func[width: Int, alignment: Int = 1](idx: Coord) {var}:
        var i = Int(idx[0].value())

        var res = op.elementwise[dtype, out_dtype, width](
            in_ptr.load[width=width](i)
        )
        out_ptr.store[width=width](i, res)

    if ctx.api() == "cpu":
        elementwise[simd_width=simd_width_of[dtype]()](func, Coord(size), ctx)
    else:
        # GPU execution - check GPU availability and op/dtype support
        comptime if has_accelerator():
            comptime if _is_gpu_allowed_mixed_unary_op[
                op
            ]() and dtype != DType.float64:
                elementwise[simd_width=1, target="gpu"](func, Coord(size), ctx)
            else:
                raise Error(
                    "GPU execution not supported for this mixed-type unary"
                    " op or dtype"
                )
        else:
            raise Error("No GPU accelerator available")
