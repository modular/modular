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

"""Mojo kernel wrapper for the Select (ternary) MO interpreter operation.

Performs elementwise ``out = cond ? true_val : false_val``. The binary
comparison ops that previously shared this module now route through the graph
compiler (see ``elementwise_binary_gc``); only the ternary select remains
hand-rolled.
"""

from std.os import abort
from std.gpu.host import DeviceContext
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys.info import has_accelerator, simd_width_of
from std.utils.coord import Coord

from std.algorithm.functional import elementwise

from builtin_kernels import Select

from op_utils import _get_dtype, _get_buffer_ptr, _get_size, _get_ctx


# =============================================================================
# Python bindings
# =============================================================================


@export
def PyInit_select_ops() abi("C") -> PythonObject:
    """Create a Python module with the select kernel function binding."""
    try:
        var b = PythonModuleBuilder("select_ops")

        # Select operation (ternary: cond ? x : y)
        b.def_function[select_dispatcher](
            "Select", docstring="Elementwise select (cond ? x : y)"
        )

        return b.finalize()
    except e:
        abort(t"failed to create select op bindings module: {e}")


# =============================================================================
# Dispatchers
# =============================================================================


def select_dispatcher(
    out_buffer: PythonObject,
    cond_buffer: PythonObject,
    true_buffer: PythonObject,
    false_buffer: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    """Select dispatcher with dtype dispatch.

    Performs element-wise: out = cond ? true_val : false_val.

    Args:
        out_buffer: The output buffer object.
        cond_buffer: Boolean condition buffer.
        true_buffer: Values selected where condition is true.
        false_buffer: Values selected where condition is false.
        device_context_ptr: Device context pointer.
    """
    var dtype = _get_dtype(true_buffer)
    var false_dtype = _get_dtype(false_buffer)
    if dtype != false_dtype:
        raise Error(
            "Mismatched input dtypes for select: "
            + String(dtype)
            + " and "
            + String(false_dtype)
        )

    var cond_dtype = _get_dtype(cond_buffer)
    if cond_dtype != DType.bool:
        raise Error("Select condition must be bool, got: " + String(cond_dtype))

    var size = _get_size(out_buffer)
    var ctx = _get_ctx(device_context_ptr)

    if dtype == DType.float16:
        select_elementwise_op[DType.float16](
            _get_buffer_ptr[DType.float16](out_buffer),
            _get_buffer_ptr[DType.bool](cond_buffer),
            _get_buffer_ptr[DType.float16](true_buffer),
            _get_buffer_ptr[DType.float16](false_buffer),
            size,
            ctx,
        )
    elif dtype == DType.float32:
        select_elementwise_op[DType.float32](
            _get_buffer_ptr[DType.float32](out_buffer),
            _get_buffer_ptr[DType.bool](cond_buffer),
            _get_buffer_ptr[DType.float32](true_buffer),
            _get_buffer_ptr[DType.float32](false_buffer),
            size,
            ctx,
        )
    elif dtype == DType.float64:
        select_elementwise_op[DType.float64](
            _get_buffer_ptr[DType.float64](out_buffer),
            _get_buffer_ptr[DType.bool](cond_buffer),
            _get_buffer_ptr[DType.float64](true_buffer),
            _get_buffer_ptr[DType.float64](false_buffer),
            size,
            ctx,
        )
    elif dtype == DType.bfloat16:
        select_elementwise_op[DType.bfloat16](
            _get_buffer_ptr[DType.bfloat16](out_buffer),
            _get_buffer_ptr[DType.bool](cond_buffer),
            _get_buffer_ptr[DType.bfloat16](true_buffer),
            _get_buffer_ptr[DType.bfloat16](false_buffer),
            size,
            ctx,
        )
    elif dtype == DType.int8:
        select_elementwise_op[DType.int8](
            _get_buffer_ptr[DType.int8](out_buffer),
            _get_buffer_ptr[DType.bool](cond_buffer),
            _get_buffer_ptr[DType.int8](true_buffer),
            _get_buffer_ptr[DType.int8](false_buffer),
            size,
            ctx,
        )
    elif dtype == DType.int16:
        select_elementwise_op[DType.int16](
            _get_buffer_ptr[DType.int16](out_buffer),
            _get_buffer_ptr[DType.bool](cond_buffer),
            _get_buffer_ptr[DType.int16](true_buffer),
            _get_buffer_ptr[DType.int16](false_buffer),
            size,
            ctx,
        )
    elif dtype == DType.int32:
        select_elementwise_op[DType.int32](
            _get_buffer_ptr[DType.int32](out_buffer),
            _get_buffer_ptr[DType.bool](cond_buffer),
            _get_buffer_ptr[DType.int32](true_buffer),
            _get_buffer_ptr[DType.int32](false_buffer),
            size,
            ctx,
        )
    elif dtype == DType.int64:
        select_elementwise_op[DType.int64](
            _get_buffer_ptr[DType.int64](out_buffer),
            _get_buffer_ptr[DType.bool](cond_buffer),
            _get_buffer_ptr[DType.int64](true_buffer),
            _get_buffer_ptr[DType.int64](false_buffer),
            size,
            ctx,
        )
    elif dtype == DType.uint8:
        select_elementwise_op[DType.uint8](
            _get_buffer_ptr[DType.uint8](out_buffer),
            _get_buffer_ptr[DType.bool](cond_buffer),
            _get_buffer_ptr[DType.uint8](true_buffer),
            _get_buffer_ptr[DType.uint8](false_buffer),
            size,
            ctx,
        )
    elif dtype == DType.uint16:
        select_elementwise_op[DType.uint16](
            _get_buffer_ptr[DType.uint16](out_buffer),
            _get_buffer_ptr[DType.bool](cond_buffer),
            _get_buffer_ptr[DType.uint16](true_buffer),
            _get_buffer_ptr[DType.uint16](false_buffer),
            size,
            ctx,
        )
    elif dtype == DType.uint32:
        select_elementwise_op[DType.uint32](
            _get_buffer_ptr[DType.uint32](out_buffer),
            _get_buffer_ptr[DType.bool](cond_buffer),
            _get_buffer_ptr[DType.uint32](true_buffer),
            _get_buffer_ptr[DType.uint32](false_buffer),
            size,
            ctx,
        )
    elif dtype == DType.uint64:
        select_elementwise_op[DType.uint64](
            _get_buffer_ptr[DType.uint64](out_buffer),
            _get_buffer_ptr[DType.bool](cond_buffer),
            _get_buffer_ptr[DType.uint64](true_buffer),
            _get_buffer_ptr[DType.uint64](false_buffer),
            size,
            ctx,
        )
    elif dtype == DType.bool:
        select_elementwise_op[DType.bool](
            _get_buffer_ptr[DType.bool](out_buffer),
            _get_buffer_ptr[DType.bool](cond_buffer),
            _get_buffer_ptr[DType.bool](true_buffer),
            _get_buffer_ptr[DType.bool](false_buffer),
            size,
            ctx,
        )
    else:
        raise Error("Unsupported dtype for select operation: " + String(dtype))


# =============================================================================
# Kernel implementation
# =============================================================================


@always_inline
def select_elementwise_op[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    cond_ptr: UnsafePointer[Scalar[DType.bool], MutUntrackedOrigin],
    true_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    false_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    size: Int,
    ctx: DeviceContext,
) raises:
    """Select elementwise operation: out = cond ? true_val : false_val.

    Parameters:
        dtype: The data type of the value arrays.

    Args:
        out_ptr: Pointer to the output buffer data.
        cond_ptr: Pointer to the condition buffer data (bool).
        true_ptr: Pointer to the true-case buffer data.
        false_ptr: Pointer to the false-case buffer data.
        size: Number of elements to process.
        ctx: Device context.
    """

    @always_inline
    def func[width: Int, alignment: Int = 1](idx: Coord) {var}:
        var i = Int(idx[0].value())

        var cond = cond_ptr.load[width=width](i)
        var tc = true_ptr.load[width=width](i)
        var fc = false_ptr.load[width=width](i)
        var res = Select.elementwise[DType.bool, dtype, width](cond, tc, fc)
        out_ptr.store[width=width](i, res)

    if ctx.api() == "cpu":
        elementwise[simd_width=simd_width_of[dtype]()](func, Coord(size), ctx)
    else:
        # GPU execution
        comptime if has_accelerator():
            comptime if dtype != DType.float64:
                elementwise[simd_width=1, target="gpu"](func, Coord(size), ctx)
            else:
                raise Error(
                    "GPU execution not supported for select with dtype float64"
                )
        else:
            raise Error("No GPU accelerator available")
