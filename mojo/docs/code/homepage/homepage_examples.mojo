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

# Note: this contains snippets currently displayed on the modular homepage,
# outside of the purview of the docsite. We keep them here so we're notified
# if they break.

from std.algorithm import vectorize
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.host.info import is_cpu, is_gpu
from std.math import ceildiv
from std.python import Python, PythonObject
from std.runtime.asyncrt import DeviceContextPtr
from std.sys import has_accelerator, simd_width_of

from layout import Layout, LayoutTensor
from tensor import InputTensor, OutputTensor


comptime float_dtype = DType.float32

comptime size = 8

comptime layout = Layout.row_major(size)

# GPU programming example


def vector_add(
    result: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    a: LayoutTensor[float_dtype, layout, MutAnyOrigin],
    b: LayoutTensor[float_dtype, layout, MutAnyOrigin],
):
    i = global_idx.x
    if i < size:
        result[i] = a[i] + b[i]


def run_gpu_programming_example() raises:
    comptime assert (
        has_accelerator()
    ), "This example requires a supported accelerator"

    var ctx = DeviceContext()
    var a_buffer = ctx.enqueue_create_buffer[float_dtype](
        comptime (layout.size())
    )
    var b_buffer = ctx.enqueue_create_buffer[float_dtype](
        comptime (layout.size())
    )
    var result_buffer = ctx.enqueue_create_buffer[float_dtype](
        comptime (layout.size())
    )

    # Map input buffers to host to fill with values from CPU
    with a_buffer.map_to_host() as host_buffer:
        var a_tensor = LayoutTensor[float_dtype, layout](host_buffer)
        for i in range(size):
            a_tensor[i] = Float32(i)
        print("a vector:", a_tensor)

    with b_buffer.map_to_host() as host_buffer:
        var b_tensor = LayoutTensor[float_dtype, layout](host_buffer)
        for i in range(size):
            b_tensor[i] = Float32(i)
        print("b vector:", b_tensor)

    # Wrap device buffers in `LayoutTensor`
    var a_tensor = LayoutTensor[float_dtype, layout](a_buffer)
    var b_tensor = LayoutTensor[float_dtype, layout](b_buffer)
    var result_tensor = LayoutTensor[float_dtype, layout](result_buffer)

    # The grid is divided up into blocks, making sure there's an extra
    # full block for any remainder. This hasn't been tuned for any specific
    # GPU.
    comptime BLOCK_SIZE = 16
    comptime num_blocks = ceildiv(size, BLOCK_SIZE)

    # Launch the compiled function on the GPU. The target device is specified
    # first, followed by all function arguments. The last two named parameters
    # are the dimensions of the grid in blocks, and the block dimensions.
    ctx.enqueue_function[vector_add, vector_add](
        result_tensor,
        a_tensor,
        b_tensor,
        grid_dim=(num_blocks),
        block_dim=(BLOCK_SIZE),
    )

    # Move the output tensor back onto the CPU so that we can read the results.
    with result_buffer.map_to_host() as host_buffer:
        var host_tensor = LayoutTensor[float_dtype, layout](host_buffer)
        print("Resulting vector:", host_tensor)


# Python interop example


def mojo_square_array(array_obj: PythonObject) raises:
    comptime simd_width = simd_width_of[DType.int64]()
    ptr = array_obj.ctypes.data.unsafe_get_as_pointer[DType.int64]()

    def pow[width: Int](i: Int) {mut ptr}:
        elem = ptr.load[width=width](i)
        ptr.store[width=width](i, elem * elem)

    vectorize[simd_width](len(array_obj), pow)


def run_python_interop_example() raises:
    np = Python.import_module("numpy")
    # values: List[Int64] = [1, 3, 5, 7, 11]
    pylist = Python.list(1, 3, 5, 7, 11)
    nparray = np.array(pylist, np.int64)
    mojo_square_array(nparray)
    print(nparray)


# Metaprogramming example


# This is basically copy-pasted from the max custom ops example,
# and inserted here to make sure the edited code still compiles.
# The @compiler.register decorator needs to be commented out as it only
# works in a custom ops context.
# @compiler.register("vector_addition")
struct VectorAddition:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[rank=1, ...],
        lhs: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        rhs: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContextPtr,
    ) raises:
        comptime if is_cpu(target):
            vector_addition_cpu(output, lhs, rhs, ctx)
        elif is_gpu(target):
            vector_addition_gpu(output, lhs, rhs, ctx)
        else:
            raise Error("No known target:", target)


def vector_addition_gpu(
    result: OutputTensor[...],
    lhs: InputTensor[...],
    rhs: InputTensor[...],
    ctx: DeviceContextPtr,
):
    pass


def vector_addition_cpu(
    result: OutputTensor[...],
    lhs: InputTensor[...],
    rhs: InputTensor[...],
    ctx: DeviceContextPtr,
):
    pass


def main() raises:
    run_gpu_programming_example()
    run_python_interop_example()
