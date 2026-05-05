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
from std.reflection import reflect
from std.runtime.asyncrt import DeviceContextPtr
from std.sys import has_accelerator, simd_width_of

from layout import TileTensor
from layout.tile_layout import Layout, row_major
from tensor import InputTensor, OutputTensor


comptime float_dtype = DType.float32

comptime size = 8

comptime layout = row_major[size]()

# GPU programming example


def vector_add(
    a: TileTensor[float_dtype, type_of(layout), MutAnyOrigin],
    b: TileTensor[float_dtype, type_of(layout), MutAnyOrigin],
    result: TileTensor[float_dtype, type_of(layout), MutAnyOrigin],
    size: Int,
):
    var i = global_idx.x
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
        var a_tensor = TileTensor(host_buffer, layout)
        for i in range(size):
            a_tensor[i] = Float32(i)
        print("a vector:", a_tensor)

    with b_buffer.map_to_host() as host_buffer:
        var b_tensor = TileTensor(host_buffer, layout)
        for i in range(size):
            b_tensor[i] = Float32(i)
        print("b vector:", b_tensor)

    # Wrap device buffers in `LayoutTensor`
    var a_tensor = TileTensor(a_buffer, layout)
    var b_tensor = TileTensor(b_buffer, layout)
    var result_tensor = TileTensor(result_buffer, layout)

    # The grid is divided up into blocks, making sure there's an extra
    # full block for any remainder. This hasn't been tuned for any specific
    # GPU.
    comptime BLOCK_SIZE = 16
    comptime num_blocks = ceildiv(size, BLOCK_SIZE)

    # Launch the compiled function on the GPU. The target device is specified
    # first, followed by all function arguments. The last two named parameters
    # are the dimensions of the grid in blocks, and the block dimensions.
    ctx.enqueue_function[vector_add](
        a_tensor,
        b_tensor,
        result_tensor,
        size,
        grid_dim=(num_blocks),
        block_dim=(BLOCK_SIZE),
    )

    # Move the output tensor back onto the CPU so that we can read the results.
    with result_buffer.map_to_host() as host_buffer:
        var host_tensor = TileTensor(host_buffer, layout)
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


trait FauxEquatable(ImplicitlyDestructible):
    # Generic implementation using reflection: compare all fields
    @always_inline
    def __eq__(self, other: Self) -> Bool:
        comptime r = reflect[Self]()
        comptime names = r.field_names()
        comptime types = r.field_types()

        comptime for i in range(names.size):
            comptime T = types[i]
            comptime assert conforms_to(T, Equatable)
            if trait_downcast[Equatable](
                r.field_ref[i](self)
            ) != trait_downcast[Equatable](r.field_ref[i](other)):
                return False
        return True


@fieldwise_init
struct EqTest(Copyable, FauxEquatable):
    var i: Int
    var s: String


def run_metaprogramming_example():
    v1 = EqTest(1, "Lucy")
    v2 = EqTest(1, "Lucy")
    v3 = EqTest(2, "Linus")
    print(v1 == v2)
    print(v1 == v3)


def main() raises:
    run_gpu_programming_example()
    run_python_interop_example()
    run_metaprogramming_example()
