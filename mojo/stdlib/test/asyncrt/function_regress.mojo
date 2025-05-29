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

from sys import is_gpu

from asyncrt_test_utils import create_test_device_context, expect_eq
from builtin.device_passable import DevicePassable
from gpu import *
from gpu.host import DeviceContext
from memory import UnsafePointer

alias T = DType.float64
alias S = Scalar[T]


@register_passable("trivial")
trait MaybeZeroSized:
    fn value(self) -> S:
        ...


@fieldwise_init
@register_passable("trivial")
struct ZeroSized(MaybeZeroSized, DevicePassable, Writable):
    alias device_type: AnyTrivialRegType = Self

    fn _to_device_type(self, target: UnsafePointer[NoneType]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return "ZeroSized"

    @staticmethod
    fn get_device_type_name() -> String:
        return Self.get_type_name()

    @always_inline
    fn value(self) -> S:
        return 2

    fn write_to[W: Writer](self, mut writer: W):
        constrained[
            not is_gpu(),
            "ZeroSized is not supported on GPUs",
        ]()
        writer.write("ZeroSized(")
        writer.write(self.value())
        writer.write(")")


@fieldwise_init
@register_passable("trivial")
struct NotZeroSized(MaybeZeroSized, DevicePassable, Writable):
    alias device_type: AnyTrivialRegType = Self

    fn _to_device_type(self, target: UnsafePointer[NoneType]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return "ZeroSized"

    @staticmethod
    fn get_device_type_name() -> String:
        return Self.get_type_name()

    var val: S

    fn __init__(out self):
        self.val = 2

    @always_inline
    fn value(self) -> S:
        return self.val

    fn write_to[W: Writer](self, mut writer: W):
        constrained[
            not is_gpu(),
            "ZeroSized is not supported on GPUs",
        ]()
        writer.write("NotZeroSized(")
        writer.write(self.value())
        writer.write(")")


fn _vec_func_zero(
    zs: ZeroSized,
    in0: UnsafePointer[S],
    in1: UnsafePointer[S],
    out: UnsafePointer[S],
    len: Int,
):
    var tid = global_idx.x
    if tid >= len:
        return
    out[tid] = in0[tid] + in1[tid] + zs.value()


fn _vec_func_not_zero(
    zs: NotZeroSized,
    in0: UnsafePointer[S],
    in1: UnsafePointer[S],
    out: UnsafePointer[S],
    len: Int,
):
    var tid = global_idx.x
    if tid >= len:
        return
    out[tid] = in0[tid] + in1[tid] + zs.value()


fn _vec_func[
    zero_sized_t: MaybeZeroSized
](
    zs: zero_sized_t,
    in0: UnsafePointer[S],
    in1: UnsafePointer[S],
    out: UnsafePointer[S],
    len: Int,
):
    var tid = global_idx.x
    if tid >= len:
        return
    out[tid] = in0[tid] + in1[tid] + zs.value()


fn test_function_compilation(ctx: DeviceContext) raises:
    print("-------")
    print("Running test_function_unchecked(" + ctx.name() + "):")

    # Compile all combinations with and without declaring the trait in
    # the signature.

    print("Compiling _vec_func[NotZeroSized]")
    var compiled_vec_func_0 = ctx.compile_function[_vec_func[NotZeroSized]]()

    print("Compiling _vec_func[ZeroSizet]")
    var compiled_vec_func_1 = ctx.compile_function[_vec_func[ZeroSized]]()

    print("Compiling _vec_func_not_zero")
    var compiled_vec_func_2 = ctx.compile_function[_vec_func_not_zero]()

    print("Compiling _vec_func_zero")
    var compiled_vec_func_3 = ctx.compile_function[_vec_func_zero]()

    _ = compiled_vec_func_0
    _ = compiled_vec_func_1
    _ = compiled_vec_func_2
    _ = compiled_vec_func_3


fn test_function_unchecked(ctx: DeviceContext) raises:
    print("-------")
    print("Running test_function_unchecked(" + ctx.name() + "):")

    alias length = 1024
    alias block_dim = 32

    alias zero_sized_t = NotZeroSized
    alias vec_func = _vec_func[zero_sized_t]
    # alias vec_func = _vec_func_not_zero
    # alias vec_func = _vec_func_zero

    var zs = zero_sized_t()
    print(zs)

    var scalar: S = 2

    # Initialize the input and outputs with known values.
    var in0 = ctx.enqueue_create_buffer[T](length)
    var out = ctx.enqueue_create_buffer[T](length)
    with in0.map_to_host() as in0_host, out.map_to_host() as out_host:
        for i in range(length):
            in0_host[i] = i
            out_host[i] = length + i
    var in1 = ctx.enqueue_create_buffer[T](length).enqueue_fill(scalar)

    print("compiling vec_func")
    var compiled_vec_func = ctx.compile_function_unchecked[vec_func]()
    print("calling vec_func")
    ctx.enqueue_function_unchecked(
        compiled_vec_func,
        zs,
        in0,
        in1,
        out,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    with out.map_to_host() as out_host:
        for i in range(length):
            if i < 10:
                print("at index", i, "the value is", out_host[i])
            expect_eq(
                out_host[i],
                i + 4,
                "at index ",
                i,
                " the value is ",
                out_host[i],
            )

    print("Done.")


fn test_function_checked(ctx: DeviceContext) raises:
    print("-------")
    print("Running test_function_checked(" + ctx.name() + "):")

    alias length = 1024
    alias block_dim = 32

    alias zero_sized_t = ZeroSized
    alias vec_func = _vec_func[zero_sized_t]
    # alias vec_func = _vec_func_not_zero
    # alias vec_func = _vec_func_zero

    var zs = zero_sized_t()
    print(zs)

    var scalar: S = 2

    # Initialize the input and outputs with known values.
    var in0 = ctx.enqueue_create_buffer[T](length)
    var out = ctx.enqueue_create_buffer[T](length)
    with in0.map_to_host() as in0_host, out.map_to_host() as out_host:
        for i in range(length):
            in0_host[i] = i
            out_host[i] = length + i
    var in1 = ctx.enqueue_create_buffer[T](length).enqueue_fill(scalar)

    print("compiling vec_func")
    var compiled_vec_func = ctx.compile_function_checked[vec_func, vec_func]()
    print("calling vec_func")
    ctx.enqueue_function_checked(
        compiled_vec_func,
        zs,
        in0,
        in1,
        out,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    with out.map_to_host() as out_host:
        for i in range(length):
            if i < 10:
                print("at index", i, "the value is", out_host[i])
            expect_eq(
                out_host[i],
                i + 4,
                "at index ",
                i,
                " the value is ",
                out_host[i],
            )

    print("Done.")


fn main() raises:
    var ctx = create_test_device_context()
    test_function_compilation(ctx)
    test_function_unchecked(ctx)
    test_function_checked(ctx)
