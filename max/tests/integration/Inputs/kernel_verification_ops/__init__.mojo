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

import extensibility
from max.gpu.host import DeviceContext, Dim
from max.gpu.host import DeviceExternalFunction
from max.gpu import global_idx
from std.math import ceildiv
from std.os import abort, getenv
from extensibility import (
    Input,
    MutableInput,
    Output,
    Tensor,
    TensorTuple,
    TileTensorable,
)
from layout import Coord
from std.utils.index import IndexList


@extensibility.register("my_add")
struct MyAdd:
    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        x: Some[Input & Tensor],
        y: Some[Input & Tensor],
    ):
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(x), TileTensorable)
        comptime assert conforms_to(type_of(y), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(x).dtype == dtype
        comptime assert type_of(y).dtype == dtype
        comptime assert type_of(x).rank == type_of(output).rank
        comptime assert type_of(y).rank == type_of(output).rank
        var x_tt = x.to_tile_tensor[dtype]().as_imm()
        var y_tt = y.to_tile_tensor[dtype]().as_imm()
        output.to_tile_tensor[dtype]().store[width=1](
            Coord(0),
            x_tt.load[width=1](Coord(0)) + y_tt.load[width=1](Coord(0)),
        )


@extensibility.register_shape_function("my_add")
def my_add_shape(
    x: Some[Tensor],
    y: Some[Tensor],
) raises -> IndexList[type_of(x).rank]:
    raise "NotImplemented"


@extensibility.register("my_add_cpu_api", type="cpu", api="cpu")
struct MyAddCpuApi:
    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        x: Some[Input & Tensor],
        y: Some[Input & Tensor],
    ):
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(x), TileTensorable)
        comptime assert conforms_to(type_of(y), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(x).dtype == dtype
        comptime assert type_of(y).dtype == dtype
        comptime assert type_of(x).rank == type_of(output).rank
        comptime assert type_of(y).rank == type_of(output).rank
        var x_tt = x.to_tile_tensor[dtype]().as_imm()
        var y_tt = y.to_tile_tensor[dtype]().as_imm()
        output.to_tile_tensor[dtype]().store[width=1](
            Coord(0),
            x_tt.load[width=1](Coord(0)) + y_tt.load[width=1](Coord(0)),
        )


@extensibility.register("op_with_device_context")
struct OpWidthDeviceContext:
    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        x: Some[Input & Tensor],
        ctx: DeviceContext,
    ):
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(x), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(x).dtype == dtype
        comptime assert type_of(x).rank == type_of(output).rank
        output.to_tile_tensor[dtype]().store[width=1](
            Coord(0), x.to_tile_tensor[dtype]().as_imm().load[width=1](Coord(0))
        )


@extensibility.register_shape_function("op_with_device_context")
def op_with_device_context_shape(
    x: Some[Tensor],
) raises -> IndexList[type_of(x).rank]:
    raise "NotImplemented"


@extensibility.register("op_with_multiple_outputs")
struct OpWithMultipleOutputs:
    @staticmethod
    def execute(
        out0: Some[Output & Tensor],
        out1: Some[Output & Tensor],
        x: Some[Input & Tensor],
    ):
        comptime assert conforms_to(type_of(out0), TileTensorable)
        comptime assert conforms_to(type_of(out1), TileTensorable)
        comptime assert conforms_to(type_of(x), TileTensorable)
        comptime dtype = type_of(out0).dtype
        comptime assert type_of(out1).dtype == dtype
        comptime assert type_of(x).dtype == dtype
        comptime assert type_of(out1).rank == type_of(out0).rank
        comptime assert type_of(x).rank == type_of(out0).rank
        var x0 = x.to_tile_tensor[dtype]().as_imm().load[width=1](Coord(0))
        out0.to_tile_tensor[dtype]().store[width=1](Coord(0), 2 * x0)
        out1.to_tile_tensor[dtype]().store[width=1](Coord(0), 4 * x0)


@extensibility.register_shape_function("op_with_multiple_outputs")
def op_with_multiple_outputs_shape(
    x: Some[Tensor],
) raises -> IndexList[type_of(x).rank]:
    raise "NotImplemented"


@extensibility.register("op_without_outputs")
struct OpWithoutOutputs:
    @staticmethod
    def execute(
        x: Some[Input & Tensor],
    ):
        comptime assert conforms_to(type_of(x), TileTensorable)
        print(x.to_tile_tensor().as_imm().load[width=1](Coord(0)))


struct MyIntMemory(Movable):
    var val: Int

    def __init__(out self, val: Int):
        self.val = val

    def __deinit__(deinit self):
        print("MyInt ")


@extensibility.register("make_my_int_memory")
struct MakeMyIntMemory:
    @staticmethod
    def execute(x: Some[Input & Tensor]) -> MyIntMemory:
        comptime assert conforms_to(type_of(x), TileTensorable)
        comptime assert type_of(x).dtype == .int32
        comptime assert type_of(x).rank == 1
        return MyIntMemory(
            Int(x.to_tile_tensor[.int32]().as_imm().load[width=1](Coord(0)))
        )


@fieldwise_init
struct MyIntReg(TrivialRegisterPassable):
    var val: Int


@extensibility.register("make_my_int_reg")
struct MakeMyIntReg:
    @staticmethod
    def execute(x: Some[Input & Tensor]) -> MyIntReg:
        comptime assert conforms_to(type_of(x), TileTensorable)
        comptime assert type_of(x).dtype == .int32
        comptime assert type_of(x).rank == 1
        return MyIntReg(
            Int(x.to_tile_tensor[.int32]().as_imm().load[width=1](Coord(0)))
        )


@extensibility.register("variadic_input_to_output")
struct VariadicInputToOutput:
    @staticmethod
    def execute[
        Outs: TypeList[Trait=Output & Tensor, ...],
        Ins: TypeList[Trait=Input & Tensor, ...],
        //,
    ](
        output: TensorTuple[*Outs],
        bias: Some[Input & Tensor],
        input: TensorTuple[*Ins],
    ):
        comptime assert conforms_to(type_of(bias), TileTensorable)
        comptime dtype = type_of(bias).dtype
        comptime assert type_of(bias).rank == 1
        comptime assert len(Outs) == len(Ins)
        var bias0 = (
            bias.to_tile_tensor[dtype]().as_imm().load[width=1](Coord(0))
        )
        comptime for i in range(len(Ins)):
            comptime assert conforms_to(type_of(output[i]), TileTensorable)
            comptime assert conforms_to(type_of(input[i]), TileTensorable)
            comptime assert type_of(output[i]).dtype == dtype
            comptime assert type_of(input[i]).dtype == dtype
            comptime assert type_of(output[i]).rank == 1
            comptime assert type_of(input[i]).rank == 1
            var out_tt = output[i].to_tile_tensor[dtype]()
            var in_tt = input[i].to_tile_tensor[dtype]().as_imm()
            for j in range(Int(input[i].shape()[0].value())):
                out_tt.store[width=1](Coord(j), in_tt.load[width=1](Coord(j)))
            out_tt.store[width=1](
                Coord(0), out_tt.load[width=1](Coord(0)) + bias0
            )


@extensibility.register("variadic_add")
struct VariadicAdd:
    @staticmethod
    def execute[
        Ins: TypeList[Trait=Input & Tensor, ...],
        //,
    ](
        output: Some[Output & Tensor],
        bias: Some[Input & Tensor],
        input: TensorTuple[*Ins],
    ):
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(bias), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(output).rank == 1
        comptime assert type_of(bias).dtype == dtype
        comptime assert type_of(bias).rank == 1
        var out_tt = output.to_tile_tensor[dtype]()
        var bias_tt = bias.to_tile_tensor[dtype]().as_imm()
        for i in range(Int(output.shape()[0].value())):
            var acc = bias_tt.load[width=1](Coord(i))

            comptime for j in range(len(Ins)):
                comptime assert conforms_to(type_of(input[j]), TileTensorable)
                comptime assert type_of(input[j]).dtype == dtype
                comptime assert type_of(input[j]).rank == 1
                acc += (
                    input[j]
                    .to_tile_tensor[dtype]()
                    .as_imm()
                    .load[width=1](Coord(i))
                )
            out_tt.store[width=1](Coord(i), acc)


@extensibility.register("binary_kernel_with_raises")
struct BinaryKernelWithRaises:
    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        x: Some[Input & Tensor],
        y: Some[Input & Tensor],
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(x), TileTensorable)
        comptime assert conforms_to(type_of(y), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(x).dtype == dtype
        comptime assert type_of(y).dtype == dtype
        comptime assert type_of(x).rank == type_of(output).rank
        comptime assert type_of(y).rank == type_of(output).rank
        var x_tt = x.to_tile_tensor[dtype]().as_imm()
        var y_tt = y.to_tile_tensor[dtype]().as_imm()
        output.to_tile_tensor[dtype]().store[width=1](
            Coord(0),
            x_tt.load[width=1](Coord(0)) + y_tt.load[width=1](Coord(0)),
        )


@extensibility.register_shape_function("binary_kernel_with_raises")
def binary_kernel_with_raises_shape(
    x: Some[Tensor],
    y: Some[Tensor],
) raises -> IndexList[type_of(x).rank]:
    raise "NotImplemented"


@extensibility.register("mutable_input_tensor")
struct MutableInputTensorKernel:
    @staticmethod
    def execute(in_place_tensor: Some[MutableInput & Tensor]) raises:
        comptime assert conforms_to(type_of(in_place_tensor), TileTensorable)
        in_place_tensor.to_tile_tensor().ptr.unsafe_store(0, 0)


@extensibility.register("op_with_int_parameter")
struct OpWithIntParameter[IntParameter: Int]:
    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        x: Some[Input & Tensor],
    ):
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(x), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(x).dtype == dtype
        comptime assert type_of(x).rank == type_of(output).rank
        output.to_tile_tensor[dtype]().store[width=1](
            Coord(0), x.to_tile_tensor[dtype]().as_imm().load[width=1](Coord(0))
        )
        print(Self.IntParameter)


@extensibility.register("op_with_dtype_parameter")
struct OpWithDTypeParameter[DTypeParameter: DType]:
    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        x: Some[Input & Tensor],
    ):
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(x), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(x).dtype == dtype
        comptime assert type_of(x).rank == type_of(output).rank
        output.to_tile_tensor[dtype]().store[width=1](
            Coord(0), x.to_tile_tensor[dtype]().as_imm().load[width=1](Coord(0))
        )
        print(Self.DTypeParameter)


@extensibility.register("op_with_string_parameter")
struct OpWithStringParameter[StringParameter: String]:
    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        x: Some[Input & Tensor],
    ):
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(x), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(x).dtype == dtype
        comptime assert type_of(x).rank == type_of(output).rank
        output.to_tile_tensor[dtype]().store[width=1](
            Coord(0), x.to_tile_tensor[dtype]().as_imm().load[width=1](Coord(0))
        )
        print(Self.StringParameter)


@extensibility.register("op_with_string_slice_parameter")
struct OpWithStringSliceParameter[StringParameter: StringSpan]:
    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        x: Some[Input & Tensor],
    ):
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(x), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(x).dtype == dtype
        comptime assert type_of(x).rank == type_of(output).rank
        output.to_tile_tensor[dtype]().store[width=1](
            Coord(0), x.to_tile_tensor[dtype]().as_imm().load[width=1](Coord(0))
        )
        print(Self.StringParameter)


# What a declarative kernel's `execute_configuration` returns: the graph
# compiler calls only `get_grid_dim()` and `get_block_dim()`.
@fieldwise_init
struct DeclarativeLaunchConfig(Copyable):
    var grid_dim: Dim
    var block_dim: Dim

    def get_grid_dim(self) -> Dim:
        return self.grid_dim

    def get_block_dim(self) -> Dim:
        return self.block_dim


# A declarative kernel has no `execute`, so it declares its parameters on its
# entry points.
@extensibility.register("declarative_op_with_int_parameter")
struct DeclarativeOpWithIntParameter:
    @staticmethod
    def execute_configuration(
        output: Some[Tensor], x: Some[Tensor]
    ) -> DeclarativeLaunchConfig:
        return DeclarativeLaunchConfig(grid_dim=Dim(1), block_dim=Dim(1))

    @staticmethod
    def execute_device[
        IntParameter: Int
    ](output: Some[Output & TileTensorable], x: Some[Input & TileTensorable]):
        pass


@extensibility.register("op_with_static_string_parameter")
struct OpWithStaticStringParameter[StringParameter: StaticString]:
    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        x: Some[Input & Tensor],
    ):
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(x), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(x).dtype == dtype
        comptime assert type_of(x).rank == type_of(output).rank
        output.to_tile_tensor[dtype]().store[width=1](
            Coord(0), x.to_tile_tensor[dtype]().as_imm().load[width=1](Coord(0))
        )
        print(Self.StringParameter)


@extensibility.register("op_with_external_cubin")
struct ExternalCubinVecAdd:
    """Custom op that uses an external cubin for vector addition."""

    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        lhs: Some[Input & Tensor],
        rhs: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert ctx.target.is_gpu()
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(lhs), TileTensorable)
        comptime assert conforms_to(type_of(rhs), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(output).rank == 1
        comptime assert type_of(lhs).dtype == dtype
        comptime assert type_of(lhs).rank == 1
        comptime assert type_of(rhs).dtype == dtype
        comptime assert type_of(rhs).rank == 1
        var gpu_ctx = ctx

        var external_func: DeviceExternalFunction[target=type_of(ctx).target]
        with open(getenv("CUBIN_PATH"), "r") as file:
            var cubin_data = file.read_bytes()

            external_func = DeviceExternalFunction(
                gpu_ctx,
                function_name="vec_add",  # matches extern "C" name
                # DeviceExternalFunction takes a StringSpan, which is probably wrong.
                # The cubin is [very, very likely] invalid UTF8.
                asm=String(StringSpan(unsafe_from_utf8=cubin_data)),
            )

        var length = Int(output.shape()[0].value())
        var block_dim = 32
        var grid_dim = ceildiv(length, block_dim)

        # Execute the external cubin kernel
        gpu_ctx.enqueue_function(
            external_func,
            lhs.to_tile_tensor[dtype]().ptr,
            rhs.to_tile_tensor[dtype]().ptr,
            output.to_tile_tensor[dtype]().ptr,
            length,
            grid_dim=(grid_dim,),
            block_dim=(block_dim,),
        )


@extensibility.register("intentional_gpu_crash")
struct IntentionalGpuCrash:
    """A custom op that launches a GPU kernel which executes a trap instruction.

    This causes a real GPU hardware fault (e.g. CUDA_ERROR_ILLEGAL_INSTRUCTION)
    to test that runtime error reporting includes source notes. Only supported
    on NVIDIA GPUs; ROCm handles GPU traps by calling host-side abort() which
    kills the process.
    """

    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        x: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert ctx.target.is_gpu()
        comptime assert type_of(output).rank == 1
        comptime assert type_of(x).dtype == type_of(output).dtype
        comptime assert type_of(x).rank == 1
        var gpu_ctx = ctx

        def crash_kernel():
            abort()

        gpu_ctx.enqueue_function[crash_kernel](grid_dim=(1,), block_dim=(1,))
        gpu_ctx.synchronize()


@extensibility.register("fill_from_host_scalar")
struct FillFromHostScalar:
    """Fills the GPU output with the value of a host-resident scalar input.

    The value is read on the host at enqueue time -- the dispatch-metadata
    pattern (attention cache lengths and friends). Under device-graph
    synthesis the read happens while recording, so the value bakes into the
    recorded kernel and the input's contents must key the graph cache:
    a changed value must re-record, never replay stale dispatch.
    """

    @staticmethod
    def execute(
        output: Some[Output & Tensor],
        h: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(h), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime assert type_of(h).dtype == dtype
        comptime assert type_of(h).rank == 1
        var value = h.to_tile_tensor[dtype]().as_imm().load[width=1](Coord(0))
        var ptr = output.to_tile_tensor[dtype]().ptr
        var length = Int32(Int(output.shape().product()))

        def fill_kernel() {var}:
            var tid = global_idx.x
            if tid >= Int(length):
                return
            ptr[unsafe_offset=tid] = value

        comptime block_dim = 256
        ctx.enqueue_function(
            fill_kernel,
            grid_dim=ceildiv(length, block_dim),
            block_dim=block_dim,
        )
