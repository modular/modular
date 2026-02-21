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

import compiler_internal as compiler
from gpu.host.device_context import DeviceExternalFunction
from os import getenv
from tensor import (
    foreach,
    DynamicTensor,
    VariadicTensors,
    InputTensor,
    OutputTensor,
    InputVariadicTensors,
)
from tensor import OutputVariadicTensors
from tensor.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)
from utils.index import IndexList
from runtime.asyncrt import DeviceContextPtr


@compiler.register("my_add")
struct MyAdd:
    @staticmethod
    fn execute(
        output: OutputTensor,
        x: InputTensor[dtype = output.dtype, rank = output.rank],
        y: InputTensor[dtype = output.dtype, rank = output.rank],
    ):
        output[0] = x[0] + y[0]

    @staticmethod
    fn shape(
        x: InputTensor,
        y: InputTensor,
    ) raises -> IndexList[x.rank]:
        raise "NotImplemented"


@compiler.register("op_with_device_context")
struct OpWidthDeviceContext:
    @staticmethod
    fn execute(
        output: OutputTensor,
        x: InputTensor[dtype = output.dtype, rank = output.rank],
        ctx: DeviceContextPtr,
    ):
        output[0] = x[0]

    @staticmethod
    fn shape(
        x: InputTensor,
    ) raises -> IndexList[x.rank]:
        raise "NotImplemented"


@compiler.register("op_with_multiple_outputs")
struct OpWithMultipleOutputs:
    @staticmethod
    fn execute(
        out0: OutputTensor,
        out1: OutputTensor[dtype = out0.dtype, rank = out0.rank],
        x: InputTensor[dtype = out0.dtype, rank = out0.rank],
    ):
        out0[0] = 2 * x[0]
        out1[0] = 4 * x[0]

    @staticmethod
    fn shape(
        x: InputTensor,
    ) raises -> IndexList[x.rank]:
        raise "NotImplemented"


@compiler.register("op_without_outputs")
struct OpWithoutOutputs:
    @staticmethod
    fn execute(
        x: InputTensor,
    ):
        print(x[0])


struct MyIntMemory(Movable):
    var val: Int

    fn __init__(out self, val: Int):
        self.val = val

    fn __del__(deinit self):
        print("MyInt del")


@compiler.register("make_my_int_memory")
struct MakeMyIntMemory:
    @staticmethod
    fn execute(x: InputTensor[dtype = DType.int32, rank=1]) -> MyIntMemory:
        return MyIntMemory(Int(x[0]))


@fieldwise_init
struct MyIntReg(TrivialRegisterPassable):
    var val: Int


@compiler.register("make_my_int_reg")
struct MakeMyIntReg:
    @staticmethod
    fn execute(x: InputTensor[dtype = DType.int32, rank=1]) -> MyIntReg:
        return MyIntReg(Int(x[0]))


@compiler.register("variadic_input_to_output")
struct VariadicInputToOutput:
    @staticmethod
    fn execute[
        dtype: DType,
        size: Int,
    ](
        output: OutputVariadicTensors[dtype, rank=1, size=size],
        bias: InputTensor[dtype=dtype, rank=1],
        input: InputVariadicTensors[dtype, rank=1, size=size],
    ):
        comptime for i in range(size):
            for j in range(input[i].size()):
                output[i][j] = input[i][j]
            output[i][0] += bias[0]


@compiler.register("variadic_add")
struct VariadicAdd:
    @staticmethod
    fn execute[
        dtype: DType,
        size: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=1],
        bias: InputTensor[dtype=dtype, rank=1],
        input: InputVariadicTensors[dtype, rank=1, size=size],
    ):
        for i in range(output.size()):
            output[i] = bias[i]

            comptime for j in range(size):
                output[i] += input[j][i]


@compiler.register("binary_kernel_with_raises")
struct BinaryKernelWithRaises:
    @staticmethod
    fn execute(
        output: OutputTensor,
        x: InputTensor[dtype = output.dtype, rank = output.rank],
        y: InputTensor[dtype = output.dtype, rank = output.rank],
    ) raises:
        output[0] = x[0] + y[0]

    @staticmethod
    fn shape(
        x: InputTensor,
        y: InputTensor,
    ) raises -> IndexList[x.rank]:
        raise "NotImplemented"


@compiler.register("mutable_input_tensor")
struct MutableInputTensorKernel:
    @staticmethod
    fn execute(in_place_tensor: MutableInputTensor) raises:
        in_place_tensor._ptr.store(0, 0)


@compiler.register("op_with_int_parameter")
struct OpWithIntParameter[IntParameter: Int]:
    @staticmethod
    fn execute(
        output: OutputTensor,
        x: InputTensor[dtype = output.dtype, rank = output.rank],
    ):
        output[0] = x[0]
        print(Self.IntParameter)


@compiler.register("op_with_dtype_parameter")
struct OpWithDTypeParameter[DTypeParameter: DType]:
    @staticmethod
    fn execute(
        output: OutputTensor,
        x: InputTensor[dtype = output.dtype, rank = output.rank],
    ):
        output[0] = x[0]
        print(Self.DTypeParameter)


@compiler.register("op_with_string_parameter")
struct OpWithStringParameter[StringParameter: String]:
    @staticmethod
    fn execute(
        output: OutputTensor,
        x: InputTensor[dtype = output.dtype, rank = output.rank],
    ):
        output[0] = x[0]
        print(Self.StringParameter)


@compiler.register("op_with_string_slice_parameter")
struct OpWithStringSliceParameter[StringParameter: StringSlice]:
    @staticmethod
    fn execute(
        output: OutputTensor,
        x: InputTensor[dtype = output.dtype, rank = output.rank],
    ):
        output[0] = x[0]
        print(Self.StringParameter)


@compiler.register("op_with_static_string_parameter")
struct OpWithStaticStringParameter[StringParameter: StaticString]:
    @staticmethod
    fn execute(
        output: OutputTensor,
        x: InputTensor[dtype = output.dtype, rank = output.rank],
    ):
        output[0] = x[0]
        print(Self.StringParameter)


@compiler.register("op_with_external_cubin")
struct ExternalCubinVecAdd:
    """Custom op that uses an external cubin for vector addition."""

    @staticmethod
    def execute[
        target: StaticString
    ](
        output: OutputTensor[rank=1],
        lhs: InputTensor[dtype = output.dtype, rank = output.rank],
        rhs: InputTensor[dtype = output.dtype, rank = output.rank],
        ctx: DeviceContextPtr,
    ):
        constrained[target == "gpu"]()
        gpu_ctx = ctx.get_device_context()

        with open(getenv("CUBIN_PATH"), "r") as file:
            cubin_data = file.read_bytes()

            external_func = DeviceExternalFunction(
                gpu_ctx,
                function_name="vec_add",  # matches extern "C" name
                # DeviceExternalFunction takes a StringSlice, which is probably wrong.
                # The cubin is [very, very likely] invalid UTF8.
                asm=StringSlice(unsafe_from_utf8=cubin_data),
            )

        length = output.dim_size(0)
        block_dim = 32
        grid_dim = (length + block_dim - 1) // block_dim

        # Execute the external cubin kernel
        gpu_ctx.enqueue_function(
            external_func,
            lhs.unsafe_ptr(),
            rhs.unsafe_ptr(),
            output.unsafe_ptr(),
            length,
            grid_dim=(grid_dim,),
            block_dim=(block_dim,),
        )
