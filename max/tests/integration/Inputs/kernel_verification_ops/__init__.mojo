# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import compiler_internal as compiler
from tensor import (
    foreach,
    DynamicTensor,
    VariadicTensors,
    InputTensor,
    OutputTensor,
    InputVariadicTensors,
)
from tensor_internal import OutputVariadicTensors
from tensor_internal.managed_tensor_slice import (
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

    fn __moveinit__(out self, owned other: Self):
        self.val = other.val

    fn __del__(owned self):
        print("MyInt del")


@compiler.register("make_my_int_memory")
struct MakeMyIntMemory:
    @staticmethod
    fn execute(x: InputTensor[dtype = DType.int32, rank=1]) -> MyIntMemory:
        return MyIntMemory(Int(x[0]))


@value
@register_passable("trivial")
struct MyIntReg(Movable):
    var val: Int

    fn __init__(out self, val: Int):
        self.val = val


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
        @parameter
        for i in range(size):
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

            @parameter
            for j in range(size):
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
        print(IntParameter)


@compiler.register("op_with_dtype_parameter")
struct OpWithDTypeParameter[DTypeParameter: DType]:
    @staticmethod
    fn execute(
        output: OutputTensor,
        x: InputTensor[dtype = output.dtype, rank = output.rank],
    ):
        output[0] = x[0]
        print(DTypeParameter)


@compiler.register("op_with_string_parameter")
struct OpWithStringParameter[StringParameter: String]:
    @staticmethod
    fn execute(
        output: OutputTensor,
        x: InputTensor[dtype = output.dtype, rank = output.rank],
    ):
        output[0] = x[0]
        print(StringParameter)


@compiler.register("op_with_string_slice_parameter")
struct OpWithStringSliceParameter[StringParameter: StringSlice]:
    @staticmethod
    fn execute(
        output: OutputTensor,
        x: InputTensor[dtype = output.dtype, rank = output.rank],
    ):
        output[0] = x[0]
        print(StringParameter)


@compiler.register("op_with_static_string_parameter")
struct OpWithStaticStringParameter[StringParameter: StaticString]:
    @staticmethod
    fn execute(
        output: OutputTensor,
        x: InputTensor[dtype = output.dtype, rank = output.rank],
    ):
        output[0] = x[0]
        print(StringParameter)
