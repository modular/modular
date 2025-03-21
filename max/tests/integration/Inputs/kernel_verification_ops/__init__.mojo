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
        out: OutputTensor,
        x: InputTensor[type = out.type, rank = out.rank],
        y: InputTensor[type = out.type, rank = out.rank],
    ):
        out[0] = x[0] + y[0]

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
        out: OutputTensor,
        x: InputTensor[type = out.type, rank = out.rank],
        ctx: DeviceContextPtr,
    ):
        out[0] = x[0]

    @staticmethod
    fn shape(
        x: InputTensor,
    ) raises -> IndexList[x.rank]:
        raise "NotImplemented"


@compiler.register("op_with_wrong_device_context_pos")
struct OpWithWrongDeviceContextPos:
    @staticmethod
    fn execute(
        out: OutputTensor,
        ctx: DeviceContextPtr,
        x: InputTensor[type = out.type, rank = out.rank],
    ):
        out[0] = x[0]

    @staticmethod
    fn shape(
        x: InputTensor,
    ) raises -> IndexList[x.rank]:
        raise "NotImplemented"


@compiler.register("op_with_multiples_device_context")
struct OpWithMultiplesDeviceContext:
    @staticmethod
    fn execute(
        out: OutputTensor,
        x: InputTensor[type = out.type, rank = out.rank],
        ctx: DeviceContextPtr,
        ctx1: DeviceContextPtr,
    ):
        out[0] = x[0]

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
        out1: OutputTensor[type = out0.type, rank = out0.rank],
        x: InputTensor[type = out0.type, rank = out0.rank],
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
    fn execute(x: InputTensor[type = DType.int32, rank=1]) -> MyIntMemory:
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
    fn execute(x: InputTensor[type = DType.int32, rank=1]) -> MyIntReg:
        return MyIntReg(Int(x[0]))


@compiler.register("op_with_return_tensor")
struct OpWithReturnTensor:
    @staticmethod
    fn execute(
        x: InputTensor,
    ) -> DynamicTensor[x.type, 1].Type:
        var res = DynamicTensor[x.type, 1].Type(x._ptr, IndexList[1](1))
        res[0] = x[0]
        return res

    @staticmethod
    fn shape(
        x: InputTensor,
    ) raises -> IndexList[x.rank]:
        raise "NotImplemented"


@compiler.register("variadic_input_to_output")
struct VariadicInputToOutput:
    @staticmethod
    fn execute[
        type: DType,
        size: Int,
    ](
        output: OutputVariadicTensors[type, rank=1, size=size],
        bias: InputTensor[type=type, rank=1],
        input: InputVariadicTensors[type, rank=1, size=size],
    ):
        @parameter
        for i in range(size):
            for j in range(input[i].size()):
                output[i][j] = input[i][j]
            output[i][0] += bias[0]


@compiler.register("multiple_variadic_inputs")
struct MultipleVariadicInputs:
    @staticmethod
    fn execute[
        type: DType,
        size: Int,
        size1: Int,
    ](
        out: OutputTensor[type=type, rank=1],
        input0: InputVariadicTensors[type=type, rank=1, size=size],
        input1: InputVariadicTensors[type=type, rank=1, size=size1],
    ):
        @parameter
        for i in range(size):
            for j in range(out.size()):
                out[j] += input0[i][j]

        @parameter
        for i in range(size1):
            for j in range(out.size()):
                out[j] += input1[i][j]


@compiler.register("multiple_variadic_outputs")
struct MultipleVariadicOutputs:
    @staticmethod
    fn execute[
        type: DType,
        size: Int,
        size1: Int,
    ](
        out0: OutputVariadicTensors[type=type, rank=1, size=size],
        out1: OutputVariadicTensors[type=type, rank=1, size=size1],
        x: InputTensor[type=type, rank=1],
    ):
        @parameter
        for i in range(size):
            for j in range(x.size()):
                out0[i][j] = x[j] * i

        @parameter
        for i in range(size1):
            for j in range(x.size()):
                out1[i][j] = -x[j] * i


@compiler.register("variadic_add")
struct VariadicAdd:
    @staticmethod
    fn execute[
        type: DType,
        size: Int,
    ](
        out: OutputTensor[type=type, rank=1],
        bias: InputTensor[type=type, rank=1],
        input: InputVariadicTensors[type, rank=1, size=size],
    ):
        for i in range(out.size()):
            out[i] = bias[i]

            @parameter
            for j in range(size):
                out[i] += input[j][i]


@compiler.register("binary_kernel_with_raises")
struct BinaryKernelWithRaises:
    @staticmethod
    fn execute(
        out: OutputTensor,
        x: InputTensor[type = out.type, rank = out.rank],
        y: InputTensor[type = out.type, rank = out.rank],
    ) raises:
        out[0] = x[0] + y[0]

    @staticmethod
    fn shape(
        x: InputTensor,
        y: InputTensor,
    ) raises -> IndexList[x.rank]:
        raise "NotImplemented"


@compiler.register("make_my_int_memory_with_raises")
struct MakeMyIntMemoryWithRaises:
    @staticmethod
    fn execute(
        x: InputTensor[type = DType.int32, rank=1]
    ) raises -> MyIntMemory:
        return MyIntMemory(Int(x[0]))


@compiler.register("make_my_int_reg_with_raises")
struct MakeMyIntRegWithRaises:
    @staticmethod
    fn execute(x: InputTensor[type = DType.int32, rank=1]) raises -> MyIntReg:
        return MyIntReg(Int(x[0]))


@compiler.register("mutable_input_tensor")
struct MutableInputTensorKernel:
    @staticmethod
    fn execute(in_place_tensor: MutableInputTensor) raises:
        in_place_tensor._ptr.store(0, 0)
