# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys import external_call
from sys.info import simdwidthof

import compiler_internal as compiler
from buffer import NDBuffer
from buffer.dimlist import DimList
from compiler_internal import StaticTensorSpec
from runtime.asyncrt import DeviceContextPtr, MojoCallContextPtr
from tensor import (
    ManagedTensorSlice,
    VariadicTensors,
    foreach,
)
from tensor_internal import (
    _input_fusion_hook_impl,
    _output_fusion_hook_impl,
    simd_load_from_managed_tensor_slice,
    simd_store_into_managed_tensor_slice,
    view_copy_impl,
)
from tensor_internal import IOUnknown

from utils import IndexList, StaticTuple
from collections import Optional


# TODO(MOCO-1413): remove this need to keep imported exported funcs alive.
@export
fn export():
    alias _simd_load_from_managed_tensor_slice = simd_load_from_managed_tensor_slice
    alias _simd_store_into_managed_tensor_slice = simd_store_into_managed_tensor_slice
    alias __input_fusion_hook_impl = _input_fusion_hook_impl
    alias __output_fusion_hook_impl = _output_fusion_hook_impl


# ===-----------------------------------------------------------------------===#
# Opaque Reg Types
# ===-----------------------------------------------------------------------===#


@value
@register_passable
struct MyCustomScalarRegSI32:
    var val: Scalar[DType.int32]

    @implicit
    fn __init__(out self, val: Scalar[DType.int32]):
        print("MyCustomScalarRegSI32.__init__", val)
        self.val = val

    fn __del__(owned self):
        print("MyCustomScalarRegSI32.__del__", self.val)


# It is intentional there are no methods which consume this.
# It is here to support some level of type checking.
@value
@register_passable
struct MyCustomScalarRegF32:
    var val: Scalar[DType.float32]

    @implicit
    fn __init__(out self, val: Scalar[DType.float32]):
        print("MyCustomScalarRegF32.__init__", val)
        self.val = val

    fn __del__(owned self):
        print("MyCustomScalarRegF32.__del__", self.val)


@compiler.register("tensor_to_custom_scalar_si32_reg", num_dps_outputs=0)
struct OpaqueToCustomScalarSI32Reg:
    @staticmethod
    fn execute(
        x: ManagedTensorSlice[type = DType.int32, rank=1]
    ) -> MyCustomScalarRegSI32:
        return MyCustomScalarRegSI32(x[0])


# Adds two custom scalar types (one of which is register passable) and writes
# to a tensor
@compiler.register("opaque_add_to_tensor_si32_reg")
struct OpaqueAddToTensorSI32Reg:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](
        out: ManagedTensorSlice[type = DType.int32, rank=1],
        x: MyCustomScalarRegSI32,
        y: MyCustomScalarRegSI32,
    ):
        out[0] = x.val + y.val


@compiler.register("opaque_add_to_tensor_f32_reg")
struct OpaqueAddToTensorF32:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](
        out: ManagedTensorSlice[type = DType.float32, rank=1],
        x: MyCustomScalarRegF32,
        y: MyCustomScalarRegF32,
    ):
        out[0] = x.val + y.val


# ===-----------------------------------------------------------------------===#
# Opaque Mem. Types
# ===-----------------------------------------------------------------------===#


@value
struct MyCustomScalarSI32:
    var val: Scalar[DType.int32]

    @implicit
    fn __init__(out self, val: Scalar[DType.int32]):
        print("MyCustomScalarSI32.__init__", val)
        self.val = val

    fn __del__(owned self):
        print("MyCustomScalarSI32.__del__", self.val)


@compiler.register("tensor_to_custom_scalar_si32", num_dps_outputs=0)
struct OpaqueToCustomScalarSI32:
    @staticmethod
    fn execute(
        x: ManagedTensorSlice[type = DType.int32, rank=1]
    ) -> MyCustomScalarSI32:
        return MyCustomScalarSI32(x[0])


# Adds two custom scalar types (one of which is register passable) and writes
# to a tensor
@compiler.register("opaque_add_to_tensor_si32")
struct OpaqueAddToTensorSI32:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](
        out: ManagedTensorSlice[type = DType.int32, rank=1],
        x: MyCustomScalarSI32,
        y: MyCustomScalarSI32,
    ):
        out[0] = x.val + y.val


@compiler.register("opaque_add_to_tensor_si32_raises")
struct OpaqueAddToTensorSI32Raises:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](
        out: ManagedTensorSlice[type = DType.int32, rank=1],
        x: MyCustomScalarSI32,
        y: MyCustomScalarSI32,
    ) raises:
        out[0] = x.val + y.val


# ===-----------------------------------------------------------------------===#
# Other Kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("imposter_add")
struct Foo:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](z: ManagedTensorSlice, x: ManagedTensorSlice, y: ManagedTensorSlice):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.type, width]:
            return rebind[SIMD[z.type, width]](x.load[width](idx)) + rebind[
                SIMD[z.type, width]
            ](y.load[width](idx))

        foreach[func](z)

    @staticmethod
    fn shape(x: ManagedTensorSlice, y: ManagedTensorSlice) -> IndexList[x.rank]:
        return x.shape()


@always_inline
fn toNDBuffer[
    out_dtype: DType, out_rank: Int
](tensor: ManagedTensorSlice) -> NDBuffer[out_dtype, out_rank]:
    # TODO(GEX-734): forward other static params automatically
    return rebind[NDBuffer[out_dtype, out_rank]](
        NDBuffer[tensor.type, tensor.rank](tensor._ptr, tensor.shape())
    )


fn _matmul[
    elementwise_lambda_fn: Optional[
        fn[
            type: DType, width: Int, *, alignment: Int = 1
        ] (IndexList[2], SIMD[type, width]) capturing -> None
    ] = None,
](c: ManagedTensorSlice, a: ManagedTensorSlice, b: ManagedTensorSlice) raises:
    var m = a.dim_size(0)
    var n = a.dim_size(1)
    var k = b.dim_size(1)

    for i in range(m):
        for j in range(n):
            var c_val = Scalar[c.type](0)
            for l in range(k):
                var a_val = a.load[1](IndexList[2](i, l)).cast[c.type]()
                var b_val = b.load[1](IndexList[2](l, j)).cast[c.type]()
                c_val += a_val * b_val

            @parameter
            if elementwise_lambda_fn:
                alias func = elementwise_lambda_fn.value()
                func(IndexList[2](i, j), c_val)
            else:
                c.store[1](IndexList[2](i, j), c_val)


# c = a @ b, should support CPU and GPU
@compiler.register("imposter_matmul")
struct ImposterMatmul:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](
        c: ManagedTensorSlice,
        a: ManagedTensorSlice,
        b: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ) raises:
        _matmul(c, a, b)

    @staticmethod
    fn shape(
        a: ManagedTensorSlice,
        b: ManagedTensorSlice,
    ) -> IndexList[2]:
        var shape = a.shape()
        shape[1] = b.dim_size[1]()
        return rebind[IndexList[2]](shape)


@compiler.register("print_tensor_spec")
struct PrintTensorSpecOp:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](out: ManagedTensorSlice, x: ManagedTensorSlice):
        print("x.shape =", x._static_shape)
        print("x.strides =", x._static_strides)
        print("x.alignment =", x.alignment)
        print("x.address_space =", x.address_space)
        print("x.exclusive =", x.exclusive)

        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[out.rank]) -> SIMD[out.type, width]:
            return rebind[SIMD[out.type, width]](x.load[width](idx))

        foreach[func](out)

    @staticmethod
    fn shape(x: ManagedTensorSlice) -> IndexList[x.rank]:
        return x.shape()


@compiler.register("print_tensor_spec_view")
@compiler.view_kernel
struct PrintTensorSpecViewOp:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](out: ManagedTensorSlice, x: ManagedTensorSlice):
        print("x.shape =", x._static_shape)
        print("x.strides =", x._static_strides)
        print("x.alignment =", x.alignment)
        print("x.address_space =", x.address_space)
        print("x.exclusive =", x.exclusive)

        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[out.rank]) -> SIMD[out.type, width]:
            return rebind[SIMD[out.type, width]](x.load[width](idx))

        foreach[func](out)

    @staticmethod
    fn shape(x: ManagedTensorSlice) -> IndexList[x.rank]:
        return x.shape()


@compiler.register("print_tensor_spec_fused")
@compiler.elementwise
struct PrintTensorSpecFusedOp:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](out: ManagedTensorSlice, x: ManagedTensorSlice):
        print("x.shape =", x._static_shape)
        print("x.strides =", x._static_strides)
        print("x.alignment =", x.alignment)
        print("x.address_space =", x.address_space)
        print("x.exclusive =", x.exclusive)

        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[out.rank]) -> SIMD[out.type, width]:
            return rebind[SIMD[out.type, width]](x._fused_load[width](idx))

        foreach[func](out)

    @staticmethod
    fn shape(x: ManagedTensorSlice) -> IndexList[x.rank]:
        return x.shape()


@compiler.register("imposter_add_elementwise")
@compiler.elementwise
struct AddElementwise:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](z: ManagedTensorSlice, x: ManagedTensorSlice, y: ManagedTensorSlice):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.type, width]:
            return rebind[SIMD[z.type, width]](
                x._fused_load[width](idx)
            ) + rebind[SIMD[z.type, width]](y._fused_load[width](idx))

        foreach[func](z)


@compiler.register("imposter_add_lhs")
struct AddFuseLHS:
    @compiler.enable_fusion_for("x")
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](z: ManagedTensorSlice, x: ManagedTensorSlice, y: ManagedTensorSlice):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.type, width]:
            return rebind[SIMD[z.type, width]](
                x._fused_load[width](idx)
            ) + rebind[SIMD[z.type, width]](y.load[width](idx))

        # Wrapper to hide the foreach call from MOGGPreElab.
        # Otherwhise it would still be detected as an elementwise kernel.
        fn foo():
            foreach[func](z)

        foo()


@compiler.register("imposter_add_fuse_inputs")
struct AddFuseInputs:
    @compiler.enable_fusion_for("x", "y")
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](z: ManagedTensorSlice, x: ManagedTensorSlice, y: ManagedTensorSlice):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.type, width]:
            return rebind[SIMD[z.type, width]](
                x._fused_load[width](idx)
            ) + rebind[SIMD[z.type, width]](y._fused_load[width](idx))

        # Wrapper to hide the foreach call from MOGGPreElab.
        # Otherwhise it would still be detected as an elementwise kernel.
        fn foo():
            foreach[func](z)

        foo()


# c = a @ b, should support CPU and GPU
@compiler.register("matmul_fuse_out")
struct MatmulFuseOut:
    @compiler.enable_fusion_for("c")
    @staticmethod
    fn execute[
        target: StringLiteral,
        lambdas_have_fusion: Bool,
        _synchronous: Bool,
    ](
        c: ManagedTensorSlice,
        a: ManagedTensorSlice,
        b: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ) raises:
        alias rank = a.rank
        alias a_dtype = a.type
        alias b_dtype = b.type
        alias c_dtype = c.type

        # Convert everything to NDBuffer
        var c_buffer = toNDBuffer[c_dtype, 2](c)
        var a_buffer = toNDBuffer[a_dtype, 2](a)
        var b_buffer = toNDBuffer[b_dtype, 2](b)

        @parameter
        @always_inline
        fn out_func[
            type: DType, width: Int, *, alignment: Int = 1
        ](idx: IndexList[2], val: SIMD[type, width]):
            c._fused_store(idx, rebind[SIMD[c.type, width]](val))

        print("lambdas_have_fusion =", lambdas_have_fusion)

        _matmul[elementwise_lambda_fn=out_func](c, a, b)

    @staticmethod
    fn shape(
        a: ManagedTensorSlice,
        b: ManagedTensorSlice,
    ) -> IndexList[2]:
        var shape = a.shape()
        shape[1] = b.dim_size[1]()
        return rebind[IndexList[2]](shape)


@compiler.register("op_with_synchronous")
struct WithSynchronous:
    @staticmethod
    fn execute[
        _synchronous: Bool,
    ](out: ManagedTensorSlice, input: ManagedTensorSlice):
        print("what up ", _synchronous)


@compiler.register("op_without_synchronous")
struct WithoutSynchronous:
    @staticmethod
    fn execute(out: ManagedTensorSlice, input: ManagedTensorSlice):
        print("what up")


# Simple, expects variadics to have the same size, and simply copies the first
# number from the associated inputs to outputs, plus a bias
@compiler.register("variadic_input_to_output")
struct VariadicInputToOutput:
    @staticmethod
    fn execute[
        type: DType,
        _synchronous: Bool,
        size: Int,
        target: StringLiteral,
    ](
        output: VariadicTensors[type, rank=1, size=size, io_spec=IOUnknown],
        bias: ManagedTensorSlice[type=type, rank=1],
        input: VariadicTensors[type, rank=1, size=size, io_spec=IOUnknown],
    ):
        @parameter
        for i in range(size):
            for j in range(input[i].size()):
                output[i][j] = input[i][j]
            output[i][0] += bias[0]


# Simply adds the first number of bias to the first number of boath outputs
# Mainly here to test logic with multiple DPS outputs
@compiler.register("add_bias_to_two_tensors", num_dps_outputs=2)
struct AddBiasToDouble:
    @staticmethod
    fn execute[
        rank: Int,
        type: DType,
        _synchronous: Bool,
    ](
        output1: ManagedTensorSlice[type=type, rank=rank],
        output2: ManagedTensorSlice[type=type, rank=rank],
        input1: ManagedTensorSlice[type=type, rank=rank],
        input2: ManagedTensorSlice[type=type, rank=rank],
        bias: ManagedTensorSlice[type=type, rank=rank],
    ):
        output1[0] = input1[0] + bias[0]
        output2[0] = input2[0] + bias[0]


@compiler.register("inplace_increment_elem", num_dps_outputs=0)
struct BasicInplace:
    @compiler.mutable("input")
    @staticmethod
    fn execute[
        type: DType,
    ](input: ManagedTensorSlice[type=type, rank=2]):
        x = input[0, 0]
        x += 1
        input[0, 0] = x


# Have this nearly identical version as having a raise changes the Mojo function's signature
@compiler.register("inplace_increment_elem_raises", num_dps_outputs=0)
struct BasicInplaceRaises:
    @compiler.mutable("input")
    @staticmethod
    fn execute[
        type: DType,
    ](input: ManagedTensorSlice[type=type, rank=2]) raises:
        x = input[0, 0]
        x += 1
        input[0, 0] = x


@compiler.register("variadic_add")
struct VariadicAdd:
    # TODO(GEX-1591): Re-enable fusion of variadic operands
    # @compiler.enable_fusion_for("inputs")
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
        target: StringLiteral,
        _synchronous: Bool,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        inputs: VariadicTensors[type, rank, io_spec=IOUnknown, *_],
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[rank]) -> SIMD[type, width]:
            var acc = SIMD[type, width](0)

            @parameter
            for i in range(inputs.size):
                acc += inputs[i]._fused_load[width](idx)

            return acc

        # Wrapper to hide the foreach call from MOGGPreElab.
        # Otherwhise it would still be detected as an elementwise kernel.
        fn foo():
            foreach[func](output)

        foo()


@compiler.register("transpose_2d")
@compiler.view_kernel
struct Transpose2DOp:
    @staticmethod
    fn build_view[
        type: DType,
    ](x: ManagedTensorSlice[type=type, rank=2]) -> StaticTuple[IndexList[2], 2]:
        var new_stride = IndexList[2]()
        var new_shape = IndexList[2]()
        new_stride[0] = x._runtime_strides[1]
        new_stride[1] = x._runtime_strides[0]
        new_shape[0] = x._spec.shape[1]
        new_shape[1] = x._spec.shape[0]

        return StaticTuple[IndexList[2], 2](new_shape, new_stride)

    @staticmethod
    fn get_view_strides(input_strides: DimList) -> DimList:
        # transpose the strides of the input
        return DimList(input_strides.at[1](), input_strides.at[0]())

    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
        type: DType,
    ](
        z: ManagedTensorSlice[type=type, rank=2],
        x: ManagedTensorSlice[type=type, rank=2],
        ctx: MojoCallContextPtr,
    ):
        alias view_strides = Self.get_view_strides(x._static_strides)
        var shape_and_strides = Self.build_view(x)

        var x_view = x.with_layout[
            new_static_shape = z._static_shape,
            new_static_strides=view_strides,
        ](shape_and_strides[0], shape_and_strides[1])

        view_copy_impl[target=target, _synchronous=_synchronous](z, x_view, ctx)


@compiler.register("elementwise_print_shape")
@compiler.elementwise
struct ElementwisePrintShape:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](z: ManagedTensorSlice, x: ManagedTensorSlice):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.type, width]:
            return rebind[SIMD[z.type, width]](x._fused_load[width](idx))

        print("input.shape =", x._spec.shape)
        print("output.shape =", z._spec.shape)

        foreach[func](z)

    @staticmethod
    fn shape(x: ManagedTensorSlice) -> IndexList[x.rank]:
        return x.shape()


# Raises if input shape is 10
@compiler.register("custom_op_that_raises")
struct CustomOpThatRaises:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](z: ManagedTensorSlice, x: ManagedTensorSlice) raises:
        if x.shape()[0] == 10:
            raise ("input_shape[0] == 10")

        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.type, width]:
            return rebind[SIMD[z.type, width]](x._fused_load[width](idx))

        foreach[func](z)

    @staticmethod
    fn shape(x: ManagedTensorSlice) raises -> IndexList[x.rank]:
        print("Hello")
        var out_shape = x.shape()
        if out_shape[0] == 20:
            raise ("data.get_shape()[0] == 20")
        return out_shape


@compiler.register("mo.test.failing_constraint")
struct OpThatAlwaysFailsConstraint:
    @staticmethod
    fn execute[
        type: DType, rank: Int
    ](
        out_tensor: ManagedTensorSlice[type=type, rank=rank],
        in_tensor: ManagedTensorSlice[type=type, rank=rank],
    ):
        constrained[
            1 == 2,
            "Expected constraint failure for error message testing",
        ]()


@compiler.register("mo.test.return_error")
struct OpThatAlwaysRaises:
    @staticmethod
    fn execute[
        type: DType, rank: Int
    ](
        out_tensor: ManagedTensorSlice[type=type, rank=rank],
        in_tensor: ManagedTensorSlice[type=type, rank=rank],
    ) raises:
        out_tensor[0] = in_tensor[0]
        raise Error("This is an error")


@compiler.register("monnx.abs_v13")
struct MONNXAbsOverload:
    @staticmethod
    fn execute[
        type: DType, rank: Int
    ](
        out_tensor: ManagedTensorSlice[type=type, rank=rank],
        in_tensor: ManagedTensorSlice[type=type, rank=rank],
    ) raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[rank]) -> SIMD[type, width]:
            return abs(in_tensor._fused_load[width](idx))

        print("The custom identity op is running!")
        foreach[func](out_tensor)


@compiler.register("torch.aten.abs")
struct MTorchAbsOverload:
    @staticmethod
    fn execute[
        type: DType, rank: Int
    ](
        out_tensor: ManagedTensorSlice[type=type, rank=rank],
        in_tensor: ManagedTensorSlice[type=type, rank=rank],
    ) raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[rank]) -> SIMD[type, width]:
            return abs(in_tensor._fused_load[width](idx))

        print("The custom identity op is running!")
        foreach[func](out_tensor)


@compiler.register("op_with_custom_params")
struct OpWithCustomParams:
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
        custom_int: Int,
        custom_str: StringLiteral,
        custom_dtype: DType,
    ](
        out_tensor: ManagedTensorSlice[type=type, rank=rank],
        in_tensor: ManagedTensorSlice[type=type, rank=rank],
    ) raises:
        out_tensor[0] = in_tensor[0]
        print("custom_int =", custom_int)
        print("custom_str =", custom_str)
        print("custom_dtype =", custom_dtype)


@compiler.register("mgprt_test_func")
struct MGPRTTestFunc:
    @staticmethod
    fn execute(out_tensor: ManagedTensorSlice) raises:
        external_call["MGP_RT_TEST", NoneType]()


@compiler.register("mutable_test_op", num_dps_outputs=0)
struct MutableTestOp:
    @staticmethod
    fn execute(in_place_tensor: ManagedTensorSlice) raises:
        in_place_tensor._ptr.store(0, 0)


# For testing support for Scalar[...] in Mojo
@compiler.register("supports_scalar_kernel")
struct SupportsScalarKernel:
    @staticmethod
    fn execute[
        type: DType
    ](
        out: ManagedTensorSlice[type=type, rank=1],
        x: ManagedTensorSlice[type=type, rank=1],
        y: Scalar[type],
    ) raises:
        print("datatype is", type)


@compiler.register("kernel_with_no_target")
struct KernelWithNoTarget:
    @staticmethod
    fn execute[
        type: DType
    ](
        out: ManagedTensorSlice[type=type, *_],
        x: ManagedTensorSlice[type=type, *_],
    ) raises:
        print("hello from kernel with no target")


@compiler.register("basic_target")
struct BasicTarget:
    @staticmethod
    fn execute[
        type: DType, target: StringLiteral
    ](
        out: ManagedTensorSlice[type=type, *_],
        x: ManagedTensorSlice[type=type, *_],
    ) raises:
        print("hello from kernel on", target)


@value
@register_passable
struct MyCustomScalarReg[type: DType]:
    var val: Scalar[type]

    @implicit
    fn __init__(out self, val: Scalar[type]):
        print("MyCustomScalarReg.__init__", val)
        self.val = val

    fn __del__(owned self):
        print("MyCustomScalarReg.__del__", self.val)


@compiler.register("buff_to_my_custom_scalar_reg", num_dps_outputs=0)
struct BuffToMyCustomScalarReg:
    @staticmethod
    fn execute[
        target: StringLiteral
    ](x: ManagedTensorSlice[type = DType.int32, rank=1]) -> MyCustomScalarReg[
        DType.int32
    ]:
        return MyCustomScalarReg(x[0])


@compiler.register("my_custom_scalar_reg_to_buff")
struct CustomScalarRegToBuff:
    @staticmethod
    fn execute[
        target: StringLiteral
    ](
        input: ManagedTensorSlice[type = DType.int32, rank=1],
        x: MyCustomScalarReg[DType.int32],
    ):
        input[0] = x.val


@compiler.register("test_custom_op")
struct TestCustomOp:
    @staticmethod
    fn execute[
        target: StringLiteral, type: DType, rank: Int
    ](
        out: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
    ):
        print("World!")

    @staticmethod
    fn shape[
        type: DType, rank: Int
    ](input: ManagedTensorSlice[type=type, rank=rank]) -> IndexList[rank]:
        print("Hello")
        return input.shape()


@compiler.register("invalid_kernel_owned_arg", num_dps_outputs=0)
struct InvalidOwnedArgConvention:
    @staticmethod
    fn execute[
        target: StringLiteral, type: DType, rank: Int
    ](owned input: MyCustomScalarSI32) -> MyCustomScalarSI32:
        return MyCustomScalarSI32(input.val)


@compiler.register("single_device_context")
struct SingleDeviceContext:
    @staticmethod
    fn execute[
        type: DType
    ](
        out: ManagedTensorSlice[type=type, *_],
        x: ManagedTensorSlice[type=type, *_],
        dev_ctx: DeviceContextPtr,
    ) raises:
        dev_ctx[].synchronize()


@compiler.register("multi_device_context", num_dps_outputs=1)
struct MultiDeviceContext:
    @staticmethod
    fn execute[
        type: DType
    ](
        out: ManagedTensorSlice[type=type, *_],
        x: ManagedTensorSlice[type=type, *_],
        dev_ctx0: DeviceContextPtr,
        dev_ctx1: DeviceContextPtr,
        ctx: MojoCallContextPtr,
    ) raises:
        print("dev_ctx0.id() =", dev_ctx0[].id())
        print("dev_ctx1.id() =", dev_ctx1[].id())
        dev_ctx0[].synchronize()
        dev_ctx1[].synchronize()


@compiler.register("multi_device_context_dedup")
struct MultiDeviceContextDedup:
    @staticmethod
    fn execute[
        type: DType
    ](
        out: ManagedTensorSlice[type=type, *_],
        x: ManagedTensorSlice[type=type, *_],
        y: ManagedTensorSlice[type=type, *_],
        dev_ctx0: DeviceContextPtr,
        dev_ctx1: DeviceContextPtr,
    ) raises:
        dev_ctx0[].synchronize()
        dev_ctx1[].synchronize()


@compiler.register("variadic_device_context", num_dps_outputs=1)
struct VariadicDeviceContext:
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
    ](
        outputs: VariadicTensors[type, rank, *_],
        inputs: VariadicTensors[type, rank, *_],
        dev_ctxs: StaticTuple[DeviceContextPtr, *_],
    ) raises:
        for i in range(len(dev_ctxs)):
            print("dev_ctxs[", i, "].id() =", dev_ctxs[i][].id())
            dev_ctxs[i][].synchronize()


@compiler.register("imposter_cast_elementwise")
@compiler.elementwise
struct CastElementwise:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](
                x._fused_load[width](idx).cast[y.type]()
            )

        foreach[func, target=target, _synchronous=_synchronous](y, ctx)


@compiler.register("print_vector_size")
@compiler.elementwise
struct PrintVectorSize:
    @staticmethod
    fn execute[
        target: StringLiteral,
        _synchronous: Bool,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.type, width]:
            if idx[0] == 0:
                # Only print once
                # We can't directly print `width` as it's HW-dependant.
                # Instead compare it with the width of i8 / f32.
                print(
                    "width == simdwidthof[DType.int8]():",
                    width == simdwidthof[DType.int8](),
                )
                print(
                    "width == simdwidthof[DType.float32]():",
                    width == simdwidthof[DType.float32](),
                )
            return rebind[SIMD[y.type, width]](x._fused_load[width](idx))

        foreach[func, target=target, _synchronous=_synchronous](y, ctx)
