# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import compiler_internal as compiler
from buffer import NDBuffer
from buffer.dimlist import DimList
from linalg.matmul import matmul as _matmul
from nn.mha import fused_attention as cpu_fused_attention_impl
from register import uses_opaque
from runtime.asyncrt import MojoCallContextPtr
from tensor_utils import ManagedTensorSlice, foreach
from tensor_utils_internal import view_copy_impl

from utils import IndexList, StaticTuple

# ===----------------------------------------------------------------------===#
# Opaque Reg Types
# ===----------------------------------------------------------------------===#


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
    @uses_opaque
    @staticmethod
    fn execute(
        x: ManagedTensorSlice[DType.int32, rank=1]
    ) -> MyCustomScalarRegSI32:
        return MyCustomScalarRegSI32(x[0])


# Adds two custom scalar types (one of which is register passable) and writes
# to a tensor
@compiler.register("opaque_add_to_tensor_si32_reg")
struct OpaqueAddToTensorSI32Reg:
    @uses_opaque
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        out: ManagedTensorSlice[DType.int32, rank=1],
        x: MyCustomScalarRegSI32,
        y: MyCustomScalarRegSI32,
    ):
        out[0] = x.val + y.val


@compiler.register("opaque_add_to_tensor_f32_reg")
struct OpaqueAddToTensorF32:
    @uses_opaque
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        out: ManagedTensorSlice[DType.float32, rank=1],
        x: MyCustomScalarRegF32,
        y: MyCustomScalarRegF32,
    ):
        out[0] = x.val + y.val


# ===----------------------------------------------------------------------===#
# Opaque Mem. Types
# ===----------------------------------------------------------------------===#


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
    @uses_opaque
    @staticmethod
    fn execute(
        x: ManagedTensorSlice[DType.int32, rank=1]
    ) -> MyCustomScalarSI32:
        return MyCustomScalarSI32(x[0])


# Adds two custom scalar types (one of which is register passable) and writes
# to a tensor
@compiler.register("opaque_add_to_tensor_si32")
struct OpaqueAddToTensorSI32:
    @uses_opaque
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        out: ManagedTensorSlice[DType.int32, rank=1],
        x: MyCustomScalarSI32,
        y: MyCustomScalarSI32,
    ):
        out[0] = x.val + y.val


# ===----------------------------------------------------------------------===#
# Other Kernels
# ===----------------------------------------------------------------------===#


@compiler.register("imposter_add")
struct Foo:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
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
        return x.get_runtime_spec().shape


@always_inline
fn toNDBuffer[
    out_dtype: DType, out_rank: Int
](tensor: ManagedTensorSlice) -> NDBuffer[out_dtype, out_rank]:
    # TODO(GEX-734): forward other static params automatically
    return rebind[NDBuffer[out_dtype, out_rank]](
        NDBuffer[tensor.type, tensor.rank](
            tensor._ptr, tensor.get_runtime_spec().shape
        )
    )


# Analogous to no_mask_flash_attention_cpu
@compiler.register("imposter_no_mask_flash_attention_cpu")
struct ImposterMHANoMask:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice,
        q: ManagedTensorSlice,
        k: ManagedTensorSlice,
        v: ManagedTensorSlice,
        scale: ManagedTensorSlice,
    ) raises:
        alias qkv_rank = q.rank
        alias qkv_dtype = q.type

        # Convert everything to NDBuffer
        var q_buffer = toNDBuffer[qkv_dtype, qkv_rank](q)
        var k_buffer = toNDBuffer[qkv_dtype, qkv_rank](k)
        var v_buffer = toNDBuffer[qkv_dtype, qkv_rank](v)
        var output_buffer = toNDBuffer[qkv_dtype, qkv_rank](output)
        var scale_buffer = toNDBuffer[qkv_dtype, 1](scale)

        alias mask_shape = DimList()
        var mask = NDBuffer[qkv_dtype, qkv_rank, mask_shape]()
        var scale_f32 = scale_buffer[0].cast[DType.float32]()
        var causal_mask: Float32 = 0

        cpu_fused_attention_impl[
            qkv_rank,
            q_buffer.shape,
            k_buffer.shape,
            v_buffer.shape,
            mask_shape,
            DimList.create_unknown[qkv_rank](),
            qkv_dtype,
            qkv_dtype,
            qkv_dtype,
            qkv_dtype,
            qkv_dtype,
            transpose_k=False,
            add_attn_mask=False,
            add_causal_mask=False,
        ](
            output_buffer,
            q_buffer,
            k_buffer,
            v_buffer,
            mask,
            scale_f32,
            causal_mask,
        )

    @staticmethod
    fn shape(
        q: ManagedTensorSlice,
        k: ManagedTensorSlice,
        v: ManagedTensorSlice,
        scale: ManagedTensorSlice,
    ) -> IndexList[q.rank]:
        return q.get_runtime_spec().shape


# c = a @ b, should support CPU and GPU
@compiler.register("imposter_matmul")
struct ImposterMatmul:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
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
        _matmul[
            False,
            False,
            False,
            None,
            saturated_vnni=False,
            single_thread_blocking_override=synchronous,
            target=target,
        ](
            c_buffer,
            a_buffer,
            b_buffer,
            ctx,
        )

    @staticmethod
    fn shape(
        a: ManagedTensorSlice,
        b: ManagedTensorSlice,
    ) -> IndexList[2]:
        var shape = a.get_runtime_spec().shape
        shape[1] = b.get_runtime_spec().shape[1]
        return rebind[IndexList[2]](shape)


@compiler.register("print_tensor_spec")
struct PrintTensorSpecOp:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](out: ManagedTensorSlice, x: ManagedTensorSlice):
        alias x_shape = compiler.specsof[x.type, x.rank]("x").shape
        alias x_strides = compiler.specsof[x.type, x.rank]("x").strides
        alias x_alignment = compiler.specsof[x.type, x.rank]("x").alignment
        alias x_address_space = compiler.specsof[x.type, x.rank](
            "x"
        ).address_space
        alias x_exclusive = compiler.specsof[x.type, x.rank]("x").exclusive

        print("x.shape =", x_shape)
        print("x.strides =", x_strides)
        print("x.alignment =", x_alignment)
        print("x.address_space =", x_address_space)
        print("x.exclusive =", x_exclusive)

        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[out.rank]) -> SIMD[out.type, width]:
            return rebind[SIMD[out.type, width]](x.load[width](idx))

        foreach[func](out)

    @staticmethod
    fn shape(x: ManagedTensorSlice) -> IndexList[x.rank]:
        return x.get_runtime_spec().shape


@compiler.register("print_tensor_spec_view")
@compiler.view_kernel
struct PrintTensorSpecViewOp:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](out: ManagedTensorSlice, x: ManagedTensorSlice):
        alias x_shape = compiler.specsof[x.type, x.rank]("x").shape
        alias x_strides = compiler.specsof[x.type, x.rank]("x").strides
        alias x_alignment = compiler.specsof[x.type, x.rank]("x").alignment
        alias x_address_space = compiler.specsof[x.type, x.rank](
            "x"
        ).address_space
        alias x_exclusive = compiler.specsof[x.type, x.rank]("x").exclusive

        print("x.shape =", x_shape)
        print("x.strides =", x_strides)
        print("x.alignment =", x_alignment)
        print("x.address_space =", x_address_space)
        print("x.exclusive =", x_exclusive)

        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[out.rank]) -> SIMD[out.type, width]:
            return rebind[SIMD[out.type, width]](x.load[width](idx))

        foreach[func](out)

    @staticmethod
    fn shape(x: ManagedTensorSlice) -> IndexList[x.rank]:
        return x.get_runtime_spec().shape


@compiler.register("print_tensor_spec_fused")
struct PrintTensorSpecFusedOp:
    @compiler.elementwise
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](out: ManagedTensorSlice, x: ManagedTensorSlice):
        alias x_shape = compiler.specsof[x.type, x.rank]("x").shape
        alias x_strides = compiler.specsof[x.type, x.rank]("x").strides
        alias x_alignment = compiler.specsof[x.type, x.rank]("x").alignment
        alias x_address_space = compiler.specsof[x.type, x.rank](
            "x"
        ).address_space
        alias x_exclusive = compiler.specsof[x.type, x.rank]("x").exclusive

        print("x.shape =", x_shape)
        print("x.strides =", x_strides)
        print("x.alignment =", x_alignment)
        print("x.address_space =", x_address_space)
        print("x.exclusive =", x_exclusive)

        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[out.rank]) -> SIMD[out.type, width]:
            return rebind[SIMD[out.type, width]](x._fused_load[width](idx))

        foreach[func](out)

    @staticmethod
    fn shape(x: ManagedTensorSlice) -> IndexList[x.rank]:
        return x.get_runtime_spec().shape


@compiler.register("imposter_add_elementwise")
@compiler.elementwise
struct AddElementwise:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
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
        synchronous: Bool,
        target: StringLiteral,
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
        synchronous: Bool,
        target: StringLiteral,
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
        synchronous: Bool,
        target: StringLiteral,
        lambdas_have_fusion: Bool,
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

        _matmul[
            False,
            False,
            False,
            elementwise_lambda_fn=out_func,
            saturated_vnni=False,
            single_thread_blocking_override=synchronous,
            target=target,
        ](
            c_buffer,
            a_buffer,
            b_buffer,
            ctx,
        )

    @staticmethod
    fn shape(
        a: ManagedTensorSlice,
        b: ManagedTensorSlice,
    ) -> IndexList[2]:
        var shape = a.get_runtime_spec().shape
        shape[1] = b.get_runtime_spec().shape[1]
        return rebind[IndexList[2]](shape)


@compiler.register("op_with_synchronous")
struct WithSynchronous:
    @staticmethod
    fn execute[
        synchronous: Bool,
    ](out: ManagedTensorSlice, input: ManagedTensorSlice):
        print("what up ", synchronous)


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
        synchronous: Bool,
        size: Int,
        target: StringLiteral,
    ](
        output: StaticTuple[ManagedTensorSlice[type, rank=1], size],
        bias: ManagedTensorSlice[type, rank=1],
        input: StaticTuple[ManagedTensorSlice[type, rank=1], size],
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
        synchronous: Bool,
    ](
        output1: ManagedTensorSlice[type, rank],
        output2: ManagedTensorSlice[type, rank],
        input1: ManagedTensorSlice[type, rank],
        input2: ManagedTensorSlice[type, rank],
        bias: ManagedTensorSlice[type, rank],
    ):
        output1[0] = input1[0] + bias[0]
        output2[0] = input2[0] + bias[0]


@compiler.register("inplace_increment_elem", num_dps_outputs=0)
struct BasicInplace:
    @compiler.mutable("input")
    @staticmethod
    fn execute[
        type: DType,
    ](input: ManagedTensorSlice[type, rank=2]):
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
    ](input: ManagedTensorSlice[type, rank=2]) raises:
        x = input[0, 0]
        x += 1
        input[0, 0] = x


@compiler.register("variadic_add")
struct VariadicAdd:
    @compiler.enable_fusion_for("inputs")
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
        synchronous: Bool,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice[type, rank],
        inputs: StaticTuple[ManagedTensorSlice[type, rank], *_],
    ):
        alias inputs_specs = compiler.specsof[type, rank, inputs.size]("inputs")

        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[rank]) -> SIMD[type, width]:
            var acc = SIMD[type, width](0)

            @parameter
            for i in range(inputs.size):
                alias in_lambda = inputs_specs[i].in_lambda

                @parameter
                if in_lambda:
                    alias in_fn = in_lambda.value()
                    acc += in_fn[width](idx)
                else:
                    acc += inputs[i].load[width](idx)

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
    ](x: ManagedTensorSlice[type, 2],) -> ManagedTensorSlice[type, 2]:
        var new_stride = IndexList[2]()
        var new_shape = IndexList[2]()
        new_stride[0] = x._strides[1]
        new_stride[1] = x._strides[0]
        new_shape[0] = x._spec.shape[1]
        new_shape[1] = x._spec.shape[0]

        return ManagedTensorSlice[type, 2](x._ptr, new_shape, new_stride)

    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
        type: DType,
    ](
        z: ManagedTensorSlice[type, 2],
        x: ManagedTensorSlice[type, 2],
        ctx: MojoCallContextPtr,
    ):
        var x_view = Self.build_view(x)
        view_copy_impl[synchronous, target](z, x_view, ctx)
