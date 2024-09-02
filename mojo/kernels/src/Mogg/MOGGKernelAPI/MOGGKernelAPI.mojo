# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import compiler_internal as compiler
from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from nn.mha import fused_attention as cpu_fused_attention_impl
from tensor_utils import ManagedTensorSlice, foreach

from utils import StaticIntTuple, unroll
from linalg.matmul import matmul as _matmul
from runtime.asyncrt import MojoCallContextPtr


@value
@register_passable
struct MyCustomScalarReg[type: DType]:
    var val: Scalar[type]

    fn __init__(inout self, val: Scalar[type]):
        print("MyCustomScalarReg.__init__", val)
        self.val = val

    fn __del__(owned self):
        print("MyCustomScalarReg.__del__", self.val)


@compiler.register("tensor_to_custom_scalar_reg")
struct OpaqueToCustomScalarReg:
    @staticmethod
    fn initialize_output(x: ManagedTensorSlice) -> MyCustomScalarReg[x.type]:
        return MyCustomScalarReg(x[0])


# Adds two custom scalar types (one of which is register passable) and writes
# to a tensor
@compiler.register("opaque_add_to_tensor")
struct OpaqueAddToTensor:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](out: ManagedTensorSlice, x: MyCustomScalarReg, y: MyCustomScalarReg):
        var scalar_x = x.val
        var scalar_y = rebind[Scalar[x.type]](y.val)
        out[0] = rebind[Scalar[out.type]](scalar_x + scalar_y)

    @staticmethod
    fn shape(x: MyCustomScalarReg, y: MyCustomScalarReg) -> StaticIntTuple[1]:
        return StaticIntTuple[1](1)


@compiler.register("imposter_add")
struct Foo:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](z: ManagedTensorSlice, x: ManagedTensorSlice, y: ManagedTensorSlice):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[z.rank]) -> SIMD[z.type, width]:
            return rebind[SIMD[z.type, width]](x.load[width](idx)) + rebind[
                SIMD[z.type, width]
            ](y.load[width](idx))

        foreach[func](z)

    @staticmethod
    fn shape(
        x: ManagedTensorSlice, y: ManagedTensorSlice
    ) -> StaticIntTuple[x.rank]:
        return x.get_static_spec().shape


@always_inline
fn toNDBuffer[
    out_dtype: DType, out_rank: Int
](tensor: ManagedTensorSlice) -> NDBuffer[out_dtype, out_rank]:
    # TODO(GRA-734): forward other static params automatically
    return rebind[NDBuffer[out_dtype, out_rank]](
        NDBuffer[tensor.type, tensor.rank](
            tensor._ptr, tensor.get_static_spec().shape
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
    ):
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

        try:
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
        except e:
            e = Error("Something went wrong!")

    @staticmethod
    fn shape(
        q: ManagedTensorSlice,
        k: ManagedTensorSlice,
        v: ManagedTensorSlice,
        scale: ManagedTensorSlice,
    ) -> StaticIntTuple[q.rank]:
        return q.get_static_spec().shape


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
    ):
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
    ) -> StaticIntTuple[2]:
        var shape = a.get_static_spec().shape
        shape[1] = b.get_static_spec().shape[1]
        return rebind[StaticIntTuple[2]](shape)


@compiler.register("print_shape_strides")
struct PrintShapeStridesOp:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](out: ManagedTensorSlice, x: ManagedTensorSlice):
        alias x_shape = compiler.specsof[x.type, x.rank]("x").shape
        alias x_strides = compiler.specsof[x.type, x.rank]("x").strides

        print("x.shape = ", x_shape)
        print("x.strides = ", x_strides)

        @parameter
        @always_inline
        fn func[
            width: Int
        ](idx: StaticIntTuple[out.rank]) -> SIMD[out.type, width]:
            return rebind[SIMD[out.type, width]](x.load[width](idx))

        foreach[func](out)

    @staticmethod
    fn shape(x: ManagedTensorSlice) -> StaticIntTuple[x.rank]:
        return x.get_static_spec().shape


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
        fn func[width: Int](idx: StaticIntTuple[z.rank]) -> SIMD[z.type, width]:
            return rebind[SIMD[z.type, width]](x.load[width](idx)) + rebind[
                SIMD[z.type, width]
            ](y.load[width](idx))

        foreach[func](z)


# c = a @ b, should support CPU and GPU
@compiler.register("matmul_fuse_out")
struct MatmulFuseOut:
    @compiler.enable_fusion_for("c")
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        c: ManagedTensorSlice,
        a: ManagedTensorSlice,
        b: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
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
            type: DType, width: Int
        ](idx: StaticIntTuple[2], val: SIMD[type, width]):
            c.store(idx, rebind[SIMD[c.type, width]](val))

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
    ) -> StaticIntTuple[2]:
        var shape = a.get_static_spec().shape
        shape[1] = b.get_static_spec().shape[1]
        return rebind[StaticIntTuple[2]](shape)


# TensorCopy intrinsic used by view kernels.
# z is a kernel output, and x a view of the input.
@no_inline
fn view_copy_impl[
    synchronous: Bool, target: StringLiteral, type: DType, rank: Int
](z: ManagedTensorSlice[type, rank], x: ManagedTensorSlice[type, rank]):
    @parameter
    @always_inline
    fn func[width: Int](idx: StaticIntTuple[z.rank]) -> SIMD[z.type, width]:
        return x._simd_load_internal[width](idx)

    foreach[func](z)


@compiler.register("mo.static.broadcast_to")
@compiler.view_kernel
struct BroadcastTo:
    @staticmethod
    fn build_view[
        type: DType,
        in_rank: Int,
        out_rank: Int,
    ](
        x: ManagedTensorSlice[type, in_rank],
        output_shape: StaticIntTuple[out_rank],
    ) -> ManagedTensorSlice[type, out_rank]:
        var new_strides = StaticIntTuple[out_rank]()
        alias delta = out_rank - in_rank

        print("Hit new kernel API!")

        @parameter
        for i in range(out_rank):
            if i < delta:
                new_strides[i] = 0
            elif x.dim_size(i - delta) <= 1:
                new_strides[i] = 0
            else:
                new_strides[i] = x._strides[i - delta]

        return ManagedTensorSlice[type, out_rank](
            x._ptr, output_shape, new_strides
        )

    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
        type: DType,
        in_rank: Int,
        out_rank: Int,
    ](
        z: ManagedTensorSlice[type, out_rank],
        x: ManagedTensorSlice[type, in_rank],
        output_shape: StaticIntTuple[out_rank],
    ):
        # We need the extra output_shape argument.
        # Using `z.shape` instead will prevent the compiler from fusing the kernels.
        var x_view = Self.build_view(x, output_shape)
        view_copy_impl[synchronous, target](z, x_view)
