# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import compiler
from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from nn.mha import fused_attention as cpu_fused_attention_impl
from tensor_utils import UnsafeTensorSlice

from utils.index import StaticIntTuple


@compiler.register("imposter_add")
struct Foo:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](z: UnsafeTensorSlice, x: UnsafeTensorSlice, y: UnsafeTensorSlice):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[z.rank]) -> SIMD[z.type, width]:
            return rebind[SIMD[z.type, width]](x.load[width](idx)) + rebind[
                SIMD[z.type, width]
            ](y.load[width](idx))

        compiler.foreach[func](z)

    @staticmethod
    fn shape(
        x: UnsafeTensorSlice, y: UnsafeTensorSlice
    ) -> StaticIntTuple[x.rank]:
        return x.get_static_spec().shape


@always_inline
fn toNDBuffer[
    out_dtype: DType, out_rank: Int
](tensor: UnsafeTensorSlice) -> NDBuffer[out_dtype, out_rank]:
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
        output: UnsafeTensorSlice,
        q: UnsafeTensorSlice,
        k: UnsafeTensorSlice,
        v: UnsafeTensorSlice,
        scale: UnsafeTensorSlice,
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
        q: UnsafeTensorSlice,
        k: UnsafeTensorSlice,
        v: UnsafeTensorSlice,
        scale: UnsafeTensorSlice,
    ) -> StaticIntTuple[q.rank]:
        return q.get_static_spec().shape
