# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from kv_cache.types import KVCacheStaticParams, KVCacheLayout


@value
struct KVCacheKernelNames:
    var matmul_kernel: StringLiteral
    var flash_attention_kernel: StringLiteral
    var kv_cache_length_kernel: StringLiteral
    var key_cache_for_layer_kernel: StringLiteral
    var value_cache_for_layer_kernel: StringLiteral
    var fused_qkv_matmul_kernel: StringLiteral
    var fused_qk_rope_kernel: StringLiteral


fn _kv_cache_kernel_names[
    type: DType, params: KVCacheStaticParams
]() -> KVCacheKernelNames:
    @parameter
    if type == DType.float32 and params == KVCacheStaticParams(
        num_heads=6, head_size=48, layout=KVCacheLayout.BSHD
    ):
        return KVCacheKernelNames(
            matmul_kernel="matmul_kv_cache_h6_d48_bshd",
            flash_attention_kernel="flash_attention_kv_cache_h6_d48_bshd",
            kv_cache_length_kernel="kv_cache_length_h6_d48_bshd_f32",
            key_cache_for_layer_kernel="key_cache_for_layer_h6_d48_bshd_f32",
            value_cache_for_layer_kernel=(
                "value_cache_for_layer_h6_d48_bshd_f32"
            ),
            fused_qkv_matmul_kernel="fused_qkv_matmul_kv_cache_h6_d48_bshd",
            fused_qk_rope_kernel="fused_qk_rope_h6_d48_bshd",
        )
    elif type == DType.float32 and params == KVCacheStaticParams(
        num_heads=8, head_size=128, layout=KVCacheLayout.BSHD
    ):
        return KVCacheKernelNames(
            matmul_kernel="matmul_kv_cache_h8_d128_bshd",
            flash_attention_kernel="flash_attention_kv_cache_h8_d128_bshd",
            kv_cache_length_kernel="kv_cache_length_h8_d128_bshd_f32",
            key_cache_for_layer_kernel="key_cache_for_layer_h8_d128_bshd_f32",
            value_cache_for_layer_kernel=(
                "value_cache_for_layer_h8_d128_bshd_f32"
            ),
            fused_qkv_matmul_kernel="fused_qkv_matmul_kv_cache_h8_d128_bshd",
            fused_qk_rope_kernel="fused_qk_rope_h8_d128_bshd",
        )
    elif type == DType.bfloat16 and params == KVCacheStaticParams(
        num_heads=8, head_size=128, layout=KVCacheLayout.BSHD
    ):
        return KVCacheKernelNames(
            matmul_kernel="matmul_kv_cache_h8_d128_bshd",
            flash_attention_kernel="flash_attention_kv_cache_h8_d128_bshd",
            kv_cache_length_kernel="kv_cache_length_h8_d128_bshd_bf16",
            key_cache_for_layer_kernel="key_cache_for_layer_h8_d128_bshd_bf16",
            value_cache_for_layer_kernel=(
                "value_cache_for_layer_h8_d128_bshd_bf16"
            ),
            fused_qkv_matmul_kernel="fused_qkv_matmul_kv_cache_h8_d128_bshd",
            fused_qk_rope_kernel="fused_qk_rope_h8_d128_bshd",
        )
    else:
        constrained[False, "Unsupported KV Cache configuration"]()

    return KVCacheKernelNames(
        matmul_kernel="",
        flash_attention_kernel="",
        kv_cache_length_kernel="",
        key_cache_for_layer_kernel="",
        value_cache_for_layer_kernel="",
        fused_qkv_matmul_kernel="",
        fused_qk_rope_kernel="",
    )
