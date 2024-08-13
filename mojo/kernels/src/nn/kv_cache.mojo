# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from buffer import DimList, NDBuffer, Dim
from collections import OptionalReg
from math import isqrt
from sys.info import _current_target
from utils import Index
from utils.numerics import min_finite, isnan

from algorithm.functional import _elementwise_impl
from gpu.host import Stream, DeviceContext, DeviceBuffer
from gpu.host._compile import _get_nvptx_target
from linalg import transpose
from linalg.matmul_gpu import _matmul_gpu
from linalg.matmul import matmul
from nn.flash_attention import flash_attention as cpu_flash_attention
from nn.mha import flash_attention as gpu_flash_attention
from register import mogg_register
from runtime.asyncrt import (
    MojoCallContextPtr,
)
from utils import StaticIntTuple

from .types import (
    ContiguousKVCache,
    ContiguousKVCacheCollection,
    KVCacheLayout,
    KVCacheStaticParams,
)


@value
struct KVCacheKernelNames:
    var matmul_kernel: StringLiteral
    var flash_attention_kernel: StringLiteral
    var kv_cache_length_kernel: StringLiteral
    var key_cache_for_layer_kernel: StringLiteral
    var value_cache_for_layer_kernel: StringLiteral
    var fused_qkv_matmul_kernel: StringLiteral
    var fused_qkv_matmul_w_rope_kernel: StringLiteral


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
            fused_qkv_matmul_w_rope_kernel=(
                "fused_qkv_matmul_kv_cache_w_rope_h6_d48_bshd"
            ),
        )
    elif type == DType.float32 and params == KVCacheStaticParams(
        num_heads=6, head_size=48, layout=KVCacheLayout.BHSD
    ):
        return KVCacheKernelNames(
            matmul_kernel="matmul_kv_cache_h6_d48_bhsd",
            flash_attention_kernel="flash_attention_kv_cache_h6_d48_bhsd",
            kv_cache_length_kernel="kv_cache_length_h6_d48_bhsd_f32",
            key_cache_for_layer_kernel="key_cache_for_layer_h6_d48_bhsd_f32",
            value_cache_for_layer_kernel=(
                "value_cache_for_layer_h6_d48_bhsd_f32"
            ),
            fused_qkv_matmul_kernel="fused_qkv_matmul_kv_cache_h6_d48_bhsd",
            fused_qkv_matmul_w_rope_kernel=(
                "fused_qkv_matmul_kv_cache_w_rope_h6_d48_bhsd"
            ),
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
            fused_qkv_matmul_w_rope_kernel=(
                "fused_qkv_matmul_kv_cache_w_rope_h8_d128_bshd"
            ),
        )
    elif type == DType.float32 and params == KVCacheStaticParams(
        num_heads=8, head_size=128, layout=KVCacheLayout.BHSD
    ):
        return KVCacheKernelNames(
            matmul_kernel="matmul_kv_cache_h8_d128_bhsd",
            flash_attention_kernel="flash_attention_kv_cache_h8_d128_bhsd",
            kv_cache_length_kernel="kv_cache_length_h8_d128_bhsd_f32",
            key_cache_for_layer_kernel="key_cache_for_layer_h8_d128_bhsd_f32",
            value_cache_for_layer_kernel=(
                "value_cache_for_layer_h8_d128_bhsd_f32"
            ),
            fused_qkv_matmul_kernel="fused_qkv_matmul_kv_cache_h8_d128_bhsd",
            fused_qkv_matmul_w_rope_kernel=(
                "fused_qkv_matmul_kv_cache_w_rope_h8_d128_bhsd"
            ),
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
            fused_qkv_matmul_w_rope_kernel=(
                "fused_qkv_matmul_kv_cache_w_rope_h8_d128_bshd"
            ),
        )
    elif type == DType.bfloat16 and params == KVCacheStaticParams(
        num_heads=8, head_size=128, layout=KVCacheLayout.BHSD
    ):
        return KVCacheKernelNames(
            matmul_kernel="matmul_kv_cache_h8_d128_bhsd",
            flash_attention_kernel="flash_attention_kv_cache_h8_d128_bhsd",
            kv_cache_length_kernel="kv_cache_length_h8_d128_bhsd_bf16",
            key_cache_for_layer_kernel="key_cache_for_layer_h8_d128_bhsd_bf16",
            value_cache_for_layer_kernel=(
                "value_cache_for_layer_h8_d128_bhsd_bf16"
            ),
            fused_qkv_matmul_kernel="fused_qkv_matmul_kv_cache_h8_d128_bhsd",
            fused_qkv_matmul_w_rope_kernel=(
                "fused_qkv_matmul_kv_cache_w_rope_h8_d128_bhsd"
            ),
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
        fused_qkv_matmul_w_rope_kernel="",
    )


@mogg_register("kv_cache_length_h8_d128_bshd_bf16")
@export
fn kv_cache_length_h8_d128_bshd_bf16(
    kv_collection: ContiguousKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BSHD
        ),
    ],
    output: NDBuffer[DType.int64, 1],
):
    return _kv_cache_length(kv_collection, output)


@mogg_register("kv_cache_length_h8_d128_bhsd_bf16")
@export
fn kv_cache_length_h8_d128_bhsd_bf16(
    kv_collection: ContiguousKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BHSD
        ),
    ],
    output: NDBuffer[DType.int64, 1],
):
    return _kv_cache_length(kv_collection, output)


@mogg_register("kv_cache_length_h6_d48_bshd_f32")
@export
fn kv_cache_length_h6_d48_bshd_f32(
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BSHD
        ),
    ],
    output: NDBuffer[DType.int64, 1],
):
    return _kv_cache_length(kv_collection, output)


@mogg_register("kv_cache_length_h6_d48_bhsd_f32")
@export
fn kv_cache_length_h6_d48_bhsd_f32(
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BHSD
        ),
    ],
    output: NDBuffer[DType.int64, 1],
):
    return _kv_cache_length(kv_collection, output)


@mogg_register("kv_cache_length_h8_d128_bshd_f32")
@export
fn kv_cache_length_h8_d128_bshd_f32(
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BSHD
        ),
    ],
    output: NDBuffer[DType.int64, 1],
):
    return _kv_cache_length(kv_collection, output)


@mogg_register("kv_cache_length_h8_d128_bhsd_f32")
@export
fn kv_cache_length_h8_d128_bhsd_f32(
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BHSD
        ),
    ],
    output: NDBuffer[DType.int64, 1],
):
    return _kv_cache_length(kv_collection, output)


fn _kv_cache_length[
    type: DType, kv_params: KVCacheStaticParams
](
    kv_collection: ContiguousKVCacheCollection[type, kv_params],
    output: NDBuffer[DType.int64, 1],
):
    """Returns the size of the cache in a ContiguousKVCacheCollection mo.opaque object.
    """
    for bs in range(output.dim[0]()):
        output.store[width=1](Index(bs), Int64(kv_collection.cache_length(bs)))


@mogg_register("key_cache_for_layer_h8_d128_bhsd_bf16")
@export
fn key_cache_for_layer_h8_d128_bhsd_bf16(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BHSD
        ),
    ],
) -> ContiguousKVCache[
    DType.bfloat16,
    KVCacheStaticParams(num_heads=6, head_size=48, layout=KVCacheLayout.BHSD),
]:
    return _key_cache_for_layer(layer_idx, kv_collection)


@mogg_register("key_cache_for_layer_h8_d128_bshd_bf16")
@export
fn key_cache_for_layer_h8_d128_bshd_bf16(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BSHD
        ),
    ],
) -> ContiguousKVCache[
    DType.bfloat16,
    KVCacheStaticParams(num_heads=6, head_size=48, layout=KVCacheLayout.BSHD),
]:
    return _key_cache_for_layer(layer_idx, kv_collection)


@mogg_register("key_cache_for_layer_h6_d48_bshd_f32")
@export
fn key_cache_for_layer_h6_d48_bshd_f32(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BSHD
        ),
    ],
) -> ContiguousKVCache[
    DType.float32,
    KVCacheStaticParams(num_heads=6, head_size=48, layout=KVCacheLayout.BSHD),
]:
    return _key_cache_for_layer(layer_idx, kv_collection)


@mogg_register("key_cache_for_layer_h6_d48_bhsd_f32")
@export
fn key_cache_for_layer_h6_d48_bhsd_f32(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BHSD
        ),
    ],
) -> ContiguousKVCache[
    DType.float32,
    KVCacheStaticParams(num_heads=6, head_size=48, layout=KVCacheLayout.BHSD),
]:
    return _key_cache_for_layer(layer_idx, kv_collection)


@mogg_register("key_cache_for_layer_h8_d128_bshd_f32")
@export
fn key_cache_for_layer_h8_d128_bshd_f32(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BSHD
        ),
    ],
) -> ContiguousKVCache[
    DType.float32,
    KVCacheStaticParams(num_heads=8, head_size=128, layout=KVCacheLayout.BSHD),
]:
    return _key_cache_for_layer(layer_idx, kv_collection)


@mogg_register("key_cache_for_layer_h8_d128_bhsd_f32")
@export
fn key_cache_for_layer_h8_d128_bhsd_f32(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BHSD
        ),
    ],
) -> ContiguousKVCache[
    DType.float32,
    KVCacheStaticParams(num_heads=8, head_size=128, layout=KVCacheLayout.BHSD),
]:
    return _key_cache_for_layer(layer_idx, kv_collection)


fn _key_cache_for_layer[
    type: DType, kv_params: KVCacheStaticParams
](
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[type, kv_params],
) -> ContiguousKVCache[type, kv_params]:
    """Retrieves the Key cache for the given layer."""
    return kv_collection.get_key_cache(int(layer_idx))


@mogg_register("value_cache_for_layer_h8_d128_bshd_bf16")
@export
fn value_cache_for_layer_h8_d128_bshd_bf16(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BSHD
        ),
    ],
) -> ContiguousKVCache[
    DType.bfloat16,
    KVCacheStaticParams(num_heads=6, head_size=48, layout=KVCacheLayout.BSHD),
]:
    return _value_cache_for_layer(layer_idx, kv_collection)


@mogg_register("value_cache_for_layer_h8_d128_bhsd_bf16")
@export
fn value_cache_for_layer_h8_d128_bhsd_bf16(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BHSD
        ),
    ],
) -> ContiguousKVCache[
    DType.bfloat16,
    KVCacheStaticParams(num_heads=6, head_size=48, layout=KVCacheLayout.BHSD),
]:
    return _value_cache_for_layer(layer_idx, kv_collection)


@mogg_register("value_cache_for_layer_h6_d48_bshd_f32")
@export
fn value_cache_for_layer_h6_d48_bshd_f32(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BSHD
        ),
    ],
) -> ContiguousKVCache[
    DType.float32,
    KVCacheStaticParams(num_heads=6, head_size=48, layout=KVCacheLayout.BSHD),
]:
    return _value_cache_for_layer(layer_idx, kv_collection)


@mogg_register("value_cache_for_layer_h6_d48_bhsd_f32")
@export
fn value_cache_for_layer_h6_d48_bhsd_f32(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BHSD
        ),
    ],
) -> ContiguousKVCache[
    DType.float32,
    KVCacheStaticParams(num_heads=6, head_size=48, layout=KVCacheLayout.BHSD),
]:
    return _value_cache_for_layer(layer_idx, kv_collection)


@mogg_register("value_cache_for_layer_h8_d128_bshd_f32")
@export
fn value_cache_for_layer_h8_d128_bshd_f32(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BSHD
        ),
    ],
) -> ContiguousKVCache[
    DType.float32,
    KVCacheStaticParams(num_heads=8, head_size=128, layout=KVCacheLayout.BSHD),
]:
    return _value_cache_for_layer(layer_idx, kv_collection)


@mogg_register("value_cache_for_layer_h8_d128_bhsd_f32")
@export
fn value_cache_for_layer_h8_d128_bhsd_f32(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BHSD
        ),
    ],
) -> ContiguousKVCache[
    DType.float32,
    KVCacheStaticParams(num_heads=8, head_size=128, layout=KVCacheLayout.BHSD),
]:
    return _value_cache_for_layer(layer_idx, kv_collection)


fn _value_cache_for_layer[
    type: DType,
    kv_params: KVCacheStaticParams,
](
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[type, kv_params],
) -> ContiguousKVCache[type, kv_params]:
    """Retrieves the Value cache for the given layer."""
    return kv_collection.get_value_cache(int(layer_idx))


@mogg_register("matmul_kv_cache_h6_d48_bshd")
@export
fn matmul_kv_cache_h6_d48_bshd[
    type: DType,
    hidden_state_shape: DimList,
    weight_shape: DimList,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[type, 3, hidden_state_shape],
    weight: NDBuffer[type, 2, weight_shape],
    cache: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BSHD
        ),
    ],
    ctx: MojoCallContextPtr,
) -> ContiguousKVCache[
    type,
    KVCacheStaticParams(num_heads=6, head_size=48, layout=KVCacheLayout.BSHD),
]:
    """Performs a matmul, writing the output into a mutable ContiguousKVCache object.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        cache: The historical ContiguousKVCache, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        ctx: The call context pointer, passed by the graph compiler.
    """
    return _matmul_kv_cache[target=target](hidden_state, weight, cache, ctx)


@mogg_register("matmul_kv_cache_h6_d48_bhsd")
@export
fn matmul_kv_cache_h6_d48_bhsd[
    type: DType,
    hidden_state_shape: DimList,
    weight_shape: DimList,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[type, 3, hidden_state_shape],
    weight: NDBuffer[type, 2, weight_shape],
    cache: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BHSD
        ),
    ],
    ctx: MojoCallContextPtr,
) -> ContiguousKVCache[
    type,
    KVCacheStaticParams(num_heads=6, head_size=48, layout=KVCacheLayout.BHSD),
]:
    """Performs a matmul, writing the output into a mutable ContiguousKVCache object.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        cache: The historical ContiguousKVCache, with logical shape:
            (batch_size, num_kv_heads, max_seq_len, head_size).
        ctx: The call context pointer, passed by the graph compiler.
    """
    return _matmul_kv_cache[target=target](hidden_state, weight, cache, ctx)


@mogg_register("matmul_kv_cache_h8_d128_bshd")
@export
fn matmul_kv_cache_h8_d128_bshd[
    type: DType,
    hidden_state_shape: DimList,
    weight_shape: DimList,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[type, 3, hidden_state_shape],
    weight: NDBuffer[type, 2, weight_shape],
    cache: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BSHD
        ),
    ],
    ctx: MojoCallContextPtr,
) -> ContiguousKVCache[
    type,
    KVCacheStaticParams(num_heads=8, head_size=128, layout=KVCacheLayout.BSHD),
]:
    """Performs a matmul, writing the output into a mutable ContiguousKVCache object.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        cache: The historical ContiguousKVCache, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        ctx: The call context pointer, passed by the graph compiler.
    """
    return _matmul_kv_cache[target=target](hidden_state, weight, cache, ctx)


@mogg_register("matmul_kv_cache_h8_d128_bhsd")
@export
fn matmul_kv_cache_h8_d128_bhsd[
    type: DType,
    hidden_state_shape: DimList,
    weight_shape: DimList,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[type, 3, hidden_state_shape],
    weight: NDBuffer[type, 2, weight_shape],
    cache: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BHSD
        ),
    ],
    ctx: MojoCallContextPtr,
) -> ContiguousKVCache[
    type,
    KVCacheStaticParams(num_heads=8, head_size=128, layout=KVCacheLayout.BHSD),
]:
    """Performs a matmul, writing the output into a mutable ContiguousKVCache object.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        cache: The historical ContiguousKVCache, with logical shape:
            (batch_size, num_kv_heads, max_seq_len, head_size).
        ctx: The call context pointer, passed by the graph compiler.
    """
    return _matmul_kv_cache[target=target](hidden_state, weight, cache, ctx)


@always_inline
fn _matmul_kv_cache[
    type: DType,
    hidden_state_shape: DimList,
    weight_shape: DimList,
    kv_params: KVCacheStaticParams,
    *,
    target: StringLiteral,
](
    hidden_state: NDBuffer[type, 3, hidden_state_shape],
    weight: NDBuffer[type, 2, weight_shape],
    cache: ContiguousKVCache[type, kv_params],
    context: MojoCallContextPtr = MojoCallContextPtr(),
) -> ContiguousKVCache[type, kv_params]:
    """Helper for performing matmul with custom ContiguousKVCache types.

    Parameters:
        type: The data type of all inputs and outputs
        hidden_state_shape: The static shapes for the hidden_state tensor
        weight_shape: The static shapes for the weight tensor
        kv_params: The static parameters for our KVCache object
        target: StringLiteral identifying the device target (cpu vs cuda)

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size)
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size)
        cache: The ContiguousKVCache, with shape determined by `kv_params.layout`
        context: Pointer containing the runtime context for the target device.
    """
    var BS = hidden_state.dim[0]()
    var SEQ_LEN = hidden_state.dim[1]()
    var N = weight.dim[0]()
    var K = weight.dim[1]()

    @parameter
    @__copy_capture(cache, SEQ_LEN)
    fn write_to_cache[
        type_: DType, width: Int
    ](idx: StaticIntTuple[2], val: SIMD[type_, width]):
        var bs_and_seq = divmod(idx[0], SEQ_LEN)
        var b_idx = bs_and_seq[0]
        var t_idx = bs_and_seq[1]
        var head_and_dim = divmod(idx[1], kv_params.head_size)
        var h_idx = head_and_dim[0]
        var hd_idx = head_and_dim[1]

        var valid_len = cache.cache_length(b_idx)
        var cache_t_idx = t_idx + valid_len
        cache.store[width](
            b_idx,
            h_idx,
            cache_t_idx,
            hd_idx,
            rebind[SIMD[type, width]](val),
        )

    var hidden_state_2d = NDBuffer[
        type, 2, DimList(Dim(), hidden_state.shape.get[2]())
    ](
        hidden_state.data,
        StaticIntTuple[2](BS * SEQ_LEN, K),
    )

    var c_ptr = UnsafePointer[Scalar[type]].alloc(
        BS * SEQ_LEN * N
    ) if target == "cpu" else UnsafePointer[Scalar[type]]()

    var c_nd = NDBuffer[type, 2, DimList(Dim(), weight.shape.get[1]())](
        c_ptr,
        StaticIntTuple[2](BS * SEQ_LEN, N),
    )
    matmul[
        type,
        hidden_state_2d.shape,
        type,
        weight.shape,
        type,
        c_nd.shape,
        transpose_b=True,
        elementwise_lambda_fn=write_to_cache,
        target=target,
    ](c_nd, hidden_state_2d, weight, ctx=context)
    c_nd.data.free()
    return cache


@mogg_register("fused_qkv_matmul_kv_cache_h6_d48_bshd")
@export
fn fused_qkv_matmul_kv_cache_h6_d48_bshd[
    type: DType,
    hidden_state_shape: DimList,
    weight_shape: DimList,
    output_shape: DimList,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[type, 3, hidden_state_shape],
    weight: NDBuffer[type, 2, weight_shape],
    k_cache: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BSHD
        ),
    ],
    v_cache: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BSHD
        ),
    ],
    output: NDBuffer[type, 3, output_shape],
    ctx: MojoCallContextPtr,
):
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        k_cache: The historical ContiguousKVCache for keys, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        v_cache: The historical ContiguousKVCache for values, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        ctx: The call context pointer, passed by the graph compiler.
    """
    return _fused_qkv_matmul_kv_cache[target=target](
        hidden_state, weight, k_cache, v_cache, output, ctx
    )


@mogg_register("fused_qkv_matmul_kv_cache_h6_d48_bhsd")
@export
fn fused_qkv_matmul_kv_cache_h6_d48_bhsd[
    type: DType,
    hidden_state_shape: DimList,
    weight_shape: DimList,
    output_shape: DimList,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[type, 3, hidden_state_shape],
    weight: NDBuffer[type, 2, weight_shape],
    k_cache: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BHSD
        ),
    ],
    v_cache: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BHSD
        ),
    ],
    output: NDBuffer[type, 3, output_shape],
    ctx: MojoCallContextPtr,
):
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        k_cache: The historical ContiguousKVCache for keys, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        v_cache: The historical ContiguousKVCache for values, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        ctx: The call context pointer, passed by the graph compiler.
    """
    return _fused_qkv_matmul_kv_cache[target=target](
        hidden_state, weight, k_cache, v_cache, output, ctx
    )


@mogg_register("fused_qkv_matmul_kv_cache_h8_d128_bshd")
@export
fn fused_qkv_matmul_kv_cache_h8_d128_bshd[
    type: DType,
    hidden_state_shape: DimList,
    weight_shape: DimList,
    output_shape: DimList,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[type, 3, hidden_state_shape],
    weight: NDBuffer[type, 2, weight_shape],
    k_cache: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BSHD
        ),
    ],
    v_cache: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BSHD
        ),
    ],
    output: NDBuffer[type, 3, output_shape],
    ctx: MojoCallContextPtr,
):
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        k_cache: The historical ContiguousKVCache for keys, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        v_cache: The historical ContiguousKVCache for values, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        ctx: The call context pointer, passed by the graph compiler.
    """
    return _fused_qkv_matmul_kv_cache[target=target](
        hidden_state, weight, k_cache, v_cache, output, ctx
    )


@mogg_register("fused_qkv_matmul_kv_cache_h8_d128_bhsd")
@export
fn fused_qkv_matmul_kv_cache_h8_d128_bhsd[
    type: DType,
    hidden_state_shape: DimList,
    weight_shape: DimList,
    output_shape: DimList,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[type, 3, hidden_state_shape],
    weight: NDBuffer[type, 2, weight_shape],
    k_cache: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BHSD
        ),
    ],
    v_cache: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BHSD
        ),
    ],
    output: NDBuffer[type, 3, output_shape],
    ctx: MojoCallContextPtr,
):
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        k_cache: The historical ContiguousKVCache for keys, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        v_cache: The historical ContiguousKVCache for values, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        ctx: The call context pointer, passed by the graph compiler.
    """
    return _fused_qkv_matmul_kv_cache[target=target](
        hidden_state, weight, k_cache, v_cache, output, ctx
    )


alias embed_fn_type = fn[type: DType, width: Int] (
    StaticIntTuple[4], SIMD[type, width]
) capturing -> SIMD[type, width]


@always_inline
fn _fused_qkv_matmul_kv_cache[
    type: DType,
    hidden_state_shape: DimList,
    weight_shape: DimList,
    output_shape: DimList,
    kv_params: KVCacheStaticParams,
    *,
    target: StringLiteral,
    q_embed_fn: OptionalReg[embed_fn_type] = None,
    k_embed_fn: OptionalReg[embed_fn_type] = None,
](
    hidden_state: NDBuffer[type, 3, hidden_state_shape],
    weight: NDBuffer[type, 2, weight_shape],
    k_cache: ContiguousKVCache[type, kv_params],
    v_cache: ContiguousKVCache[type, kv_params],
    output: NDBuffer[type, 3, output_shape],
    context: MojoCallContextPtr = MojoCallContextPtr(),
):
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        k_cache: The historical ContiguousKVCache for keys, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        v_cache: The historical ContiguousKVCache for values, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        context: The call context pointer, passed by the graph compiler.
    """
    var BS = hidden_state.dim[0]()
    var SEQ_LEN = hidden_state.dim[1]()
    alias N = weight_shape.get[0]()
    alias K = weight_shape.get[1]()

    var q_dim = output.dim[2]()
    var k_dim = kv_params.head_size * kv_params.num_heads
    var qk_offset = q_dim + k_dim

    @parameter
    @__copy_capture(k_cache, v_cache, output, q_dim, qk_offset, SEQ_LEN)
    fn write_to_cache[
        type_: DType, width: Int
    ](idx: StaticIntTuple[2], val: SIMD[type_, width]):
        var bs_and_seq = divmod(idx[0], SEQ_LEN)
        var b_idx = bs_and_seq[0]
        var t_idx = bs_and_seq[1]
        if idx[1] < q_dim:
            var output_val = val

            @parameter
            if q_embed_fn:
                alias q_fn_val = q_embed_fn.value()

                var head_and_dim = divmod(idx[1], kv_params.head_size)
                var h_idx = head_and_dim[0]
                var hd_idx = head_and_dim[1]
                output_val = q_fn_val((b_idx, t_idx, h_idx, hd_idx), output_val)

            output.store(
                (b_idx, t_idx, idx[1]), rebind[SIMD[type, width]](output_val)
            )
            return

        var h_idx: Int
        var hd_idx: Int
        var cache: ContiguousKVCache[type, kv_params]
        var output_val = val
        if idx[1] < qk_offset:
            var head_and_dim = divmod(idx[1] - q_dim, kv_params.head_size)
            h_idx = head_and_dim[0]
            hd_idx = head_and_dim[1]
            cache = k_cache

            @parameter
            if k_embed_fn:
                alias k_fn_val = k_embed_fn.value()
                output_val = k_fn_val((b_idx, t_idx, h_idx, hd_idx), output_val)
        else:
            cache = v_cache
            var head_and_dim = divmod(idx[1] - qk_offset, kv_params.head_size)
            h_idx = head_and_dim[0]
            hd_idx = head_and_dim[1]

        var valid_len = cache.cache_length(b_idx)
        var cache_t_idx = t_idx + valid_len
        cache.store(
            b_idx,
            h_idx,
            cache_t_idx,
            hd_idx,
            rebind[SIMD[type, width]](output_val),
        )

    var hidden_state_2d = NDBuffer[
        type, 2, DimList(Dim(), hidden_state.shape.get[2]())
    ](
        hidden_state.data,
        StaticIntTuple[2](BS * SEQ_LEN, K),
    )

    var c_ptr = UnsafePointer[Scalar[type]].alloc(
        BS * SEQ_LEN * N
    ) if target == "cpu" else UnsafePointer[Scalar[type]]()

    var c_nd = NDBuffer[type, 2, DimList(Dim(), N)](
        c_ptr,
        StaticIntTuple[2](BS * SEQ_LEN, N),
    )
    matmul[
        type,
        hidden_state_2d.shape,
        type,
        weight.shape,
        type,
        c_nd.shape,
        transpose_b=True,
        elementwise_lambda_fn=write_to_cache,
        target=target,
    ](c_nd, hidden_state_2d, weight, ctx=context)
    c_nd.data.free()


@mogg_register("fused_qkv_matmul_kv_cache_w_rope_h6_d48_bshd")
@export
fn fused_qkv_matmul_kv_cache_w_rope_h6_d48_bshd[
    type: DType,
    hidden_state_shape: DimList,
    weight_shape: DimList,
    freqs_shape: DimList,
    output_shape: DimList,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[type, 3, hidden_state_shape],
    weight: NDBuffer[type, 2, weight_shape],
    freqs: NDBuffer[type, 2, freqs_shape],
    k_cache: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BSHD
        ),
    ],
    v_cache: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BSHD
        ),
    ],
    output: NDBuffer[type, 3, output_shape],
    context: MojoCallContextPtr = MojoCallContextPtr(),
):
    return _fused_qkv_matmul_kv_cache_w_rope[target=target](
        hidden_state, weight, freqs, k_cache, v_cache, output, context
    )


@mogg_register("fused_qkv_matmul_kv_cache_w_rope_h6_d48_bhsd")
@export
fn fused_qkv_matmul_kv_cache_w_rope_h6_d48_bhsd[
    type: DType,
    hidden_state_shape: DimList,
    weight_shape: DimList,
    freqs_shape: DimList,
    output_shape: DimList,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[type, 3, hidden_state_shape],
    weight: NDBuffer[type, 2, weight_shape],
    freqs: NDBuffer[type, 2, freqs_shape],
    k_cache: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BHSD
        ),
    ],
    v_cache: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BHSD
        ),
    ],
    output: NDBuffer[type, 3, output_shape],
    context: MojoCallContextPtr = MojoCallContextPtr(),
):
    return _fused_qkv_matmul_kv_cache_w_rope[target=target](
        hidden_state, weight, freqs, k_cache, v_cache, output, context
    )


@mogg_register("fused_qkv_matmul_kv_cache_w_rope_h8_d128_bshd")
@export
fn fused_qkv_matmul_kv_cache_w_rope_h8_d128_bshd[
    type: DType,
    hidden_state_shape: DimList,
    weight_shape: DimList,
    freqs_shape: DimList,
    output_shape: DimList,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[type, 3, hidden_state_shape],
    weight: NDBuffer[type, 2, weight_shape],
    freqs: NDBuffer[type, 2, freqs_shape],
    k_cache: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BSHD
        ),
    ],
    v_cache: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BSHD
        ),
    ],
    output: NDBuffer[type, 3, output_shape],
    context: MojoCallContextPtr = MojoCallContextPtr(),
):
    return _fused_qkv_matmul_kv_cache_w_rope[target=target](
        hidden_state, weight, freqs, k_cache, v_cache, output, context
    )


@mogg_register("fused_qkv_matmul_kv_cache_w_rope_h8_d128_bhsd")
@export
fn fused_qkv_matmul_kv_cache_w_rope_h8_d128_bhsd[
    type: DType,
    hidden_state_shape: DimList,
    weight_shape: DimList,
    freqs_shape: DimList,
    output_shape: DimList,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[type, 3, hidden_state_shape],
    weight: NDBuffer[type, 2, weight_shape],
    freqs: NDBuffer[type, 2, freqs_shape],
    k_cache: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BHSD
        ),
    ],
    v_cache: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BHSD
        ),
    ],
    output: NDBuffer[type, 3, output_shape],
    context: MojoCallContextPtr = MojoCallContextPtr(),
):
    return _fused_qkv_matmul_kv_cache_w_rope[target=target](
        hidden_state, weight, freqs, k_cache, v_cache, output, context
    )


@always_inline
fn _fused_qkv_matmul_kv_cache_w_rope[
    type: DType,
    hidden_state_shape: DimList,
    weight_shape: DimList,
    freqs_shape: DimList,
    output_shape: DimList,
    kv_params: KVCacheStaticParams,
    *,
    target: StringLiteral,
](
    hidden_state: NDBuffer[type, 3, hidden_state_shape],
    weight: NDBuffer[type, 2, weight_shape],
    freqs: NDBuffer[type, 2, freqs_shape],
    k_cache: ContiguousKVCache[type, kv_params],
    v_cache: ContiguousKVCache[type, kv_params],
    output: NDBuffer[type, 3, output_shape],
    context: MojoCallContextPtr = MojoCallContextPtr(),
):
    @always_inline
    @parameter
    @__copy_capture(freqs)
    fn rope_fn[
        type_: DType, width: Int
    ](idx: StaticIntTuple[4], val: SIMD[type_, width]) -> SIMD[type_, width]:
        @parameter
        if width == 1:
            print("ROPE KERNEL CALLED WITH SINGLE VALUE, EXPECTED AT LEAST 2")
            return val
        else:
            var t_idx = idx[1]
            var hd_idx = idx[3]

            var x_c = val.deinterleave()
            var x_re = x_c[0]
            var x_im = x_c[1]

            var f_idx = StaticIntTuple[2](t_idx, hd_idx)
            var f_c_temp = rebind[SIMD[type_, width]](
                freqs.load[width=width](f_idx)
            )

            var f_c = f_c_temp.deinterleave()
            var f_re = f_c[0]
            var f_im = f_c[1]

            var r_re = (x_re * f_re) - (x_im * f_im)
            var r_im = (x_re * f_im) + (x_im * f_re)

            return rebind[SIMD[type_, width]](r_re.interleave(r_im))

    return _fused_qkv_matmul_kv_cache[
        target=target, q_embed_fn=rope_fn, k_embed_fn=rope_fn
    ](hidden_state, weight, k_cache, v_cache, output, context)


@mogg_register("flash_attention_kv_cache_h6_d48_bshd")
@export
fn flash_attention_kv_cache_h6_d48_bshd[
    type: DType,
    q_shape: DimList,
    mask_shape: DimList,
    scale_shape: DimList,
    output_shape: DimList,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[type, 4, q_shape],
    k: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BSHD
        ),
    ],
    v: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BSHD
        ),
    ],
    mask: NDBuffer[type, 2, mask_shape],
    scale: NDBuffer[DType.float32, 1, scale_shape],
    output: NDBuffer[type, 4, output_shape],
    context: MojoCallContextPtr,
):
    return _flash_attention_kv_cache[target=target](
        q, k, v, mask, scale, output, context
    )


@mogg_register("flash_attention_kv_cache_h6_d48_bhsd")
@export
fn flash_attention_kv_cache_h6_d48_bhsd[
    type: DType,
    q_shape: DimList,
    mask_shape: DimList,
    scale_shape: DimList,
    output_shape: DimList,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[type, 4, q_shape],
    k: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BHSD
        ),
    ],
    v: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BHSD
        ),
    ],
    mask: NDBuffer[type, 2, mask_shape],
    scale: NDBuffer[DType.float32, 1, scale_shape],
    output: NDBuffer[type, 4, output_shape],
    context: MojoCallContextPtr,
):
    return _flash_attention_kv_cache[target=target](
        q, k, v, mask, scale, output, context
    )


@mogg_register("flash_attention_kv_cache_h8_d128_bshd")
@export
fn flash_attention_kv_cache_h8_d128_bshd[
    type: DType,
    q_shape: DimList,
    mask_rank: Int,
    mask_shape: DimList,
    scale_shape: DimList,
    output_shape: DimList,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[type, 4, q_shape],
    k: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BSHD
        ),
    ],
    v: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BSHD
        ),
    ],
    mask: NDBuffer[type, mask_rank, mask_shape],
    scale: NDBuffer[DType.float32, 1, scale_shape],
    output: NDBuffer[type, 4, output_shape],
    context: MojoCallContextPtr,
):
    return _flash_attention_kv_cache[target=target](
        q, k, v, mask, scale, output, context
    )


@mogg_register("flash_attention_kv_cache_h8_d128_bhsd")
@export
fn flash_attention_kv_cache_h8_d128_bhsd[
    type: DType,
    q_shape: DimList,
    mask_rank: Int,
    mask_shape: DimList,
    scale_shape: DimList,
    output_shape: DimList,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[type, 4, q_shape],
    k: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BHSD
        ),
    ],
    v: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BHSD
        ),
    ],
    mask: NDBuffer[type, mask_rank, mask_shape],
    scale: NDBuffer[DType.float32, 1, scale_shape],
    output: NDBuffer[type, 4, output_shape],
    context: MojoCallContextPtr,
):
    return _flash_attention_kv_cache[target=target](
        q, k, v, mask, scale, output, context
    )


fn _flash_attention_kv_cache[
    type: DType,
    mask_rank: Int,
    q_shape: DimList,
    mask_shape: DimList,
    scale_shape: DimList,
    output_shape: DimList,
    kv_params: KVCacheStaticParams,
    target: StringLiteral,
](
    q: NDBuffer[type, 4, q_shape],
    k: ContiguousKVCache[type, kv_params],
    v: ContiguousKVCache[type, kv_params],
    mask: NDBuffer[type, mask_rank, mask_shape],
    scale: NDBuffer[DType.float32, 1, scale_shape],
    output: NDBuffer[type, 4, output_shape],
    context: MojoCallContextPtr,
):
    """Performs flash attention using k and v caches from ContiguousKVCache custom types.

    Args:
        q: NDBuffer with shape (batch_size, num_heads, seq_len, head_size).
        k: ContiguousKVCache type with logical shape (batch_size, num_heads, max_seq_len, head_size).
        v: ContiguousKVCache type with logical shape (batch_size, num_heads, max_seq_len, head_size).
        mask: The attention mask to apply to the score matrix.
        scale: The scaled factor in scaled-dot product attention. Usually isqrt(head_size).
        output: The Pre-allocated output buffer to write results to. Has shape:
            (batch_size, num_heads, seq_len, head_size).
        context: Pointer containing the runtime context for the target device.
    """

    @parameter
    if target == "cpu":
        return _flash_attention_kv_cache_cpu[
            type,
            mask_rank,
            q_shape,
            mask_shape,
            scale_shape,
            output_shape,
        ](q, k, v, mask, scale, output, context)
    else:
        return _flash_attention_kv_cache_gpu[
            type,
            mask_rank,
            q_shape,
            mask_shape,
            scale_shape,
            output_shape,
        ](q, k, v, mask, scale, output, context)


fn _flash_attention_kv_cache_cpu[
    type: DType,
    mask_rank: Int,
    q_shape: DimList,
    mask_shape: DimList,
    scale_shape: DimList,
    output_shape: DimList,
    kv_params: KVCacheStaticParams,
](
    q: NDBuffer[type, 4, q_shape],
    k: ContiguousKVCache[type, kv_params],
    v: ContiguousKVCache[type, kv_params],
    mask: NDBuffer[type, mask_rank, mask_shape],
    scale: NDBuffer[DType.float32, 1, scale_shape],
    output: NDBuffer[type, 4, output_shape],
    context: MojoCallContextPtr = MojoCallContextPtr(),
):
    constrained[
        kv_params.layout == KVCacheLayout.BHSD,
        "CPU flash attention only supports BHSD layout",
    ]()

    @parameter
    @__copy_capture(k)
    fn input_k_fn[
        width: Int, rank: Int
    ](idx: StaticIntTuple[rank]) -> SIMD[type, width]:
        var bs = idx[0]
        var head_idx = idx[1]
        var seq = idx[2]
        var head_d_idx = idx[3]

        var retval = k.load[width=width](bs, head_idx, seq, head_d_idx)
        return retval

    @parameter
    @__copy_capture(v)
    fn input_v_fn[
        width: Int, rank: Int
    ](idx: StaticIntTuple[rank]) -> SIMD[type, width]:
        var bs = idx[0]
        var head_idx = idx[1]
        var seq = idx[2]
        var head_d_idx = idx[3]
        var retval = v.load[width=width](bs, head_idx, seq, head_d_idx)
        return retval

    @parameter
    @__copy_capture(mask)
    fn input_mask_fn[
        width: Int, rank: Int
    ](idx: StaticIntTuple[rank]) -> SIMD[type, width]:
        @parameter
        if mask_rank == 4:
            return mask.load[width=width](
                rebind[StaticIntTuple[mask_rank]](idx)
            )
        else:
            return mask.load[width=width]((idx[2], idx[3]))

    var batch_size = q.dim[0]()
    var num_heads = q.dim[1]()
    var depth = q.dim[3]()
    var new_seq_len = q.dim[2]()
    var cache_seq_len = int(k.cache_length(0))
    var seq_len = new_seq_len + cache_seq_len
    var fa_k_shape = StaticIntTuple[4](
        batch_size, kv_params.num_heads, seq_len, depth
    )
    var fa_v_shape = StaticIntTuple[4](
        batch_size, kv_params.num_heads, seq_len, depth
    )

    cpu_flash_attention[
        type,
        4,
        input_k_fn,
        input_v_fn,
        input_mask_fn,
        output.shape,
        transpose_k=True,
    ](
        q.make_dims_unknown(),
        fa_k_shape,
        fa_v_shape,
        output,
        scale.load[width=1](0),
    )


fn _flash_attention_kv_cache_gpu[
    type: DType,
    mask_rank: Int,
    q_shape: DimList,
    mask_shape: DimList,
    scale_shape: DimList,
    output_shape: DimList,
    kv_params: KVCacheStaticParams,
](
    q: NDBuffer[type, 4, q_shape],
    k: ContiguousKVCache[type, kv_params],
    v: ContiguousKVCache[type, kv_params],
    mask: NDBuffer[type, mask_rank, mask_shape],
    scale: NDBuffer[DType.float32, 1, scale_shape],
    output: NDBuffer[type, 4, output_shape],
    context: MojoCallContextPtr = MojoCallContextPtr(),
):
    constrained[
        kv_params.layout == KVCacheLayout.BSHD,
        "GPU Flash attention only supports the BSHD layout.",
    ]()

    alias wrapped_mask_rank = mask_rank if mask_rank == 4 else 3
    var mask_nd: NDBuffer[
        type,
        wrapped_mask_rank,
        DimList.create_unknown[wrapped_mask_rank](),
    ]

    @parameter
    if mask_rank == 2:
        mask_nd = NDBuffer[
            type,
            wrapped_mask_rank,
            DimList.create_unknown[wrapped_mask_rank](),
        ](
            mask.data,
            StaticIntTuple[wrapped_mask_rank](
                q.dim[0](), mask.dim[0](), mask.dim[1]()
            ),
        )
    else:
        mask_nd = rebind[
            NDBuffer[
                type,
                wrapped_mask_rank,
                DimList.create_unknown[wrapped_mask_rank](),
            ]
        ](mask)

    # GPU flash attention kernel gets the cache length from the k tensor shape
    # TODO remove this an instead pass in explicit KVCache lengths to the GPU kernel.
    # KERN-725
    var valid_length = int(k.cache_length(0) + q.dim[1]())
    var k_shape = k._block.dynamic_shape
    var kv_shape = StaticIntTuple[4](
        k_shape[0], valid_length, k_shape[2], k_shape[3]
    )
    var k_nd = NDBuffer[type, 4, k._internal_block_shape](
        k._block.data, kv_shape
    )
    var v_nd = NDBuffer[type, 4, v._internal_block_shape](
        v._block.data, kv_shape
    )

    try:
        gpu_flash_attention[
            4,
            wrapped_mask_rank,
            q.shape,
            k_nd.shape,
            v_nd.shape,
            mask_nd.shape,
            output.shape,
            q.type,
            k.type,
            v.type,
            mask_nd.type,
            output.type,
            add_attn_mask=True,
            use_tensor_core=True,
            target="cuda",
        ](
            output,
            q,
            k_nd,
            v_nd,
            mask_nd,
            # TODO take scale from argument GRA-750
            isqrt(Float32(kv_params.head_size)),
            context,
        )
    except e:
        print("Error in GPU Flash Attention:", e)
