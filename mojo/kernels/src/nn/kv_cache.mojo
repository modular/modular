# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from buffer import DimList, NDBuffer, Dim
from math import gcd, isqrt
from memory import UnsafePointer
from collections import Optional, OptionalReg, InlineArray
from sys.info import _current_target, simdwidthof
from utils import Index
from utils.numerics import min_finite, isnan
from os import abort
from memory import memcpy

from algorithm.functional import elementwise
from gpu.host import Stream, DeviceContext, DeviceBuffer
from gpu.host._compile import _get_nvptx_target
from linalg import transpose
from linalg.matmul_gpu import _matmul_gpu
from linalg.matmul import _matmul_cpu, elementwise_epilogue_type
from nn.flash_attention import flash_attention as cpu_flash_attention
from nn.mha import flash_attention as gpu_flash_attention
from register import mogg_register
from runtime.asyncrt import (
    MojoCallContextPtr,
)
from utils import StaticIntTuple

from kv_cache.types import (
    _default_max_batch_size,
    ContiguousKVCache,
    ContiguousKVCacheCollection,
    KVCacheLayout,
    KVCacheStaticParams,
)
from runtime.tracing import Trace, TraceLevel


@mogg_register("kv_cache_length_h8_d128_bshd_bf16")
@export
fn kv_cache_length_h8_d128_bshd_bf16[
    target: StringLiteral = "cpu"
](
    kv_collection: ContiguousKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BSHD
        ),
    ],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
):
    return _kv_cache_length[target=target](kv_collection, output, ctx)


@mogg_register("kv_cache_length_h8_d128_bhsd_bf16")
@export
fn kv_cache_length_h8_d128_bhsd_bf16[
    target: StringLiteral = "cpu"
](
    kv_collection: ContiguousKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BHSD
        ),
    ],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
):
    return _kv_cache_length[target=target](kv_collection, output, ctx)


@mogg_register("kv_cache_length_h6_d48_bshd_f32")
@export
fn kv_cache_length_h6_d48_bshd_f32[
    target: StringLiteral = "cpu"
](
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BSHD
        ),
    ],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
):
    return _kv_cache_length[target=target](kv_collection, output, ctx)


@mogg_register("kv_cache_length_h6_d48_bhsd_f32")
@export
fn kv_cache_length_h6_d48_bhsd_f32[
    target: StringLiteral = "cpu"
](
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BHSD
        ),
    ],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
):
    return _kv_cache_length[target=target](kv_collection, output, ctx)


@mogg_register("kv_cache_length_h8_d128_bshd_f32")
@export
fn kv_cache_length_h8_d128_bshd_f32[
    target: StringLiteral = "cpu"
](
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BSHD
        ),
    ],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
):
    return _kv_cache_length[target=target](kv_collection, output, ctx)


@mogg_register("kv_cache_length_h8_d128_bhsd_f32")
@export
fn kv_cache_length_h8_d128_bhsd_f32[
    target: StringLiteral = "cpu"
](
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BHSD
        ),
    ],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
):
    return _kv_cache_length[target=target](kv_collection, output, ctx)


@mogg_register("kv_cache_length_h1_d10_bhsd_f32")
@export
fn kv_cache_length_h1_d10_bhsd_f32[
    target: StringLiteral = "cpu"
](
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=1, head_size=10, layout=KVCacheLayout.BHSD
        ),
    ],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
):
    return _kv_cache_length[target=target](kv_collection, output, ctx)


@mogg_register("kv_cache_length_h1_d10_bshd_f32")
@export
fn kv_cache_length_h1_d10_bshd_f32[
    target: StringLiteral = "cpu"
](
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=1, head_size=10, layout=KVCacheLayout.BSHD
        ),
    ],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
):
    return _kv_cache_length[target=target](kv_collection, output, ctx)


@mogg_register("kv_cache_length_h1_d10_bhsd_bf16")
@export
fn kv_cache_length_h1_d10_bhsd_bf16[
    target: StringLiteral = "cpu"
](
    kv_collection: ContiguousKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(
            num_heads=1, head_size=10, layout=KVCacheLayout.BHSD
        ),
    ],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
):
    return _kv_cache_length[target=target](kv_collection, output, ctx)


@mogg_register("kv_cache_length_h1_d10_bshd_bf16")
@export
fn kv_cache_length_h1_d10_bshd_bf16[
    target: StringLiteral = "cpu"
](
    kv_collection: ContiguousKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(
            num_heads=1, head_size=10, layout=KVCacheLayout.BSHD
        ),
    ],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
):
    return _kv_cache_length[target=target](kv_collection, output, ctx)


@mogg_register("kv_cache_length_h8_d64_bshd_bf16")
@export
fn kv_cache_length_h8_d64_bshd_bf16[
    target: StringLiteral = "cpu"
](
    kv_collection: ContiguousKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(
            num_heads=8, head_size=64, layout=KVCacheLayout.BSHD
        ),
    ],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
):
    return _kv_cache_length[target=target](kv_collection, output, ctx)


@mogg_register("kv_cache_length_h8_d64_bhsd_bf16")
@export
fn kv_cache_length_h8_d64_bhsd_bf16[
    target: StringLiteral = "cpu"
](
    kv_collection: ContiguousKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(
            num_heads=8, head_size=64, layout=KVCacheLayout.BHSD
        ),
    ],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
):
    return _kv_cache_length[target=target](kv_collection, output, ctx)


@mogg_register("kv_cache_length_h8_d64_bshd_f32")
@export
fn kv_cache_length_h8_d64_bshd_f32[
    target: StringLiteral = "cpu"
](
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=8, head_size=64, layout=KVCacheLayout.BSHD
        ),
    ],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
):
    return _kv_cache_length[target=target](kv_collection, output, ctx)


@mogg_register("kv_cache_length_h8_d64_bhsd_f32")
@export
fn kv_cache_length_h8_d64_bhsd_f32[
    target: StringLiteral = "cpu"
](
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=8, head_size=64, layout=KVCacheLayout.BHSD
        ),
    ],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
):
    return _kv_cache_length[target=target](kv_collection, output, ctx)


@always_inline
fn _kv_cache_length[
    type: DType, kv_params: KVCacheStaticParams, target: StringLiteral
](
    kv_collection: ContiguousKVCacheCollection[type, kv_params],
    output: NDBuffer[DType.uint32, 1],
    ctx: MojoCallContextPtr,
):
    """Returns the size of the cache in a ContiguousKVCacheCollection mo.opaque object.
    """

    @parameter
    if target != "cpu":
        var cuda_ctx = ctx.get_device_context()
        try:
            cuda_ctx.enqueue_copy_device_to_device(
                output.data,
                kv_collection.cache_lengths.data,
                output.dim[0](),
            )
        except e:
            abort("Failed to enqueue device_to_device copy:" + str(e))
    else:
        memcpy(
            output.data,
            kv_collection.cache_lengths.data,
            output.dim[0](),
        )


@mogg_register("key_cache_for_layer_h8_d128_bhsd_bf16")
@export
fn key_cache_for_layer_h8_d128_bhsd_bf16(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BHSD
        ),
    ],
) -> ContiguousKVCache[
    DType.bfloat16,
    KVCacheStaticParams(num_heads=8, head_size=128, layout=KVCacheLayout.BHSD),
]:
    return _key_cache_for_layer(layer_idx, kv_collection)


@mogg_register("key_cache_for_layer_h8_d128_bshd_bf16")
@export
fn key_cache_for_layer_h8_d128_bshd_bf16(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BSHD
        ),
    ],
) -> ContiguousKVCache[kv_collection.type, kv_collection.kv_params]:
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
) -> ContiguousKVCache[kv_collection.type, kv_collection.kv_params]:
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
) -> ContiguousKVCache[kv_collection.type, kv_collection.kv_params]:
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
) -> ContiguousKVCache[kv_collection.type, kv_collection.kv_params]:
    return _key_cache_for_layer(layer_idx, kv_collection)


@mogg_register("key_cache_for_layer_h1_d10_bhsd_f32")
@export
fn key_cache_for_layer_h1_d10_bhsd_f32(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=1, head_size=10, layout=KVCacheLayout.BHSD
        ),
    ],
) -> ContiguousKVCache[kv_collection.type, kv_collection.kv_params]:
    return _key_cache_for_layer(layer_idx, kv_collection)


@mogg_register("key_cache_for_layer_h1_d10_bshd_f32")
@export
fn key_cache_for_layer_h1_d10_bshd_f32(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=1, head_size=10, layout=KVCacheLayout.BSHD
        ),
    ],
) -> ContiguousKVCache[
    DType.float32,
    KVCacheStaticParams(num_heads=1, head_size=10, layout=KVCacheLayout.BSHD),
]:
    return _key_cache_for_layer(layer_idx, kv_collection)


@mogg_register("key_cache_for_layer_h1_d10_bhsd_bf16")
@export
fn key_cache_for_layer_h1_d10_bhsd_bf16(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(
            num_heads=1, head_size=10, layout=KVCacheLayout.BHSD
        ),
    ],
) -> ContiguousKVCache[kv_collection.type, kv_collection.kv_params]:
    return _key_cache_for_layer(layer_idx, kv_collection)


@mogg_register("key_cache_for_layer_h1_d10_bshd_bf16")
@export
fn key_cache_for_layer_h1_d10_bshd_bf16(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(
            num_heads=1, head_size=10, layout=KVCacheLayout.BSHD
        ),
    ],
) -> ContiguousKVCache[
    DType.bfloat16,
    KVCacheStaticParams(num_heads=1, head_size=10, layout=KVCacheLayout.BSHD),
]:
    return _key_cache_for_layer(layer_idx, kv_collection)


@mogg_register("key_cache_for_layer_h8_d64_bhsd_bf16")
@export
fn key_cache_for_layer_h8_d64_bhsd_bf16(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(
            num_heads=8, head_size=64, layout=KVCacheLayout.BHSD
        ),
    ],
) -> ContiguousKVCache[kv_collection.type, kv_collection.kv_params]:
    return _key_cache_for_layer(layer_idx, kv_collection)


@mogg_register("key_cache_for_layer_h8_d64_bshd_bf16")
@export
fn key_cache_for_layer_h8_d64_bshd_bf16(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(
            num_heads=8, head_size=64, layout=KVCacheLayout.BSHD
        ),
    ],
) -> ContiguousKVCache[kv_collection.type, kv_collection.kv_params]:
    return _key_cache_for_layer(layer_idx, kv_collection)


@mogg_register("key_cache_for_layer_h8_d64_bhsd_f32")
@export
fn key_cache_for_layer_h8_d64_bhsd_f32(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=8, head_size=64, layout=KVCacheLayout.BHSD
        ),
    ],
) -> ContiguousKVCache[kv_collection.type, kv_collection.kv_params]:
    return _key_cache_for_layer(layer_idx, kv_collection)


@mogg_register("key_cache_for_layer_h8_d64_bshd_f32")
@export
fn key_cache_for_layer_h8_d64_bshd_f32(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=8, head_size=64, layout=KVCacheLayout.BSHD
        ),
    ],
) -> ContiguousKVCache[kv_collection.type, kv_collection.kv_params]:
    return _key_cache_for_layer(layer_idx, kv_collection)


@always_inline
fn _key_cache_for_layer[
    type: DType, kv_params: KVCacheStaticParams
](
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[type, kv_params],
) -> ContiguousKVCache[type, kv_params] as result:
    """Retrieves the Key cache for the given layer."""
    return kv_collection.get_key_cache[__type_of(result)](int(layer_idx))


@mogg_register("value_cache_for_layer_h8_d128_bshd_bf16")
@export
fn value_cache_for_layer_h8_d128_bshd_bf16(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BSHD
        ),
    ],
) -> ContiguousKVCache[kv_collection.type, kv_collection.kv_params]:
    return _value_cache_for_layer(layer_idx, kv_collection)


@mogg_register("value_cache_for_layer_h8_d128_bhsd_bf16")
@export
fn value_cache_for_layer_h8_d128_bhsd_bf16(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BHSD
        ),
    ],
) -> ContiguousKVCache[kv_collection.type, kv_collection.kv_params]:
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
) -> ContiguousKVCache[kv_collection.type, kv_collection.kv_params]:
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
) -> ContiguousKVCache[kv_collection.type, kv_collection.kv_params]:
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
) -> ContiguousKVCache[kv_collection.type, kv_collection.kv_params]:
    return _value_cache_for_layer(layer_idx, kv_collection)


@mogg_register("value_cache_for_layer_h1_d10_bshd_f32")
@export
fn value_cache_for_layer_h1_d10_bshd_f32(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=1, head_size=10, layout=KVCacheLayout.BSHD
        ),
    ],
) -> ContiguousKVCache[kv_collection.type, kv_collection.kv_params]:
    return _value_cache_for_layer(layer_idx, kv_collection)


@mogg_register("value_cache_for_layer_h1_d10_bhsd_f32")
@export
fn value_cache_for_layer_h1_d10_bhsd_f32(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=1, head_size=10, layout=KVCacheLayout.BHSD
        ),
    ],
) -> ContiguousKVCache[kv_collection.type, kv_collection.kv_params]:
    return _value_cache_for_layer(layer_idx, kv_collection)


@mogg_register("value_cache_for_layer_h1_d10_bshd_bf16")
@export
fn value_cache_for_layer_h1_d10_bshd_bf16(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(
            num_heads=1, head_size=10, layout=KVCacheLayout.BSHD
        ),
    ],
) -> ContiguousKVCache[kv_collection.type, kv_collection.kv_params]:
    return _value_cache_for_layer(layer_idx, kv_collection)


@mogg_register("value_cache_for_layer_h1_d10_bhsd_bf16")
@export
fn value_cache_for_layer_h1_d10_bhsd_bf16(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(
            num_heads=1, head_size=10, layout=KVCacheLayout.BHSD
        ),
    ],
) -> ContiguousKVCache[kv_collection.type, kv_collection.kv_params]:
    return _value_cache_for_layer(layer_idx, kv_collection)


@mogg_register("value_cache_for_layer_h8_d64_bshd_bf16")
@export
fn value_cache_for_layer_h8_d64_bshd_bf16(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(
            num_heads=8, head_size=64, layout=KVCacheLayout.BSHD
        ),
    ],
) -> ContiguousKVCache[kv_collection.type, kv_collection.kv_params]:
    return _value_cache_for_layer(layer_idx, kv_collection)


@mogg_register("value_cache_for_layer_h8_d64_bhsd_bf16")
@export
fn value_cache_for_layer_h8_d64_bhsd_bf16(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(
            num_heads=8, head_size=64, layout=KVCacheLayout.BHSD
        ),
    ],
) -> ContiguousKVCache[kv_collection.type, kv_collection.kv_params]:
    return _value_cache_for_layer(layer_idx, kv_collection)


@mogg_register("value_cache_for_layer_h8_d64_bshd_f32")
@export
fn value_cache_for_layer_h8_d64_bshd_f32(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=8, head_size=64, layout=KVCacheLayout.BSHD
        ),
    ],
) -> ContiguousKVCache[kv_collection.type, kv_collection.kv_params]:
    return _value_cache_for_layer(layer_idx, kv_collection)


@mogg_register("value_cache_for_layer_h8_d64_bhsd_f32")
@export
fn value_cache_for_layer_h8_d64_bhsd_f32(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=8, head_size=64, layout=KVCacheLayout.BHSD
        ),
    ],
) -> ContiguousKVCache[kv_collection.type, kv_collection.kv_params]:
    return _value_cache_for_layer(layer_idx, kv_collection)


@always_inline
fn _value_cache_for_layer[
    type: DType,
    kv_params: KVCacheStaticParams,
](
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[type, kv_params],
) -> ContiguousKVCache[type, kv_params] as result:
    """Retrieves the Value cache for the given layer."""
    return kv_collection.get_value_cache[__type_of(result)](int(layer_idx))


@mogg_register("matmul_kv_cache_h6_d48_bshd")
@export
fn matmul_kv_cache_h6_d48_bshd[
    type: DType,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[type, 3, _],
    weight: NDBuffer[type, 2, _],
    cache: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BSHD
        ),
    ],
    ctx: MojoCallContextPtr,
) -> ContiguousKVCache[cache.type, cache.kv_params]:
    """Performs a matmul, writing the output into a mutable ContiguousKVCache object.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        cache: The historical ContiguousKVCache, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        ctx: The call context pointer, passed by the graph compiler.
    """
    with Trace[TraceLevel.OP, target=target]("matmul_kv_cache_h6_d48_bshd"):
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
) -> ContiguousKVCache[cache.type, cache.kv_params]:
    """Performs a matmul, writing the output into a mutable ContiguousKVCache object.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        cache: The historical ContiguousKVCache, with logical shape:
            (batch_size, num_kv_heads, max_seq_len, head_size).
        ctx: The call context pointer, passed by the graph compiler.
    """
    with Trace[TraceLevel.OP, target=target]("matmul_kv_cache_h6_d48_bhsd"):
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
) -> ContiguousKVCache[type, cache.kv_params]:
    """Performs a matmul, writing the output into a mutable ContiguousKVCache object.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        cache: The historical ContiguousKVCache, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        ctx: The call context pointer, passed by the graph compiler.
    """
    with Trace[TraceLevel.OP, target=target]("matmul_kv_cache_h8_d128_bshd"):
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
) -> ContiguousKVCache[type, cache.kv_params]:
    """Performs a matmul, writing the output into a mutable ContiguousKVCache object.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        cache: The historical ContiguousKVCache, with logical shape:
            (batch_size, num_kv_heads, max_seq_len, head_size).
        ctx: The call context pointer, passed by the graph compiler.
    """
    with Trace[TraceLevel.OP, target=target]("matmul_kv_cache_h8_d128_bhsd"):
        return _matmul_kv_cache[target=target](hidden_state, weight, cache, ctx)


@mogg_register("matmul_kv_cache_h1_d10_bshd")
@export
fn matmul_kv_cache_h1_d10_bshd[
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
            num_heads=1, head_size=10, layout=KVCacheLayout.BSHD
        ),
    ],
    ctx: MojoCallContextPtr,
) -> ContiguousKVCache[type, cache.kv_params]:
    """Performs a matmul, writing the output into a mutable ContiguousKVCache object.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        cache: The historical ContiguousKVCache, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        ctx: The call context pointer, passed by the graph compiler.
    """
    with Trace[TraceLevel.OP, target=target]("matmul_kv_cache_h1_d10_bshd"):
        return _matmul_kv_cache[target=target](hidden_state, weight, cache, ctx)


@mogg_register("matmul_kv_cache_h1_d10_bhsd")
@export
fn matmul_kv_cache_h1_d10_bhsd[
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
            num_heads=1, head_size=10, layout=KVCacheLayout.BHSD
        ),
    ],
    ctx: MojoCallContextPtr,
) -> ContiguousKVCache[type, cache.kv_params]:
    """Performs a matmul, writing the output into a mutable ContiguousKVCache object.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        cache: The historical ContiguousKVCache, with logical shape:
            (batch_size, num_kv_heads, max_seq_len, head_size).
        ctx: The call context pointer, passed by the graph compiler.
    """
    with Trace[TraceLevel.OP, target=target]("matmul_kv_cache_h1_d10_bhsd"):
        return _matmul_kv_cache[target=target](hidden_state, weight, cache, ctx)


@mogg_register("matmul_kv_cache_h8_d64_bshd")
@export
fn matmul_kv_cache_h8_d64_bshd[
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
            num_heads=8, head_size=64, layout=KVCacheLayout.BSHD
        ),
    ],
    ctx: MojoCallContextPtr,
) -> ContiguousKVCache[type, cache.kv_params]:
    """Performs a matmul, writing the output into a mutable ContiguousKVCache object.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        cache: The historical ContiguousKVCache, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        ctx: The call context pointer, passed by the graph compiler.
    """
    with Trace[TraceLevel.OP, target=target]("matmul_kv_cache_h8_d64_bshd"):
        return _matmul_kv_cache[target=target](hidden_state, weight, cache, ctx)


@mogg_register("matmul_kv_cache_h8_d64_bhsd")
@export
fn matmul_kv_cache_h8_d64_bhsd[
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
            num_heads=8, head_size=64, layout=KVCacheLayout.BHSD
        ),
    ],
    ctx: MojoCallContextPtr,
) -> ContiguousKVCache[type, cache.kv_params]:
    """Performs a matmul, writing the output into a mutable ContiguousKVCache object.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        cache: The historical ContiguousKVCache, with logical shape:
            (batch_size, num_kv_heads, max_seq_len, head_size).
        ctx: The call context pointer, passed by the graph compiler.
    """
    with Trace[TraceLevel.OP, target=target]("matmul_kv_cache_h8_d64_bhsd"):
        return _matmul_kv_cache[target=target](hidden_state, weight, cache, ctx)


@always_inline
fn _matmul_kv_cache[
    type: DType,
    kv_params: KVCacheStaticParams,
    *,
    target: StringLiteral,
](
    hidden_state: NDBuffer[type, 3, _],
    weight: NDBuffer[type, 2, _],
    cache: ContiguousKVCache[type, kv_params],
    context: MojoCallContextPtr,
) -> ContiguousKVCache[type, kv_params]:
    """Helper for performing matmul with custom ContiguousKVCache types.

    Parameters:
        type: The data type of all inputs and outputs
        kv_params: The static parameters for our KVCache object
        target: StringLiteral identifying the device target (cpu vs cuda)

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size)
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size)
        cache: The ContiguousKVCache, with shape determined by `kv_params.layout`
        context: Pointer containing the runtime context for the target device.
    """

    var cuda_ctx: Optional[DeviceContext] = None

    @parameter
    if target != "cpu":
        cuda_ctx = context.get_device_context()

    return _matmul_kv_cache_impl[target=target](
        hidden_state, weight, cache, cuda_ctx
    )


@always_inline
fn _matmul_kv_cache_impl[
    type: DType,
    kv_params: KVCacheStaticParams,
    *,
    target: StringLiteral,
](
    hidden_state: NDBuffer[type, 3, _],
    weight: NDBuffer[type, 2, _],
    cache: ContiguousKVCache[type, kv_params],
    ctx: Optional[DeviceContext],
) -> ContiguousKVCache[type, kv_params]:
    """Helper for performing matmul with custom ContiguousKVCache types.

    Parameters:
        type: The data type of all inputs and outputs
        kv_params: The static parameters for our KVCache object
        target: StringLiteral identifying the device target (cpu vs cuda)

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size)
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size)
        cache: The ContiguousKVCache, with shape determined by `kv_params.layout`
        ctx: Pointer containing the runtime context for the target device.
    """
    var SEQ_LEN = hidden_state.dim[1]()
    alias N = weight.shape.get[0]()
    alias K = weight.shape.get[1]()

    @parameter
    @__copy_capture(cache, SEQ_LEN)
    fn write_to_cache[
        type_: DType, width: Int, *, alignment: Int = 1
    ](idx: StaticIntTuple[2], val: SIMD[type_, width]):
        b_idx, t_idx = divmod(idx[0], SEQ_LEN)
        h_idx, hd_idx = divmod(idx[1], kv_params.head_size)

        var valid_len = cache.cache_length(b_idx)
        var cache_t_idx = t_idx + valid_len
        cache.store(
            b_idx,
            h_idx,
            cache_t_idx,
            hd_idx,
            rebind[SIMD[type, width]](val),
        )

    _matmul_common[target=target, elementwise_lambda_fn=write_to_cache](
        hidden_state, weight, ctx
    )
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
    v_cache: ContiguousKVCache[type, k_cache.kv_params],
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
    with Trace[TraceLevel.OP, target=target](
        "fused_qkv_matmul_kv_cache_h6_d48_bshd"
    ):
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
    v_cache: ContiguousKVCache[type, k_cache.kv_params],
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
    with Trace[TraceLevel.OP, target=target](
        "fused_qkv_matmul_kv_cache_h6_d48_bhsd"
    ):
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
    v_cache: ContiguousKVCache[type, k_cache.kv_params],
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
    with Trace[TraceLevel.OP, target=target](
        "fused_qkv_matmul_kv_cache_h8_d128_bshd"
    ):
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
    v_cache: ContiguousKVCache[type, k_cache.kv_params],
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
    with Trace[TraceLevel.OP, target=target](
        "fused_qkv_matmul_kv_cache_h8_d128_bhsd"
    ):
        return _fused_qkv_matmul_kv_cache[target=target](
            hidden_state, weight, k_cache, v_cache, output, ctx
        )


@mogg_register("fused_qkv_matmul_kv_cache_h1_d10_bshd")
@export
fn fused_qkv_matmul_kv_cache_h1_d10_bshd[
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
            num_heads=1, head_size=10, layout=KVCacheLayout.BSHD
        ),
    ],
    v_cache: ContiguousKVCache[type, k_cache.kv_params],
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
    with Trace[TraceLevel.OP, target=target](
        "fused_qkv_matmul_kv_cache_h1_d10_bshd"
    ):
        return _fused_qkv_matmul_kv_cache[target=target](
            hidden_state, weight, k_cache, v_cache, output, ctx
        )


@mogg_register("fused_qkv_matmul_kv_cache_h1_d10_bhsd")
@export
fn fused_qkv_matmul_kv_cache_h1_d10_bhsd[
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
            num_heads=1, head_size=10, layout=KVCacheLayout.BHSD
        ),
    ],
    v_cache: ContiguousKVCache[type, k_cache.kv_params],
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
    with Trace[TraceLevel.OP, target=target](
        "fused_qkv_matmul_kv_cache_h1_d10_bhsd"
    ):
        return _fused_qkv_matmul_kv_cache[target=target](
            hidden_state, weight, k_cache, v_cache, output, ctx
        )


alias embed_fn_type = fn[type: DType, width: Int] (
    StaticIntTuple[4], SIMD[type, width]
) capturing -> SIMD[type, width]


@mogg_register("fused_qkv_matmul_kv_cache_h8_d64_bshd")
@export
fn fused_qkv_matmul_kv_cache_h8_d64_bshd[
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
            num_heads=8, head_size=64, layout=KVCacheLayout.BSHD
        ),
    ],
    v_cache: ContiguousKVCache[type, k_cache.kv_params],
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
    with Trace[TraceLevel.OP, target=target](
        "fused_qkv_matmul_kv_cache_h8_d64_bshd"
    ):
        return _fused_qkv_matmul_kv_cache[target=target](
            hidden_state, weight, k_cache, v_cache, output, ctx
        )


@mogg_register("fused_qkv_matmul_kv_cache_h8_d64_bhsd")
@export
fn fused_qkv_matmul_kv_cache_h8_d64_bhsd[
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
            num_heads=8, head_size=64, layout=KVCacheLayout.BHSD
        ),
    ],
    v_cache: ContiguousKVCache[type, k_cache.kv_params],
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
    with Trace[TraceLevel.OP, target=target](
        "fused_qkv_matmul_kv_cache_h8_d64_bhsd"
    ):
        return _fused_qkv_matmul_kv_cache[target=target](
            hidden_state, weight, k_cache, v_cache, output, ctx
        )


@always_inline
fn _fused_qkv_matmul_kv_cache[
    type: DType,
    hidden_state_shape: DimList,
    weight_shape: DimList,
    output_shape: DimList,
    kv_params: KVCacheStaticParams,
    *,
    target: StringLiteral,
](
    hidden_state: NDBuffer[type, 3, hidden_state_shape],
    weight: NDBuffer[type, 2, weight_shape],
    k_cache: ContiguousKVCache[type, kv_params],
    v_cache: ContiguousKVCache[type, kv_params],
    output: NDBuffer[type, 3, output_shape],
    context: MojoCallContextPtr,
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

    var cuda_ctx: Optional[DeviceContext] = None

    @parameter
    if target != "cpu":
        cuda_ctx = context.get_device_context()

    return _fused_qkv_matmul_kv_cache_impl[target=target](
        hidden_state, weight, k_cache, v_cache, output, cuda_ctx
    )


@always_inline
fn _fused_qkv_matmul_kv_cache_impl[
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
    context: Optional[DeviceContext],
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
        context: The DeviceContext. This is unused if target == "cpu".
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
        type_: DType, width: Int, *, alignment: Int = 1
    ](idx: StaticIntTuple[2], val: SIMD[type_, width]):
        var bs_and_seq = divmod(idx[0], SEQ_LEN)
        var b_idx = bs_and_seq[0]
        var t_idx = bs_and_seq[1]
        if idx[1] < q_dim:
            output.store((b_idx, t_idx, idx[1]), rebind[SIMD[type, width]](val))
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

    _matmul_common[target=target, elementwise_lambda_fn=write_to_cache](
        hidden_state, weight, context
    )


@always_inline
fn _matmul_common[
    type: DType, //,
    *,
    target: StringLiteral,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    hidden_state: NDBuffer[type, 3, _],
    weight: NDBuffer[type, 2, _],
    context: Optional[DeviceContext],
):
    var BS = hidden_state.dim[0]()
    var SEQ_LEN = hidden_state.dim[1]()
    alias N = weight.shape.get[0]()
    alias K = weight.shape.get[1]()

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

    # TODO unify with other matmul
    @parameter
    if target == "cpu":
        var kernel_type_m = hidden_state_2d.shape.at[0]().or_else(0)

        _matmul_cpu[
            transpose_b=True,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](c_nd, hidden_state_2d, weight, kernel_type_m)

    else:
        _matmul_gpu[
            elementwise_lambda_fn=elementwise_lambda_fn,
            use_tensor_core=True,
            transpose_b=True,
            target=target,
        ](c_nd, hidden_state_2d, weight, context.value())

    c_nd.data.free()


@mogg_register("fused_qk_rope_h6_d48_bshd")
fn fused_qk_rope_h6_d48_bshd[
    type: DType, //,
    *,
    target: StringLiteral,
](
    q_proj: NDBuffer[type, 4, *_],
    k_cache: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BSHD
        ),
    ],
    freqs_cis: NDBuffer[type, 2, *_],
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr = MojoCallContextPtr(),
):
    """Performs a fused RoPE projection for Q and K projections.

    We have a manually fused QKV projection with mo.opaque types in our Llama model.
    Due to a limitation in custom op definitions, we can't declare both a tensor
    and opaque type as output from a custom kernel. This requires us to only note
    Q_proj as an output from the QKV projection. If we immediately follow the
    QKV proj kernel with a RoPE kernel applied to K, we'll get a race condition
    because the graph compiler doesn't know about the dependency between these
    kernels in the graph definition. Here we fuse the RoPE kernel applied to
    Q_proj with K_proj, so K_proj RoPE is only excuted after QKV completes.
    """
    with Trace[TraceLevel.OP, target=target]("fused_qk_rope_h6_d48_bshd"):
        _fused_qk_rope[target=target](
            q_proj, k_cache, freqs_cis, output, context
        )


@mogg_register("fused_qk_rope_h6_d48_bhsd")
fn fused_qk_rope_h6_d48_bhsd[
    type: DType, //,
    *,
    target: StringLiteral,
](
    q_proj: NDBuffer[type, 4, *_],
    k_cache: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BHSD
        ),
    ],
    freqs_cis: NDBuffer[type, 2, *_],
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr = MojoCallContextPtr(),
):
    """Performs a fused RoPE projection for Q and K projections.

    We have a manually fused QKV projection with mo.opaque types in our Llama model.
    Due to a limitation in custom op definitions, we can't declare both a tensor
    and opaque type as output from a custom kernel. This requires us to only note
    Q_proj as an output from the QKV projection. If we immediately follow the
    QKV proj kernel with a RoPE kernel applied to K, we'll get a race condition
    because the graph compiler doesn't know about the dependency between these
    kernels in the graph definition. Here we fuse the RoPE kernel applied to
    Q_proj with K_proj, so K_proj RoPE is only excuted after QKV completes.
    """
    with Trace[TraceLevel.OP, target=target]("fused_qk_rope_h6_d48_bhsd"):
        _fused_qk_rope[target=target](
            q_proj, k_cache, freqs_cis, output, context
        )


@mogg_register("fused_qk_rope_h8_d128_bshd")
fn fused_qk_rope_h8_d128_bshd[
    type: DType, //,
    *,
    target: StringLiteral,
](
    q_proj: NDBuffer[type, 4, *_],
    k_cache: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BSHD
        ),
    ],
    freqs_cis: NDBuffer[type, 2, *_],
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr = MojoCallContextPtr(),
):
    """Performs a fused RoPE projection for Q and K projections.

    We have a manually fused QKV projection with mo.opaque types in our Llama model.
    Due to a limitation in custom op definitions, we can't declare both a tensor
    and opaque type as output from a custom kernel. This requires us to only note
    Q_proj as an output from the QKV projection. If we immediately follow the
    QKV proj kernel with a RoPE kernel applied to K, we'll get a race condition
    because the graph compiler doesn't know about the dependency between these
    kernels in the graph definition. Here we fuse the RoPE kernel applied to
    Q_proj with K_proj, so K_proj RoPE is only excuted after QKV completes.
    """
    with Trace[TraceLevel.OP, target=target]("fused_qk_rope_h8_d128_bshd"):
        _fused_qk_rope[target=target](
            q_proj, k_cache, freqs_cis, output, context
        )


@mogg_register("fused_qk_rope_h8_d128_bhsd")
fn fused_qk_rope_h8_d128_bhsd[
    type: DType, //,
    *,
    target: StringLiteral,
](
    q_proj: NDBuffer[type, 4, *_],
    k_cache: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BHSD
        ),
    ],
    freqs_cis: NDBuffer[type, 2, *_],
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr = MojoCallContextPtr(),
):
    """Performs a fused RoPE projection for Q and K projections.

    We have a manually fused QKV projection with mo.opaque types in our Llama model.
    Due to a limitation in custom op definitions, we can't declare both a tensor
    and opaque type as output from a custom kernel. This requires us to only note
    Q_proj as an output from the QKV projection. If we immediately follow the
    QKV proj kernel with a RoPE kernel applied to K, we'll get a race condition
    because the graph compiler doesn't know about the dependency between these
    kernels in the graph definition. Here we fuse the RoPE kernel applied to
    Q_proj with K_proj, so K_proj RoPE is only excuted after QKV completes.
    """
    with Trace[TraceLevel.OP, target=target]("fused_qk_rope_h8_d128_bhsd"):
        _fused_qk_rope[target=target](
            q_proj, k_cache, freqs_cis, output, context
        )


@mogg_register("fused_qk_rope_h1_d10_bshd")
fn fused_qk_rope_h1_d10_bshd[
    type: DType, //,
    *,
    target: StringLiteral,
](
    q_proj: NDBuffer[type, 4, *_],
    k_cache: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=1, head_size=10, layout=KVCacheLayout.BSHD
        ),
    ],
    freqs_cis: NDBuffer[type, 2, *_],
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr = MojoCallContextPtr(),
):
    """Performs a fused RoPE projection for Q and K projections.

    We have a manually fused QKV projection with mo.opaque types in our Llama model.
    Due to a limitation in custom op definitions, we can't declare both a tensor
    and opaque type as output from a custom kernel. This requires us to only note
    Q_proj as an output from the QKV projection. If we immediately follow the
    QKV proj kernel with a RoPE kernel applied to K, we'll get a race condition
    because the graph compiler doesn't know about the dependency between these
    kernels in the graph definition. Here we fuse the RoPE kernel applied to
    Q_proj with K_proj, so K_proj RoPE is only excuted after QKV completes.
    """
    with Trace[TraceLevel.OP, target=target]("fused_qk_rope_h1_d10_bshd"):
        _fused_qk_rope[target=target](
            q_proj, k_cache, freqs_cis, output, context
        )


@mogg_register("fused_qk_rope_h1_d10_bhsd")
fn fused_qk_rope_h1_d10_bhsd[
    type: DType, //,
    *,
    target: StringLiteral,
](
    q_proj: NDBuffer[type, 4, *_],
    k_cache: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=1, head_size=10, layout=KVCacheLayout.BHSD
        ),
    ],
    freqs_cis: NDBuffer[type, 2, *_],
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr = MojoCallContextPtr(),
):
    """Performs a fused RoPE projection for Q and K projections.

    We have a manually fused QKV projection with mo.opaque types in our Llama model.
    Due to a limitation in custom op definitions, we can't declare both a tensor
    and opaque type as output from a custom kernel. This requires us to only note
    Q_proj as an output from the QKV projection. If we immediately follow the
    QKV proj kernel with a RoPE kernel applied to K, we'll get a race condition
    because the graph compiler doesn't know about the dependency between these
    kernels in the graph definition. Here we fuse the RoPE kernel applied to
    Q_proj with K_proj, so K_proj RoPE is only excuted after QKV completes.
    """
    with Trace[TraceLevel.OP, target=target]("fused_qk_rope_h1_d10_bhsd"):
        _fused_qk_rope[target=target](
            q_proj, k_cache, freqs_cis, output, context
        )


@mogg_register("fused_qk_rope_h8_d64_bshd")
fn fused_qk_rope_h8_d64_bshd[
    type: DType, //,
    *,
    target: StringLiteral,
](
    q_proj: NDBuffer[type, 4, *_],
    k_cache: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=8, head_size=64, layout=KVCacheLayout.BSHD
        ),
    ],
    freqs_cis: NDBuffer[type, 2, *_],
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr = MojoCallContextPtr(),
):
    """Performs a fused RoPE projection for Q and K projections.

    We have a manually fused QKV projection with mo.opaque types in our Llama model.
    Due to a limitation in custom op definitions, we can't declare both a tensor
    and opaque type as output from a custom kernel. This requires us to only note
    Q_proj as an output from the QKV projection. If we immediately follow the
    QKV proj kernel with a RoPE kernel applied to K, we'll get a race condition
    because the graph compiler doesn't know about the dependency between these
    kernels in the graph definition. Here we fuse the RoPE kernel applied to
    Q_proj with K_proj, so K_proj RoPE is only excuted after QKV completes.
    """
    with Trace[TraceLevel.OP, target=target]("fused_qk_rope_h8_d64_bshd"):
        _fused_qk_rope[target=target](
            q_proj, k_cache, freqs_cis, output, context
        )


@mogg_register("fused_qk_rope_h8_d64_bhsd")
fn fused_qk_rope_h8_d64_bhsd[
    type: DType, //,
    *,
    target: StringLiteral,
](
    q_proj: NDBuffer[type, 4, *_],
    k_cache: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=8, head_size=64, layout=KVCacheLayout.BHSD
        ),
    ],
    freqs_cis: NDBuffer[type, 2, *_],
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr = MojoCallContextPtr(),
):
    """Performs a fused RoPE projection for Q and K projections.

    We have a manually fused QKV projection with mo.opaque types in our Llama model.
    Due to a limitation in custom op definitions, we can't declare both a tensor
    and opaque type as output from a custom kernel. This requires us to only note
    Q_proj as an output from the QKV projection. If we immediately follow the
    QKV proj kernel with a RoPE kernel applied to K, we'll get a race condition
    because the graph compiler doesn't know about the dependency between these
    kernels in the graph definition. Here we fuse the RoPE kernel applied to
    Q_proj with K_proj, so K_proj RoPE is only excuted after QKV completes.
    """
    with Trace[TraceLevel.OP, target=target]("fused_qk_rope_h8_d64_bhsd"):
        _fused_qk_rope[target=target](
            q_proj, k_cache, freqs_cis, output, context
        )


@always_inline
fn _fused_qk_rope[
    type: DType, //,
    *,
    target: StringLiteral,
](
    q_proj: NDBuffer[type, 4, *_],
    k_cache: ContiguousKVCache[type, *_],
    freqs_cis: NDBuffer[type, 2, *_],
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr,
):
    var batch_size = q_proj.dim[0]()
    var new_seq_len = q_proj.dim[1]()
    alias num_q_heads = q_proj.shape.get[2]()
    alias num_k_heads = k_cache.kv_params.num_heads
    alias head_size = q_proj.shape.get[3]()

    @always_inline
    @parameter
    @__copy_capture(freqs_cis, q_proj, k_cache, output)
    fn rope_fn[width: Int, rank: Int](idx_arg: StaticIntTuple[rank]):
        constrained[rank == 4, "Invalid rank passed to rope kernel"]()

        @parameter
        if width == 1:
            print("ROPE KERNEL CALLED WITH SINGLE VALUE, EXPECTED AT LEAST 2")
            return
        else:
            var idx = rebind[StaticIntTuple[4]](idx_arg)
            var bs_idx = idx[0]
            var seq_idx = idx[1]
            var head_idx = idx[2]
            var head_dim_idx = idx[3]

            # WARN assumes head_size % simd_width == 0
            # guarded by constrained statement below
            var is_q_proj = head_idx < num_q_heads
            var val: SIMD[type, width]

            if is_q_proj:
                val = q_proj.load[width=width](idx)
            else:
                head_idx -= num_q_heads

                var cache_seq_idx = seq_idx + k_cache.cache_length(bs_idx)
                val = k_cache.load[type, width=width](
                    bs_idx, head_idx, cache_seq_idx, head_dim_idx
                )

            var x_c = val.deinterleave()
            var x_re = x_c[0]
            var x_im = x_c[1]

            var f_idx = StaticIntTuple[2](seq_idx, head_dim_idx)
            var f_c_temp = freqs_cis.load[width=width](f_idx)

            var f_c = f_c_temp.deinterleave()
            var f_re = f_c[0]
            var f_im = f_c[1]

            var r_re = (x_re * f_re) - (x_im * f_im)
            var r_im = (x_re * f_im) + (x_im * f_re)

            var result = r_re.interleave(r_im)
            if is_q_proj:
                output.store(idx, result)
            else:
                var cache_seq_idx = seq_idx + k_cache.cache_length(bs_idx)
                k_cache.store(
                    bs_idx, head_idx, cache_seq_idx, head_dim_idx, result
                )

    alias compile_target = _current_target() if target == "cpu" else _get_nvptx_target()
    alias target_simd_width = simdwidthof[type, target=compile_target]()
    alias kernel_simd_width = gcd(
        target_simd_width, k_cache.kv_params.head_size
    )
    constrained[kernel_simd_width >= 2, "invalid simd_width and head size"]()

    var launch_shape = StaticIntTuple[4](
        batch_size,
        new_seq_len,
        num_q_heads + num_k_heads,  # concat q and k along head dim
        head_size,
    )
    elementwise[func=rope_fn, simd_width=kernel_simd_width, target=target](
        launch_shape, context.get_device_context()
    )


@mogg_register("flash_attention_kv_cache_h6_d48_bshd")
@export
fn flash_attention_kv_cache_h6_d48_bshd[
    type: DType, //,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[type, 4, *_],
    k: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BSHD
        ),
    ],
    v: ContiguousKVCache[type, k.kv_params],
    mask: NDBuffer[type, *_],
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr,
):
    with Trace[TraceLevel.OP, target=target](
        "flash_attention_kv_cache_h6_d48_bshd"
    ):
        return _flash_attention_kv_cache[target=target](
            q, k, v, mask, valid_lengths, scale, output, context
        )


@mogg_register("flash_attention_kv_cache_h6_d48_bhsd")
@export
fn flash_attention_kv_cache_h6_d48_bhsd[
    type: DType, //,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[type, 4, *_],
    k: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BHSD
        ),
    ],
    v: ContiguousKVCache[type, k.kv_params],
    mask: NDBuffer[type, *_],
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr,
):
    with Trace[TraceLevel.OP, target=target](
        "flash_attention_kv_cache_h6_d48_bhsd"
    ):
        return _flash_attention_kv_cache[target=target](
            q, k, v, mask, valid_lengths, scale, output, context
        )


@mogg_register("flash_attention_kv_cache_h8_d128_bshd")
@export
fn flash_attention_kv_cache_h8_d128_bshd[
    type: DType, //,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[type, 4, *_],
    k: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BSHD
        ),
    ],
    v: ContiguousKVCache[type, k.kv_params],
    mask: NDBuffer[type, *_],
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr,
):
    with Trace[TraceLevel.OP, target=target](
        "flash_attention_kv_cache_h8_d128_bshd"
    ):
        return _flash_attention_kv_cache[target=target](
            q, k, v, mask, valid_lengths, scale, output, context
        )


@mogg_register("flash_attention_kv_cache_h8_d128_bhsd")
@export
fn flash_attention_kv_cache_h8_d128_bhsd[
    type: DType, //,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[type, 4, *_],
    k: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=8, head_size=128, layout=KVCacheLayout.BHSD
        ),
    ],
    v: ContiguousKVCache[type, k.kv_params],
    mask: NDBuffer[type, *_],
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr,
):
    with Trace[TraceLevel.OP, target=target](
        "flash_attention_kv_cache_h8_d128_bhsd"
    ):
        return _flash_attention_kv_cache[target=target](
            q, k, v, mask, valid_lengths, scale, output, context
        )


@mogg_register("flash_attention_kv_cache_h1_d10_bshd")
@export
fn flash_attention_kv_cache_h1_d10_bshd[
    type: DType, //,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[type, 4, *_],
    k: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=1, head_size=10, layout=KVCacheLayout.BSHD
        ),
    ],
    v: ContiguousKVCache[type, k.kv_params],
    mask: NDBuffer[type, *_],
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr,
):
    with Trace[TraceLevel.OP, target=target](
        "flash_attention_kv_cache_h1_d10_bshd"
    ):
        return _flash_attention_kv_cache[target=target](
            q, k, v, mask, valid_lengths, scale, output, context
        )


@mogg_register("flash_attention_kv_cache_h1_d10_bhsd")
@export
fn flash_attention_kv_cache_h1_d10_bhsd[
    type: DType, //,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[type, 4, *_],
    k: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=1, head_size=10, layout=KVCacheLayout.BHSD
        ),
    ],
    v: ContiguousKVCache[type, k.kv_params],
    mask: NDBuffer[type, *_],
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr,
):
    with Trace[TraceLevel.OP, target=target](
        "flash_attention_kv_cache_h1_d10_bhsd"
    ):
        return _flash_attention_kv_cache[target=target](
            q, k, v, mask, valid_lengths, scale, output, context
        )


@mogg_register("flash_attention_kv_cache_h8_d64_bshd")
@export
fn flash_attention_kv_cache_h8_d64_bshd[
    type: DType, //,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[type, 4, *_],
    k: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=8, head_size=64, layout=KVCacheLayout.BSHD
        ),
    ],
    v: ContiguousKVCache[type, k.kv_params],
    mask: NDBuffer[type, *_],
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr,
):
    with Trace[TraceLevel.OP, target=target](
        "flash_attention_kv_cache_h8_d64_bshd"
    ):
        return _flash_attention_kv_cache[target=target](
            q, k, v, mask, valid_lengths, scale, output, context
        )


@mogg_register("flash_attention_kv_cache_h8_d64_bhsd")
@export
fn flash_attention_kv_cache_h8_d64_bhsd[
    type: DType, //,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[type, 4, *_],
    k: ContiguousKVCache[
        type,
        KVCacheStaticParams(
            num_heads=8, head_size=64, layout=KVCacheLayout.BHSD
        ),
    ],
    v: ContiguousKVCache[type, k.kv_params],
    mask: NDBuffer[type, *_],
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr,
):
    with Trace[TraceLevel.OP, target=target](
        "flash_attention_kv_cache_h8_d64_bhsd"
    ):
        return _flash_attention_kv_cache[target=target](
            q, k, v, mask, valid_lengths, scale, output, context
        )


@always_inline
fn _flash_attention_kv_cache[
    type: DType, //,
    target: StringLiteral,
](
    q: NDBuffer[type, 4, *_],
    k: ContiguousKVCache[type, *_],
    v: ContiguousKVCache[type, *_],
    mask: NDBuffer[type, *_],
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
    context: MojoCallContextPtr,
):
    """Performs flash attention using k and v caches from ContiguousKVCache custom types.

    Args:
        q: NDBuffer with shape (batch_size, num_heads, seq_len, head_size).
        k: ContiguousKVCache type with logical shape (batch_size, num_heads, max_seq_len, head_size).
        v: ContiguousKVCache type with logical shape (batch_size, num_heads, max_seq_len, head_size).
        mask: The attention mask to apply to the score matrix.
        valid_lengths: The unpadded lengths of the sequences contained in q
        scale: The scaled factor in scaled-dot product attention. Usually isqrt(head_size).
        output: The Pre-allocated output buffer to write results to. Has shape:
            (batch_size, num_heads, seq_len, head_size).
        context: Pointer containing the runtime context for the target device.
    """

    var cuda_ctx: Optional[DeviceContext] = None

    @parameter
    if target != "cpu":
        cuda_ctx = context.get_device_context()

    _flash_attention_kv_cache_impl[target=target](
        q, k, v, mask, valid_lengths, scale, output, cuda_ctx
    )


@always_inline
fn _flash_attention_kv_cache_impl[
    type: DType, //,
    target: StringLiteral,
](
    q: NDBuffer[type, 4, *_],
    k: ContiguousKVCache[type, *_],
    v: ContiguousKVCache[type, *_],
    mask: NDBuffer[type, *_],
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
    context: Optional[DeviceContext],
):
    """Performs flash attention using k and v caches from ContiguousKVCache custom types.

    Args:
        q: NDBuffer with shape (batch_size, num_heads, seq_len, head_size).
        k: ContiguousKVCache type with logical shape (batch_size, num_heads, max_seq_len, head_size).
        v: ContiguousKVCache type with logical shape (batch_size, num_heads, max_seq_len, head_size).
        mask: The attention mask to apply to the score matrix.
        valid_lengths: The unpadded lengths of the sequences contained in q
        scale: The scaled factor in scaled-dot product attention. Usually isqrt(head_size).
        output: The Pre-allocated output buffer to write results to. Has shape:
            (batch_size, num_heads, seq_len, head_size).
        context: CUDA DeviceContext. This is not used if target == "cpu"
    """

    @parameter
    if target == "cpu":
        return _flash_attention_kv_cache_cpu(q, k, v, mask, scale, output)
    else:
        return _flash_attention_kv_cache_gpu[target=target](
            q, k, v, mask, valid_lengths, scale, output, context.value()
        )


@always_inline
fn _flash_attention_kv_cache_cpu[
    type: DType, //
](
    q: NDBuffer[type, 4, *_],
    k: ContiguousKVCache[type, *_],
    v: ContiguousKVCache[type, *_],
    mask: NDBuffer[type, *_],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
):
    constrained[
        k.kv_params.layout == KVCacheLayout.BHSD,
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

        var retval = k.load[type, width=width](bs, head_idx, seq, head_d_idx)
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
        var retval = v.load[type, width=width](bs, head_idx, seq, head_d_idx)
        return retval

    @parameter
    @__copy_capture(mask)
    fn input_mask_fn[
        width: Int, rank: Int
    ](idx: StaticIntTuple[rank]) -> SIMD[type, width]:
        @parameter
        if mask.rank == 4:
            return mask.load[width=width](
                rebind[StaticIntTuple[mask.rank]](idx)
            )
        elif mask.rank == 3:
            return mask.load[width=width]((idx[0], idx[2], idx[3]))
        else:
            return mask.load[width=width]((idx[2], idx[3]))

    var batch_size = q.dim[0]()
    var depth = q.dim[3]()
    var new_seq_len = q.dim[2]()
    var cache_seq_len = int(k.cache_length(0))
    var seq_len = new_seq_len + cache_seq_len
    var fa_k_shape = StaticIntTuple[4](
        batch_size, k.kv_params.num_heads, seq_len, depth
    )
    var fa_v_shape = StaticIntTuple[4](
        batch_size, k.kv_params.num_heads, seq_len, depth
    )

    cpu_flash_attention[
        input_k_fn,
        input_v_fn,
        input_mask_fn,
        transpose_k=True,
    ](
        q.make_dims_unknown(),
        fa_k_shape,
        fa_v_shape,
        output,
        scale,
    )


@always_inline
fn _flash_attention_kv_cache_gpu[
    type: DType, //, *, target: StringLiteral
](
    q: NDBuffer[type, 4, *_],
    k: ContiguousKVCache[type, *_],
    v: ContiguousKVCache[type, *_],
    mask: NDBuffer[type, *_],
    valid_lengths: NDBuffer[DType.uint32, 1],
    scale: Float32,
    output: NDBuffer[type, 4, *_],
    context: DeviceContext,
):
    constrained[
        k.kv_params.layout == KVCacheLayout.BSHD,
        "GPU Flash attention only supports the BSHD layout.",
    ]()

    alias wrapped_mask_rank = mask.rank if mask.rank == 4 else 3
    var mask_nd: NDBuffer[
        type,
        wrapped_mask_rank,
        DimList.create_unknown[wrapped_mask_rank](),
    ]

    @parameter
    if mask.rank == 2:
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
    try:
        gpu_flash_attention[
            add_attn_mask=True,
            use_tensor_core=True,
            target=target,
        ](
            output,
            q,
            k,
            v,
            mask_nd,
            valid_lengths,
            scale,
            context,
        )
    except e:
        print("Error in GPU Flash Attention:", e)


fn _contiguous_kv_cache_collection[
    type: DType, //, kv_params: KVCacheStaticParams
](
    key_cache: NDBuffer[type, 5],
    value_cache: NDBuffer[type, 5],
    cache_lengths: NDBuffer[DType.uint32, 1],
    is_cache_empty: NDBuffer[DType.bool, 1],
    seq_ids: NDBuffer[DType.int32, 1],
    # Note that num_layers and batch_size are scalars.
    num_layers: NDBuffer[DType.int32, 1],
    batch_size: NDBuffer[DType.int32, 1],
) -> ContiguousKVCacheCollection[type, kv_params]:
    # Marshal NDBuffers into arguments expected by the
    # ContiguousKVCacheCollection constructor.

    seq_ids_list = List[Int]()
    for id in range(seq_ids.size()):
        seq_ids_list.append(id)

    return ContiguousKVCacheCollection[
        type,
        kv_params,
        _default_max_batch_size,
    ](
        key_cache,
        value_cache,
        cache_lengths,
        is_cache_empty[0],
        seq_ids_list,
        int(num_layers[0]),
        int(batch_size[0]),
    )


# Boilerplate: stub out interface for every combination of KV cache parameters.
alias kv_params_h1_d10_bshd = KVCacheStaticParams(
    num_heads=1, head_size=10, layout=KVCacheLayout.BSHD
)
alias kv_params_h1_d10_bhsd = KVCacheStaticParams(
    num_heads=1, head_size=10, layout=KVCacheLayout.BHSD
)
alias kv_params_h6_d48_bshd = KVCacheStaticParams(
    num_heads=6, head_size=48, layout=KVCacheLayout.BSHD
)
alias kv_params_h6_d48_bhsd = KVCacheStaticParams(
    num_heads=6, head_size=48, layout=KVCacheLayout.BHSD
)
alias kv_params_h8_d128_bshd = KVCacheStaticParams(
    num_heads=8, head_size=128, layout=KVCacheLayout.BSHD
)
alias kv_params_h8_d128_bhsd = KVCacheStaticParams(
    num_heads=8, head_size=128, layout=KVCacheLayout.BHSD
)
alias kv_params_h8_d64_bshd = KVCacheStaticParams(
    num_heads=8, head_size=64, layout=KVCacheLayout.BSHD
)
alias kv_params_h8_d64_bhsd = KVCacheStaticParams(
    num_heads=8, head_size=64, layout=KVCacheLayout.BHSD
)


@mogg_register("contiguous_kv_cache_collection_h6_d48_bshd")
@export
fn contiguous_kv_cache_collection_h6_d48_bshd[
    type: DType, //, target: StringLiteral
](
    key_cache: NDBuffer[type, 5],
    value_cache: NDBuffer[type, 5],
    cache_lengths: NDBuffer[DType.uint32, 1],
    is_cache_empty: NDBuffer[DType.bool, 1],
    seq_ids: NDBuffer[DType.int32, 1],
    num_layers: NDBuffer[DType.int32, 1],
    batch_size: NDBuffer[DType.int32, 1],
) -> ContiguousKVCacheCollection[
    type,
    kv_params_h6_d48_bshd,
]:
    return _contiguous_kv_cache_collection[kv_params_h6_d48_bshd](
        key_cache,
        value_cache,
        cache_lengths,
        is_cache_empty,
        seq_ids,
        num_layers,
        batch_size,
    )


@mogg_register("contiguous_kv_cache_collection_h6_d48_bhsd")
@export
fn contiguous_kv_cache_collection_h6_d48_bhsd[
    type: DType, //, target: StringLiteral
](
    key_cache: NDBuffer[type, 5],
    value_cache: NDBuffer[type, 5],
    cache_lengths: NDBuffer[DType.uint32, 1],
    is_cache_empty: NDBuffer[DType.bool, 1],
    seq_ids: NDBuffer[DType.int32, 1],
    num_layers: NDBuffer[DType.int32, 1],
    batch_size: NDBuffer[DType.int32, 1],
) -> ContiguousKVCacheCollection[
    type,
    kv_params_h6_d48_bhsd,
]:
    return _contiguous_kv_cache_collection[kv_params_h6_d48_bhsd](
        key_cache,
        value_cache,
        cache_lengths,
        is_cache_empty,
        seq_ids,
        num_layers,
        batch_size,
    )


@mogg_register("contiguous_kv_cache_collection_h8_d128_bshd")
@export
fn contiguous_kv_cache_collection_h8_d128_bshd[
    type: DType, //, target: StringLiteral
](
    key_cache: NDBuffer[type, 5],
    value_cache: NDBuffer[type, 5],
    cache_lengths: NDBuffer[DType.uint32, 1],
    is_cache_empty: NDBuffer[DType.bool, 1],
    seq_ids: NDBuffer[DType.int32, 1],
    num_layers: NDBuffer[DType.int32, 1],
    batch_size: NDBuffer[DType.int32, 1],
) -> ContiguousKVCacheCollection[
    type,
    kv_params_h8_d128_bshd,
]:
    return _contiguous_kv_cache_collection[kv_params_h8_d128_bshd](
        key_cache,
        value_cache,
        cache_lengths,
        is_cache_empty,
        seq_ids,
        num_layers,
        batch_size,
    )


@mogg_register("contiguous_kv_cache_collection_h8_d128_bhsd")
@export
fn contiguous_kv_cache_collection_h8_d128_bhsd[
    type: DType, //, target: StringLiteral
](
    key_cache: NDBuffer[type, 5],
    value_cache: NDBuffer[type, 5],
    cache_lengths: NDBuffer[DType.uint32, 1],
    is_cache_empty: NDBuffer[DType.bool, 1],
    seq_ids: NDBuffer[DType.int32, 1],
    num_layers: NDBuffer[DType.int32, 1],
    batch_size: NDBuffer[DType.int32, 1],
) -> ContiguousKVCacheCollection[
    type,
    kv_params_h8_d128_bhsd,
]:
    return _contiguous_kv_cache_collection[kv_params_h8_d128_bhsd](
        key_cache,
        value_cache,
        cache_lengths,
        is_cache_empty,
        seq_ids,
        num_layers,
        batch_size,
    )


@mogg_register("contiguous_kv_cache_collection_h1_d10_bshd")
@export
fn contiguous_kv_cache_collection_h1_d10_bshd[
    type: DType, //, target: StringLiteral
](
    key_cache: NDBuffer[type, 5],
    value_cache: NDBuffer[type, 5],
    cache_lengths: NDBuffer[DType.uint32, 1],
    is_cache_empty: NDBuffer[DType.bool, 1],
    seq_ids: NDBuffer[DType.int32, 1],
    num_layers: NDBuffer[DType.int32, 1],
    batch_size: NDBuffer[DType.int32, 1],
) -> ContiguousKVCacheCollection[
    type,
    kv_params_h1_d10_bshd,
]:
    return _contiguous_kv_cache_collection[kv_params_h1_d10_bshd](
        key_cache,
        value_cache,
        cache_lengths,
        is_cache_empty,
        seq_ids,
        num_layers,
        batch_size,
    )


@mogg_register("contiguous_kv_cache_collection_h1_d10_bhsd")
@export
fn contiguous_kv_cache_collection_h1_d10_bhsd[
    type: DType, //, target: StringLiteral
](
    key_cache: NDBuffer[type, 5],
    value_cache: NDBuffer[type, 5],
    cache_lengths: NDBuffer[DType.uint32, 1],
    is_cache_empty: NDBuffer[DType.bool, 1],
    seq_ids: NDBuffer[DType.int32, 1],
    num_layers: NDBuffer[DType.int32, 1],
    batch_size: NDBuffer[DType.int32, 1],
) -> ContiguousKVCacheCollection[
    type,
    kv_params_h1_d10_bhsd,
]:
    return _contiguous_kv_cache_collection[kv_params_h1_d10_bhsd](
        key_cache,
        value_cache,
        cache_lengths,
        is_cache_empty,
        seq_ids,
        num_layers,
        batch_size,
    )


@mogg_register("contiguous_kv_cache_collection_h8_d64_bshd")
@export
fn contiguous_kv_cache_collection_h8_d64_bshd[
    type: DType, //, target: StringLiteral
](
    key_cache: NDBuffer[type, 5],
    value_cache: NDBuffer[type, 5],
    cache_lengths: NDBuffer[DType.uint32, 1],
    is_cache_empty: NDBuffer[DType.bool, 1],
    seq_ids: NDBuffer[DType.int32, 1],
    num_layers: NDBuffer[DType.int32, 1],
    batch_size: NDBuffer[DType.int32, 1],
) -> ContiguousKVCacheCollection[
    type,
    kv_params_h8_d64_bshd,
]:
    return _contiguous_kv_cache_collection[kv_params_h8_d64_bshd](
        key_cache,
        value_cache,
        cache_lengths,
        is_cache_empty,
        seq_ids,
        num_layers,
        batch_size,
    )


@mogg_register("contiguous_kv_cache_collection_h8_d64_bhsd")
@export
fn contiguous_kv_cache_collection_h8_d64_bhsd[
    type: DType, //, target: StringLiteral
](
    key_cache: NDBuffer[type, 5],
    value_cache: NDBuffer[type, 5],
    cache_lengths: NDBuffer[DType.uint32, 1],
    is_cache_empty: NDBuffer[DType.bool, 1],
    seq_ids: NDBuffer[DType.int32, 1],
    num_layers: NDBuffer[DType.int32, 1],
    batch_size: NDBuffer[DType.int32, 1],
) -> ContiguousKVCacheCollection[
    type,
    kv_params_h8_d64_bhsd,
]:
    return _contiguous_kv_cache_collection[kv_params_h8_d64_bhsd](
        key_cache,
        value_cache,
        cache_lengths,
        is_cache_empty,
        seq_ids,
        num_layers,
        batch_size,
    )
