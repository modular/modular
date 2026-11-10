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
"""GLM-5.3-Flash's hybrid schedule, cache tree and checkpoint-bound sizing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, NamedTuple

from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache.cache_params import (
    CacheLeafParamInterface,
    KVCacheParamInterface,
    KVCacheQuantizationConfig,
    MLAKVCacheParams,
    MultiKVCacheParams,
    RecurrentStateParams,
    recurrent_leaf,
)
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from max.pipelines.lib.memory_estimation import FreeMemory
from transformers import AutoConfig

from .layers.indexer_inputs import indexer_selection_width
from .ring import ring_len_for_window
from .state_cache import (
    INDEXER_CACHE_KEY,
    MLA_CACHE_KEY,
    STATE_CACHE_KEY,
    state_regions,
    with_verify_ring,
)

__all__ = [
    "LINEAR_ATTENTION",
    "MLA_KERNEL_LATENT_WIDTH",
    "SPARSE_ATTENTION",
    "construct_kv_params",
    "declared_kda_dims",
    "declared_state_dtype",
    "get_num_layers",
    "get_text_config",
    "layer_indices",
    "linear_attn_field",
    "recurrent_state_bytes",
    "residual_bytes_per_token",
    "resolve_layer_types",
    "state_bounded_batch_size",
]


#: ``layer_types`` value for a KDA linear-attention layer.
LINEAR_ATTENTION = "linear_attention"


#: ``layer_types`` value for a sparse-MLA + DSA-indexer layer.
SPARSE_ATTENTION = "deepseek_sparse_attention"


# TODO(KERN-3520): Pad to this width until we have kernel support for 512.
MLA_KERNEL_LATENT_WIDTH = 576


def _resolve_mla_latent_width(text_config: AutoConfig) -> int:
    native = text_config.kv_lora_rank + text_config.qk_rope_head_dim
    return max(native, MLA_KERNEL_LATENT_WIDTH)


_POOL_DTYPE_MAP = {"bfloat16": DType.bfloat16, "float32": DType.float32}
"""Storage dtypes ``--state-pool-dtype`` accepts."""


class _KdaDims(NamedTuple):
    """The KDA dimensions that size both the layers and the state pools."""

    num_heads: int
    head_dim: int
    conv_kernel_dim: int


def layer_indices(layer_types: Sequence[str], kind: str) -> tuple[int, ...]:
    """Returns the indices of the layers of type ``kind``."""
    return tuple(i for i, t in enumerate(layer_types) if t == kind)


def recurrent_state_bytes(
    kv_params: KVCacheParamInterface, num_devices: int
) -> int:
    """Returns the bytes one request's recurrent state occupies.

    The total is summed over ``num_devices``, read off the leaf the cache
    declares.
    """
    state = recurrent_leaf(kv_params)
    if state is None:
        return 0
    return state.bytes_per_state * num_devices


def residual_bytes_per_token(
    hc_mult: int, hidden_size: int, dtype: DType
) -> int:
    """Returns the bytes one live ``hc_mult``-stream residual takes per token."""
    return hc_mult * hidden_size * dtype.size_in_bytes


def state_bounded_batch_size(
    per_request_bytes: int,
    devices: Sequence[Device],
    *,
    weights_size: int,
    device_memory_utilization: float,
) -> int:
    """Returns a memory-safe default ``max_batch_size``.

    The state occupies pages of the KV pool, so it is reserved nowhere here.
    The states get up to half the post-weights budget and the KV absorbs the
    rest; the halving bounds concurrency, it allocates nothing.
    """
    if per_request_bytes == 0:
        return 32
    try:
        free_bytes = FreeMemory.summed(devices).hi
    except Exception:
        # Only the minimized bring-up fixture lands here.
        return 8
    budget = int(free_bytes * device_memory_utilization) - weights_size
    if budget <= 0:
        return 1
    return min(512, max(1, (budget // 2) // per_request_bytes))


def get_text_config(huggingface_config: AutoConfig) -> AutoConfig:
    """Extracts the text sub-config, tolerating a flat text-only config."""
    return getattr(huggingface_config, "text_config", huggingface_config)


def resolve_layer_types(text_config: AutoConfig) -> list[str]:
    """Returns the per-layer attention family list.

    Prefers the checkpoint's explicit ``layer_types``. Falls back to
    ``linear_attn_config.full_attn_layers``, then to the period-4 rule the
    family uses (sparse where ``idx % 4 == 3``).
    """
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types:
        return list(layer_types)

    num_layers = text_config.num_hidden_layers
    linear_attn_config = getattr(text_config, "linear_attn_config", None) or {}
    if not isinstance(linear_attn_config, Mapping):
        linear_attn_config = vars(linear_attn_config)
    full_attn_layers = linear_attn_config.get("full_attn_layers")
    if full_attn_layers:
        full = set(full_attn_layers)
        return [
            SPARSE_ATTENTION if i in full else LINEAR_ATTENTION
            for i in range(num_layers)
        ]
    return [
        SPARSE_ATTENTION if (i % 4) == 3 else LINEAR_ATTENTION
        for i in range(num_layers)
    ]


def declared_state_dtype(kv_cache_config: KVCacheConfig) -> DType:
    """Returns :attr:`state_dtype` as far as it is knowable pre-``finalize``.

    The state leaves are declared before a quantization scheme resolves,
    and ``cache_dtype`` is not a stand-in: ``--kv-cache-format`` may put
    the MLA latent in fp8 while the KDA pools stay float32.

    Raises:
        ValueError: If ``--state-pool-dtype`` names a dtype the pools
            cannot be stored in.
    """
    override = kv_cache_config.state_pool_dtype
    if override is None:
        return DType.float32
    if override not in _POOL_DTYPE_MAP:
        raise ValueError(
            "state_pool_dtype must be 'bfloat16' or 'float32', got "
            f"{override!r}"
        )
    return _POOL_DTYPE_MAP[override]


def declared_kda_dims(text_config: AutoConfig) -> _KdaDims:
    """Returns the KDA dimensions, keyed by the names HF's config uses.

    The state leaves are declared before an instance exists, so the pools
    cannot read these off :class:`Glm5NextConfig`. Both readers go through
    here instead: sizing the pools off one set of names and the layers off
    another is not a shape error, just a mis-sized pool.
    """
    read = linear_attn_field
    return _KdaDims(
        num_heads=read(text_config, "num_heads", 64),
        head_dim=read(text_config, "head_dim", 128),
        conv_kernel_dim=read(text_config, "short_conv_kernel_size", 4),
    )


def linear_attn_field(text_config: AutoConfig, name: str, default: Any) -> Any:
    """Reads one field out of the nested ``linear_attn_config``."""
    linear_attn_config = getattr(text_config, "linear_attn_config", None) or {}
    if not isinstance(linear_attn_config, Mapping):
        linear_attn_config = vars(linear_attn_config)
    return linear_attn_config.get(name, default)


def get_num_layers(huggingface_config: AutoConfig) -> int:
    """The text config's hidden layer count."""
    text_config = get_text_config(huggingface_config)
    return text_config.num_hidden_layers


def construct_kv_params(
    huggingface_config: AutoConfig,
    pipeline_config: PipelineConfig,
    devices: list[DeviceRef],
    kv_cache_config: KVCacheConfig,
    cache_dtype: DType,
) -> KVCacheParamInterface:
    """Builds the MLA, indexer and recurrent-state children of the cache.

    The graph's input types are built from these params, so a state
    derived any later cannot appear among them.
    """
    text_config = get_text_config(huggingface_config)
    layer_types = resolve_layer_types(text_config)
    num_cached_layers = sum(1 for t in layer_types if t == SPARSE_ATTENTION)
    if num_cached_layers == 0:
        raise ValueError(
            "GLM-5.3-Flash needs at least one sparse-attention layer; "
            f"layer_types resolved to {len(layer_types)} entries with none."
        )
    if pipeline_config.speculative:
        num_draft_layers = getattr(text_config, "num_nextn_predict_layers", 1)
        if num_draft_layers == 0:
            raise ValueError(
                "GLM-5.3-Flash speculates with the MTP layer its "
                "checkpoint ships, but this checkpoint declares none "
                "(num_nextn_predict_layers is 0)."
            )
        # The MTP draft layer is itself sparse MLA and owns a latent.
        num_cached_layers += num_draft_layers

    speculative_method = None
    num_draft_tokens = 0
    if pipeline_config.speculative:
        speculative_method = pipeline_config.speculative.speculative_method
        num_draft_tokens = (
            pipeline_config.speculative.num_speculative_tokens or 0
        )

    mla_quant_config = None
    if cache_dtype in (DType.float8_e4m3fn, DType.float8_e4m3fnuz):
        mla_quant_config = KVCacheQuantizationConfig(
            scale_dtype=DType.int8, quantization_granularity=32
        )

    # NoPE: the latent is `kv_lora_rank` wide with no rotary tail, but the
    # SM100 kernels comptime-assert 576, so it is stored padded. See
    # the config's ``mla_latent_pad_to``.
    latent_width = _resolve_mla_latent_width(text_config)
    mla_kv_params = kv_cache_config.to_params(
        dtype=cache_dtype,
        n_kv_heads=1,
        head_dim=latent_width,
        num_layers=num_cached_layers,
        devices=devices,
        data_parallel_degree=pipeline_config.model.data_parallel_degree,
        is_mla=True,
        num_q_heads=text_config.num_attention_heads,
        kvcache_quant_config=mla_quant_config,
        speculative_method=speculative_method,
        num_draft_tokens=num_draft_tokens,
    )
    index_kpool = getattr(text_config, "index_kpool", 4)
    # Each token attends the indexer's selection, k-pool tail included.
    assert isinstance(mla_kv_params, MLAKVCacheParams)
    mla_kv_params = replace(
        mla_kv_params,
        sparse_indices_stride=indexer_selection_width(
            text_config.index_topk,
            index_kpool,
            getattr(text_config, "index_kpool_always_select_tail", True),
        ),
        # The sparse decode passes per-token top-k lengths.
        sparse_fixed_topk=False,
    )

    # The indexer's K cache holds one *pooled* key per `index_kpool` tokens.
    pooled_head_dim, remainder = divmod(text_config.index_head_dim, index_kpool)
    if remainder:
        raise ValueError(
            f"index_head_dim ({text_config.index_head_dim}) must be "
            f"divisible by index_kpool ({index_kpool}) to declare the "
            "pooled key cache."
        )
    if mla_kv_params.page_size % index_kpool != 0:
        raise ValueError(
            f"kv_cache_page_size ({mla_kv_params.page_size}) must be a "
            f"multiple of index_kpool ({index_kpool}): a page that ends "
            "mid-pool would split a pooled key across two pages, and every "
            "leaf of the multi-cache shares one page table."
        )
    indexer_kv_params = kv_cache_config.to_params(
        dtype=DType.float8_e4m3fn,
        n_kv_heads=1,
        head_dim=pooled_head_dim,
        num_layers=mla_kv_params.num_layers,
        devices=devices,
        data_parallel_degree=pipeline_config.model.data_parallel_degree,
        is_mla=True,
        num_q_heads=text_config.num_attention_heads,
        kvcache_quant_config=KVCacheQuantizationConfig(
            scale_dtype=DType.float32,
            quantization_granularity=pooled_head_dim,
        ),
        speculative_method=speculative_method,
        num_draft_tokens=num_draft_tokens,
    )
    assert isinstance(indexer_kv_params, MLAKVCacheParams)
    indexer_kv_params = indexer_kv_params.joining_sparse_dispatch_of(
        mla_kv_params
    )
    children: dict[str, CacheLeafParamInterface] = {
        MLA_CACHE_KEY: mla_kv_params,
        INDEXER_CACHE_KEY: indexer_kv_params,
    }
    num_kda_layers = sum(1 for t in layer_types if t == LINEAR_ATTENTION)
    if num_kda_layers:
        kda_dims = declared_kda_dims(text_config)
        # A leaf of zero bytes would be a page the pool cannot tile, so a
        # checkpoint with no KDA layers gets the attention children alone.
        children[STATE_CACHE_KEY] = RecurrentStateParams(
            regions=state_regions(
                num_kda_layers=num_kda_layers,
                num_heads=kda_dims.num_heads,
                head_dim=kda_dims.head_dim,
                conv_kernel_dim=kda_dims.conv_kernel_dim,
                # One ring per indexer, so the ring's layer set is the
                # indexer cache's own -- which includes the MTP draft
                # layer, itself a sparse-MLA layer with an indexer.
                num_sparse_layers=indexer_kv_params.num_layers,
                index_kpool=index_kpool,
                index_head_dim=text_config.index_head_dim,
                dtype=declared_state_dtype(kv_cache_config),
                num_devices=len(devices),
            ),
            devices=mla_kv_params.devices,
            data_parallel_degree=mla_kv_params.data_parallel_degree,
        )
        if pipeline_config.speculative:
            # A speculative verify records its window's KDA updates in
            # this scratch leaf, which the rollback folds once acceptance
            # is known.
            state = children[STATE_CACHE_KEY]
            assert isinstance(state, RecurrentStateParams)
            children[STATE_CACHE_KEY] = replace(
                state,
                regions=with_verify_ring(
                    state.regions,
                    ring_len_for_window(
                        1 + pipeline_config.speculative.draft_width
                    ),
                ),
            )
    return MultiKVCacheParams.from_params(children)
