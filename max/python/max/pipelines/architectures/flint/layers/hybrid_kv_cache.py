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

"""KV cache parameters for architectures that mix sliding and full attention.

Each attention type owns a cache child, so a layer's index inside its child
differs from its index in the model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

from max.dtype import DType
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams
from max.pipelines.lib import KVCacheConfig, PipelineConfig

SLIDING = "sliding_attention"
FULL = "full_attention"
_SLIDING_ALIASES = frozenset(
    {
        "sliding_attention",
        "sliding_window",
        "sliding",
        "local",
        "local_attention",
    }
)
_FULL_ALIASES = frozenset(
    {"full_attention", "full", "global", "global_attention", "attention"}
)


def alternating_sliding_first(num_layers: int) -> list[str]:
    """Returns a pattern that slides on even layers."""
    return [SLIDING if i % 2 == 0 else FULL for i in range(num_layers)]


def full_every(k: int, final_full: bool = True) -> Callable[[int], list[str]]:
    """Returns a pattern with full attention every ``k`` layers."""

    def pattern(num_layers: int) -> list[str]:
        return [
            FULL
            if (i % k == 0 or (final_full and i == num_layers - 1))
            else SLIDING
            for i in range(num_layers)
        ]

    return pattern


def normalize_layer_types(
    hf_cfg: Any,
    default: Callable[[int], list[str]] | None = None,
) -> list[str]:
    """Returns the per-layer attention types in the cache vocabulary.

    Accepts the ``sliding_window`` spelling as an alias.
    """
    num_layers = hf_cfg.num_hidden_layers
    raw = getattr(hf_cfg, "layer_types", None)
    if not raw:
        if default is None:
            raise ValueError(
                "config carries no layer_types and no default pattern was "
                "given (pass default=alternating_sliding_first / "
                "full_every(k))"
            )
        raw = default(num_layers)
    raw = list(raw)
    if len(raw) != num_layers:
        raise ValueError(
            f"layer_types covers {len(raw)} layers, config has {num_layers}"
        )
    out = []
    for t in raw:
        if t in _SLIDING_ALIASES:
            out.append(SLIDING)
        elif t in _FULL_ALIASES:
            out.append(FULL)
        else:
            raise ValueError(
                f"unknown layer type {t!r}: not a sliding/full attention "
                "alias (a different cache kind needs its own tree)"
            )
    return out


def hybrid_swa_full_kv_params(
    *,
    layer_types: Sequence[str],
    sliding_window: int,
    pipeline_config: PipelineConfig,
    devices: list[DeviceRef],
    kv_cache_config: KVCacheConfig,
    cache_dtype: DType,
    n_kv_heads: int,
    head_dim: int,
    allow_kv_head_replication: bool = False,
    sliding_n_kv_heads: int | None = None,
    sliding_head_dim: int | None = None,
    full_n_kv_heads: int | None = None,
    full_head_dim: int | None = None,
) -> MultiKVCacheParams:
    """Builds the sliding and full KV cache tree.

    The sliding child carries the window; the full child does not.
    """
    sliding_layers = 0
    full_layers = 0
    for attention_type in layer_types:
        if attention_type == SLIDING:
            sliding_layers += 1
        elif attention_type == FULL:
            full_layers += 1
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

    num_spec_tokens = (
        (pipeline_config.speculative.num_speculative_tokens or 0)
        if pipeline_config.speculative
        else 0
    )
    speculative_method = (
        pipeline_config.speculative.speculative_method
        if pipeline_config.speculative
        else None
    )
    sliding_kv_params = kv_cache_config.to_params(
        dtype=cache_dtype,
        n_kv_heads=(
            n_kv_heads if sliding_n_kv_heads is None else sliding_n_kv_heads
        ),
        head_dim=head_dim if sliding_head_dim is None else sliding_head_dim,
        num_layers=sliding_layers,
        devices=devices,
        data_parallel_degree=pipeline_config.model.data_parallel_degree,
        speculative_method=speculative_method,
        num_draft_tokens=num_spec_tokens,
        allow_kv_head_replication=allow_kv_head_replication,
        window_size=sliding_window,
    )
    full_kv_params = kv_cache_config.to_params(
        dtype=cache_dtype,
        n_kv_heads=n_kv_heads if full_n_kv_heads is None else full_n_kv_heads,
        head_dim=head_dim if full_head_dim is None else full_head_dim,
        num_layers=full_layers,
        devices=devices,
        data_parallel_degree=pipeline_config.model.data_parallel_degree,
        speculative_method=speculative_method,
        num_draft_tokens=num_spec_tokens,
        allow_kv_head_replication=allow_kv_head_replication,
        window_size=None,
    )
    # Children order matters: unflatten_basic_kv_tree yields sliding first.
    return MultiKVCacheParams.from_params(
        {SLIDING: sliding_kv_params, FULL: full_kv_params}
    )


@dataclass(frozen=True)
class HybridLayerSlot:
    """The cache child, leaf params, and index for one layer."""

    layer_type: str
    is_sliding: bool
    kv_params: KVCacheParams
    layer_idx_in_cache: int


def hybrid_layer_plan(
    kv_params: MultiKVCacheParams, layer_types: Sequence[str]
) -> list[HybridLayerSlot]:
    """Returns one slot per layer, indexed inside its own cache child."""
    # `hybrid_swa_full_kv_params` builds both children from `KVCacheParams`
    # leaves, and the tree is frozen once constructed.
    by_type = cast(dict[str, KVCacheParams], kv_params.children)
    counts = {SLIDING: 0, FULL: 0}
    plan = []
    for t in layer_types:
        plan.append(
            HybridLayerSlot(
                layer_type=t,
                is_sliding=(t == SLIDING),
                kv_params=by_type[t],
                layer_idx_in_cache=counts[t],
            )
        )
        counts[t] += 1
    return plan


def unflatten_hybrid_kv(
    kv_params: MultiKVCacheParams, variadic_args: Iterable[Any], tokens: Any
) -> tuple[PagedCacheValues, PagedCacheValues]:
    """Splits the graph's KV arguments into sliding and full values."""
    kv_inputs = iter(x._graph_value for x in variadic_args)
    sliding_inputs, full_inputs = kv_params.unflatten_basic_kv_tree(kv_inputs)
    return (
        PagedCacheValues.from_upstream(sliding_inputs, tokens.mapping),
        PagedCacheValues.from_upstream(full_inputs, tokens.mapping),
    )
