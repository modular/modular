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
"""The indexer's construction contracts, against the resolver's facts.

``Indexer`` derives its store and select roles from the layer's resolved
:class:`~...layer_resolver.LayerInfo`, never from the block-group string
alone: the resolver accepts layouts the published checkpoint does not
produce (a ratio-1 owner beside the candidate source, a ratio-2 candidate
source, a ratio-2 reindex row), and those must keep their own ratio and
their own streams. These tests build each layout and pin the derived
flags, the kind cross-check, the MXFP4 head-geometry rejection, and the
checkpoint fields the constructor reads (``candidate_block_size``,
``rms_norm_eps``) -- all construction-time, so they run on CPU.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    V41Config,
    build_kv_params,
)
from max.pipelines.architectures.deepseekV41_modulev3.layer_resolver import (
    block_group,
    resolve,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.attention import (
    Attention,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.compressor import (
    Compressor,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.indexer import (
    Indexer,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.rope import (
    RopeParams,
)
from max.pipelines.architectures.deepseekV41_modulev3.model_config import (
    DeepseekV41Config,
)
from max.pipelines.lib import KVCacheConfig
from max.tensor.sharding import DeviceMesh
from runtime_config import text_config

_MESH = DeviceMesh((CPU(),), (1,), ("tp",))
_ROPE = RopeParams(64, 32.0, 1.0, 4096)


def _model_config(text: SimpleNamespace) -> DeepseekV41Config:
    """The resolved model config of a synthetic text layout."""
    kv = build_kv_params(
        V41Config.from_hf({"text_config": vars(text)}),
        KVCacheConfig(kv_cache_page_size=128),
        [DeviceRef.CPU()],
    )
    return DeepseekV41Config(
        dtype=DType.bfloat16,
        kv_params=kv,
        devices=[DeviceRef.CPU()],
        max_seq_len=256,
        **DeepseekV41Config.resolve_text_fields(text),
    )


def _layout(
    text: SimpleNamespace, layer: int, **overrides: Any
) -> tuple[DeepseekV41Config, str, dict[str, Any]]:
    """The layer's ``(config, block group, indexer kwargs)``."""
    config = _model_config(text)
    return (
        config,
        block_group(config, layer),
        dict(
            layer_info=resolve(config)[layer],
            rope=_ROPE,
            max_position=64,
            **overrides,
        ),
    )


# A ratio-1 KV owner that is not the candidate source: resolver-valid,
# absent from the published checkpoint.
_OWNER_BESIDE_CANDIDATE = dict(
    compress_ratios=[2, 1],
    kv_source_layer_ids=[0, 1],
    index_source_layer_ids=[0, 1],
    candidate_source_layer_id=0,
)
# The candidate source itself on a ratio-2 layer, with a second owner.
_RATIO_TWO_CANDIDATE_SOURCE = dict(
    compress_ratios=[2, 2],
    kv_source_layer_ids=[0, 1],
    index_source_layer_ids=[0, 1],
    candidate_source_layer_id=1,
)
# A ratio-2 reindex row rescoring a ratio-2 candidate source.
_RATIO_TWO_REINDEX = dict(
    compress_ratios=[2, 2],
    kv_source_layer_ids=[0],
    index_source_layer_ids=[0, 1],
    candidate_source_layer_id=0,
)


def test_owner_beside_the_candidate_source_keeps_its_streams() -> None:
    """A ``full_r1`` owner stores its own index keys at ratio 1."""
    config, kind, kwargs = _layout(text_config(**_OWNER_BESIDE_CANDIDATE), 1)
    indexer = Indexer(config, _MESH, kind, **kwargs)
    assert indexer.owns_k
    assert indexer.ratio == 1
    assert not indexer.is_candidate_source
    assert not indexer.uses_candidates


def test_ratio_two_candidate_source_stores_at_ratio_two() -> None:
    """A ``full_r2_cand`` source owns keys, pools blocks, at ratio 2."""
    config, kind, kwargs = _layout(
        text_config(**_RATIO_TWO_CANDIDATE_SOURCE), 1
    )
    indexer = Indexer(config, _MESH, kind, **kwargs)
    assert indexer.owns_k
    assert indexer.ratio == 2
    assert indexer.is_candidate_source
    assert not indexer.uses_candidates


def test_ratio_two_reindex_selects_through_the_candidate_pool() -> None:
    """A ``reindex_r2`` row scores the pool's blocks at ratio 2."""
    config, kind, kwargs = _layout(text_config(**_RATIO_TWO_REINDEX), 1)
    indexer = Indexer(config, _MESH, kind, **kwargs)
    assert not indexer.owns_k
    assert indexer.ratio == 2
    assert indexer.uses_candidates


def test_kind_disagreeing_with_the_resolved_facts_is_rejected() -> None:
    """A kind string from another layer's role never builds an indexer."""
    config, _, kwargs = _layout(text_config(**_RATIO_TWO_REINDEX), 1)
    with pytest.raises(ValueError, match="does not match"):
        Indexer(config, _MESH, "full_r1_cand", **kwargs)


def test_geometry_outside_the_mxfp4_head_layout_is_rejected() -> None:
    """The MXFP4 kernels read 128-wide keys under a 64-wide rotation."""
    narrow = text_config(**_RATIO_TWO_REINDEX, index_head_dim=112)
    config, kind, kwargs = _layout(narrow, 1)
    with pytest.raises(ValueError, match="index_head_dim 128"):
        Indexer(config, _MESH, kind, **kwargs)
    config, kind, kwargs = _layout(text_config(**_RATIO_TWO_REINDEX), 1)
    with pytest.raises(ValueError, match="rope_head_dim 64"):
        Indexer(
            config,
            _MESH,
            kind,
            **dict(kwargs, rope=RopeParams(32, 32.0, 1.0, 4096)),
        )


def test_candidate_block_size_comes_from_the_config() -> None:
    """The default block size is the checkpoint's, not a module constant."""
    config, kind, kwargs = _layout(text_config(**_OWNER_BESIDE_CANDIDATE), 1)
    indexer = Indexer(config, _MESH, kind, **kwargs)
    assert indexer.candidate_block_size == 64
    indexer = Indexer(
        config, _MESH, kind, **dict(kwargs, candidate_block_size=4)
    )
    assert indexer.candidate_block_size == 4


def test_norms_read_the_checkpoint_epsilon() -> None:
    """Every attention-half norm takes ``rms_norm_eps`` from the config."""
    text = text_config(**_OWNER_BESIDE_CANDIDATE, rms_norm_eps=1e-12)
    config, kind, kwargs = _layout(text, 1)
    indexer = Indexer(config, _MESH, kind, **kwargs)
    attn = Attention(config, kwargs["layer_info"], _MESH, rope=_ROPE)
    compressor = Compressor(config, 1, _MESH)
    assert indexer.k_norm.eps == 1e-12
    assert attn.q_norm.eps == 1e-12
    assert attn.kv_norm.eps == 1e-12
    assert compressor.norm.eps == 1e-12
