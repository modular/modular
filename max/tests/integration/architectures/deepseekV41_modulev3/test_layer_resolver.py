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
"""Resolver regressions against the published checkpoint's layer mapping."""

from __future__ import annotations

from pathlib import Path

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures import register_all_models
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    V41Config,
    block_group,
    build_kv_params,
)
from max.pipelines.architectures.deepseekV41_modulev3.layer_resolver import (
    resolve,
)
from max.pipelines.architectures.deepseekV41_modulev3.model_config import (
    DeepseekV41Config,
)
from max.pipelines.lib import KVCacheConfig
from transformers import AutoConfig

_V2_TESTDATA = (
    Path(__file__).resolve().parent.parent / "deepseekV41" / "testdata"
)


def _resolver_config(
    num_hidden_layers: int,
    compress_ratios: list[int],
    kv_source_layer_ids: list[int],
    index_source_layer_ids: list[int],
    candidate_source_layer_id: int,
) -> DeepseekV41Config:
    """A config bare enough for the pure resolver: no graph, cache or weights.

    :func:`resolve` is a pure function of the mapping fields, so the
    rejection tests set exactly the fields it reads rather than building
    the full pipeline config.
    """
    config = object.__new__(DeepseekV41Config)
    config.num_hidden_layers = num_hidden_layers
    config.compress_ratios = compress_ratios
    config.kv_source_layer_ids = kv_source_layer_ids
    config.index_source_layer_ids = index_source_layer_ids
    config.candidate_source_layer_id = candidate_source_layer_id
    config.engram_layer_ids = []
    config.index_topk = 4
    config.rope_scaling = None
    config.rope_theta = 10000.0
    config.compress_rope_theta = 160000.0
    return config


def test_published_checkpoint_resolves_trunk_and_draft_stages() -> None:
    """The registered checkpoint's real mapping: 43 ratio rows, 40 layers.

    The published ``config.json`` carries three trailing DSpark draft rows
    past the 40 trunk layers; ``resolve`` must accept it, resolve the trunk
    and the three draft stages, and reproduce the checkpoint's attention
    plan exactly.
    """
    # Registers the AutoConfig shim for the model_type the bazel-pinned
    # transformers does not know natively.
    register_all_models()
    hf = AutoConfig.from_pretrained(
        str(_V2_TESTDATA / "config.json"), local_files_only=True
    )
    text = DeepseekV41Config.get_text_config(hf)
    interface_config = V41Config.from_hf({"text_config": vars(text)})
    kv_params = build_kv_params(
        interface_config,
        KVCacheConfig(kv_cache_page_size=128),
        [DeviceRef.CPU()],
    )
    config = DeepseekV41Config(
        dtype=DType.bfloat16,
        kv_params=kv_params,
        devices=[DeviceRef.CPU()],
        max_seq_len=256,
        **DeepseekV41Config.resolve_text_fields(text),
    )
    rows = resolve(config)
    assert [row.layer_id for row in rows] == list(range(43))
    assert [row.compress_ratio for row in rows] == (
        [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
    )
    # The rows past the trunk are the drafter's stages: SWA-only windows.
    assert [row.side for row in rows[40:]] == ["drafter"] * 3
    assert [row.swa_only for row in rows[40:]] == [True] * 3
    assert [row.dspark_stage for row in rows] == [None] * 40 + [0, 1, 2]
    # The four KV owners, and only they, run a full attention pass.
    assert [row.layer_id for row in rows if row.attention_mode == "Full"] == [
        2,
        8,
        14,
        20,
    ]
    assert [
        row.layer_id for row in rows if row.attention_mode == "Reindex"
    ] == [
        24,
        28,
        32,
        36,
    ]
    assert [row.layer_id for row in rows if row.has_indexer] == [
        2,
        8,
        14,
        20,
        24,
        28,
        32,
        36,
    ]
    # The candidate pool: layer 20 sources it, the Reindex layers consume it.
    assert [row.layer_id for row in rows if row.candidate_role == "source"] == [
        20
    ]
    assert [
        row.layer_id for row in rows if row.candidate_role == "consumer"
    ] == [24, 28, 32, 36]
    # A Reindex layer rescoring the candidate source's KV address domain
    # (kv source 20) with a fresh index of its own.
    assert rows[24].kv_source_layer_id == 20
    assert rows[24].index_source_layer_id == 24
    # Two Engram layers, on the checkpoint's own IDs.
    assert [row.layer_id for row in rows if row.has_engram] == [1, 14]
    # SWA-only layers use plain rope; compressed layers use YaRN.
    assert rows[0].rope_regime == "plain"
    assert rows[2].rope_regime == "yarn"
    # The shared-subgraph bodies the trunk runs, from the interface's own
    # frozen grouping.
    assert block_group(interface_config, 0) == "swa_only"
    assert block_group(interface_config, 2) == "full_r2"
    assert block_group(interface_config, 3) == "reuse_r2"
    assert block_group(interface_config, 20) == "full_r1_cand"
    assert block_group(interface_config, 21) == "reuse_r1"
    assert block_group(interface_config, 24) == "reindex_r1"


def test_the_ratio_list_drives_the_resolved_rows() -> None:
    """``resolve`` yields one row per ratio entry, stages included.

    The stage count is not a separate knob: the trailing entries past
    ``num_hidden_layers`` resolve as DSpark stages, so the resolver stays
    correct if DSpark ever changes its stage count.
    """
    config = _resolver_config(4, [0, 2, 2, 2, 0, 0, 0], [1, 2], [1, 2, 3], 2)
    rows = resolve(config)
    assert [row.layer_id for row in rows] == list(range(7))
    assert [row.compress_ratio for row in rows] == [0, 2, 2, 2, 0, 0, 0]
    assert [row.side for row in rows] == ["encoder"] * 2 + ["decoder"] * 2 + [
        "drafter"
    ] * 3
    assert [row.dspark_stage for row in rows] == [None] * 4 + [0, 1, 2]


def test_a_bad_draft_row_fails_the_drafter_not_the_trunk() -> None:
    """A stage row the drafter cannot run is refused only where it is read.

    A DSpark stage is window-only, so the drafter refuses any other ratio;
    a plain trunk boot resolves its own rows and never reads the stage rows.
    """
    config = _resolver_config(4, [0, 2, 2, 2, 7], [1, 2], [1, 2, 3], 2)
    with pytest.raises(ValueError, match="window-only"):
        resolve(config)
    rows = resolve(config, include_drafter=False)
    assert [row.layer_id for row in rows] == [0, 1, 2, 3]


def test_short_ratio_list_is_rejected() -> None:
    """A ratio list short of the trunk would else resolve a partial trunk.

    The KV leaf specs size the cache for ``num_hidden_layers`` plus the
    draft stages, so nothing downstream would catch the missing rows.
    """
    config = _resolver_config(4, [0, 2], [1], [1], 1)
    with pytest.raises(ValueError, match="one entry per trunk layer"):
        resolve(config)


def test_owner_beyond_the_ratio_rows_is_rejected() -> None:
    config = _resolver_config(4, [0, 2, 2, 2, 0], [2, 5], [2, 5], 2)
    with pytest.raises(ValueError, match="sorted unique trunk"):
        resolve(config)


def test_swa_only_owner_is_rejected() -> None:
    config = _resolver_config(4, [0, 0, 2, 2, 0], [0, 2], [0, 2], 2)
    with pytest.raises(ValueError, match="SWA-only"):
        resolve(config)
