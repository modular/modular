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
"""Tests the retained DeepSeek-V4.1-Flash ModuleV3 config resolution."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.legacy.kv_cache import MLAKVCacheParams
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    PAGE_SIZE,
    V41Config,
)
from max.pipelines.architectures.deepseekV41_modulev3.model_config import (
    DeepseekV41Config,
)
from max.pipelines.lib import KVCacheConfig
from max.pipelines.speculative.config import (
    SpeculativeConfig,
    SpeculativeMethod,
)


def _text_config(**overrides: Any) -> dict[str, Any]:
    """A small synthetic V4.1 ``text_config`` mapping."""
    text_config: dict[str, Any] = {
        "vocab_size": 128,
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "rms_norm_eps": 1e-6,
        "num_attention_heads": 4,
        "head_dim": 512,
        "q_lora_rank": 32,
        "o_groups": 2,
        "o_lora_rank": 16,
        "hc_mult": 1,
        "sliding_window": 16,
        "n_routed_experts": 8,
        "moe_intermediate_size": 32,
        "num_experts_per_tok": 2,
        "compress_ratios": [0, 2, 1, 2],
        "kv_source_layer_ids": [1, 2],
        "index_source_layer_ids": [3],
        "candidate_source_layer_id": 1,
        "engram_layer_ids": [3],
        "engram_num_embeddings": [2],
        "engram_head_dim": 256,
        "dspark_target_layer_ids": [3],
        "index_topk": 4,
        # The indexer geometry keys are required here as they are in
        # ``V41Config.from_hf``; the published checkpoint defines both.
        "index_n_heads": 32,
        "index_head_dim": 128,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "compress_rope_theta": 160000.0,
    }
    text_config.update(overrides)
    return text_config


def test_get_text_config_normalizes_each_form() -> None:
    # A raw mapping with a nested text_config unwraps to a namespace.
    nested = DeepseekV41Config.get_text_config({"text_config": _text_config()})
    assert nested.vocab_size == 128

    # A bare text mapping becomes a namespace of itself.
    bare = DeepseekV41Config.get_text_config(_text_config())
    assert isinstance(bare, SimpleNamespace)
    assert bare.vocab_size == 128

    # An object with a text_config attribute unwraps; without one, the
    # input already is the text config and passes through unchanged.
    text_config = SimpleNamespace(**_text_config())
    wrapper = SimpleNamespace(text_config=text_config)
    assert DeepseekV41Config.get_text_config(wrapper) is text_config
    assert DeepseekV41Config.get_text_config(text_config) is text_config


def test_resolve_text_fields() -> None:
    fields = DeepseekV41Config.resolve_text_fields(
        SimpleNamespace(**_text_config())
    )
    assert fields["vocab_size"] == 128
    assert fields["sliding_window"] == 16
    assert fields["compress_ratios"] == [0, 2, 1, 2]
    assert fields["kv_source_layer_ids"] == [1, 2]
    assert fields["engram_num_embeddings"] == [2]
    # The DSpark drafter reads the target taps; the SWA leaf's draft rows
    # are sized by the draft-stage count, not this list.
    assert fields["dspark_target_layer_ids"] == [3]
    # Optional checkpoint keys resolve to their defaults, not errors.
    assert fields["n_shared_experts"] == 1
    assert fields["num_hash_layers"] == 0
    assert fields["routed_scaling_factor"] == 1.5
    assert fields["candidate_topk_blocks"] == 0
    # The indexer geometry is required, not defaulted, in this view as
    # in the interface's.
    assert fields["index_n_heads"] == 32
    assert fields["index_head_dim"] == 128


def _construct_kv_params(speculative: Any) -> Any:
    """Runs ``construct_kv_params`` under the given speculative config.

    The interface's ``from_hf`` requires the candidate block count like
    the published checkpoint does, while the shared fixture omits it to
    pin the trunk view's own default of zero.
    """
    pipeline_config = Mock()
    pipeline_config.model = Mock(data_parallel_degree=1)
    pipeline_config.speculative = speculative
    huggingface_config = Mock()
    huggingface_config.to_dict.return_value = {
        "text_config": _text_config(candidate_topk_blocks=2)
    }
    return DeepseekV41Config.construct_kv_params(
        huggingface_config=huggingface_config,
        pipeline_config=pipeline_config,
        devices=[DeviceRef.CPU()],
        kv_cache_config=KVCacheConfig(kv_cache_page_size=PAGE_SIZE),
        cache_dtype=DType.bfloat16,
    )


def test_construct_kv_params_without_a_speculative_method() -> None:
    """Builds the leaf params when no speculative method is named."""
    _construct_kv_params(speculative=None)
    _construct_kv_params(speculative=SpeculativeConfig())


def test_construct_kv_params_sizes_the_pool_for_a_speculative_method() -> None:
    """A speculative config sizes every leaf's pool for the draft rows.

    The V2 pattern: the method and width resolve before the KV cache is
    sized and reach every leaf's params, so the pool reserves each
    request's draft blocks, while the leaf geometry itself stays the
    fixed-ring shape both spec and non-spec boots share.
    """
    params = _construct_kv_params(
        SpeculativeConfig(speculative_method="dspark", num_speculative_tokens=5)
    )
    for child in params.children.values():
        assert child.num_draft_tokens == 5
        assert child.speculative_method == "dspark"
    # The state ring stays at its fixed depth, spec config or not.
    state = params.children["c2a_state"]
    assert isinstance(state, MLAKVCacheParams)
    assert state.slots_per_page == 8


def test_construct_kv_params_needs_a_resolved_draft_width() -> None:
    """A named method without a resolved width cannot size the pool.

    The draft width sizes the per-request blocks and the state ring's
    depth check, so an unset width fails here rather than sizing a
    draft-free pool for a speculative run.
    """
    speculative = SpeculativeConfig(speculative_method="dspark")
    assert speculative.num_speculative_tokens is None
    with pytest.raises(
        ValueError, match="resolved before the KV cache is sized"
    ):
        _construct_kv_params(speculative)


@pytest.mark.parametrize("method", ["mtp", "eagle", "dflash"])
def test_construct_kv_params_refuses_non_dspark_methods(
    method: SpeculativeMethod,
) -> None:
    """DSpark is the trunk's only draft pairing, so every other method
    fails at startup rather than booting a plain trunk with draft-sized
    pages."""
    speculative = SpeculativeConfig(
        speculative_method=method, num_speculative_tokens=3
    )
    with pytest.raises(ValueError, match="speculates only with dspark"):
        _construct_kv_params(speculative)


def test_v41_config_from_hf_reads_the_nested_text_config() -> None:
    """The interface config reads the published ``text_config`` nesting.

    Every field is required as the published checkpoint defines it, so
    the candidate block count is passed here explicitly -- the shared
    fixture omits it to pin the trunk view's own default. The draft
    stages resolve the way V2's ``num_dspark_stages`` resolves them, so
    a config that names no MTP key serves the default three.
    """
    config = V41Config.from_hf(
        {"text_config": _text_config(candidate_topk_blocks=2)}
    )
    assert config.vocab_size == 128
    assert config.compress_ratios == (0, 2, 1, 2)
    assert config.kv_source_layer_ids == (1, 2)
    assert config.engram_num_embeddings == (2,)
    assert config.num_dspark_stages == 3
    assert (
        V41Config.from_hf(
            {
                "text_config": _text_config(
                    candidate_topk_blocks=2, n_mtp_layers=1
                )
            }
        ).num_dspark_stages
        == 1
    )
