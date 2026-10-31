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
"""Synthetic text configuration shared by runtime integration tests."""

from types import SimpleNamespace
from typing import Any


def text_config(**overrides: Any) -> SimpleNamespace:
    """The synthetic text config, with any field overridden.

    Tests that need a layout the published checkpoint does not produce
    (a ratio-1 owner beside the candidate source, a ratio-2 reindex row)
    override the layer-role fields; everything else keeps the values
    the runtime tests share.
    """
    base: dict[str, Any] = dict(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        rms_norm_eps=1e-20,
        num_attention_heads=4,
        head_dim=512,
        q_lora_rank=64,
        o_groups=2,
        # The grouped wo_a's production arm (the block-diagonal launch)
        # reads one B-scales interleave per shard, viewed per group over
        # whole 128-row interleave blocks, so this fixture stages the
        # production geometry. Smaller ranks (a multiple of 32) keep the
        # grouped reference arm.
        o_lora_rank=128,
        hc_mult=4,
        sliding_window=128,
        n_routed_experts=8,
        n_shared_experts=1,
        moe_intermediate_size=64,
        num_experts_per_tok=6,
        compress_ratios=[0, 1],
        kv_source_layer_ids=[1],
        index_source_layer_ids=[1],
        candidate_source_layer_id=1,
        candidate_topk_blocks=2,
        candidate_block_size=64,
        engram_layer_ids=[],
        engram_num_embeddings=[],
        engram_head_dim=256,
        dspark_target_layer_ids=[],
        index_topk=4,
        index_n_heads=2,
        index_head_dim=128,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        qk_rope_head_dim=64,
        rope_scaling={
            "factor": 16.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 4096,
        },
    )
    base.update(overrides)
    return SimpleNamespace(**base)
