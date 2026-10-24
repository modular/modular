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
"""Tests that DeepSeek-V4.1's rotary tables cover speculative positions.

With speculative decoding, a request at the length cap verifies and drafts
rows up to ``spec_decode_cache_slack`` positions past ``max_seq_len``: the
DSpark block of a request that commits its last tokens starts at the cap. Every
rotary table the trunk and the DSpark drafter gather from needs a row for each
of those positions, or the GPU gather traps ("Gather index out of bounds").
The memory planner must also reserve the indexer's candidate keep step at the
width the graph builds it.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, ops
from max.nn.kv_cache import spec_decode_cache_slack
from max.pipelines.architectures import hf_config_shims  # noqa: F401
from max.pipelines.architectures.deepseekV41 import memory_planner
from max.pipelines.architectures.deepseekV41.deepseekV41 import (
    DeepseekV41Block,
)
from max.pipelines.architectures.deepseekV41.dspark_drafter import (
    DeepseekV41WithDSpark,
)
from max.pipelines.architectures.deepseekV41.layers.attention import (
    CSA2Attention,
)
from max.pipelines.architectures.deepseekV41.model_config import (
    DeepseekV41Config,
)
from transformers import AutoConfig

TESTDATA = Path(__file__).parent / "testdata"
MAX_SEQ_LEN = 64
TRUNK_LAYERS = 2
STAGES = 3


def _config(num_draft_tokens: int) -> DeepseekV41Config:
    # One compressed (YaRN) layer and one sliding-window layer, so both rotary
    # schedules are built. The CSA2 and indexer kernels fix the window, head,
    # rope and index widths.
    return DeepseekV41Config(
        dtype=DType.bfloat16,
        kv_params=MagicMock(
            num_draft_tokens=num_draft_tokens,
            num_draft_tokens_per_step=num_draft_tokens,
        ),
        devices=[DeviceRef.CPU()],
        max_seq_len=MAX_SEQ_LEN,
        vocab_size=64,
        hidden_size=128,
        num_hidden_layers=TRUNK_LAYERS,
        rms_norm_eps=1e-6,
        sliding_window=128,
        n_routed_experts=8,
        moe_intermediate_size=128,
        num_experts_per_tok=2,
        num_dspark_layers=STAGES,
        dspark_target_layer_ids=[0],
        dspark_block_size=5,
        dspark_markov_rank=32,
        dspark_n_routed_experts=4,
        dspark_num_experts_per_tok=2,
        compress_ratios=[1, 0] + [0] * STAGES,
        kv_source_layer_ids=[0],
        index_source_layer_ids=[0],
        candidate_source_layer_id=0,
        candidate_block_size=8,
        candidate_topk_blocks=4,
        num_attention_heads=2,
        head_dim=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        q_lora_rank=64,
        o_groups=2,
        o_lora_rank=32,
        rope_scaling={
            "factor": 16.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "original_max_position_embeddings": MAX_SEQ_LEN,
        },
    )


def _model(num_draft_tokens: int) -> DeepseekV41WithDSpark:
    return DeepseekV41WithDSpark(
        _config(num_draft_tokens), DeviceRef.CPU(), engram_host_tables={}
    )


def _trunk_attentions(model: DeepseekV41WithDSpark) -> list[CSA2Attention]:
    attentions = []
    for layer in model.trunk.layers:
        assert isinstance(layer, DeepseekV41Block)
        assert isinstance(layer.attn, CSA2Attention)
        attentions.append(layer.attn)
    return attentions


@pytest.mark.parametrize(
    "num_draft_tokens, keep_blocks",
    # 64 positions, or 64 + 17 with the slack, in blocks of 8.
    [(0, 8), (5, 11)],
)
def test_rotary_tables_cover_speculative_slack(
    num_draft_tokens: int, keep_blocks: int
) -> None:
    """Every trunk and drafter table has ``max_seq_len`` plus the slack rows."""
    model = _model(num_draft_tokens)
    want = MAX_SEQ_LEN + spec_decode_cache_slack(model.config.kv_params)
    trunk = _trunk_attentions(model)
    drafter = list(model.dspark._attentions())
    assert len(trunk) == TRUNK_LAYERS and len(drafter) == STAGES
    tables = [(f"trunk layer {i}", a.rope) for i, a in enumerate(trunk)]
    tables += [(f"drafter stage {i}", a.rope) for i, a in enumerate(drafter)]
    for where, rope in tables:
        assert rope.max_seq_len >= want, (
            f"{where} rotary table has {rope.max_seq_len} rows; speculative"
            f" positions reach {want - 1}"
        )
    assert trunk[0].layer_info.candidate_role == "source"
    assert trunk[0].candidate_blocks == keep_blocks


def test_drafter_gathers_last_speculative_position_on_cpu() -> None:
    """The drafter's and trunk's rotary gathers accept the last slack position.

    Runs the drafter's own ``_freqs_cis`` (the gather that trapped on the GPU)
    and the trunk's equivalent gather on CPU, where an out-of-range index
    raises instead of trapping.
    """
    model = _model(num_draft_tokens=5)
    last = MAX_SEQ_LEN + spec_decode_cache_slack(model.config.kv_params) - 1
    positions = np.array([MAX_SEQ_LEN - 1, MAX_SEQ_LEN, last], dtype=np.int32)
    drafter = model.dspark._attentions()[0]
    trunk = _trunk_attentions(model)[0]
    with Graph("dsv41_rope_gather_at_capacity", input_types=[]) as graph:
        pos = ops.constant(positions, DType.int32, DeviceRef.CPU())
        graph.output(
            drafter._freqs_cis(pos),
            ops.gather(trunk.rope.freqs_cis_base(), pos, axis=0),
        )
    session = InferenceSession(devices=[CPU()])
    drafted, trunk_rows = session.load(graph).execute()
    assert drafted.to_numpy().shape == (3, 32, 2)
    assert trunk_rows.to_numpy().shape == (3, 32, 2)


@pytest.mark.parametrize(
    "max_length, keep_blocks",
    [
        # ceil(16384 / 8) == candidate_topk_blocks: no committed row can
        # reach a block past the top k, so the graph skips the keep step.
        (16384, 0),
        # From here the step runs; its blocks cover the slack's keys too.
        (16400, 2053),
        (40000, 5003),
    ],
)
def test_planner_reserves_the_graphs_candidate_blocks(
    max_length: int, keep_blocks: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The planner reserves the keep step the graph builds.

    Speculative decoding is on, and the candidate-source layer uses the real
    checkpoint's block sizes.
    """
    hf_config = AutoConfig.from_pretrained(str(TESTDATA / "config.json"))
    text_config = DeepseekV41Config.get_text_config(hf_config)
    config = dataclasses.replace(
        _config(num_draft_tokens=5),
        max_seq_len=max_length,
        candidate_source_layer_id=0,
        candidate_block_size=text_config.candidate_block_size,
        candidate_topk_blocks=text_config.candidate_topk_blocks,
    )
    source = DeepseekV41WithDSpark(
        config, DeviceRef.CPU(), engram_host_tables={}
    ).trunk.layers[0]
    assert isinstance(source, DeepseekV41Block)
    assert isinstance(source.attn, CSA2Attention)
    assert source.attn.layer_info.candidate_role == "source"
    assert source.attn.candidate_blocks == keep_blocks

    planned: list[int] = []
    reserve = memory_planner.indexer_activation_bytes

    def spy(
        *,
        live_rows: int,
        candidate_blocks: int,
        kept_blocks: int,
        block_size: int,
    ) -> int:
        planned.append(candidate_blocks)
        return reserve(
            live_rows=live_rows,
            candidate_blocks=candidate_blocks,
            kept_blocks=kept_blocks,
            block_size=block_size,
        )

    monkeypatch.setattr(memory_planner, "indexer_activation_bytes", spy)
    pipeline_config = MagicMock()
    pipeline_config.model.vision_config_overrides = {
        "max_images_per_request": 0
    }
    pipeline_config.model.max_length = max_length
    pipeline_config.model.device_specs = [MagicMock()]
    pipeline_config.model.data_parallel_degree = 1
    pipeline_config.runtime.max_num_input_tokens = 8192
    pipeline_config.runtime.max_batch_input_tokens = 8192
    memory_planner.DeepseekV41MemoryPlanner(config).estimate_activation_memory(
        pipeline_config, hf_config
    )
    assert planned and set(planned) == {keep_blocks}
