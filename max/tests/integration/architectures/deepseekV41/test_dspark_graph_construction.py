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
"""Traces the DSpark pairing's speculative graph at the real V4.1 config.

Tracing only, but the test needs a GPU because the MXFP8 scale layout is
chosen from the host's accelerator rather than graph metadata.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.nn.legacy.comm.ep import EPConfig
from max.nn.legacy.kv_cache import MultiKVCacheParams
from max.nn.legacy.transformer import ReturnHiddenStates
from max.pipelines.architectures import hf_config_shims  # noqa: F401
from max.pipelines.architectures.deepseekV41 import kv_leaves
from max.pipelines.architectures.deepseekV41.deepseekV41 import DeepseekV41
from max.pipelines.architectures.deepseekV41.dspark_drafter import (
    DeepseekV41DSparkDrafter,
)
from max.pipelines.architectures.deepseekV41.layers.engram_hash import (
    EngramLayout,
)
from max.pipelines.architectures.deepseekV41.layers.engram_lookup import (
    EngramLookup,
)
from max.pipelines.architectures.deepseekV41.model_config import (
    DeepseekV41Config,
    RoutedExpertFormat,
)
from max.pipelines.architectures.dspark_deepseekV41 import (
    DSparkDeepseekV41DraftConfig,
)
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    build_spec_graph,
)
from max.pipelines.speculative.config import SpeculativeConfig
from transformers import AutoConfig

TESTDATA = Path(__file__).parent / "testdata"
NUM_SPECULATIVE_TOKENS = 5


def _speculative(mixed_width: int | None = None) -> SpeculativeConfig:
    return SpeculativeConfig(
        speculative_method="dspark",
        num_speculative_tokens=NUM_SPECULATIVE_TOKENS,
        num_speculative_tokens_mixed_batch=mixed_width,
    )


def _model_config(
    config: AutoConfig,
    devices: list[DeviceRef],
    data_parallel_degree: int,
    expert_parallel: bool = False,
    expert_format: RoutedExpertFormat = RoutedExpertFormat.NVFP4,
) -> DeepseekV41Config:
    text_config = DeepseekV41Config.get_text_config(config)
    pipeline_config = Mock()
    pipeline_config.model.data_parallel_degree = data_parallel_degree
    pipeline_config.speculative = _speculative()
    kv_params = kv_leaves.construct_kv_params(
        huggingface_config=text_config,
        pipeline_config=pipeline_config,
        devices=devices,
        kv_cache_config=KVCacheConfig(),
        cache_dtype=DType.bfloat16,
    )
    text_fields = DeepseekV41Config.resolve_text_fields(text_config)
    ep_config = (
        EPConfig(
            dispatch_dtype=expert_format.dispatch_dtype,
            dispatch_quant_config=expert_format.activation_quant_config,
            combine_dtype=DType.bfloat16,
            hidden_size=text_fields["hidden_size"],
            top_k=text_fields["num_experts_per_tok"],
            n_experts=text_fields["n_routed_experts"],
            max_tokens_per_rank=256,
            n_gpus_per_node=len(devices),
            n_nodes=1,
        )
        if expert_parallel
        else None
    )
    return DeepseekV41Config(
        dtype=DType.bfloat16,
        ep_config=ep_config,
        kv_params=kv_params,
        devices=devices,
        max_seq_len=4096,
        data_parallel_degree=data_parallel_degree,
        kv_leaf_specs=kv_leaves.kv_leaf_specs(text_config),
        **text_fields,
        routed_expert_format=expert_format,
    )


def _pipeline_config(
    config: AutoConfig, data_parallel_degree: int, mixed_width: int | None
) -> PipelineConfig:
    """What the pairing reads of a served config."""
    return cast(
        PipelineConfig,
        SimpleNamespace(
            models=SimpleNamespace(
                main_architecture_name="DeepseekV41ForCausalLM"
            ),
            draft_model=None,
            speculative=_speculative(mixed_width),
            needs_bitmask_constraints=False,
            runtime=SimpleNamespace(
                reasoning_parser=None, prefer_module_v3=False
            ),
            model=SimpleNamespace(
                data_parallel_degree=data_parallel_degree,
                pipeline_stage=None,
                huggingface_config=config,
            ),
        ),
    )


_NVFP4 = RoutedExpertFormat.NVFP4
_MXFP4 = RoutedExpertFormat.MXFP4


@pytest.mark.parametrize(
    (
        "num_devices",
        "data_parallel_degree",
        "expert_parallel",
        "mixed_width",
        "expert_format",
    ),
    [
        (1, 1, False, None, _NVFP4),
        (4, 1, False, None, _NVFP4),
        (4, 4, False, None, _NVFP4),
        (4, 4, True, None, _NVFP4),
        (1, 1, False, 2, _NVFP4),
        (4, 4, False, 2, _NVFP4),
        (1, 1, False, None, _MXFP4),
        (4, 1, False, None, _MXFP4),
        (4, 4, True, None, _MXFP4),
    ],
)
def test_unified_dspark_graph_traces(
    num_devices: int,
    data_parallel_degree: int,
    expert_parallel: bool,
    mixed_width: int | None,
    expert_format: RoutedExpertFormat,
) -> None:
    """One spec-decode step traces, its weights named but not loaded.

    With a data-parallel degree of 4 each device verifies and drafts its own
    replica's requests; with ``expert_parallel`` the verification's routed
    experts dispatch their rows to the owning devices, which reads the
    expert-parallel graph inputs. With ``mixed_width`` set, a mixed prefill
    and decode step verifies fewer drafts than a decode step; the step's
    verification width is a graph dimension, so the same graph serves both
    kinds of step and still drafts the full block. ``expert_format`` picks
    the trunk's routed experts: NVFP4 rows dispatch to the fused expert
    kernel, MXFP4 rows dispatch as MXFP8 to the W4A8 GEMMs.
    """
    config = AutoConfig.from_pretrained(str(TESTDATA / "config.json"))
    devices = [DeviceRef.GPU(i) for i in range(num_devices)]
    model_config = _model_config(
        config, devices, data_parallel_degree, expert_parallel, expert_format
    )
    kv_params = model_config.kv_params
    assert isinstance(kv_params, MultiKVCacheParams)
    assert model_config.engram_layer_ids, "the real config has Engram layers"
    n = max(model_config.vocab_size, model_config.engram_vocab_size)
    token_map = (
        np.arange(n, dtype=np.int32) % model_config.engram_compressed_vocab_size
    )
    model_config.engram_input_types = _engram_table_types(
        model_config, devices, token_map, sharded=data_parallel_degree == 1
    )
    target = DeepseekV41(
        model_config, engram_host_tables={}, engram_token_map=token_map
    )
    draft = DeepseekV41DSparkDrafter(model_config, devices[0])
    spec = build_spec_graph(
        _pipeline_config(config, data_parallel_degree, mixed_width),
        MultiKVCacheParams.from_params({"target": kv_params}),
        devices,
        include_signal_buffers=num_devices > 1,
        target=target,
        target_config=model_config,
        draft=draft,
        draft_config=DSparkDeepseekV41DraftConfig(model_config),
        hidden_states=ReturnHiddenStates.SELECTED_LAYERS,
        draft_weights=None,
    )

    assert spec.graph.name == "unified_dspark_deepseekV41"
    # The Engram lookback, the tables, then the merged-rows marker end it.
    assert str(spec.input_types[-1].shape[0]) == "merged_seq_len"
    # (num_accepted, next_tokens, next_draft_tokens), argmax drafts.
    outputs = spec.graph.output_types
    assert len(outputs) == 3
    drafts = outputs[2]
    assert isinstance(drafts, TensorType)
    assert drafts.shape[1] == NUM_SPECULATIVE_TOKENS


def _engram_table_types(
    model_config: DeepseekV41Config,
    devices: list[DeviceRef],
    token_map: np.ndarray,
    *,
    sharded: bool,
) -> list[TensorType]:
    """Per device, each Engram layer's weight and scale table types.

    With ``sharded`` and more than one device, tables are row-sharded per
    rank; otherwise every device reads them whole.
    """
    layout = EngramLayout.from_config(model_config)
    assert layout is not None
    lookup = EngramLookup(layout, token_map)
    tables: list[TensorType] = []
    for rank in range(len(devices)):
        for index, rows in enumerate(model_config.engram_num_embeddings):
            if sharded and len(devices) > 1:
                _, rows = lookup.shard_rows(index, rank, len(devices))
            tables.append(
                TensorType(
                    DType.float8_e4m3fn,
                    (rows, model_config.engram_head_dim),
                    device=DeviceRef.CPU(),
                )
            )
            tables.append(
                TensorType(
                    DType.float8_e8m0fnu,
                    (rows, model_config.engram_head_dim // 32),
                    device=DeviceRef.CPU(),
                )
            )
    return tables
