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
"""Traces the unified DSpark spec-decode graph at the real V4.1 config.

Tracing only, but the test needs a GPU because the MXFP8 scale layout is
chosen from the host's accelerator rather than graph metadata.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.nn.comm.ep import EPConfig
from max.nn.kv_cache import MultiKVCacheParams
from max.nn.layer import Module
from max.pipelines.architectures import hf_config_shims  # noqa: F401
from max.pipelines.architectures.deepseekV41 import kv_leaves
from max.pipelines.architectures.deepseekV41.dspark_drafter import (
    DeepseekV41WithDSpark,
)
from max.pipelines.architectures.deepseekV41.layers.engram_hash import (
    EngramLayout,
)
from max.pipelines.architectures.deepseekV41.layers.engram_lookup import (
    EngramLookup,
)
from max.pipelines.architectures.deepseekV41.model_config import (
    NVFP4_ACTIVATION_QUANT_CONFIG,
    DeepseekV41Config,
)
from max.pipelines.architectures.unified_dspark_deepseekV41.model import (
    driver_outputs,
)
from max.pipelines.architectures.unified_dspark_deepseekV41.unified_dspark_deepseekV41 import (
    UnifiedDSparkDeepseekV41,
)
from max.pipelines.lib import KVCacheConfig
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
            dispatch_dtype=DType.uint8,
            dispatch_quant_config=NVFP4_ACTIVATION_QUANT_CONFIG,
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
        **text_fields,
        **DeepseekV41Config.resolve_quantization_fields(config),
    )


def _fix_up_weight_names(module: Module) -> None:
    """Sets each ``Weight.name`` to its dotted state-dict key."""
    for name, weight in module.raw_state_dict().items():
        weight.name = name


@pytest.mark.parametrize(
    ("num_devices", "data_parallel_degree", "expert_parallel", "mixed_width"),
    [
        (1, 1, False, None),
        (4, 1, False, None),
        (4, 4, False, None),
        (4, 4, True, None),
        (1, 1, False, 2),
        (4, 4, False, 2),
    ],
)
def test_unified_dspark_graph_traces(
    num_devices: int,
    data_parallel_degree: int,
    expert_parallel: bool,
    mixed_width: int | None,
) -> None:
    """One spec-decode step traces; a weightless driver declares its inputs.

    With a data-parallel degree of 4 each device verifies and drafts its own
    replica's requests; with ``expert_parallel`` the verification's routed
    experts dispatch their rows to the owning devices, which reads the
    expert-parallel graph inputs. With ``mixed_width`` set, a mixed prefill
    and decode step verifies fewer drafts than a decode step; the step's
    verification width is a graph dimension, so the same graph serves both
    kinds of step and still drafts the full block.
    """
    config = AutoConfig.from_pretrained(str(TESTDATA / "config.json"))
    devices = [DeviceRef.GPU(i) for i in range(num_devices)]
    model_config = _model_config(
        config, devices, data_parallel_degree, expert_parallel
    )
    kv_params = model_config.kv_params
    assert isinstance(kv_params, MultiKVCacheParams)
    specs = kv_leaves.kv_leaf_specs(DeepseekV41Config.get_text_config(config))
    layer_ids = model_config.engram_layer_ids
    assert layer_ids, "the real config has Engram layers"

    lookback = TensorType(
        DType.int64,
        ["engram_lookback_batch", model_config.engram_max_ngram_size - 1],
        device=devices[0],
    )
    n = max(model_config.vocab_size, model_config.engram_vocab_size)
    token_map = (
        np.arange(n, dtype=np.int32) % model_config.engram_compressed_vocab_size
    )
    tables = _engram_table_types(
        model_config, devices, token_map, sharded=data_parallel_degree == 1
    )
    trailing_types = [lookback, *tables]
    signature = UnifiedDSparkDeepseekV41(
        model_config,
        _speculative(mixed_width),
        specs,
        model=None,
        trailing_input_types=trailing_types,
    )
    input_types = signature.input_types(kv_params)

    with Graph(
        "unified_dspark_deepseekV41_trace", input_types=input_types
    ) as graph:
        trailing = signature.decode_inputs(graph.inputs, kv_params).trailing
        # The model's own inputs follow the standard ones exactly once.
        assert len(trailing) == len(trailing_types)
        values = trailing[1:]
        per_device: list[dict[int, tuple[TensorValue, TensorValue]]] = []
        idx = 0
        for _ in devices:
            per_dev: dict[int, tuple[TensorValue, TensorValue]] = {}
            for layer_id in layer_ids:
                per_dev[layer_id] = (values[idx].tensor, values[idx + 1].tensor)
                idx += 2
            per_device.append(per_dev)
        nn_model = DeepseekV41WithDSpark(
            model_config,
            engram_host_tables=per_device[0],
            engram_token_map=token_map,
        )
        _fix_up_weight_names(nn_model)
        driver = UnifiedDSparkDeepseekV41(
            model_config,
            _speculative(mixed_width),
            specs,
            model=nn_model,
            trailing_input_types=trailing_types,
        )
        assert tuple(driver.input_types(kv_params)) == tuple(input_types)
        if num_devices > 1:
            driver.shard(devices, per_device)
        outputs = driver_outputs(
            driver,
            driver.decode_inputs(graph.inputs, kv_params),
            specs,
            has_engram=True,
        )
        graph.output(*outputs)

    # (num_accepted, next_tokens, next_draft_tokens), argmax drafts.
    assert len(outputs) == 3
    assert outputs[2].shape[1] == NUM_SPECULATIVE_TOKENS


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
