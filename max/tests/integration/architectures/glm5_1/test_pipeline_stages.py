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
"""Where GLM-5.3 cuts into pipeline stages, and what each stage caches."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams
from max.pipelines.architectures.deepseekV3_2_modulev3.model_config import (
    indexer_cache_layer_map,
    resolve_indexer_types,
    stage_indexer_types,
)
from max.pipelines.architectures.glm5_1_modulev3.glm5_1 import Glm5_1Decoder
from max.pipelines.architectures.glm5_1_modulev3.model_config import (
    Glm5_1Config,
)
from max.pipelines.lib import KVCacheConfig
from max.pipelines.lib.graph_specs import GraphRole
from max.pipelines.lib.module_registration import supports_pipeline_stages
from max.pipelines.lib.pipeline_parallel import (
    PipelinedModel,
    pipeline_stages,
    split_layers,
)
from max.pipelines.lib.pipeline_stage import PipelineStage
from max.pipelines.lib.registration_graph_api import registered_pipeline_model
from max.pipelines.modeling.types import PipelineTask


def _glm_5_3() -> Any:
    """GLM-5.3's config: 78 layers, full at 0-2, then every fourth from 6."""
    return SimpleNamespace(
        num_hidden_layers=78,
        num_attention_heads=64,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        index_topk=2048,
        index_head_dim=128,
        indexer_types=None,
        index_topk_pattern=None,
        index_topk_freq=4,
        index_skip_topk_offset=3,
    )


def _cached_layers(stage: PipelineStage | None) -> tuple[int, int]:
    """Returns the MLA and indexer layer counts a stage's caches hold."""
    pipeline_config: Any = SimpleNamespace(
        model=SimpleNamespace(
            data_parallel_degree=1,
            decode_context_parallel_degree=1,
            pipeline_stage=stage,
        ),
        speculative=None,
    )
    params = Glm5_1Config.construct_kv_params(
        huggingface_config=_glm_5_3(),
        pipeline_config=pipeline_config,
        devices=[DeviceRef.CPU()],
        kv_cache_config=KVCacheConfig(),
        cache_dtype=DType.bfloat16,
    )
    assert isinstance(params, MultiKVCacheParams)
    mla, indexer = params.children["mla"], params.children["indexer"]
    assert isinstance(mla, KVCacheParams)
    assert isinstance(indexer, KVCacheParams)
    return mla.num_layers, indexer.num_layers


def test_every_pp8_stage_starts_on_a_full_layer() -> None:
    spec = Glm5_1Config.construct_partition_spec(_glm_5_3())
    schedule = resolve_indexer_types(_glm_5_3(), 78)
    stages = pipeline_stages(split_layers(spec, 8))
    assert stages[-1].layers.stop == 78
    assert all(schedule[stage.layers.start] == "full" for stage in stages)


def test_a_stage_may_not_start_on_a_shared_layer() -> None:
    spec = Glm5_1Config.construct_partition_spec(_glm_5_3())
    with pytest.raises(ValueError, match="forbids a cut"):
        split_layers(spec, 2, [3, 75])


def test_every_stage_keeps_the_latent_to_indexer_proportion() -> None:
    """Stages share one KV pool, so each holds four latent slots per full
    layer: layers 10-21 hold three full layers, 10, 14 and 18, and the first
    stage's four full layers (0, 1, 2 and 6) get six spare slots."""
    assert _cached_layers(PipelineStage(1, 8, range(10, 22))) == (12, 3)
    assert _cached_layers(PipelineStage(0, 8, range(10))) == (16, 4)
    assert _cached_layers(PipelineStage(0, 1, range(78))) == (78, 21)
    assert _cached_layers(None) == (78, 21)


def test_a_stage_numbers_its_indexer_slots_from_its_first_layer() -> None:
    schedule = resolve_indexer_types(_glm_5_3(), 78)
    stage = stage_indexer_types(schedule, range(10, 20))
    assert indexer_cache_layer_map(stage, 10) == [0, 0, 0, 0, 1, 0, 0, 0, 2, 0]
    assert stage_indexer_types(schedule, range(78)) == schedule


def test_the_registered_glm_decoder_builds_stages() -> None:
    assert supports_pipeline_stages(Glm5_1Decoder)
    model = registered_pipeline_model(
        PipelineTask.TEXT_GENERATION,
        Glm5_1Config,
        ((GraphRole.DECODER, Glm5_1Decoder),),
        None,
    )
    assert issubclass(model, PipelinedModel)
