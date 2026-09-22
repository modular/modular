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

from __future__ import annotations

from unittest.mock import MagicMock, NonCallableMock

from max.driver import DeviceSpec
from max.pipelines.architectures.deepseekV3.memory_planner import (
    DeepseekV3MemoryPlanner,
)
from max.pipelines.architectures.unified_mtp_glm5_2.arch import (
    unified_mtp_glm5_2_arch,
)
from max.pipelines.kv_cache.memory_planner import ModelConfigWithKVCache
from max.pipelines.lib import PipelineConfig, SupportedEncoding

NUM_RANKS = 8


def _mock_pipeline_config(
    quantization_encoding: SupportedEncoding = "float8_e4m3fn",
) -> NonCallableMock:
    pipeline_config = NonCallableMock(spec=PipelineConfig)
    pipeline_config.model = MagicMock()
    pipeline_config.runtime = MagicMock()
    pipeline_config.model.quantization_encoding = quantization_encoding
    pipeline_config.model.kv_cache.kv_cache_format = None
    pipeline_config.model.data_parallel_degree = NUM_RANKS
    pipeline_config.model.device_specs = [
        NonCallableMock(spec=DeviceSpec) for _ in range(NUM_RANKS)
    ]
    pipeline_config.runtime.pipeline_role = "prefill_and_decode"
    pipeline_config.model.max_length = 1024 * 1024
    pipeline_config.runtime.max_batch_total_tokens = None
    pipeline_config.runtime.ep_size = NUM_RANKS
    pipeline_config.runtime.max_batch_input_tokens = 128
    pipeline_config.runtime.device_graph_capture = True
    pipeline_config.runtime.ep_use_allreduce = False
    pipeline_config.runtime.ep_fuse_ffn_combine_send = False
    pipeline_config.speculative = None
    return pipeline_config


def _mock_huggingface_config() -> MagicMock:
    huggingface_config = MagicMock()
    huggingface_config.num_attention_heads = 128
    huggingface_config.qk_nope_head_dim = 128
    huggingface_config.n_routed_experts = 256
    huggingface_config.moe_intermediate_size = 2048
    huggingface_config.hidden_size = 7168
    huggingface_config.num_hidden_layers = 61
    huggingface_config.first_k_dense_replace = 1
    huggingface_config.num_nextn_predict_layers = 1
    huggingface_config.vocab_size = 129280
    huggingface_config.n_shared_experts = 1
    huggingface_config.num_experts_per_tok = 8
    return huggingface_config


def _planner(cls: type[DeepseekV3MemoryPlanner]) -> DeepseekV3MemoryPlanner:
    config = MagicMock(spec=ModelConfigWithKVCache)
    config.get_kv_params.return_value = MagicMock()
    return cls(config)


def test_glm_mtp_plans_through_the_deepseek_planner() -> None:
    """GLM has no planner of its own, so DeepSeek's terms are GLM's terms.

    Pinned because the coupling is invisible from the GLM package: a memory
    term added to the DeepSeek planner lands on GLM's recipes without anyone
    editing GLM.
    """
    assert unified_mtp_glm5_2_arch.memory_planner is DeepseekV3MemoryPlanner


def test_graph_capture_does_not_move_the_glm_estimate() -> None:
    """GLM carries allocator slack in the recipe, not the activation estimate.

    The recipes hold ~3% back through `device_memory_utilization`. A capture
    headroom term in the planner would withhold that same slack a second
    time, out of the KV budget, where the recipe cannot see it.
    """
    huggingface_config = _mock_huggingface_config()
    planner_cls = unified_mtp_glm5_2_arch.memory_planner
    assert planner_cls is not None
    assert issubclass(planner_cls, DeepseekV3MemoryPlanner)
    planner = _planner(planner_cls)

    with_capture = _mock_pipeline_config()
    with_capture.runtime.device_graph_capture = True
    without_capture = _mock_pipeline_config()
    without_capture.runtime.device_graph_capture = False

    assert planner.estimate_activation_memory(
        with_capture, huggingface_config
    ) == planner.estimate_activation_memory(without_capture, huggingface_config)
