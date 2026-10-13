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

import pytest
from max.driver import DeviceSpec
from max.dtype import DType
from max.nn.comm.ep.ep_config import EPConfig
from max.nn.comm.ep.ep_manager import (
    _bound_fused_moe_workspace,
    _ep_sync_counter_bytes,
)
from max.pipelines.architectures.deepseekV3.memory_planner import (
    DeepseekV3MemoryPlanner,
)
from max.pipelines.architectures.deepseekV3_modulev3.memory_planner import (
    DeepseekV3ModuleV3MemoryPlanner,
)
from max.pipelines.architectures.glm5_1.arch import glm5_1_arch
from max.pipelines.architectures.glm5_1.memory_planner import (
    Glm5_1MemoryPlanner,
)
from max.pipelines.architectures.glm5_1_modulev3.arch import (
    glm5_1_modulev3_arch,
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


def test_glm_without_mtp_plans_what_ep_init_adds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the MTP draft GLM keeps the paged planner, plus what EP init
    allocates for the fused MoE: its counters' reserve and, when
    MODULAR_EP_FUSED_MOE=1 asks for it, the backend's bound for its
    workspace. The EP communication buffers and the rest of the counter
    buffers stay unplanned, as on main."""
    assert glm5_1_arch.memory_planner is Glm5_1MemoryPlanner
    config = MagicMock(spec=ModelConfigWithKVCache)
    config.get_kv_params.return_value = MagicMock()
    planner = Glm5_1MemoryPlanner(config)
    huggingface_config = _mock_huggingface_config()
    huggingface_config.hidden_size = 6144
    # NVFP4 GLM-5.3 at EP8 with TP attention: 192 input tokens are 24 per rank.
    pipeline_config = _mock_pipeline_config("float4_e2m1fnx2")
    pipeline_config.model.data_parallel_degree = 1
    pipeline_config.runtime.max_batch_input_tokens = 192
    reserve = NUM_RANKS * (
        _ep_sync_counter_bytes(256, NUM_RANKS, False, 2048)
        - _ep_sync_counter_bytes(256, NUM_RANKS, False, 0)
    )
    assert reserve == NUM_RANKS * 2 * (1 << 20)
    arena, refusal = _bound_fused_moe_workspace(
        EPConfig(
            dispatch_dtype=DType.bfloat16,
            combine_dtype=DType.bfloat16,
            hidden_size=6144,
            top_k=8,
            n_experts=256,
            max_tokens_per_rank=24,
            n_gpus_per_node=NUM_RANKS,
            n_nodes=1,
            moe_dim=2048,
        ),
        DType.uint8,
    )
    assert refusal == "" and arena > 0

    def plan() -> int:
        return planner.estimate_activation_memory(
            pipeline_config, huggingface_config
        )

    monkeypatch.delenv("MODULAR_EP_FUSED_MOE", raising=False)
    assert plan() == reserve
    monkeypatch.setenv("MODULAR_EP_FUSED_MOE", "1")
    assert plan() == reserve + NUM_RANKS * arena
    # The default 8192 input tokens are 1024 per rank, which the backend
    # refuses: no arena.
    pipeline_config.runtime.max_batch_input_tokens = 8192
    assert plan() == reserve
    pipeline_config.runtime.ep_size = 1
    assert plan() == 0


def test_glm_modulev3_plans_through_the_deepseek_planner() -> None:
    """Without DeepSeek's terms the EP heap and MoE activations are unplanned."""
    assert (
        glm5_1_modulev3_arch.memory_planner is DeepseekV3ModuleV3MemoryPlanner
    )


def test_modulev3_reserves_the_ffn_output_even_when_fusion_is_requested() -> (
    None
):
    """ModuleV3 never fuses the combine send, so it must keep the reserve.

    The graph-API planner drops the FFN output term when the fusion is on; the
    ModuleV3 graph still materializes that tensor, so dropping it would be an
    under-reserve.
    """
    huggingface_config = _mock_huggingface_config()
    fused = _mock_pipeline_config(quantization_encoding="float4_e2m1fnx2")
    fused.runtime.ep_fuse_ffn_combine_send = True
    unfused = _mock_pipeline_config(quantization_encoding="float4_e2m1fnx2")
    unfused.runtime.ep_fuse_ffn_combine_send = False
    # Short context keeps the MLA term below the MoE term, which is the one
    # the fusion changes.
    for config in (fused, unfused):
        config.model.max_length = 4096

    v2 = _planner(DeepseekV3MemoryPlanner)
    v3 = _planner(DeepseekV3ModuleV3MemoryPlanner)

    assert v2.estimate_activation_memory(
        fused, huggingface_config
    ) < v2.estimate_activation_memory(unfused, huggingface_config)
    assert v3.estimate_activation_memory(
        fused, huggingface_config
    ) == v2.estimate_activation_memory(unfused, huggingface_config)
