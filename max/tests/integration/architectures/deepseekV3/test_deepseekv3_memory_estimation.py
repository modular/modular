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
from max.nn.comm.ep.ep_config import NUM_GROUPS, EPConfig
from max.nn.comm.ep.ep_manager import _bound_fused_moe_workspace
from max.pipelines.architectures.deepseekV3.memory_planner import (
    DeepseekV3MemoryPlanner,
    _ep_max_rank_send_tokens_for_pipeline,
    ep_fuse_ffn_combine_send_for_pipeline,
)
from max.pipelines.kv_cache.memory_planner import ModelConfigWithKVCache
from max.pipelines.lib import (
    PipelineConfig,
    PipelineRole,
    SupportedEncoding,
)

MAX_SEND_TOKENS_PER_RANK = 128


def _make_planner() -> DeepseekV3MemoryPlanner:
    """Create a DeepseekV3MemoryPlanner with a minimal mock KV config."""
    config = MagicMock(spec=ModelConfigWithKVCache)
    config.get_kv_params.return_value = MagicMock()
    return DeepseekV3MemoryPlanner(config)


NUM_RANKS = 8


def mock_pipeline_config(
    pipeline_role: PipelineRole,
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

    # Pipeline config attributes
    pipeline_config.runtime.pipeline_role = pipeline_role
    pipeline_config.model.max_length = 1024 * 1024  # ~million tokens
    pipeline_config.runtime.max_batch_total_tokens = None
    pipeline_config.runtime.ep_size = NUM_RANKS
    pipeline_config.runtime.max_batch_input_tokens = MAX_SEND_TOKENS_PER_RANK
    pipeline_config.runtime.device_graph_capture = False
    pipeline_config.runtime.ep_use_allreduce = False
    pipeline_config.runtime.ep_fuse_ffn_combine_send = False
    pipeline_config.speculative = None

    return pipeline_config


def mock_huggingface_config() -> MagicMock:
    huggingface_config = MagicMock()

    # HuggingFace config attributes
    huggingface_config.num_attention_heads = 128
    huggingface_config.qk_nope_head_dim = 128
    huggingface_config.n_routed_experts = 256
    huggingface_config.moe_intermediate_size = 2048
    huggingface_config.hidden_size = 7168

    # Additional attributes for estimate_weights_size
    huggingface_config.num_hidden_layers = 61
    huggingface_config.first_k_dense_replace = 1
    huggingface_config.num_nextn_predict_layers = 1
    huggingface_config.vocab_size = 129280
    huggingface_config.n_shared_experts = 1
    huggingface_config.num_experts_per_tok = 8

    return huggingface_config


def test_deepseekv3_memory_estimation() -> None:
    planner = _make_planner()
    pipeline_config = mock_pipeline_config("decode_only")
    huggingface_config = mock_huggingface_config()
    assert huggingface_config is not None

    memory_estimated = planner.estimate_activation_memory(
        pipeline_config, huggingface_config
    )

    max_recv_tokens_per_rank = (
        MAX_SEND_TOKENS_PER_RANK * huggingface_config.n_routed_experts
    )
    moe_min_memory = (
        max_recv_tokens_per_rank * huggingface_config.moe_intermediate_size * 1
    )  # Float8
    moe_min_memory += (
        max_recv_tokens_per_rank * huggingface_config.hidden_size * 2
    )  # BFloat16
    moe_min_memory *= NUM_RANKS

    assert memory_estimated > moe_min_memory


def test_deepseekv3_memory_estimation_exact() -> None:
    planner = _make_planner()
    huggingface_config = mock_huggingface_config()
    assert huggingface_config is not None

    # The fused MoE counters' 1 MiB reserve at the end of every device's two
    # EP sync-counter buffers: the planner's only new term.
    reserve = NUM_RANKS * NUM_GROUPS * (1 << 20)

    # For DecodeOnly, we only need to consider moe_activation_memory
    pipeline_config = mock_pipeline_config("decode_only")
    mem = planner.estimate_activation_memory(
        pipeline_config, huggingface_config
    )
    assert mem == 5225054208 + reserve

    # For PrefillAndDecode, we also need to consider mla_activation_memory
    pipeline_config = mock_pipeline_config("prefill_and_decode")
    mem = planner.estimate_activation_memory(
        pipeline_config, huggingface_config
    )
    assert mem == 551759642624 + reserve

    # Also check model with different quantization encoding
    pipeline_config = mock_pipeline_config("decode_only", "float4_e2m1fnx2")
    mem = planner.estimate_activation_memory(
        pipeline_config, huggingface_config
    )
    assert mem == 4399759360 + reserve


def _fused_planning_case(
    max_batch_input_tokens: int,
    use_allreduce: bool = False,
    encoding: SupportedEncoding = "float4_e2m1fnx2",
    data_parallel_degree: int = NUM_RANKS,
) -> tuple[NonCallableMock, MagicMock]:
    """GLM-5.3's MoE at EP8 (hidden 6144, MoE width 2048, 256 routed experts,
    top-k 8). Data-parallel attention (the default here) puts
    ``max_batch_input_tokens`` on each rank; TP attention
    (``data_parallel_degree=1``) splits them over the 8 ranks."""
    pipeline_config = mock_pipeline_config("decode_only", encoding)
    pipeline_config.runtime.max_batch_input_tokens = max_batch_input_tokens
    pipeline_config.runtime.ep_use_allreduce = use_allreduce
    pipeline_config.model.data_parallel_degree = data_parallel_degree
    huggingface_config = mock_huggingface_config()
    huggingface_config.hidden_size = 6144
    return pipeline_config, huggingface_config


def _fused_plan_increase(
    monkeypatch: pytest.MonkeyPatch,
    pipeline_config: NonCallableMock,
    huggingface_config: MagicMock,
) -> int:
    """What MODULAR_EP_FUSED_MOE=1 adds to the activation memory plan."""
    monkeypatch.delenv("MODULAR_EP_FUSED_MOE", raising=False)
    off = _make_planner().estimate_activation_memory(
        pipeline_config, huggingface_config
    )
    monkeypatch.setenv("MODULAR_EP_FUSED_MOE", "1")
    on = _make_planner().estimate_activation_memory(
        pipeline_config, huggingface_config
    )
    return on - off


def _glm_moe_bound(
    tokens_per_rank: int,
    dispatch_dtype: DType = DType.uint8,
    use_allreduce: bool = False,
) -> tuple[int, str]:
    """The backend's bound for GLM-5.3's MoE at EP8 with this capacity."""
    return _bound_fused_moe_workspace(
        EPConfig(
            dispatch_dtype=DType.bfloat16,
            combine_dtype=DType.bfloat16,
            hidden_size=6144,
            top_k=8,
            n_experts=256,
            max_tokens_per_rank=tokens_per_rank,
            n_gpus_per_node=NUM_RANKS,
            n_nodes=1,
            moe_dim=2048,
            use_allreduce=use_allreduce,
        ),
        dispatch_dtype,
    )


@pytest.mark.parametrize(
    "max_batch_input_tokens, use_allreduce, planned",
    [(24, False, True), (8192, False, False), (24, True, False)],
)
def test_deepseekv3_plans_the_fused_workspace_the_backend_serves(
    monkeypatch: pytest.MonkeyPatch,
    max_batch_input_tokens: int,
    use_allreduce: bool,
    planned: bool,
) -> None:
    """With MODULAR_EP_FUSED_MOE=1 the plan adds exactly the arena the
    backend reports for the EP configuration, and nothing when the backend
    refuses it (here a capacity above 32 tokens per rank, or the allreduce
    backend), so a refused configuration costs the KV cache nothing."""
    pipeline_config, huggingface_config = _fused_planning_case(
        max_batch_input_tokens, use_allreduce
    )
    arena, refusal = _glm_moe_bound(
        max_batch_input_tokens, use_allreduce=use_allreduce
    )
    assert (refusal == "") == planned, refusal
    assert _fused_plan_increase(
        monkeypatch, pipeline_config, huggingface_config
    ) == (NUM_RANKS * arena if planned else 0)
    if planned:
        assert arena > 0


@pytest.mark.parametrize(
    "max_batch_input_tokens, tokens_per_rank, planned",
    [(192, 24, True), (8192, 1024, False)],
)
def test_deepseekv3_fused_plan_uses_the_tp_attention_capacity(
    monkeypatch: pytest.MonkeyPatch,
    max_batch_input_tokens: int,
    tokens_per_rank: int,
    planned: bool,
) -> None:
    """With TP attention at EP8 each rank holds ceildiv(max_batch_input_tokens,
    8): 192 input tokens give 24 per rank, which the backend serves; the
    default 8192 give 1024, which it refuses, so a worker at the default
    keeps the shipping chain and plans no arena."""
    pipeline_config, huggingface_config = _fused_planning_case(
        max_batch_input_tokens, data_parallel_degree=1
    )
    assert (
        _ep_max_rank_send_tokens_for_pipeline(pipeline_config)
        == tokens_per_rank
    )
    arena, refusal = _glm_moe_bound(tokens_per_rank)
    assert (refusal == "") == planned, refusal
    assert _fused_plan_increase(
        monkeypatch, pipeline_config, huggingface_config
    ) == (NUM_RANKS * arena if planned else 0)


def test_deepseekv3_fused_plan_bounds_a_float8_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A float8 encoding is MXFP8 (which the backend serves) or 128 x 128
    block FP8 (which it refuses), and only the parsed checkpoint tells them
    apart. The plan takes the backend's bound for the FP8 element type, so it
    can be high but never short."""
    pipeline_config, huggingface_config = _fused_planning_case(
        24, encoding="float8_e4m3fn"
    )
    fp8_bound, refusal = _glm_moe_bound(24, DType.float8_e4m3fn)
    assert refusal == "" and fp8_bound > _glm_moe_bound(24)[0]
    assert (
        _fused_plan_increase(monkeypatch, pipeline_config, huggingface_config)
        == NUM_RANKS * fp8_bound
    )


def test_deepseekv3_fused_plan_bounds_the_target_under_mtp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An MTP draft can make EP init size the shared buffers from a bfloat16
    copy of the config, which the backend refuses. Whether it does depends on
    the checkpoint (quantized MTP experts keep the target's config, which the
    backend can serve), so the plan bounds the target's encoding."""
    pipeline_config, huggingface_config = _fused_planning_case(
        192, data_parallel_degree=1
    )
    pipeline_config.speculative = MagicMock()
    pipeline_config.speculative.is_mtp.return_value = True
    pipeline_config.draft_model = None
    nvfp4_bound, refusal = _glm_moe_bound(24)
    assert refusal == "" and nvfp4_bound > 0
    assert (
        _fused_plan_increase(monkeypatch, pipeline_config, huggingface_config)
        == NUM_RANKS * nvfp4_bound
    )


def test_deepseekv3_memory_estimation_ignores_graph_capture() -> None:
    """Capture allocates from the memory manager, so it needs no reservation.

    The planner used to withhold 8 GiB per device here. It reserved nothing
    for capture -- the figure was subtracted from the KV budget, leaving
    allocator slack under a graph-capture name -- and a paired A/B measured
    no throughput difference with it on or off.
    """
    planner = _make_planner()
    huggingface_config = mock_huggingface_config()

    pipeline_config = mock_pipeline_config("decode_only")
    baseline = planner.estimate_activation_memory(
        pipeline_config, huggingface_config
    )

    pipeline_config.runtime.device_graph_capture = True
    with_capture = planner.estimate_activation_memory(
        pipeline_config, huggingface_config
    )

    assert with_capture == baseline


def mock_weights_pipeline_config(
    n_gpus: int, ep_size: int, dp_degree: int
) -> NonCallableMock:
    """Create a mock pipeline config for estimate_weights_size tests."""
    huggingface_config = mock_huggingface_config()
    assert huggingface_config is not None

    pipeline_config = NonCallableMock(spec=PipelineConfig)
    pipeline_config.model = MagicMock()
    pipeline_config.runtime = MagicMock()
    pipeline_config.model.quantization_encoding = "float8_e4m3fn"
    pipeline_config.model.data_parallel_degree = dp_degree
    pipeline_config.model.device_specs = [
        NonCallableMock(spec=DeviceSpec) for _ in range(n_gpus)
    ]
    pipeline_config.model.huggingface_config = huggingface_config
    # Use a large enough weights size to account for the algorithm's subtractions.
    # DeepSeek-V3 has ~671B parameters, ~700GB at FP8.
    pipeline_config.model.weights_size.return_value = 700 * 1024**3
    pipeline_config.runtime.ep_size = ep_size

    return pipeline_config


def compute_routing_experts_size() -> int:
    """Compute the routing_experts_size from mock_huggingface_config values.

    This matches the calculation in estimate_weights_size:
    routing_experts_size = n_sparse_layers * n_routed_experts * expert_size
    where expert_size = moe_intermediate_size * hidden_size * 3 * dtype
    """
    hf_config = mock_huggingface_config()
    dtype = 1  # float8_e4m3fn size in bytes
    n_sparse_layers = (
        hf_config.num_hidden_layers - hf_config.first_k_dense_replace
    )
    expert_size = (
        hf_config.moe_intermediate_size * hf_config.hidden_size * 3 * dtype
    )
    return n_sparse_layers * hf_config.n_routed_experts * expert_size


def test_deepseekv3_estimate_weights_size_no_expert_parallelism() -> None:
    """Test estimate_weights_size with ep_size=1 and multiple devices.

    This is a regression test for a bug where ep_size=1 with multiple GPUs
    would cause a ZeroDivisionError (n_nodes = 1 // 8 = 0).
    """
    planner = _make_planner()
    # EP=1 (no expert parallelism), 8 GPUs, DP=1
    pipeline_config = mock_weights_pipeline_config(
        n_gpus=8, ep_size=1, dp_degree=1
    )

    # This should not raise ZeroDivisionError
    mem = planner.estimate_weights_size(pipeline_config)
    assert mem > 0


def test_deepseekv3_estimate_weights_size_dp_ep_exact() -> None:
    planner = _make_planner()
    # EP=8, 8 GPUs, DP=8
    pipeline_config = mock_weights_pipeline_config(
        n_gpus=8, ep_size=8, dp_degree=8
    )

    # The result is quite large because the mock weights size is larger
    # than the actual weights size.
    mem = planner.estimate_weights_size(pipeline_config)
    assert mem == 1124551261664


def test_deepseekv3_estimate_weights_size_tp_ep_exact() -> None:
    planner = _make_planner()
    # EP=8, 8 GPUs, TP attention (DP=1)
    pipeline_config = mock_weights_pipeline_config(
        n_gpus=8, ep_size=8, dp_degree=1
    )

    mem = planner.estimate_weights_size(pipeline_config)
    assert mem == 754209345468


def test_deepseekv3_estimate_weights_size_routing_experts_scaling() -> None:
    """Verify routing experts memory scales correctly with EP configurations.

    Currently, ep_size must be either 1 (no EP) or a multiple of n_gpus_per_node
    (EP across full nodes). Mixed EP/TP strategies are not yet supported.

    For supported configurations:
    - EP=1: routing_experts_memory = routing_experts_size (full copy)
    - EP=n_gpus*n_nodes: routing_experts_memory = routing_experts_size / n_nodes
    """
    planner = _make_planner()
    routing_experts_size = compute_routing_experts_size()
    n_gpus = 8

    # Convert to int since the memory estimation involves some float arithmetic.
    # EP=1: no expert parallelism
    mem_ep1 = int(
        planner.estimate_weights_size(
            mock_weights_pipeline_config(n_gpus=n_gpus, ep_size=1, dp_degree=1)
        )
    )
    # EP=8: single node with full EP (n_nodes=1)
    mem_ep8 = int(
        planner.estimate_weights_size(
            mock_weights_pipeline_config(n_gpus=n_gpus, ep_size=8, dp_degree=1)
        )
    )
    # EP=16: two nodes (n_nodes=2)
    mem_ep16 = int(
        planner.estimate_weights_size(
            mock_weights_pipeline_config(n_gpus=n_gpus, ep_size=16, dp_degree=1)
        )
    )

    # Verify the routing experts contribution:
    # EP=1: full routing_experts_size (no split)
    # EP=8 (n_nodes=1): routing_experts_size / 1 = routing_experts_size
    # EP=16 (n_nodes=2): routing_experts_size / 2

    # EP=1 vs EP=8: EP=1 has full routing_experts_size, EP=8 has routing_experts_size
    assert mem_ep1 == mem_ep8  # Both have full routing_experts_size (n_nodes=1)

    # EP=8 vs EP=16: EP=16 splits across 2 nodes, so routing_experts_size / 2
    assert mem_ep8 - mem_ep16 == routing_experts_size // 2


def test_deepseekv3_memory_estimation_drops_c_when_ffn_sends() -> None:
    """The fused combine send removes the down-projection output term exactly.

    When the FFN scatters its output to the peers from its own epilogue, that
    `(max_recv_tokens_per_rank, hidden_size)` bf16 tensor is never
    materialized. The estimate has to drop by exactly its size and nothing
    else: reserving for a tensor that does not exist bills the KV cache for
    it, and under-dropping leaves the same bug in smaller form.
    """
    huggingface_config = mock_huggingface_config()

    # NVFP4, because that is the only encoding the send serves; the flag alone
    # must not move the estimate anywhere else.
    unfused = _make_planner().estimate_activation_memory(
        mock_pipeline_config("decode_only", "float4_e2m1fnx2"),
        huggingface_config,
    )

    fused_config = mock_pipeline_config("decode_only", "float4_e2m1fnx2")
    fused_config.runtime.ep_fuse_ffn_combine_send = True
    fused = _make_planner().estimate_activation_memory(
        fused_config, huggingface_config
    )

    max_recv_tokens_per_rank = MAX_SEND_TOKENS_PER_RANK * min(
        huggingface_config.n_routed_experts,
        NUM_RANKS * huggingface_config.num_experts_per_tok,
    )
    # The planner scales the per-device MoE term by the device count.
    expected_drop = (
        max_recv_tokens_per_rank
        * huggingface_config.hidden_size
        * 2  # bfloat16
        * NUM_RANKS
    )

    assert unfused - fused == expected_drop


@pytest.mark.parametrize(
    ("pipeline_role", "encoding", "explicit", "expected"),
    [
        # Unset: on for anything that serves prefill, off for decode-only.
        # One graph serves both phases under the default role, so it gets the
        # send for decode batches too -- deliberate, not an oversight. The
        # first row is the shipping NVFP4 MoE recipes: they set neither the
        # role nor the flag, so the whole gate rides on these two defaults.
        ("prefill_and_decode", "float4_e2m1fnx2", None, True),
        ("prefill_only", "float4_e2m1fnx2", None, True),
        ("decode_only", "float4_e2m1fnx2", None, False),
        # An explicit setting wins over the role, but NOT over a config the
        # send cannot serve -- dropping the buffer there would under-reserve.
        ("decode_only", "float4_e2m1fnx2", True, True),
        ("prefill_only", "float4_e2m1fnx2", False, False),
        ("prefill_only", "float8_e4m3fn", True, False),
    ],
)
def test_ep_fuse_ffn_combine_send_follows_role_and_capability(
    pipeline_role: PipelineRole,
    encoding: SupportedEncoding,
    explicit: bool | None,
    expected: bool,
) -> None:
    """On for prefill, and only where the graph can actually elide the tensor.

    The planner drops a buffer on this answer, so it must never outrun what
    the graph will do: an over-reserve wastes memory, an under-reserve OOMs.
    """
    pipeline_config = mock_pipeline_config(pipeline_role, encoding)
    pipeline_config.runtime.ep_fuse_ffn_combine_send = explicit
    assert (
        ep_fuse_ffn_combine_send_for_pipeline(
            pipeline_config, mock_huggingface_config()
        )
        is expected
    )


@pytest.mark.parametrize(
    ("attr", "value"),
    [
        # Allreduce routes within the device; there are no peer buffers.
        ("ep_use_allreduce", True),
    ],
)
def test_ep_fuse_ffn_combine_send_off_without_peer_buffers(
    attr: str, value: bool
) -> None:
    """A runtime setting the send cannot serve turns it off, not on."""
    pipeline_config = mock_pipeline_config("prefill_only", "float4_e2m1fnx2")
    pipeline_config.runtime.ep_fuse_ffn_combine_send = True
    setattr(pipeline_config.runtime, attr, value)
    assert not ep_fuse_ffn_combine_send_for_pipeline(
        pipeline_config, mock_huggingface_config()
    )


def test_ep_fuse_ffn_combine_send_needs_an_unfused_shared_expert() -> None:
    """The send leaves only the WAIT half of the combine to run.

    The EP forward takes that split path only when a shared expert gives it
    something to overlap, so a model without one keeps the staging buffer.
    """
    pipeline_config = mock_pipeline_config("prefill_only", "float4_e2m1fnx2")
    pipeline_config.runtime.ep_fuse_ffn_combine_send = True
    huggingface_config = mock_huggingface_config()
    huggingface_config.n_shared_experts = 0
    assert not ep_fuse_ffn_combine_send_for_pipeline(
        pipeline_config, huggingface_config
    )
