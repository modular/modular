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

import dataclasses
import re
from typing import Any

import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, Type
from max.nn.comm.ep import calculate_ep_max_tokens_per_rank, ep_kernels
from max.nn.comm.ep.ep_config import (
    _EP_LOCAL_SYNC_RESERVED_WORDS,
    NUM_GROUPS,
    EPConfig,
    _ep_fused_moe_requested,
    estimate_ep_memory_usage,
)
from max.nn.comm.ep.ep_kernels import (
    _call_mega_ffn_ep_fused,
    _call_mega_ffn_ep_fused_init,
    _call_mega_ffn_ep_fused_plan,
    _ep_dispatch_format_parameters,
    _ep_dispatch_output_types,
    _fused_moe_bound_parameters,
    _fused_moe_parameters,
    _validate_ffn_combine_send_config,
    call_ep_dispatch_async,
    call_ep_dispatch_wait,
)
from max.nn.comm.ep.ep_manager import (
    _ep_counter_words,
    _ep_sync_counter_bytes,
    _fused_moe_answer,
    _plan_fused_moe_workspace,
    get_ep_local_sync_counters_size,
)
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)


@pytest.mark.parametrize(
    "max_batch_input_tokens, ep_size, data_parallel_degree, expected",
    [
        # Evenly divisible: ceil == floor.
        (4096, 8, 1, 512),
        (1024, 4, 1, 256),
        # Non-divisible: must ceil to match ops.reducescatter.sum's
        # ceiling-biased ragged binning, which puts ceil(S/P) tokens on the
        # first (S % P) ranks. Floor would under-size the EP per-rank cap
        # and trip the dispatch assertion in ep.mojo for the over-sized
        # shards.
        (4196, 8, 1, 525),
        (4097, 8, 1, 513),
        (10, 3, 1, 4),
        # DP_EP: tp_size == 1, every rank holds the full batch.
        (4196, 8, 8, 4196),
    ],
)
def test_calculate_ep_max_tokens_per_rank_ceil(
    max_batch_input_tokens: int,
    ep_size: int,
    data_parallel_degree: int,
    expected: int,
) -> None:
    assert (
        calculate_ep_max_tokens_per_rank(
            max_batch_input_tokens=max_batch_input_tokens,
            ep_size=ep_size,
            data_parallel_degree=data_parallel_degree,
        )
        == expected
    )


def test_calculate_ep_max_tokens_per_rank_allreduce_bypasses_tp() -> None:
    # use_allreduce keeps the full batch on every rank, regardless of tp_size.
    assert (
        calculate_ep_max_tokens_per_rank(
            max_batch_input_tokens=4196,
            ep_size=8,
            data_parallel_degree=1,
            use_allreduce=True,
        )
        == 4196
    )


def _ep_memory_usage(
    dispatch_dtype: DType, dispatch_element_dtype: DType | None
) -> int:
    return estimate_ep_memory_usage(
        hidden_size=32,
        dispatch_dtype=dispatch_dtype,
        combine_dtype=DType.bfloat16,
        max_tokens_per_rank=4,
        n_experts=2,
        n_nodes=1,
        n_gpus_per_node=2,
        top_k=1,
        dispatch_element_dtype=dispatch_element_dtype,
    )


def test_estimate_ep_memory_usage_mxfp6_uses_three_quarter_byte_packing() -> (
    None
):
    """MXFP6 packs four codes per three bytes, plus one E8M0 byte per 32-block.

    ``uint8`` alone cannot tell MXFP4 from MXFP6, so a config that forgets
    ``dispatch_element_dtype`` silently falls through to the FP4/NVFP4
    packing ratio instead of failing loudly -- this pins the two to known,
    distinct byte counts for the same logical shape.
    """
    assert (
        _ep_memory_usage(DType.uint8, DType.float6_e2m3fn) == 556
    )  # d_token_size = 32*3//4 + 32//32 = 25 bytes.
    assert (
        _ep_memory_usage(DType.uint8, DType.float6_e3m2fn) == 556
    )  # Both FP6 encodings pack identically; only the codes differ.


def test_estimate_ep_memory_usage_distinguishes_fp6_from_fp4_packing() -> None:
    assert _ep_memory_usage(DType.uint8, None) == 472  # NVFP4/MXFP4 ratio.
    assert _ep_memory_usage(DType.bfloat16, None) == 1024  # Unpacked.


def _base_ep_config() -> EPConfig:
    """An EP config with every field the send's guard reads left at default.

    bfloat16 keeps the fixture valid without a dispatch quant config; the
    guard under test reads the communication topology, not the dtype.
    """
    return EPConfig(
        dispatch_dtype=DType.bfloat16,
        combine_dtype=DType.bfloat16,
        hidden_size=6144,
        top_k=8,
        n_experts=256,
        max_tokens_per_rank=1024,
        n_gpus_per_node=8,
        n_nodes=1,
    )


def _ffn_send_config(
    *, use_allreduce: bool = False, fused_shared_expert: bool = False
) -> EPConfig:
    """:func:`_base_ep_config` with the send opted in.

    The two keywords are exactly the settings the send's guard rejects, so a
    test names the one it is exercising and nothing else moves.
    """
    return dataclasses.replace(
        _base_ep_config(),
        fuse_ffn_combine_send=True,
        use_allreduce=use_allreduce,
        fused_shared_expert=fused_shared_expert,
    )


def test_ffn_combine_send_off_by_default() -> None:
    """The fused send must be opt-in: it changes what the planner reserves."""
    assert not _base_ep_config().fuse_ffn_combine_send
    assert _ffn_send_config().fuse_ffn_combine_send


def test_ffn_combine_send_accepts_the_supported_config() -> None:
    """The happy path must not raise, or the negative cases prove nothing."""
    _validate_ffn_combine_send_config(_ffn_send_config())


def test_ffn_combine_send_rejects_allreduce() -> None:
    """Allreduce routes within the device, so there are no peer buffers."""
    with pytest.raises(ValueError, match="allreduce"):
        _validate_ffn_combine_send_config(_ffn_send_config(use_allreduce=True))


def test_ffn_combine_send_rejects_fused_shared_expert() -> None:
    """A fused shared expert keeps its rows in the tensor being elided."""
    with pytest.raises(ValueError, match="fused_shared_expert"):
        _validate_ffn_combine_send_config(
            _ffn_send_config(fused_shared_expert=True)
        )


def _nvfp4_quant_config() -> QuantConfig:
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.STATIC,
            dtype=DType.float32,
            block_size=(1, 16),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float8_e4m3fn,
            block_size=(1, 16),
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        embedding_output_dtype=None,
        format=QuantFormat.NVFP4,
    )


def _mxfp8_quant_config() -> QuantConfig:
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=(1, 32),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float8_e8m0fnu,
            block_size=(1, 32),
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        embedding_output_dtype=None,
        format=QuantFormat.MXFP8,
    )


def _block_fp8_quant_config() -> QuantConfig:
    """128 x 128 block-scaled FP8, the FP8 checkpoint format of DeepSeek-V3
    and GLM."""
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=(1, 128),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float32,
            block_size=(128, 128),
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        embedding_output_dtype=None,
        format=QuantFormat.BLOCKSCALED_FP8,
    )


def _nvfp4_ep_config(*, nvfp4_dyn_global_scales: bool) -> EPConfig:
    return dataclasses.replace(
        _base_ep_config(),
        dispatch_dtype=DType.uint8,
        dispatch_quant_config=_nvfp4_quant_config(),
        nvfp4_dyn_global_scales=nvfp4_dyn_global_scales,
    )


def test_nvfp4_dyn_global_scales_requires_nvfp4_dispatch() -> None:
    """The per-token global scale exists only in the NVFP4 wire format."""
    with pytest.raises(ValueError, match="nvfp4_dyn_global_scales"):
        dataclasses.replace(_base_ep_config(), nvfp4_dyn_global_scales=True)


def test_nvfp4_dyn_global_scales_adds_rowwise_scales_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rowwise scales sit right after the block scales, one BF16 per
    received row, and `src_info` stays last for the batch manager."""
    # The NVIDIA scale layout is chosen by accelerator; pin it so the test
    # runs on hosts without a GPU.
    monkeypatch.setattr(ep_kernels, "accelerator_api", lambda: "cuda")
    device = DeviceRef.GPU()

    static_types = _ep_dispatch_output_types(
        _nvfp4_ep_config(nvfp4_dyn_global_scales=False), device
    )
    dyn_config = _nvfp4_ep_config(nvfp4_dyn_global_scales=True)
    dyn_types = _ep_dispatch_output_types(dyn_config, device)

    assert len(dyn_types) == len(static_types) + 1
    rowwise_scales = dyn_types[2]
    assert rowwise_scales.dtype == DType.bfloat16
    assert rowwise_scales.shape == [dyn_config.get_max_recv_tokens()]
    assert dyn_types[:2] + dyn_types[3:] == static_types


def test_nvfp4_dyn_global_scales_rejects_non_fused_dispatch() -> None:
    """Only the fused dispatch kernels implement the per-token global scale."""
    config = _nvfp4_ep_config(nvfp4_dyn_global_scales=True)
    device = DeviceRef.GPU()
    input_types: list[Type[Any]] = [
        BufferType(DType.int32, [16], device),
        TensorType(DType.bfloat16, [4, config.hidden_size], device),
        TensorType(DType.int32, [4, config.top_k], device),
        TensorType(DType.uint64, [config.n_gpus_per_node], DeviceRef.CPU()),
    ]
    with Graph("ep_non_fused_dispatch", input_types=input_types) as graph:
        counters = graph.inputs[0].buffer
        tokens = graph.inputs[1].tensor
        topk_ids = graph.inputs[2].tensor
        ptrs = graph.inputs[3].tensor
        with pytest.raises(ValueError, match="call_ep_dispatch_async"):
            call_ep_dispatch_async(
                tokens, topk_ids, counters, ptrs, ptrs, ptrs, config
            )
        with pytest.raises(ValueError, match="call_ep_dispatch_wait"):
            call_ep_dispatch_wait(counters, ptrs, ptrs, config)


def test_only_a_fused_capable_config_reserves_counter_words() -> None:
    """A config that gives the experts' FFN width can host the fused MoE, so
    its counter buffers end in a fixed 1 MiB reserve
    (``EPLocalSyncCounters.allocation_size()``); any other config allocates
    the regions alone, as before."""
    assert _EP_LOCAL_SYNC_RESERVED_WORDS * 4 == 1 << 20
    for n_experts in (16, 32, 256):
        regions = (2 * n_experts + 8) + (8 * n_experts + 8) + 304
        assert get_ep_local_sync_counters_size(n_experts) == regions
        assert _ep_counter_words(n_experts, 8, False, 0) == regions
        words = regions + _EP_LOCAL_SYNC_RESERVED_WORDS
        assert _ep_counter_words(n_experts, 8, False, 2048) == words
        assert _ep_sync_counter_bytes(n_experts, 8, False, 2048) == (
            NUM_GROUPS * words * 4
        )
    # The allreduce backend's counters cover the local experts.
    assert _ep_counter_words(
        256, 8, True, 0
    ) == get_ep_local_sync_counters_size(32)


def test_fused_moe_is_requested_only_by_the_developer_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Off by default; only ``MODULAR_EP_FUSED_MOE=1`` asks for it."""
    monkeypatch.delenv("MODULAR_EP_FUSED_MOE", raising=False)
    assert not _ep_fused_moe_requested()
    for value in ("0", "true", ""):
        monkeypatch.setenv("MODULAR_EP_FUSED_MOE", value)
        assert not _ep_fused_moe_requested()
    monkeypatch.setenv("MODULAR_EP_FUSED_MOE", "1")
    assert _ep_fused_moe_requested()


@pytest.mark.parametrize(
    "override",
    [
        {"fused_shared_expert": True},
        {"use_allreduce": True},
        {"eplb_enabled": True},
        {"top_k": 32},
        {"moe_dim": 1024},
        {"max_tokens_per_rank": 4096},
    ],
)
def test_ep_config_holds_no_fused_moe_scope(override: dict[str, Any]) -> None:
    """The backend decides what the fused path serves, so any EP config
    builds, and none is marked ready before EP init says so."""
    config = dataclasses.replace(
        _nvfp4_ep_config(nvfp4_dyn_global_scales=False), **override
    )
    assert not config.fused_moe_ready


def test_ep_config_has_no_fused_moe_fields() -> None:
    """The fused path's tuning lives in the backend, not in EP config fields."""
    assert not any("megaffn" in f.name for f in dataclasses.fields(EPConfig))


def _fused_init_ir(config: EPConfig) -> str:
    """The IR of the fused init op EP init adds for ``config``."""
    with Graph(
        "fused_moe_init",
        input_types=[
            BufferType(DType.int32, [16], DeviceRef.GPU()),
            TensorType(DType.uint64, [2, 3], DeviceRef.CPU()),
        ],
    ) as graph:
        _call_mega_ffn_ep_fused_init(
            graph.inputs[0].buffer, graph.inputs[1].tensor, config
        )
        return str(graph)


def test_dispatch_format_parameters_are_the_ep_init_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ep.init`` and the fused init read one dispatch format."""
    # The NVIDIA dispatch layout is chosen by accelerator; pin it so the test
    # runs on hosts without a GPU.
    monkeypatch.setattr(ep_kernels, "accelerator_api", lambda: "cuda")
    nvfp4 = _ep_dispatch_format_parameters(
        _nvfp4_ep_config(nvfp4_dyn_global_scales=False)
    )
    assert nvfp4 == {
        "dispatch_dtype": DType.uint8,
        "dispatch_fmt_str": "BLOCK_SCALED_NV",
        "dispatch_scale_dtype": DType.float8_e4m3fn,
    }
    dyn = _ep_dispatch_format_parameters(
        _nvfp4_ep_config(nvfp4_dyn_global_scales=True)
    )
    assert dyn["nvfp4_dyn_global_scales"] is True
    assert _ep_dispatch_format_parameters(_base_ep_config()) == {
        "dispatch_dtype": DType.bfloat16,
        "dispatch_fmt_str": "BF16",
        "dispatch_scale_dtype": DType.float32,
    }


def test_fused_moe_init_op_gets_the_ep_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fused init sees what ``ep.init`` sees, plus the experts' FFN width
    and the EP features it cannot serve; Python adds no fused geometry."""
    # The NVIDIA dispatch layout is chosen by accelerator; pin it so the test
    # runs on hosts without a GPU.
    monkeypatch.setattr(ep_kernels, "accelerator_api", lambda: "cuda")
    ir = _fused_init_ir(
        dataclasses.replace(
            _nvfp4_ep_config(nvfp4_dyn_global_scales=False), moe_dim=2048
        )
    )
    assert 'symbol = "mega_ffn.ep_fused_init"' in ir
    for name in (
        "dispatch_fmt_str",
        "dispatch_scale_dtype",
        "max_token_per_rank",
        "moe_dim",
        "fused_shared_expert",
        "eplb",
    ):
        assert name in ir, name
    for name in ("token_block", "nvfp4 ", "max_tpr"):
        assert name not in ir, name


def test_fused_moe_init_sees_one_rank_under_allreduce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The allreduce backend routes within each device, so its EP ops get the
    one-rank parameters; the backend's rank rule refuses the fused path for
    them before anything is allocated."""
    # The NVIDIA dispatch layout is chosen by accelerator; pin it so the test
    # runs on hosts without a GPU.
    monkeypatch.setattr(ep_kernels, "accelerator_api", lambda: "cuda")
    ir = _fused_init_ir(
        dataclasses.replace(
            _nvfp4_ep_config(nvfp4_dyn_global_scales=False),
            moe_dim=2048,
            use_allreduce=True,
        )
    )
    assert re.search(r"n_gpus_per_node = 1\b", ir), ir


def test_fused_moe_emitter_needs_ep_init_readiness() -> None:
    """Without EP init's go-ahead the emitter raises before adding the op."""
    config = _nvfp4_ep_config(nvfp4_dyn_global_scales=False)
    device = DeviceRef.GPU()
    with Graph(
        "fused_moe_not_ready",
        input_types=[
            BufferType(DType.int32, [16], device),
            TensorType(DType.float32, [4], device),
            TensorType(DType.uint64, [8], DeviceRef.CPU()),
        ],
    ) as graph:
        counters = graph.inputs[0].buffer
        t = graph.inputs[1].tensor
        ptrs = graph.inputs[2].tensor
        with pytest.raises(ValueError, match="did not set up"):
            _call_mega_ffn_ep_fused(
                counters, t, t, t, t, t, t, t, t, t, t, t, ptrs, config
            )


def _ep2_fused_config(**override: Any) -> EPConfig:
    """The EP2 integration test's geometry: 32 experts on 2 ranks, top-k 8,
    hidden 6144, MoE width 2048, NVFP4 dispatch."""
    fields: dict[str, Any] = dict(
        n_experts=32, n_gpus_per_node=2, max_tokens_per_rank=24, moe_dim=2048
    )
    fields.update(override)
    return dataclasses.replace(
        _nvfp4_ep_config(nvfp4_dyn_global_scales=False), **fields
    )


def _fused_moe_plans(
    cases: list[dict[str, bool | int | str | DType]],
) -> list[tuple[int, int, str]]:
    """The backend's plan answer for each parameter set, all from one host
    graph: every new plan graph is a compile."""
    with Graph("fused_moe_plans", input_types=[]) as graph:
        graph.output(
            *(
                answer
                for parameters in cases
                for answer in _call_mega_ffn_ep_fused_plan(parameters)
            )
        )
    outputs = InferenceSession(devices=[CPU()]).load(graph).execute()
    answers = []
    for workspace, refusal in zip(outputs[0::2], outputs[1::2], strict=True):
        assert isinstance(workspace, Buffer) and isinstance(refusal, Buffer)
        answers.append(
            _fused_moe_answer(workspace.to_numpy(), refusal.to_numpy())
        )
    return answers


def test_fused_moe_plan_follows_the_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The host plan query gives the init op's answer without allocating:
    a size and token block for a configuration the backend serves, zero and
    the backend's reason for one it refuses. The cached planner entry point
    gives the same answer."""
    # The NVIDIA dispatch layout is chosen by accelerator; pin it so the test
    # runs on hosts without a GPU.
    monkeypatch.setattr(ep_kernels, "accelerator_api", lambda: "cuda")
    refused = (
        ({"max_tokens_per_rank": 40}, "capacity"),
        ({"use_allreduce": True}, "allreduce"),
        ({"moe_dim": 0}, "moe_dim"),
        ({"fused_shared_expert": True}, "shared-expert"),
    )
    nv24, nv32, mx24, bf16, *refusals = _fused_moe_plans(
        [
            _fused_moe_parameters(config)
            for config in (
                _ep2_fused_config(),
                _ep2_fused_config(max_tokens_per_rank=32),
                _ep2_fused_config(
                    dispatch_dtype=DType.float8_e4m3fn,
                    dispatch_quant_config=_mxfp8_quant_config(),
                ),
                _base_ep_config(),
                *(_ep2_fused_config(**override) for override, _ in refused),
            )
        ]
    )
    assert nv24[0] > 0 and nv24[1:] == (8, "")
    assert nv32[1:] == (32, "") and nv32[0] > nv24[0]
    assert mx24[1:] == (8, "") and mx24[0] > nv24[0]
    assert bf16[:2] == (0, 0) and bf16[2]
    for (override, reason), (size, block, refusal) in zip(
        refused, refusals, strict=True
    ):
        assert (size, block) == (0, 0) and reason in refusal, (
            override,
            refusal,
        )
    assert _plan_fused_moe_workspace(_ep2_fused_config()) == nv24


def test_fused_moe_bound_covers_every_format_of_its_element_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A memory plan is made before the checkpoint is parsed, so it knows the
    dispatch element type but not the format, its global scales, the
    shared-expert rows or EPLB. The backend's bound for that element type
    covers every real dispatch a model can hand EP init: init allocates
    exactly the bound for the one format it serves (the same effective
    parameters) and nothing for any variant it refuses."""
    # The NVIDIA dispatch layout is chosen by accelerator; pin it so the test
    # runs on hosts without a GPU.
    monkeypatch.setattr(ep_kernels, "accelerator_api", lambda: "cuda")
    nvfp4 = _ep2_fused_config()
    mxfp8 = dataclasses.replace(
        nvfp4,
        dispatch_dtype=DType.float8_e4m3fn,
        dispatch_quant_config=_mxfp8_quant_config(),
    )
    bf16 = dataclasses.replace(
        nvfp4, dispatch_dtype=DType.bfloat16, dispatch_quant_config=None
    )
    answers = _fused_moe_plans(
        [
            _fused_moe_bound_parameters(bf16, DType.uint8),
            _fused_moe_bound_parameters(bf16, DType.float8_e4m3fn),
            _fused_moe_bound_parameters(bf16, DType.bfloat16),
            *(
                _fused_moe_parameters(config)
                for config in (
                    nvfp4,
                    dataclasses.replace(nvfp4, nvfp4_dyn_global_scales=True),
                    dataclasses.replace(nvfp4, fused_shared_expert=True),
                    dataclasses.replace(nvfp4, eplb_enabled=True),
                    mxfp8,
                    dataclasses.replace(mxfp8, fused_shared_expert=True),
                    dataclasses.replace(
                        mxfp8, dispatch_quant_config=_block_fp8_quant_config()
                    ),
                    bf16,
                )
            ),
        ]
    )
    bound_fp4, bound_fp8, bound_bf16 = answers[:3]
    nv, nv_dyn, nv_rows, nv_eplb, mx, mx_rows, block_fp8, plain = answers[3:]
    assert bound_fp4[0] > 0 and bound_fp8[0] > bound_fp4[0]
    assert bound_bf16[0] == 0 and "NVFP4 or MXFP8" in bound_bf16[2]
    assert nv == bound_fp4 and mx == bound_fp8
    for (size, block, refusal), reason in (
        (nv_dyn, "static input scale"),
        (nv_rows, "shared-expert rows"),
        (nv_eplb, "EPLB"),
        (mx_rows, "shared-expert rows"),
        (block_fp8, "NVFP4 or MXFP8"),
        (plain, "NVFP4 or MXFP8"),
    ):
        assert (size, block) == (0, 0) and reason in refusal, refusal
