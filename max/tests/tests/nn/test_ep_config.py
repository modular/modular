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
from typing import Any

import pytest
from max.dtype import DType
from max.graph import BufferType, DeviceRef, Graph, TensorType, Type
from max.nn.comm.ep import calculate_ep_max_tokens_per_rank, ep_kernels
from max.nn.comm.ep.ep_config import EPConfig, estimate_ep_memory_usage
from max.nn.comm.ep.ep_kernels import (
    _ep_dispatch_output_types,
    _validate_ffn_combine_send_config,
    call_ep_dispatch_async,
    call_ep_dispatch_wait,
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
