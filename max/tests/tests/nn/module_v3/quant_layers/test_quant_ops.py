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

"""Checks for the quant-format dispatch helpers in ``quant_ops``.

These cover the format predicates, weight construction/stacking, and the EP
dispatch-payload parsing — the plumbing that decides which kernel family a
weight or activation ends up on. The kernel calls themselves are covered by the
layer-level lazy-trace tests.
"""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

import numpy as np
import pytest
from max.driver import CPU, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn.common_layers import quant_ops
from max.experimental.nn.common_layers.quant_tensor import (
    FP8BlockActivation,
    FP8BlockTensor,
    MXFP4Quantizer,
    MXFP4Tensor,
    MXFP8Quantizer,
    MXFP8Tensor,
    NVFP4Activation,
    NVFP4Tensor,
    NVFP4TensorQuantizer,
)
from max.experimental.tensor import Tensor, default_dtype
from max.nn.comm.ep import EPConfig
from max.nn.quant_config import QuantConfig, QuantFormat

_HIDDEN_DIM = 256
_MOE_DIM = 128
_NUM_EXPERTS = 4
_NUM_LOCAL_EXPERTS = 2
_TOKENS = 8


# --------------------------------------------------------------------------- #
# Format predicates
# --------------------------------------------------------------------------- #


def test_predicates_separate_fp8_from_nvfp4(
    fp8_quant_config: QuantConfig, nvfp4_quant_config: QuantConfig
) -> None:
    """FP8 and NVFP4 are distinguished by format, not by block granularity.

    Both configs are block-scaled, so a ``weight_scale.is_block`` test would
    route NVFP4 weights onto the FP8 kernels.
    """
    assert nvfp4_quant_config.weight_scale.is_block

    assert quant_ops.is_fp8_block_quantized(fp8_quant_config)
    assert not quant_ops.is_fp8_block_quantized(nvfp4_quant_config)
    assert not quant_ops.is_fp8_block_quantized(None)

    assert quant_ops.is_nvfp4_quantized(nvfp4_quant_config)
    assert not quant_ops.is_nvfp4_quantized(fp8_quant_config)
    assert not quant_ops.is_nvfp4_quantized(None)


def test_source_is_orthogonal_to_format(
    nvfp4_quant_config: QuantConfig, nvfp4_fake_quant_config: QuantConfig
) -> None:
    """On-the-fly NVFP4 is still NVFP4; only its source differs.

    The format answers what the weight becomes, so both configs say NVFP4.
    The source answers what the checkpoint holds, and only there do they
    diverge.
    """
    assert quant_ops.is_nvfp4_quantized(nvfp4_quant_config)
    assert quant_ops.is_nvfp4_quantized(nvfp4_fake_quant_config)

    assert not nvfp4_quant_config.quantizes_on_load
    assert nvfp4_fake_quant_config.quantizes_on_load


def test_storage_dtype_follows_the_source_not_the_format(
    nvfp4_quant_config: QuantConfig, nvfp4_fake_quant_config: QuantConfig
) -> None:
    """A weight quantized on load is bf16 on disk, whatever it becomes."""
    assert quant_ops.routed_weight_dtype(nvfp4_quant_config) == DType.uint8
    assert (
        quant_ops.routed_weight_dtype(nvfp4_fake_quant_config) == DType.bfloat16
    )
    assert quant_ops.routed_weight_dtype(None) == DType.bfloat16


def test_only_nvfp4_needs_dispatch_scales_and_offsets(
    fp8_quant_config: QuantConfig, nvfp4_quant_config: QuantConfig
) -> None:
    """The EP dispatch and MoE index kernels take extra NVFP4-only inputs."""
    assert quant_ops.ep_requires_dispatch_scales(nvfp4_quant_config)
    assert not quant_ops.ep_requires_dispatch_scales(fp8_quant_config)
    assert not quant_ops.ep_requires_dispatch_scales(None)

    assert quant_ops.moe_requires_scales_offsets(nvfp4_quant_config)
    assert not quant_ops.moe_requires_scales_offsets(fp8_quant_config)
    assert not quant_ops.moe_requires_scales_offsets(None)


def test_dispatch_predicates_refuse_on_the_fly_weights(
    nvfp4_fake_quant_config: QuantConfig,
) -> None:
    """A quantize-on-load config is refused rather than silently answered.

    Returning ``False`` would decide by default a case nobody has reasoned
    about, and would surface as a wrong unpack arity in ``from_dispatch``
    well below the decision that caused it.
    """
    for predicate in (
        quant_ops.ep_requires_dispatch_scales,
        quant_ops.moe_requires_scales_offsets,
    ):
        with pytest.raises(NotImplementedError, match="quantized on load"):
            predicate(nvfp4_fake_quant_config)


def test_dispatch_predicates_answer_checkpoint_weights(
    nvfp4_quant_config: QuantConfig, fp8_quant_config: QuantConfig
) -> None:
    """Checkpoint-sourced configs still get an ordinary answer."""
    assert quant_ops.ep_requires_dispatch_scales(nvfp4_quant_config)
    assert quant_ops.moe_requires_scales_offsets(nvfp4_quant_config)
    assert not quant_ops.ep_requires_dispatch_scales(fp8_quant_config)
    assert not quant_ops.ep_requires_dispatch_scales(None)


# --------------------------------------------------------------------------- #
# quantized_weight
# --------------------------------------------------------------------------- #


def test_quantized_weight_nvfp4(nvfp4_quant_config: QuantConfig) -> None:
    """An NVFP4 config yields a packed NVFP4Tensor for the logical shape."""
    with F.lazy():
        weight = quant_ops.quantized_weight(
            _MOE_DIM, _HIDDEN_DIM, nvfp4_quant_config
        )

        assert isinstance(weight, NVFP4Tensor)
        assert weight.block_size == (1, 16)
        assert list(weight.weight.shape) == [_MOE_DIM, _HIDDEN_DIM // 2]
        assert list(weight.weight_scale.shape) == [_MOE_DIM, _HIDDEN_DIM // 16]


def test_quantized_weight_dispatch_by_format(
    fp8_quant_config: QuantConfig, nvfp4_quant_config: QuantConfig
) -> None:
    """Each supported format maps to its own weight wrapper."""
    with F.lazy():
        assert isinstance(
            quant_ops.quantized_weight(_MOE_DIM, _HIDDEN_DIM, None), Tensor
        )
        assert isinstance(
            quant_ops.quantized_weight(_MOE_DIM, _HIDDEN_DIM, fp8_quant_config),
            FP8BlockTensor,
        )
        assert isinstance(
            quant_ops.quantized_weight(
                _MOE_DIM, _HIDDEN_DIM, nvfp4_quant_config
            ),
            NVFP4Tensor,
        )


def test_quantized_weight_nvfp4_fake(
    nvfp4_fake_quant_config: QuantConfig,
) -> None:
    """Fake-quant NVFP4 wraps the logical bf16 weight instead of packed leaves."""
    # The default dtype is float32 on a CPU host and bfloat16 on a GPU one;
    # models build under bfloat16.
    with F.lazy(), default_dtype(DType.bfloat16):
        weight = quant_ops.quantized_weight(
            _MOE_DIM, _HIDDEN_DIM, nvfp4_fake_quant_config
        )

        assert isinstance(weight, NVFP4TensorQuantizer)
        assert weight.block_size == (1, 16)
        assert list(weight.shape) == [_MOE_DIM, _HIDDEN_DIM]
        # The dtype the checkpoint holds, not whatever device happens to be
        # available: this matches what `routed_weight_dtype` promises.
        assert weight._tensor.dtype == DType.bfloat16
        assert weight._tensor.dtype == quant_ops.routed_weight_dtype(
            nvfp4_fake_quant_config
        )


def test_quantized_weight_rejects_unsupported_format(
    nvfp4_quant_config: QuantConfig,
) -> None:
    """Formats this architecture has not wired up are rejected up front.

    MXFP6 stands in for the unregistered case: it is in the same MX family as
    the formats that *are* wired up, so this also records where the boundary
    currently sits. Pick another unregistered format if MXFP6 is ever added.
    """
    mxfp6 = dataclasses.replace(nvfp4_quant_config, format=QuantFormat.MXFP6)
    with F.lazy():
        with pytest.raises(ValueError, match="not yet supported"):
            quant_ops.quantized_weight(_MOE_DIM, _HIDDEN_DIM, mxfp6)


# --------------------------------------------------------------------------- #
# stack / concat
# --------------------------------------------------------------------------- #


def _nvfp4_weight(rows: int = _MOE_DIM, cols: int = _HIDDEN_DIM) -> NVFP4Tensor:
    return NVFP4Tensor.zeros((rows, cols))


def test_stack_nvfp4_adds_expert_axis_to_every_leaf() -> None:
    """Stacking per-expert weights lifts all four leaves to a leading E axis."""
    with F.lazy():
        stacked = quant_ops.stack([_nvfp4_weight() for _ in range(4)], axis=0)

        assert isinstance(stacked, NVFP4Tensor)
        assert list(stacked.weight.shape) == [4, _MOE_DIM, _HIDDEN_DIM // 2]
        assert list(stacked.weight_scale.shape) == [
            4,
            _MOE_DIM,
            _HIDDEN_DIM // 16,
        ]
        # The per-tensor scales become per-expert vectors.
        assert list(stacked.weight_scale_2.shape) == [4]
        assert list(stacked.input_scale.shape) == [4]
        assert stacked.block_size == (1, 16)


def test_concat_nvfp4_shares_the_global_scales() -> None:
    """Fusing gate||up concatenates the block leaves and shares the globals.

    modelopt emits one global scale for the fused projection, so the second
    weight's ``weight_scale_2`` / ``input_scale`` are intentionally dropped.
    """
    with F.lazy():
        gate, up = _nvfp4_weight(), _nvfp4_weight()
        fused = quant_ops.concat_weights(gate, up, axis=0)

        assert isinstance(fused, NVFP4Tensor)
        assert list(fused.weight.shape) == [2 * _MOE_DIM, _HIDDEN_DIM // 2]
        assert list(fused.weight_scale.shape) == [
            2 * _MOE_DIM,
            _HIDDEN_DIM // 16,
        ]
        assert fused.weight_scale_2 is gate.weight_scale_2
        assert fused.input_scale is gate.input_scale


def test_concat_nvfp4_rejects_non_row_axis() -> None:
    with F.lazy():
        with pytest.raises(ValueError, match="only supports axis=0"):
            quant_ops.concat_nvfp4(_nvfp4_weight(), _nvfp4_weight(), axis=1)


def test_concat_nvfp4_rejects_mixed_block_sizes() -> None:
    with F.lazy():
        other = NVFP4Tensor.zeros((_MOE_DIM, _HIDDEN_DIM), block_size=(1, 32))
        with pytest.raises(ValueError, match="same block_size"):
            quant_ops.concat_nvfp4(_nvfp4_weight(), other)


def test_concat_nvfp4_requires_a_tensor() -> None:
    with pytest.raises(ValueError, match="at least one tensor"):
        quant_ops.concat_nvfp4()


# --------------------------------------------------------------------------- #
# Fused-SwiGLU sigma permutation
# --------------------------------------------------------------------------- #


def test_sigma_permute_interleaves_gate_and_up_rows() -> None:
    """The fused SwiGLU kernel wants ``[g0, u0, g1, u1, ...]`` on the N axis.

    Checked eagerly on CPU: the stacked ``[gate; up]`` rows come back
    row-interleaved, and the same permutation is applied to the block scales.
    """
    num_experts, gate_rows, cols = 2, 3, 4
    rows = 2 * gate_rows

    data = np.arange(num_experts * rows * cols, dtype=np.uint8).reshape(
        num_experts, rows, cols
    )
    scales = np.arange(num_experts * rows * 2, dtype=np.float32).reshape(
        num_experts, rows, 2
    )

    weight = NVFP4Tensor(
        weight=Tensor(data, device=CPU()),
        weight_scale=Tensor(scales, device=CPU()),
        weight_scale_2=Tensor(np.ones(num_experts, np.float32), device=CPU()),
        input_scale=Tensor(np.ones(num_experts, np.float32), device=CPU()),
    )

    permuted = quant_ops.sigma_permute_gate_up_nvfp4(weight)

    # sigma(2i) = i (gate row i), sigma(2i + 1) = gate_rows + i (up row i).
    order = [i // 2 if i % 2 == 0 else gate_rows + i // 2 for i in range(rows)]
    np.testing.assert_array_equal(permuted.weight.to_numpy(), data[:, order])
    np.testing.assert_array_equal(
        permuted.weight_scale.to_numpy(), scales[:, order]
    )
    # The global scales are untouched by the permutation.
    assert permuted.weight_scale_2 is weight.weight_scale_2
    assert permuted.input_scale is weight.input_scale
    assert permuted.block_size == weight.block_size


# --------------------------------------------------------------------------- #
# EPDispatchPayload
# --------------------------------------------------------------------------- #


def _ep_config(
    dispatch_dtype: DType, quant_config: QuantConfig | None
) -> EPConfig:
    return EPConfig(
        dispatch_dtype=dispatch_dtype,
        combine_dtype=DType.bfloat16,
        hidden_size=_HIDDEN_DIM,
        top_k=2,
        n_experts=_NUM_EXPERTS,
        max_tokens_per_rank=_TOKENS,
        n_gpus_per_node=2,
        n_nodes=1,
        dispatch_quant_config=quant_config,
    )


def _value(shape: list[int], dtype: DType) -> Tensor:
    return Tensor.zeros(shape, dtype=dtype, device=CPU())


def _metadata() -> Tensor:
    """The host grouped-matmul metadata tuple element ``[max_m, n_active]``."""
    return Tensor.zeros([2], dtype=DType.uint32, device=CPU())


def _bf16_dispatch() -> list[tuple[Tensor, ...]]:
    return [
        (
            _value([_TOKENS, _HIDDEN_DIM], DType.bfloat16),
            _value([_NUM_LOCAL_EXPERTS + 1], DType.uint32),
            _value([_NUM_LOCAL_EXPERTS], DType.int32),
            _metadata(),
        )
        for _ in range(2)
    ]


def _fp8_dispatch() -> list[tuple[Tensor, ...]]:
    return [
        (
            _value([_TOKENS, _HIDDEN_DIM], DType.float8_e4m3fn),
            _value([_TOKENS, _HIDDEN_DIM // 128], DType.float32),
            _value([_NUM_LOCAL_EXPERTS + 1], DType.uint32),
            _value([_NUM_LOCAL_EXPERTS], DType.int32),
            _metadata(),
        )
        for _ in range(2)
    ]


def _nvfp4_dispatch() -> list[tuple[Tensor, ...]]:
    """The six NVFP4 dispatch outputs (see ``_ep_dispatch_output_types``)."""
    return [
        (
            _value([_TOKENS, _HIDDEN_DIM // 2], DType.uint8),
            _value([1, 1, 32, 4, 4], DType.float8_e4m3fn),
            _value([_NUM_LOCAL_EXPERTS + 1], DType.uint32),
            _value([_NUM_LOCAL_EXPERTS], DType.uint32),
            _value([_NUM_LOCAL_EXPERTS], DType.int32),
            _metadata(),
        )
        for _ in range(2)
    ]


def test_payload_bf16_has_no_scales() -> None:
    """A bf16 dispatch carries usage stats and no scale columns."""
    with F.lazy():
        config = _ep_config(DType.bfloat16, None)
        payload = quant_ops.EPDispatchPayload.from_dispatch(
            _bf16_dispatch(), None, config
        )

        assert payload.scales is None
        assert payload.scales_offset is None
        assert payload.usage_stats is not None
        assert len(payload.tokens) == 2
        assert list(payload.expert_start[0].shape) == [_NUM_LOCAL_EXPERTS + 1]
        assert list(payload.expert_ids[0].shape) == [_NUM_LOCAL_EXPERTS]
        # The bf16 grouped matmul reads num_active_experts off the host.
        assert payload.usage_stats[0].device.is_host

        assert payload.per_device_tokens() == payload.tokens


def test_payload_fp8_carries_activation_scales(
    fp8_quant_config: QuantConfig,
) -> None:
    """An FP8 dispatch yields pre-quantized FP8BlockActivation tokens."""
    with F.lazy():
        config = _ep_config(DType.float8_e4m3fn, fp8_quant_config)
        payload = quant_ops.EPDispatchPayload.from_dispatch(
            _fp8_dispatch(), fp8_quant_config, config
        )

        assert payload.scales is not None
        assert payload.scales_offset is None
        assert payload.usage_stats is not None

        tokens = payload.per_device_tokens()
        assert len(tokens) == 2
        first = tokens[0]
        assert isinstance(first, FP8BlockActivation)
        assert first.data is payload.tokens[0]
        assert first.scales is payload.scales[0]


def test_payload_nvfp4_carries_scales_and_offsets(
    nvfp4_quant_config: QuantConfig,
) -> None:
    """An NVFP4 dispatch adds scale offsets and drops the host usage stats."""
    with F.lazy():
        config = _ep_config(DType.float8_e4m3fn, nvfp4_quant_config)
        payload = quant_ops.EPDispatchPayload.from_dispatch(
            _nvfp4_dispatch(), nvfp4_quant_config, config
        )

        assert payload.scales is not None
        assert payload.scales_offset is not None
        # The NVFP4 grouped matmul synthesizes its own host stats.
        assert payload.usage_stats is None

        global_scale = Tensor.zeros((), dtype=DType.float32, device=CPU())
        tokens = payload.per_device_tokens(nvfp4_global_scale=global_scale)
        assert len(tokens) == 2
        first = tokens[0]
        assert isinstance(first, NVFP4Activation)
        assert first.data is payload.tokens[0]
        assert first.scales is payload.scales[0]
        assert first.scales_offset is payload.scales_offset[0]
        assert first.input_scale is global_scale


def test_payload_nvfp4_tokens_require_the_uniform_scale(
    nvfp4_quant_config: QuantConfig,
) -> None:
    """The dispatch quantized with one scale; wrapping without it is a bug."""
    with F.lazy():
        config = _ep_config(DType.float8_e4m3fn, nvfp4_quant_config)
        payload = quant_ops.EPDispatchPayload.from_dispatch(
            _nvfp4_dispatch(), nvfp4_quant_config, config
        )

        with pytest.raises(AssertionError, match="uniform scale"):
            payload.per_device_tokens()


def test_nvfp4_grouped_matmul_reports_the_routed_row_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dispatcher's row estimate is the routed row count, not a constant.

    It picks decode tiles when ``estimated_total_m / num_active_experts <= 8``
    and, given no estimate, reads ``estimated_total_m`` from the host stats.
    """
    seen: dict[str, object] = {}

    def spy(*args: Tensor, estimated_total_m: object, **_: object) -> Tensor:
        seen["stats"], seen["estimate"] = args[8], estimated_total_m
        return Tensor.zeros((_TOKENS, _MOE_DIM), dtype=DType.bfloat16)

    monkeypatch.setattr(quant_ops, "grouped_matmul_block_scaled", spy)
    monkeypatch.setattr(
        quant_ops, "interleave_grouped_scales", lambda s, **_: s
    )
    device = CPU()
    offsets = Tensor.zeros((_NUM_EXPERTS,), dtype=DType.uint32, device=device)
    tokens = NVFP4Activation(
        data=Tensor.zeros(
            (_TOKENS, _HIDDEN_DIM // 2), dtype=DType.uint8, device=device
        ),
        scales=Tensor.zeros(
            (_NUM_EXPERTS + 1, _HIDDEN_DIM // 64, 32, 4, 4),
            dtype=DType.float8_e4m3fn,
            device=device,
        ),
        input_scale=Tensor.zeros(
            (_NUM_EXPERTS,), dtype=DType.float32, device=device
        ),
        scales_offset=offsets,
    )
    quant_ops.grouped_matmul(
        tokens,
        _stacked_nvfp4(_MOE_DIM, _HIDDEN_DIM, device),
        Tensor.zeros((_NUM_EXPERTS + 1,), dtype=DType.uint32, device=device),
        Tensor.zeros((_NUM_EXPERTS,), dtype=DType.int32, device=device),
        None,
    )

    assert seen["estimate"] is None
    stats = seen["stats"]
    assert isinstance(stats, Tensor)
    assert stats.to_numpy().tolist() == [_TOKENS, _NUM_EXPERTS]


# --------------------------------------------------------------------------- #
# SwiGLU route
# --------------------------------------------------------------------------- #


def _stacked_nvfp4(rows: int, cols: int, device: Device) -> NVFP4Tensor:
    return NVFP4Tensor(
        weight=Tensor.zeros(
            (_NUM_EXPERTS, rows, cols // 2), dtype=DType.uint8, device=device
        ),
        weight_scale=Tensor.zeros(
            (_NUM_EXPERTS, rows, cols // 16),
            dtype=DType.float8_e4m3fn,
            device=device,
        ),
        weight_scale_2=Tensor.zeros(
            (_NUM_EXPERTS,), dtype=DType.float32, device=CPU()
        ),
        input_scale=Tensor.zeros(
            (_NUM_EXPERTS,), dtype=DType.float32, device=CPU()
        ),
        block_size=(1, 16),
    )


def _stacked_fp8(rows: int, cols: int, device: Device) -> FP8BlockTensor:
    return FP8BlockTensor(
        weight=Tensor.zeros(
            (_NUM_EXPERTS, rows, cols),
            dtype=DType.float8_e4m3fn,
            device=device,
        ),
        weight_scale_inv=Tensor.zeros(
            (_NUM_EXPERTS, max(rows // 128, 1), max(cols // 128, 1)),
            dtype=DType.float32,
            device=device,
        ),
        block_size=(128, 128),
    )


def _swiglu_route(
    encoding: str,
    prequantized: bool,
    accum_dtype: DType | None,
    swiglu_limit: float,
    device: Device,
    quant_config: QuantConfig,
) -> str:
    """Traces ``grouped_matmul_silu`` and names the activation it returns."""
    nvfp4 = encoding == "nvfp4"
    build = _stacked_nvfp4 if nvfp4 else _stacked_fp8
    offsets = Tensor.zeros((_NUM_EXPERTS,), dtype=DType.uint32, device=device)
    if not prequantized:
        tokens: quant_ops.QuantAwareTensor = Tensor.zeros(
            (_NUM_EXPERTS, _HIDDEN_DIM), dtype=DType.bfloat16, device=device
        )
    elif nvfp4:
        tokens = NVFP4Activation(
            data=Tensor.zeros(
                (_NUM_EXPERTS, _HIDDEN_DIM // 2),
                dtype=DType.uint8,
                device=device,
            ),
            scales=Tensor.zeros(
                (_NUM_EXPERTS + 1, _HIDDEN_DIM // 64, 32, 4, 4),
                dtype=DType.float8_e4m3fn,
                device=device,
            ),
            input_scale=Tensor.zeros(
                (_NUM_EXPERTS,), dtype=DType.float32, device=CPU()
            ),
            scales_offset=offsets,
        )
    else:
        tokens = FP8BlockActivation(
            data=Tensor.zeros(
                (_NUM_EXPERTS, _HIDDEN_DIM),
                dtype=DType.float8_e4m3fn,
                device=device,
            ),
            scales=Tensor.zeros(
                (_NUM_EXPERTS, _HIDDEN_DIM // 128),
                dtype=DType.float32,
                device=device,
            ),
        )
    out = quant_ops.grouped_matmul_silu(
        tokens,
        build(2 * _MOE_DIM, _HIDDEN_DIM, device),
        build(_HIDDEN_DIM, _MOE_DIM, device),
        Tensor.zeros((_NUM_EXPERTS + 1,), dtype=DType.uint32, device=device),
        Tensor.zeros((_NUM_EXPERTS,), dtype=DType.int32, device=device),
        Tensor.zeros((2,), dtype=DType.uint32, device=device),
        quant_config,
        scales_offset=offsets if nvfp4 else None,
        accum_dtype=accum_dtype,
        swiglu_limit=swiglu_limit,
    )
    return type(out).__name__


@pytest.mark.parametrize(
    ("arch", "encoding", "prequantized", "accum_dtype", "limit", "expected"),
    [
        # deepseekV3_modulev3 and kimik2_5_modulev3 share one MoE: no
        # gate/up accumulation dtype and no activation clamp.
        ("deepseekV3/kimik2_5 TP", "fp8", False, None, 0.0, "Tensor"),
        ("deepseekV3/kimik2_5 TP", "nvfp4", False, None, 0.0, "Tensor"),
        (
            "deepseekV3/kimik2_5 EP",
            "fp8",
            True,
            None,
            0.0,
            "FP8BlockActivation",
        ),
        (
            "deepseekV3/kimik2_5 EP",
            "nvfp4",
            True,
            None,
            0.0,
            "NVFP4Activation",
        ),
        # deepseekV3_2_modulev3 accumulates the gate/up in float32 off the EP
        # dispatch; glm5_next_modulev3 does too, and clamps the activation.
        ("deepseekV3_2 TP", "fp8", False, DType.float32, 0.0, "Tensor"),
        ("deepseekV3_2 EP", "fp8", True, None, 0.0, "FP8BlockActivation"),
        ("glm5_next TP", "fp8", False, DType.float32, 10.0, "Tensor"),
        (
            "glm5_next EP",
            "fp8",
            True,
            DType.float32,
            10.0,
            "FP8BlockActivation",
        ),
        # A float32 SwiGLU output over an NVFP4 down weight is the one case
        # that must take the fused quantized SiLU: the down matmul quantizes
        # from bf16 only.
        (
            "float32 SwiGLU, NVFP4 down",
            "nvfp4",
            False,
            DType.float32,
            0.0,
            "NVFP4Activation",
        ),
    ],
)
def test_swiglu_route_per_architecture(
    mock_accelerator: MagicMock,
    nvfp4_quant_config: QuantConfig,
    fp8_quant_config: QuantConfig,
    sm100_arch: None,
    arch: str,
    encoding: str,
    prequantized: bool,
    accum_dtype: DType | None,
    limit: float,
    expected: str,
) -> None:
    """Pins the SwiGLU each shipped MoE configuration asks for.

    Off the EP dispatch a bf16 SwiGLU output is left for the down matmul to
    quantize, because the fused quantized SiLU cannot be built for a device
    holding more experts than the GPU has SMs.
    """
    device = mock_accelerator()
    config = nvfp4_quant_config if encoding == "nvfp4" else fp8_quant_config
    with F.lazy():
        got = _swiglu_route(
            encoding, prequantized, accum_dtype, limit, device, config
        )
    assert got == expected, arch


# --------------------------------------------------------------------------- #
# MXFP8
# --------------------------------------------------------------------------- #


def test_mxfp8_predicates_and_moe_inputs(
    mxfp8_quant_config: QuantConfig, nvfp4_quant_config: QuantConfig
) -> None:
    """MXFP8 needs MoE scale offsets but no EP dispatch scales."""
    assert quant_ops.is_mxfp8_quantized(mxfp8_quant_config)
    assert not quant_ops.is_mxfp8_quantized(nvfp4_quant_config)
    assert not quant_ops.is_mxfp8_quantized(None)
    assert not quant_ops.is_nvfp4_quantized(mxfp8_quant_config)

    assert quant_ops.moe_requires_scales_offsets(mxfp8_quant_config)
    assert not quant_ops.ep_requires_dispatch_scales(mxfp8_quant_config)
    assert (
        quant_ops.routed_weight_dtype(mxfp8_quant_config) == DType.float8_e4m3fn
    )


def test_quantized_weight_mxfp8(mxfp8_quant_config: QuantConfig) -> None:
    """An MXFP8 config yields an MXFP8Tensor with one scale per 32 values."""
    with F.lazy():
        weight = quant_ops.quantized_weight(
            _MOE_DIM, _HIDDEN_DIM, mxfp8_quant_config
        )

        assert isinstance(weight, MXFP8Tensor)
        assert weight.block_size == (1, 32)
        assert list(weight.weight.shape) == [_MOE_DIM, _HIDDEN_DIM]
        assert list(weight.weight_scale.shape) == [_MOE_DIM, _HIDDEN_DIM // 32]


def test_stack_and_concat_mxfp8() -> None:
    """Stacking adds an expert axis to both leaves; concat fuses rows."""
    with F.lazy():
        weights: list[quant_ops.QuantAwareTensor] = [
            MXFP8Tensor.zeros((_MOE_DIM, _HIDDEN_DIM)) for _ in range(4)
        ]
        stacked = quant_ops.stack(weights, axis=0)
        assert isinstance(stacked, MXFP8Tensor)
        assert list(stacked.weight.shape) == [4, _MOE_DIM, _HIDDEN_DIM]
        assert list(stacked.weight_scale.shape) == [
            4,
            _MOE_DIM,
            _HIDDEN_DIM // 32,
        ]

        fused = quant_ops.concat_weights(weights[0], weights[1], axis=0)
        assert isinstance(fused, MXFP8Tensor)
        assert list(fused.weight.shape) == [2 * _MOE_DIM, _HIDDEN_DIM]
        assert list(fused.weight_scale.shape) == [
            2 * _MOE_DIM,
            _HIDDEN_DIM // 32,
        ]


def test_concat_mxfp8_rejects_non_row_axis() -> None:
    with F.lazy():
        w = MXFP8Tensor.zeros((_MOE_DIM, _HIDDEN_DIM))
        with pytest.raises(ValueError, match="only supports axis=0"):
            quant_ops.concat_mxfp8(w, w, axis=1)


def test_grouped_silu_takes_a_swiglu_limit(
    mock_accelerator: MagicMock,
) -> None:
    """The bf16 SiLU accepts a clamping limit and keeps its shape."""
    device = mock_accelerator()
    with F.lazy():
        x = Tensor.zeros(
            [_TOKENS, 2 * _MOE_DIM], dtype=DType.bfloat16, device=device
        )
        offsets = Tensor.zeros([3], dtype=DType.uint32, device=device)
        out = quant_ops.grouped_silu(x, offsets, swiglu_limit=7.0)
        assert isinstance(out, Tensor)
        assert list(out.shape) == [_TOKENS, _MOE_DIM]


# --------------------------------------------------------------------------- #
# MXFP4 and on-the-fly MX quantization
# --------------------------------------------------------------------------- #


def test_mxfp8_constructor_matches_the_parsed_config(
    mxfp8_quant_config: QuantConfig,
) -> None:
    """``QuantConfig.mxfp8()`` builds what the checkpoint parser produces."""
    assert (
        QuantConfig.mxfp8(mlp_quantized_layers={0}, attn_quantized_layers={0})
        == mxfp8_quant_config
    )


def test_quantized_weight_mxfp8_fake(
    mxfp8_fake_quant_config: QuantConfig,
) -> None:
    """Quantize-on-load MXFP8 wraps the logical bf16 weight."""
    with F.lazy():
        weight = quant_ops.quantized_weight(
            _MOE_DIM, _HIDDEN_DIM, mxfp8_fake_quant_config
        )

        assert isinstance(weight, MXFP8Quantizer)
        # It is also the container, so MXFP8 quant ops accept it as is.
        assert isinstance(weight, MXFP8Tensor)
        assert weight.block_size == (1, 32)
        assert list(weight.shape) == [_MOE_DIM, _HIDDEN_DIM]
        assert weight._tensor.dtype == DType.bfloat16
        assert weight._tensor.dtype == quant_ops.routed_weight_dtype(
            mxfp8_fake_quant_config
        )


def test_routed_weight_dtype_mx_follows_source(
    mxfp8_quant_config: QuantConfig,
    mxfp8_fake_quant_config: QuantConfig,
    mxfp4_quant_config: QuantConfig,
    mxfp4_fake_quant_config: QuantConfig,
) -> None:
    """The source decides the dtype, not the format."""
    assert (
        quant_ops.routed_weight_dtype(mxfp8_quant_config) == DType.float8_e4m3fn
    )
    assert quant_ops.routed_weight_dtype(mxfp4_quant_config) == DType.uint8
    for config in (mxfp8_fake_quant_config, mxfp4_fake_quant_config):
        assert quant_ops.routed_weight_dtype(config) == DType.bfloat16


def test_quantized_weight_mxfp4(mxfp4_quant_config: QuantConfig) -> None:
    """MXFP4 packs two codes per byte, with one E8M0 scale per 32 inputs."""
    with F.lazy():
        weight = quant_ops.quantized_weight(
            _MOE_DIM, _HIDDEN_DIM, mxfp4_quant_config
        )

        assert isinstance(weight, MXFP4Tensor)
        assert weight.block_size == (1, 32)
        # Halved trailing dim, unlike MXFP8: two e2m1 values per uint8.
        assert list(weight.weight.shape) == [_MOE_DIM, _HIDDEN_DIM // 2]
        assert weight.weight.dtype == DType.uint8
        assert list(weight.weight_scale.shape) == [_MOE_DIM, _HIDDEN_DIM // 32]
        assert weight.weight_scale.dtype == DType.float8_e8m0fnu


def test_quantized_weight_mxfp4_fake(
    mxfp4_fake_quant_config: QuantConfig,
) -> None:
    """Quantize-on-load MXFP4 wraps the logical bf16 weight."""
    with F.lazy():
        weight = quant_ops.quantized_weight(
            _MOE_DIM, _HIDDEN_DIM, mxfp4_fake_quant_config
        )

        assert isinstance(weight, MXFP4Quantizer)
        assert isinstance(weight, MXFP4Tensor)
        assert list(weight.shape) == [_MOE_DIM, _HIDDEN_DIM]
        assert weight._tensor.dtype == DType.bfloat16


def test_matmul_mxfp4_names_the_hardware_on_nvidia(
    mxfp4_quant_config: QuantConfig,
) -> None:
    """Off AMD, MXFP4 matmul refuses up front instead of failing in the kernel.

    The SM100 block-scaled op takes ``uint8`` operands only with
    ``float8_e4m3fn`` scales, so MXFP4's E8M0 scales would otherwise surface
    as a scales-dtype mismatch well below this call.
    """
    with F.lazy():
        weight = quant_ops.quantized_weight(128, 64, mxfp4_quant_config)
        x = Tensor.zeros((8, 64), dtype=DType.bfloat16)

        with pytest.raises(NotImplementedError, match="AMD CDNA"):
            quant_ops.matmul(x, weight)


def test_matmul_mxfp4_builds_the_amd_graph(
    amd_arch: None, mxfp4_quant_config: QuantConfig
) -> None:
    """On AMD the matmul traces, with rank-2 scales and no interleave.

    Graph construction only -- numerics are covered by the AMD integration
    test, not here.
    """
    del amd_arch
    with F.lazy():
        weight = quant_ops.quantized_weight(128, 64, mxfp4_quant_config)
        out = quant_ops.matmul(
            Tensor.zeros((8, 64), dtype=DType.bfloat16), weight
        )

        assert out.dtype == DType.bfloat16
        assert list(out.shape) == [8, 128]
