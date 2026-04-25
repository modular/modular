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

"""Tensor-parallel NVFP4 MoEQuantized test across 2 GPUs with allreduce."""

from __future__ import annotations

import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_api, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, Shape, ShardingStrategy, TensorType
from max.graph.weights import WeightData
from max.nn import Allreduce, Signals
from max.nn.moe import MoEGate, MoEQuantized
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from test_common.graph_utils import is_b100_b200
from torch.utils.dlpack import from_dlpack

HIDDEN_DIM = 256
MOE_DIM = 512
NUM_EXPERTS = 4
NUM_EXPERTS_PER_TOKEN = 2
SEQ_LEN = 8

_FLOAT4_EXP_BIAS = 1
_FLOAT4_MANTISSA_BITS = 1
_FP4_MAX = 6.0
_BF16_MANTISSA_BITS = 7
_BF16_EXP_BITS = 8
_BF16_EXP_BIAS = 127


# ---------------------------------------------------------------------------
# FP4 emulation (copied from test_ep_moe_fp4.py)
# ---------------------------------------------------------------------------


# Reference: https://github.com/vllm-project/vllm/blob/c57d38d603213a9acfd5e83f38d45f9d635124fb/tests/quantization/reference_mxfp4.py
def _dequantize_fp4_to_bf16(
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
) -> torch.Tensor:
    """Dequantize FP4 E2M1 packed weights to BF16 on the given device."""
    assert packed_weight.dtype == torch.uint8

    unpacked = torch.zeros(
        *packed_weight.shape[:-1],
        packed_weight.shape[-1] * 2,
        dtype=torch.uint8,
        device=packed_weight.device,
    )
    unpacked[..., 1::2] = (packed_weight >> 4) & 0x0F
    unpacked[..., ::2] = packed_weight & 0x0F

    sign = (unpacked >> 3).to(torch.int32)
    exp = ((unpacked >> 1) & 3).to(torch.int32)
    mantissa = (unpacked & 1).to(torch.int32)

    new_exp = exp - _FLOAT4_EXP_BIAS + _BF16_EXP_BIAS
    new_exp = new_exp * torch.logical_or(exp > 0, mantissa > 0).to(torch.int32)
    new_mantissa = torch.logical_and(mantissa, exp > 0).to(torch.int32)

    bf16_bits = (
        (sign << 15)
        + (new_exp << _BF16_MANTISSA_BITS)
        + (new_mantissa << (_BF16_MANTISSA_BITS - 1))
    ).to(torch.uint16)

    bf16_weight = bf16_bits.view(torch.bfloat16)

    scale = weight_scale.to(torch.float32).repeat_interleave(16, dim=1)
    return (
        bf16_weight.to(torch.float32) * scale * weight_scale_2.to(torch.float32)
    ).to(torch.bfloat16)


def _simulate_bf16_to_fp4_roundtrip(val: torch.Tensor) -> torch.Tensor:
    """Cast BF16 values to the nearest FP4 E2M1 representable value and back."""
    float_type = val.dtype
    half_mantissa_bits = _BF16_MANTISSA_BITS
    half_exp_bits = _BF16_EXP_BITS
    half_exp_bias = _BF16_EXP_BIAS

    val_view = val.view(torch.int16)

    exp = val_view >> half_mantissa_bits
    exp = exp & ((1 << half_exp_bits) - 1)
    exp = exp.view(torch.uint16).to(torch.int32)

    sign = (val_view >> (half_mantissa_bits + half_exp_bits)) & 1
    mantissa_last = (val_view >> (half_mantissa_bits - 1)) & 1

    exp_unbias = exp - half_exp_bias
    new_exp = exp_unbias + _FLOAT4_EXP_BIAS

    exp_shift = (new_exp <= 0) * (1 - new_exp)
    tail_bits = half_mantissa_bits - _FLOAT4_MANTISSA_BITS + exp_shift
    tail_bits[tail_bits >= 16] = 16

    mantissa_plus_one = val_view & ((1 << (half_mantissa_bits + 1)) - 1)
    half = 1 << (tail_bits - 1)
    tail = mantissa_plus_one & ((1 << tail_bits) - 1)

    round_close = tail < half
    round_away = tail > half
    tie = tail == half

    new_mantissa_close = (new_exp > 0) * mantissa_last
    new_exp_close = exp

    new_mantissa_away = torch.logical_and(new_exp > 0, mantissa_last == 0)
    new_exp_away = exp + torch.logical_or(new_exp <= 0, mantissa_last == 1)

    new_exp_tie = (exp > (half_exp_bias - 2)) * (exp + (mantissa_last == 1))

    new_exp = (
        round_away * new_exp_away
        + round_close * new_exp_close
        + tie * new_exp_tie
    )
    new_mantissa = (
        round_away * new_mantissa_away + round_close * new_mantissa_close
    )

    new_mantissa = new_mantissa + (new_exp > (2 + half_exp_bias)) * (
        new_mantissa == 0
    )

    new_exp = (new_exp >= (half_exp_bias - 2)) * torch.clamp(
        new_exp, half_exp_bias - 2, half_exp_bias + 2
    )

    sign = sign.to(torch.int32)
    new_mantissa = new_mantissa.to(torch.int32)

    qdq_val = (
        (sign << 15)
        + (new_exp << half_mantissa_bits)
        + (new_mantissa << (half_mantissa_bits - 1))
    )
    qdq_val = qdq_val.to(torch.uint16)
    return qdq_val.view(float_type)


def simulate_fp4_blockwise_quantize(
    x: torch.Tensor, block_size: int = 16
) -> torch.Tensor:
    """Simulate NVFP4 blockwise quantization on a BF16 tensor."""
    assert x.dtype == torch.bfloat16
    orig_shape = x.shape
    *batch, last = orig_shape
    assert last % block_size == 0

    x_blocks = x.reshape(*batch, last // block_size, block_size)

    abs_max = x_blocks.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale_f32 = abs_max / _FP4_MAX

    scale_fp8 = scale_f32.to(torch.float8_e4m3fn)
    scale_restored = scale_fp8.to(torch.float32)

    x_scaled = (x_blocks.float() / scale_restored).to(torch.bfloat16)
    x_fp4_rt = _simulate_bf16_to_fp4_roundtrip(x_scaled)
    x_deq = (x_fp4_rt.float() * scale_restored).to(torch.bfloat16)

    return x_deq.reshape(orig_shape)


# ---------------------------------------------------------------------------
# Torch reference
# ---------------------------------------------------------------------------


def torch_moe(
    input_token: torch.Tensor,
    moe_weights: dict[str, torch.Tensor],
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
) -> torch.Tensor:
    """Single-token MoE reference with FP4 quantisation simulation."""
    assert input_token.shape[0] == 1

    input_token = simulate_fp4_blockwise_quantize(input_token)

    top_k = topk_indices.shape[1]
    result = torch.zeros_like(input_token)

    for i in range(top_k):
        scores = topk_scores[0, i]
        expert_idx = topk_indices[0, i].item()
        gate_weight = moe_weights[f"experts.{expert_idx}.gate_proj.weight"]
        up_weight = moe_weights[f"experts.{expert_idx}.up_proj.weight"]
        down_weight = moe_weights[f"experts.{expert_idx}.down_proj.weight"]

        expert_gate = input_token @ gate_weight.T
        expert_up = input_token @ up_weight.T
        down_input = simulate_fp4_blockwise_quantize(
            torch.nn.functional.silu(expert_gate) * expert_up
        )
        expert_output = down_input @ down_weight.T

        result += expert_output * scores

    shared_gate_weight = moe_weights["shared_experts.gate_proj.weight"]
    shared_up_weight = moe_weights["shared_experts.up_proj.weight"]
    shared_down_weight = moe_weights["shared_experts.down_proj.weight"]
    shared_expert_gate = input_token @ shared_gate_weight.T
    shared_expert_up = input_token @ shared_up_weight.T
    shared_down_input = simulate_fp4_blockwise_quantize(
        torch.nn.functional.silu(shared_expert_gate) * shared_expert_up
    )
    shared_expert_output = shared_down_input @ shared_down_weight.T
    result += shared_expert_output

    return result


# ---------------------------------------------------------------------------
# Weight creation
# ---------------------------------------------------------------------------


def _create_fp4_weights() -> dict[str, torch.Tensor]:
    """Create FP4 weights inline for all experts and shared experts."""
    torch.manual_seed(42)

    fp4_scale_min = 50.0
    fp4_scale_max = 150.0

    def _add_fp4_proj(
        weights: dict[str, torch.Tensor],
        prefix: str,
        out_dim: int,
        in_dim: int,
        weight_scale_2: torch.Tensor | None = None,
        input_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = torch.randint(
            0, 256, (out_dim, in_dim // 2), dtype=torch.uint8
        )
        weight_scale = (
            torch.rand(out_dim, weight.shape[1] // 8, dtype=torch.float32)
            * (fp4_scale_max - fp4_scale_min)
            + fp4_scale_min
        ).to(torch.float8_e4m3fn)

        if weight_scale_2 is None:
            weight_scale_2 = torch.rand((), dtype=torch.float32) * 1e-3

        if input_scale is None:
            input_scale = torch.rand((), dtype=torch.float32) + 0.2

        weights[f"{prefix}.weight"] = weight
        weights[f"{prefix}.weight_scale"] = weight_scale
        weights[f"{prefix}.weight_scale_2"] = weight_scale_2
        weights[f"{prefix}.input_scale"] = input_scale
        return weight_scale_2, input_scale

    weights: dict[str, torch.Tensor] = {}

    weights["gate.gate_score.weight"] = (
        torch.randn(NUM_EXPERTS, HIDDEN_DIM, dtype=torch.bfloat16) * 2e-3
    )

    for expert_idx in range(NUM_EXPERTS):
        gate_up_scale_2, input_scale = _add_fp4_proj(
            weights,
            f"experts.{expert_idx}.gate_proj",
            MOE_DIM,
            HIDDEN_DIM,
        )
        _add_fp4_proj(
            weights,
            f"experts.{expert_idx}.up_proj",
            MOE_DIM,
            HIDDEN_DIM,
            weight_scale_2=gate_up_scale_2,
            input_scale=input_scale,
        )
        _add_fp4_proj(
            weights,
            f"experts.{expert_idx}.down_proj",
            HIDDEN_DIM,
            MOE_DIM,
        )

    shared_gate_up_scale_2, shared_input_scale = _add_fp4_proj(
        weights,
        "shared_experts.gate_proj",
        MOE_DIM,
        HIDDEN_DIM,
    )
    _add_fp4_proj(
        weights,
        "shared_experts.up_proj",
        MOE_DIM,
        HIDDEN_DIM,
        weight_scale_2=shared_gate_up_scale_2,
        input_scale=shared_input_scale,
    )
    _add_fp4_proj(
        weights,
        "shared_experts.down_proj",
        HIDDEN_DIM,
        MOE_DIM,
    )

    return weights


def _wrap_fp8_weights(
    raw_weights: dict[str, torch.Tensor],
) -> dict[str, WeightData | torch.Tensor]:
    """Wrap float8_e4m3fn tensors as WeightData for session.load."""
    wrapped: dict[str, WeightData | torch.Tensor] = {}
    for key, value in raw_weights.items():
        if value.dtype == torch.float8_e4m3fn:
            wrapped[key] = WeightData(
                Buffer.from_dlpack(value.view(torch.uint8)).view(
                    DType.float8_e4m3fn
                ),
                key,
                DType.float8_e4m3fn,
                Shape(value.shape),
            )
        else:
            wrapped[key] = value
    return wrapped


def _dequantize_all_weights(
    raw_weights: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Dequantize all FP4 projection weights to BF16 for torch reference."""
    bf16_weights: dict[str, torch.Tensor] = {}
    for key, value in raw_weights.items():
        if key.endswith(".weight") and value.dtype == torch.uint8:
            scale = raw_weights[f"{key}_scale"]
            scale_2 = raw_weights[f"{key}_scale_2"]
            bf16_weights[key] = _dequantize_fp4_to_bf16(
                value.to(device), scale.to(device), scale_2.to(device)
            )
        elif key == "gate.gate_score.weight":
            bf16_weights[key] = value.to(device)
    return bf16_weights


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    accelerator_api() == "hip", reason="FP4 kernel only supports Nvidia GPUs"
)
@pytest.mark.skipif(
    not is_b100_b200(),
    reason="FP4 kernel requires B100 or B200",
)
def test_tp_moe_nvfp4() -> None:
    """Verify tensor-parallel NVFP4 MoEQuantized across 2 GPUs."""
    n_devices = 2
    assert accelerator_count() >= n_devices, (
        f"Need {n_devices} GPUs, found {accelerator_count()}"
    )

    devices = [Accelerator(id) for id in range(n_devices)]
    device_refs = [DeviceRef(d.label, d.id) for d in devices]
    session = InferenceSession(devices=devices)

    fp4_config = QuantConfig(
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

    raw_weights = _create_fp4_weights()
    wrapped_weights = _wrap_fp8_weights(raw_weights)

    # Build TP-sharded MoEQuantized
    moe = MoEQuantized(
        devices=device_refs,
        hidden_dim=HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=NUM_EXPERTS_PER_TOKEN,
        moe_dim=MOE_DIM,
        has_shared_experts=True,
        shared_experts_dim=MOE_DIM,
        dtype=DType.uint8,
        quant_config=fp4_config,
    )
    moe.sharding_strategy = ShardingStrategy.tensor_parallel(n_devices)
    tp_moe_shards = moe.shard(device_refs)
    moe.load_state_dict(wrapped_weights)

    allreduce = Allreduce(num_accelerators=n_devices)
    signals = Signals(device_refs)

    input_type = TensorType(
        DType.bfloat16, [SEQ_LEN, HIDDEN_DIM], device=DeviceRef.GPU()
    )

    with Graph(
        "TP_MoE_NVFP4",
        input_types=[input_type, *signals.input_types()],
    ) as graph:
        x = graph.inputs[0].tensor
        inputs = [x.to(DeviceRef.GPU(i)) for i in range(n_devices)]
        signal_buffers = [inp.buffer for inp in graph.inputs[1:]]
        outputs = [
            shard(inp) for shard, inp in zip(tp_moe_shards, inputs, strict=True)
        ]
        outputs = allreduce(outputs, signal_buffers)
        graph.output(*outputs)

    compiled = session.load(graph, weights_registry=moe.state_dict())

    hidden_states = torch.randn(
        SEQ_LEN, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda"
    )
    result = compiled.execute(
        Buffer.from_dlpack(hidden_states).to(devices[0]),
        *signals.buffers(),
    )
    all_outputs = from_dlpack(result[0]).to("cpu")

    # Run MoEGate separately to get topk indices/scores for reference.
    moe_gate = MoEGate(
        devices=[DeviceRef.GPU()],
        hidden_dim=HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=NUM_EXPERTS_PER_TOKEN,
        dtype=DType.bfloat16,
    )
    gate_weight_dict: dict[str, WeightData | torch.Tensor] = {
        "gate_score.weight": raw_weights["gate.gate_score.weight"]
    }
    moe_gate.load_state_dict(gate_weight_dict)

    gate_input_type = TensorType(
        DType.bfloat16, [SEQ_LEN, HIDDEN_DIM], device=DeviceRef.GPU()
    )
    with Graph("MoEGate", input_types=[gate_input_type]) as gate_graph:
        gate_out = moe_gate(gate_graph.inputs[0].tensor)
        gate_graph.output(*gate_out)

    gate_compiled = session.load(
        gate_graph, weights_registry=moe_gate.state_dict()
    )
    gate_result = gate_compiled.execute(
        Buffer.from_dlpack(hidden_states).to(devices[0])
    )
    topk_indices = from_dlpack(gate_result[0]).to("cuda")
    topk_scores = from_dlpack(gate_result[1]).to("cuda")

    # Dequantize FP4 weights to BF16 for torch reference.
    bf16_weights = _dequantize_all_weights(raw_weights, torch.device("cuda"))
    all_outputs = all_outputs.to("cuda")
    hidden_states_ref = hidden_states

    for tok_idx in range(SEQ_LEN):
        torch_output = torch_moe(
            hidden_states_ref[tok_idx : tok_idx + 1],
            bf16_weights,
            topk_indices[tok_idx : tok_idx + 1],
            topk_scores[tok_idx : tok_idx + 1],
        )
        cos_sim = torch.nn.functional.cosine_similarity(
            all_outputs[tok_idx : tok_idx + 1].float(),
            torch_output.float(),
            dim=-1,
        )
        assert cos_sim.min() > 0.98, (
            f"token {tok_idx}: cosine similarity"
            f" {cos_sim.min().item():.6f} < 0.98"
        )
