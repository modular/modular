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

import os

import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_api, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    DeviceRef,
    Graph,
    Shape,
    ShardingStrategy,
    TensorType,
    TensorValue,
)
from max.graph.weights import WeightData
from max.nn.comm.ep import EPBatchManager, EPCommInitializer, EPConfig
from max.nn.moe import MoEGate, MoEQuantized
from max.nn.moe.expert_parallel import forward_moe_sharded_layers
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from test_common.graph_utils import is_b100_b200

MOE_DIM = 2048
HIDDEN_DIM = 7168
NUM_EXPERTS = 64

_FLOAT4_EXP_BIAS = 1
_FLOAT4_MANTISSA_BITS = 1
_FP4_MAX = 6.0
_BF16_MANTISSA_BITS = 7
_BF16_EXP_BITS = 8
_BF16_EXP_BIAS = 127


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


@pytest.mark.skipif(
    accelerator_api() == "hip", reason="FP4 kernel only supports Nvidia GPUs"
)
@pytest.mark.skipif(
    not is_b100_b200(),
    reason="FP4 kernel requires B100 or B200",
)
@pytest.mark.parametrize("n_devices", [2])
def test_ep_moe_fp4(
    n_devices: int,
    moe_weights_fp4: dict[str, torch.Tensor],
) -> None:
    assert n_devices <= accelerator_count(), (
        "Devices are not enough to run EP test"
    )

    # Configuration parameters
    top_k = 8
    max_tokens_per_rank = 128
    dtype = DType.uint8

    # Copy weights to CPU for session.load (moe_weights_fp4 lives on GPU).
    moe_weights_fp4_cpu = {k: v.cpu() for k, v in moe_weights_fp4.items()}

    wrapped_moe_weights_fp4: dict[str, WeightData | torch.Tensor] = {}
    for key, value in moe_weights_fp4_cpu.items():
        if value.dtype == torch.float8_e4m3fn:
            wrapped_moe_weights_fp4[key] = WeightData(
                Buffer.from_dlpack(value.view(torch.uint8)).view(
                    DType.float8_e4m3fn
                ),
                key,
                DType.float8_e4m3fn,
                Shape(value.shape),
            )
        else:
            wrapped_moe_weights_fp4[key] = value

    # Initialize devices
    devices = [Accelerator(id) for id in range(n_devices)]
    devices_ref = [DeviceRef(d.label, d.id) for d in devices]
    session = InferenceSession(devices=devices)

    # Create fp4 config (NVFP4)
    fp4_input_config = InputScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        origin=ScaleOrigin.STATIC,
        dtype=DType.float32,
        block_size=(1, 16),
    )
    fp4_weight_config = WeightScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        dtype=DType.float8_e4m3fn,
        block_size=(1, 16),
    )
    fp4_config = QuantConfig(
        input_scale=fp4_input_config,
        weight_scale=fp4_weight_config,
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        embedding_output_dtype=None,
        format=QuantFormat.NVFP4,
    )

    # Create EP configuration
    ep_config = EPConfig(
        dispatch_dtype=dtype,
        combine_dtype=DType.bfloat16,
        hidden_size=HIDDEN_DIM,
        top_k=top_k,
        n_experts=NUM_EXPERTS,
        max_tokens_per_rank=max_tokens_per_rank,
        n_gpus_per_node=n_devices,
        n_nodes=int(os.environ.get("SHMEM_TOTAL_NODES", "1")),
        dispatch_quant_config=fp4_config,
        fused_shared_expert=True,
    )

    # Initialize EP communication
    ep_comm_init = EPCommInitializer(ep_config)
    ep_batch_manager = EPBatchManager(ep_config)

    # Create MoE module with EP support
    moe = MoEQuantized(
        devices=[DeviceRef.CPU()] + devices_ref,
        hidden_dim=HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=top_k,
        moe_dim=MOE_DIM,
        has_shared_experts=True,
        shared_experts_dim=MOE_DIM,
        ep_size=n_devices,
        dtype=dtype,
        apply_router_weight_first=False,
        ep_batch_manager=ep_batch_manager,
        quant_config=fp4_config,
    )
    moe.sharding_strategy = ShardingStrategy.expert_parallel(n_devices)
    moe_shards = moe.shard(devices_ref)

    # Load weights
    moe.load_state_dict(wrapped_moe_weights_fp4)

    # Initialize EP communication infrastructure
    ep_comm_init.ep_init(session)

    per_device_input_types: list[TensorType] = [
        TensorType(
            DType.bfloat16,
            (f"input_len_{i}", HIDDEN_DIM),
            DeviceRef.GPU(i),
        )
        for i in range(n_devices)
    ]
    input_lengths = torch.randint(1, max_tokens_per_rank, (n_devices,))
    per_device_inputs_torch = [
        torch.randn(
            input_lengths[i],
            HIDDEN_DIM,
            dtype=torch.bfloat16,
            device="cpu",
        )
        for i in range(n_devices)
    ]

    per_device_inputs = [
        Buffer.from_dlpack(input).to(devices[i])
        for i, input in enumerate(per_device_inputs_torch)
    ]

    with Graph(
        "EPMoE_FP4",
        input_types=[
            *per_device_input_types,
            *ep_batch_manager.input_types(),
        ],
    ) as graph:
        inputs_tensors = [x.tensor for x in graph.inputs[:n_devices]]

        ep_batch_manager.fetch_buffers(graph.inputs[n_devices:])

        outputs = forward_moe_sharded_layers(moe_shards, inputs_tensors)

        graph.output(*outputs)

    # Compile and execute MoE
    compiled = session.load(graph, weights_registry=moe.state_dict())
    result = compiled.execute(*per_device_inputs, *ep_comm_init.model_inputs())
    torch_result = [torch.from_dlpack(x).to("cpu") for x in result]

    all_outputs = torch.cat(torch_result, dim=0)
    assert not torch.any(torch.isnan(all_outputs)), (
        "MoE output should not contain NaN"
    )

    # Build and run MoEGate to obtain topk indices/scores for reference.
    moe_gate = MoEGate(
        devices=devices_ref,
        hidden_dim=HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=top_k,
        dtype=DType.bfloat16,
    )
    moe_gate.sharding_strategy = ShardingStrategy.replicate(n_devices)
    moe_gate_shards = moe_gate.shard(devices_ref)

    gate_weight_dict = {
        "gate_score.weight": moe_weights_fp4_cpu["gate.gate_score.weight"]
    }
    moe_gate.load_state_dict(gate_weight_dict)

    with Graph(
        "MoEGate",
        input_types=per_device_input_types,
    ) as gate_graph:
        gate_inputs = [x.tensor for x in gate_graph.inputs[:n_devices]]
        gate_outputs: list[TensorValue] = []
        for moe_gate_shard, inp in zip(
            moe_gate_shards, gate_inputs, strict=False
        ):
            gate_outputs.extend(moe_gate_shard(inp))
        gate_graph.output(*gate_outputs)

    gate_compiled = session.load(
        gate_graph, weights_registry=moe_gate.state_dict()
    )
    gate_result = gate_compiled.execute(*per_device_inputs)
    topk_idxs_weights = [torch.from_dlpack(x).to("cpu") for x in gate_result]

    # Dequantize FP4 -> BF16 on GPU for torch reference.
    # Pop GPU tensors as we go to stay within the PyTorch memory budget.
    gpu = moe_weights_fp4["gate.gate_score.weight"].device
    moe_weights_gpu: dict[str, torch.Tensor] = {}
    moe_weights_gpu["gate.gate_score.weight"] = moe_weights_fp4.pop(
        "gate.gate_score.weight"
    )
    weight_keys = [
        k
        for k in moe_weights_fp4
        if k.endswith(".weight") and moe_weights_fp4[k].dtype == torch.uint8
    ]
    for key in weight_keys:
        weight = moe_weights_fp4.pop(key)
        scale = moe_weights_fp4.pop(f"{key}_scale")
        scale_2 = moe_weights_fp4.pop(f"{key}_scale_2")
        moe_weights_fp4.pop(f"{key.replace('.weight', '.input_scale')}", None)
        moe_weights_gpu[key] = _dequantize_fp4_to_bf16(weight, scale, scale_2)
        del weight, scale, scale_2
    all_outputs = all_outputs.to(gpu)
    all_inputs = torch.cat(per_device_inputs_torch, dim=0).to(gpu)
    all_topk_idxs = torch.cat(topk_idxs_weights[::2], dim=0).to(gpu)
    all_topk_weights = torch.cat(topk_idxs_weights[1::2], dim=0).to(gpu)

    for tok_idx in range(all_inputs.shape[0]):
        torch_output = torch_moe(
            all_inputs[tok_idx : tok_idx + 1],
            moe_weights_gpu,
            all_topk_idxs[tok_idx : tok_idx + 1],
            all_topk_weights[tok_idx : tok_idx + 1],
        )
        cos_sim = torch.nn.functional.cosine_similarity(
            all_outputs[tok_idx : tok_idx + 1].float(),
            torch_output.float(),
            dim=-1,
        )
        assert cos_sim.min() > 0.995, (
            f"token {tok_idx}: cosine similarity {cos_sim.min().item():.6f} < 0.995"
        )
