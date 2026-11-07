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
"""Gemma4 ModuleV3 NVFP4 MoE against a host dequantization of its checkpoint.

Guards the down projection's fold of each expert's up/gate global-scale ratio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import torch
from conftest import TorchGemma4RMSNorm
from max import functional as F
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.nn import Module
from max.nn.activation import GatedActivation
from max.nn.quant_config import QuantConfig
from max.pipelines.architectures.gemma4_modulev3.layers.moe import Gemma4MoE
from max.pipelines.weights.fp4_quantization import (
    MAX_E2M1,
    FP4Format,
    dequantize_nvfp4,
    encode_f32_to_e4m3,
    quantize_nvfp4,
    round_to_e2m1,
    unpack_e4m3_uint8_to_fp32,
)
from max.tensor import Tensor, default_device, default_dtype
from max.tensor.compilation import CompiledCallable
from max.tensor.sharding import DeviceMapping, DeviceMesh, Replicated
from numpy.typing import NDArray
from test_common.graph_utils import is_b100_b200

_NUM_EXPERTS = 8
_TOP_K = 2
_EPS = 1e-6

# The SM100 block-scaled grouped matmul's smallest tested widths.
_FP4_HIDDEN = 256
_FP4_MOE_DIM = 128
_FP4_BLOCK = 16
# Each expert's up/gate weight spread, roughly its global-scale ratio. The
# checkpoint spans 0.8 to 1.46; 0.5 and 2 make a missed ratio obvious.
_UP_OVER_GATE = (1.0, 0.5, 2.0, 0.8, 1.46, 0.5, 2.0, 0.8)
_DECODE_TOKENS = 1
_PREFILL_TOKENS = 12
# Relative to the batch norm. B200 measures 0.044 (decode) and 0.040
# (prefill), against 0.30 and 0.51 with the ratio missed.
_FP4_TOLERANCE = 0.07
# The largest E4M3 block scale, and the headroom modelopt-style calibration
# leaves below it.
_MAX_E4M3 = 448.0
_CALIBRATION_HEADROOM = 1.5

_GELU_TANH = GatedActivation("gelu_tanh")


def _bf16(*shape: int, std: float = 1.0, mean: float = 0.0) -> torch.Tensor:
    return (mean + std * torch.randn(*shape)).to(torch.bfloat16)


def _host(buffer: Buffer) -> torch.Tensor:
    return torch.from_dlpack(buffer.to(CPU()))


@pytest.fixture(scope="module")
def mesh() -> DeviceMesh:
    return DeviceMesh((Accelerator(),), (1,), ("tp",))


@dataclass(frozen=True)
class _FP4Weight:
    """One projection as the NVFP4 checkpoint stores it."""

    packed: NDArray[np.uint8]
    block_scales: NDArray[np.uint8]
    """E4M3 codes, ``[rows, cols // 16]``."""
    global_scale: float
    input_scale: float

    @classmethod
    def quantize(
        cls, weight: NDArray[np.float32], input_scale: float
    ) -> _FP4Weight:
        packed, scales, global_scale = quantize_nvfp4(weight, FP4Format.NVFP4)
        return cls(
            packed,
            scales.reshape(weight.shape[0], -1),
            float(global_scale),
            input_scale,
        )

    def dequantized(self) -> NDArray[np.float32]:
        return dequantize_nvfp4(
            self.packed,
            self.block_scales[..., None],
            self.global_scale,
            FP4Format.NVFP4,
        )

    def checkpoint(self, prefix: str) -> dict[str, Any]:
        return {
            f"{prefix}.weight": self.packed,
            f"{prefix}.weight_scale": Buffer.from_numpy(self.block_scales).view(
                DType.float8_e4m3fn
            ),
            f"{prefix}.weight_scale_2": np.array(self.global_scale, np.float32),
            f"{prefix}.input_scale": np.array(self.input_scale, np.float32),
        }


def _calibrated_input_scale(activations: NDArray[np.float32]) -> float:
    """The static activation scale modelopt calibrates, with headroom."""
    amax = float(np.abs(activations).max())
    return amax * _CALIBRATION_HEADROOM / (MAX_E2M1 * _MAX_E4M3)


def _fake_quantize(
    x: NDArray[np.float32], input_scale: float
) -> NDArray[np.float32]:
    """Quantizes rows to NVFP4 and back, as the grouped matmul's input."""
    rows, cols = x.shape
    blocks = x.reshape(rows, cols // _FP4_BLOCK, _FP4_BLOCK).astype(np.float32)
    amax = np.abs(blocks).max(axis=-1, keepdims=True)
    scale = unpack_e4m3_uint8_to_fp32(
        encode_f32_to_e4m3(amax / np.float32(MAX_E2M1 * input_scale))
    )
    step = scale * np.float32(input_scale)
    safe = np.where(step > 0, step, np.float32(1.0))
    codes = np.where(step > 0, round_to_e2m1(blocks / safe), np.float32(0.0))
    return (codes * step).reshape(rows, cols)


def _bf16_round(x: NDArray[np.float32]) -> NDArray[np.float32]:
    return torch.from_numpy(x).to(torch.bfloat16).float().numpy()


def _gelu_tanh(x: NDArray[np.float32]) -> NDArray[np.float32]:
    return torch.nn.functional.gelu(
        torch.from_numpy(x), approximate="tanh"
    ).numpy()


@dataclass(frozen=True)
class _FP4Expert:
    gate: _FP4Weight
    up: _FP4Weight
    down: _FP4Weight

    @property
    def up_over_gate(self) -> float:
        return self.up.global_scale / self.gate.global_scale

    def forward(self, h: NDArray[np.float32]) -> NDArray[np.float32]:
        """The expert on its BF16 input, quantized as the checkpoint says.

        Gate and up dequantize with their own global scales, unlike the
        module's fused weight, and round to bfloat16 where the kernels do.
        """
        h_q = _fake_quantize(h, self.gate.input_scale)
        gate = _bf16_round(h_q @ self.gate.dequantized().T)
        up = _bf16_round(h_q @ self.up.dequantized().T)
        act = _bf16_round(_gelu_tanh(gate) * up)
        return _fake_quantize(act, self.down.input_scale) @ (
            self.down.dequantized().T
        )


class _MoEHarness(Module[[Tensor], tuple[Tensor, ...]]):
    """Runs the MoE, and returns its experts' input and routing beside it."""

    def __init__(self, moe: Gemma4MoE, mesh: DeviceMesh) -> None:
        super().__init__()
        self.moe = moe
        self.mesh = mesh

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        x = x.to(self.mesh)
        idx, routed = self.moe.gate(x)
        out = F.transfer_to(
            self.moe(x), DeviceMapping(self.mesh, (Replicated(),))
        )
        h = self.moe.pre_expert_norm(x)
        device = self.mesh.devices[0]
        return tuple(t.to(device) for t in (out, h, idx, routed))


@dataclass(frozen=True)
class _FP4MoE:
    compiled: CompiledCallable[..., Any]
    tokens: torch.Tensor
    """``_DECODE_TOKENS`` then ``_PREFILL_TOKENS`` rows, in BF16."""
    experts: list[_FP4Expert]


@pytest.fixture(scope="module")
def fp4_moe(mesh: DeviceMesh) -> _FP4MoE:
    if not is_b100_b200():
        pytest.skip("NVFP4 grouped matmuls run on SM100 only")
    torch.manual_seed(3)
    rng = np.random.default_rng(3)

    def normal(*shape: int, std: float = 0.05) -> NDArray[np.float32]:
        return rng.normal(0.0, std, shape).astype(np.float32)

    tokens = _bf16(_DECODE_TOKENS + _PREFILL_TOKENS, _FP4_HIDDEN)
    norm_weight = _bf16(_FP4_HIDDEN, mean=1.0, std=0.1)
    norm = TorchGemma4RMSNorm(_FP4_HIDDEN, eps=_EPS)
    norm.load_state_dict({"weight": norm_weight})
    with torch.no_grad():
        h = norm(tokens).float().numpy()

    # Calibrated on every token; gate and up share one, as in the checkpoint.
    gate_up_input_scale = _calibrated_input_scale(h)
    experts: list[_FP4Expert] = []
    for ratio in _UP_OVER_GATE:
        gate = normal(_FP4_MOE_DIM, _FP4_HIDDEN)
        up = normal(_FP4_MOE_DIM, _FP4_HIDDEN, std=0.05 * ratio)
        down = normal(_FP4_HIDDEN, _FP4_MOE_DIM)
        act = _gelu_tanh(h @ gate.T) * (h @ up.T)
        experts.append(
            _FP4Expert(
                _FP4Weight.quantize(gate, gate_up_input_scale),
                _FP4Weight.quantize(up, gate_up_input_scale),
                _FP4Weight.quantize(down, _calibrated_input_scale(act)),
            )
        )

    weights: dict[str, Any] = {
        "moe.gate.gate_score.weight": _bf16(_NUM_EXPERTS, _FP4_HIDDEN, std=2.0),
        "moe.gate.scale": _bf16(_FP4_HIDDEN, mean=1.0, std=0.1),
        "moe.gate.per_expert_scale": _bf16(_NUM_EXPERTS, mean=1.0, std=0.2),
        "moe.pre_expert_norm.weight": norm_weight,
    }
    for e, expert in enumerate(experts):
        for name in ("gate", "up", "down"):
            projection = getattr(expert, name)
            weights |= projection.checkpoint(f"moe.experts.{e}.{name}_proj")

    quant_config = QuantConfig.nvfp4(
        mlp_quantized_layers=set(), attn_quantized_layers=set()
    )
    with F.lazy(), default_dtype(DType.bfloat16), default_device(mesh):
        moe = Gemma4MoE(
            _FP4_HIDDEN,
            _NUM_EXPERTS,
            _TOP_K,
            _FP4_MOE_DIM,
            _EPS,
            _GELU_TANH,
            quant_config=quant_config,
        )
        harness = _MoEHarness(moe, mesh)
    assert {name for name, _ in harness.parameters} == set(weights)
    # One graph serves the decode and the prefill row counts.
    compiled = harness.compile(
        TensorType(DType.bfloat16, ["tokens", _FP4_HIDDEN], DeviceRef.GPU()),
        weights=weights,
    )
    return _FP4MoE(compiled, tokens, experts)


@pytest.mark.parametrize(
    "rows",
    [
        slice(0, _DECODE_TOKENS),
        slice(_DECODE_TOKENS, _DECODE_TOKENS + _PREFILL_TOKENS),
    ],
    ids=["decode", "prefill"],
)
def test_nvfp4_moe_matches_dequantized_reference(
    fp4_moe: _FP4MoE, mesh: DeviceMesh, rows: slice
) -> None:
    """The experts match a host dequantization of their checkpoint tensors.

    The reference starts from the module's normed rows, since one ulp there
    can move an FP4 output by percents. Per-token rounding still differs by
    a few percent, so the bound is on the whole batch.
    """
    out, h_max, idx, routed = (
        _host(b)
        for b in fp4_moe.compiled.execute_raw(
            Buffer.from_dlpack(fp4_moe.tokens[rows]).to(mesh.devices[0])
        )
    )
    h = h_max.float().numpy()
    idx_np = idx.long().numpy()
    routed_np = routed.float().numpy()

    per_expert = np.stack([e.forward(h) for e in fp4_moe.experts])
    ratios = np.array([e.up_over_gate for e in fp4_moe.experts])
    tokens = np.arange(len(h))[:, None]
    contributions = routed_np[..., None] * per_expert[idx_np, tokens]
    expected = contributions.sum(axis=1)
    # What the output would be if the down projection ignored the ratio.
    unscaled = (contributions / ratios[idx_np][..., None]).sum(axis=1)

    def error(actual: NDArray[np.float32]) -> float:
        return float(
            np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        )

    got = error(out.float().numpy())
    missed = error(unscaled)
    print(f"\nrelative error {got:.4f}; with the ratio missed {missed:.4f}")
    assert len(np.unique(idx_np)) >= min(_TOP_K * len(h), _NUM_EXPERTS // 2)
    assert missed > 3 * _FP4_TOLERANCE, "the routed experts' ratios cancel"
    assert got < _FP4_TOLERANCE
