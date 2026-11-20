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
"""Tests for the Gemma4 float16 FFN loss scale (MLP + post-FFN norm)."""

from __future__ import annotations

import torch
from conftest import TEXT_RMS_NORM_EPS
from max import functional as F
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.nn import Module
from max.nn.common_layers.mlp import MLP
from max.pipelines.architectures.gemma4_modulev3.layers.rms_norm import (
    Gemma4RMSNorm,
)
from max.tensor import Tensor, default_dtype

HIDDEN = 256
INTER = 512
ROWS = 4
LOSS_SCALE = 256.0


class _FfnTail(Module[[Tensor], Tensor]):
    """The dense MLP followed by the post-feedforward norm."""

    def __init__(self, up_scale: float, eps: float) -> None:
        super().__init__()
        self.mlp = MLP(
            HIDDEN, INTER, activation_function="gelu_tanh", up_scale=up_scale
        )
        self.norm = Gemma4RMSNorm(HIDDEN, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.mlp(x))


def _weights() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "mlp.gate_proj.weight": torch.randn(INTER, HIDDEN) / HIDDEN**0.5,
        "mlp.up_proj.weight": torch.randn(INTER, HIDDEN) / HIDDEN**0.5,
        "mlp.down_proj.weight": torch.randn(HIDDEN, INTER) / INTER**0.5,
        "norm.weight": 1.0 + 0.1 * torch.randn(HIDDEN),
    }


def _run_off_and_on(
    dtype: DType, weights: dict[str, torch.Tensor], x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Runs the tail with the scale off, then on."""
    device = Accelerator(0)
    compiled = []
    # Compile both before touching the device, so precompile records both.
    for up_scale in (1.0, 1.0 / LOSS_SCALE):
        with F.lazy(), default_dtype(dtype):
            tail = _FfnTail(up_scale, TEXT_RMS_NORM_EPS * up_scale**2)
            tail.to(device)
        compiled.append(
            tail.compile(
                TensorType(dtype, tuple(x.shape), device=DeviceRef.GPU()),
                weights=weights,
            )
        )
    x_gpu = Buffer.from_dlpack(x).to(device)
    outs = []
    for model in compiled:
        result = model.execute_raw(x_gpu)[0]
        assert isinstance(result, Buffer)
        outs.append(torch.from_dlpack(result).cpu().float())
    return outs[0], outs[1]


def _reference(
    weights: dict[str, torch.Tensor], x: torch.Tensor
) -> torch.Tensor:
    w = {k: v.float() for k, v in weights.items()}
    xf = x.float()
    gate = torch.nn.functional.gelu(
        xf @ w["mlp.gate_proj.weight"].T, approximate="tanh"
    )
    y = (gate * (xf @ w["mlp.up_proj.weight"].T)) @ w["mlp.down_proj.weight"].T
    ms = y.pow(2).mean(-1, keepdim=True)
    return y * torch.rsqrt(ms + TEXT_RMS_NORM_EPS) * w["norm.weight"]


def test_scaled_matches_unscaled_in_float32() -> None:
    """With the scale on, the post-FFN norm output is unchanged in float32."""
    weights = _weights()
    x = torch.randn(ROWS, HIDDEN)
    off, on = _run_off_and_on(DType.float32, weights, x)
    torch.testing.assert_close(on, off)


def test_scale_keeps_float16_ffn_in_range() -> None:
    """Activations large enough to overflow the float16 FFN stay finite."""
    weights = {k: v.half() for k, v in _weights().items()}
    # Gated products near 4e4 on average, so their peaks exceed 65504.
    x = (200.0 * torch.randn(ROWS, HIDDEN)).half()
    off, on = _run_off_and_on(DType.float16, weights, x)
    assert not torch.isfinite(off).all()
    torch.testing.assert_close(on, _reference(weights, x), rtol=2e-2, atol=2e-2)
