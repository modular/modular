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
"""Tests for max.experimental.nn.Linear with a static FP8 quant_config."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.experimental.nn import Linear
from max.experimental.tensor import Tensor, TensorType
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)

IN_DIM = 256
OUT_DIM = 128
WEIGHT_SCALE = 0.02
INPUT_SCALE = 0.05


def _quant_config(
    granularity: ScaleGranularity = ScaleGranularity.TENSOR,
) -> QuantConfig:
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=granularity,
            origin=ScaleOrigin.STATIC,
            dtype=DType.float32,
        ),
        weight_scale=WeightScaleSpec(
            granularity=granularity, dtype=DType.float32
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        format=QuantFormat.COMPRESSED_TENSORS_FP8,
    )


def test_parameters_match_the_modelopt_layout() -> None:
    linear = Linear(IN_DIM, OUT_DIM, bias=False, quant_config=_quant_config())
    params = dict(linear.parameters)
    assert sorted(params) == ["input_scale", "weight", "weight_scale"]
    assert params["weight"].dtype == DType.float8_e4m3fn
    assert list(params["weight"].shape) == [OUT_DIM, IN_DIM]
    for scale in ("weight_scale", "input_scale"):
        assert params[scale].dtype == DType.float32
        assert params[scale].device == CPU()


@pytest.mark.parametrize("rows", [1, 64])
def test_matches_the_dequantized_reference(rows: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(rows, IN_DIM, dtype=torch.bfloat16)
    weight = (torch.randn(OUT_DIM, IN_DIM) * 4).to(torch.float8_e4m3fn)

    device = Accelerator()
    linear = Linear(IN_DIM, OUT_DIM, bias=False, quant_config=_quant_config())
    linear.to(device)
    weight_bytes = weight.view(torch.uint8).numpy()
    run = linear.compile(
        TensorType(DType.bfloat16, ["rows", IN_DIM], device),
        weights={
            "weight": Buffer.from_numpy(weight_bytes).view(
                DType.float8_e4m3fn, weight_bytes.shape
            ),
            "weight_scale": np.array(WEIGHT_SCALE, dtype=np.float32),
            "input_scale": np.array([INPUT_SCALE], dtype=np.float32),
        },
    )
    got = torch.from_dlpack(run(Tensor.from_dlpack(x).to(device))).cpu()

    x_fp8 = (x.float() / INPUT_SCALE).to(torch.float8_e4m3fn).float()
    want = (x_fp8 @ weight.float().T) * (INPUT_SCALE * WEIGHT_SCALE)
    torch.testing.assert_close(got.float(), want, rtol=2e-2, atol=2e-2)


def test_other_quant_configs_are_refused() -> None:
    with pytest.raises(NotImplementedError):
        Linear(
            IN_DIM,
            OUT_DIM,
            quant_config=_quant_config(ScaleGranularity.ROWWISE),
        )
