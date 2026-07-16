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

"""Shared fixtures for the quant_layers lazy-trace tests.

The quant layers are traced under :func:`max.experimental.functional.lazy`
against a mocked accelerator, so they exercise GPU kernel dispatch without a
real device. ``mock_accelerator`` provides that stand-in device and
``fp8_quant_config`` provides the block-scaled FP8 config used to drive the
quantized code paths.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from max.driver import Device
from max.dtype import DType
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)

# FP8 block-scaled weight block size, matching the deepseekV3 FP8 checkpoint.
# Dimensions in the tests are multiples of this so the weight-scale grid
# divides evenly.
FP8_BLOCK_SIZE = (128, 128)


def _make_fake_gpu(id: int = 0) -> MagicMock:
    """A stand-in accelerator so lazy tracing routes to GPU kernels.

    The mock reports ``label == "gpu"`` so device dispatch picks the
    accelerator path without instantiating a real ``Accelerator``.
    """
    label = "gpu"
    fake = MagicMock(spec=Device)
    fake.id = id
    fake.label = label
    fake.__eq__ = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda other: (
            getattr(other, "id", None) == id
            and getattr(other, "label", None) == label
        )
    )
    fake.__hash__ = MagicMock(return_value=hash((id, label)))  # type: ignore[method-assign]
    return fake


@pytest.fixture
def mock_accelerator() -> Iterator[MagicMock]:
    """Patches ``Accelerator`` with a fake-GPU factory for the test body.

    Call the yielded mock to mint devices: ``mock_accelerator()`` for a single
    device, or ``mock_accelerator(0)`` / ``mock_accelerator(1)`` for distinct
    ids.
    """
    with patch("max.graph.type.Accelerator") as mock:
        mock.side_effect = _make_fake_gpu
        yield mock


@pytest.fixture
def fp8_quant_config() -> QuantConfig:
    """A block-scaled FP8 config matching the deepseekV3 FP8 checkpoint."""
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
            block_size=FP8_BLOCK_SIZE,
        ),
        mlp_quantized_layers={0},
        attn_quantized_layers={0},
        format=QuantFormat.BLOCKSCALED_FP8,
    )
