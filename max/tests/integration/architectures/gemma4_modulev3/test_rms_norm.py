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
"""Tests for the Gemma4 ModuleV3 RMSNorm layer."""

from __future__ import annotations

import pytest
import torch
from conftest import (
    TEXT_HEAD_DIM,
    TEXT_HIDDEN_SIZE,
    TEXT_RMS_NORM_EPS,
    TorchGemma4RMSNorm,
)
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import default_dtype
from max.graph import DeviceRef, TensorType
from max.pipelines.architectures.gemma4_modulev3.layers.rms_norm import (
    Gemma4RMSNorm,
)

TORCH_DTYPE = torch.bfloat16
MAX_DTYPE = DType.bfloat16


def _build_and_run(
    dim: int,
    with_weight: bool,
    weights: dict[str, torch.Tensor],
    x: torch.Tensor,
) -> Buffer:
    """Compile the ModuleV3 Gemma4RMSNorm, execute it, and return output."""
    device = Accelerator(0)
    with F.lazy(), default_dtype(MAX_DTYPE):
        norm = Gemma4RMSNorm(
            dim=dim, eps=TEXT_RMS_NORM_EPS, with_weight=with_weight
        )
        norm.to(device)

    input_type = TensorType(MAX_DTYPE, tuple(x.shape), device=DeviceRef.GPU())
    compiled = norm.compile(input_type, weights=weights)
    x_gpu = Buffer.from_dlpack(x).to(device)
    result = compiled.execute_raw(x_gpu)[0]
    assert isinstance(result, Buffer)
    return result


def _assert_close(expected: torch.Tensor, actual: Buffer) -> None:
    rtol = 2e-2
    atol = 2 * torch.finfo(TORCH_DTYPE).eps
    torch.testing.assert_close(
        expected,
        torch.from_dlpack(actual).cpu(),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize(
    "dim,seq_len",
    [
        (TEXT_HIDDEN_SIZE, 1),
        (TEXT_HIDDEN_SIZE, 8),
        (TEXT_HEAD_DIM, 32),
    ],
    ids=["hidden_single_token", "hidden_short_seq", "head_dim_medium_seq"],
)
def test_with_weight_true_matches_reference(dim: int, seq_len: int) -> None:
    """Verify Gemma4RMSNorm with_weight=True matches the HF reference."""
    torch.manual_seed(42)

    weights = {"weight": torch.randn(dim, dtype=TORCH_DTYPE)}
    x = torch.randn(seq_len, dim, dtype=TORCH_DTYPE)

    max_output = _build_and_run(dim, True, weights, x)

    ref = TorchGemma4RMSNorm(
        dim=dim,
        eps=TEXT_RMS_NORM_EPS,
        with_scale=True,
    )
    ref.load_state_dict({"weight": weights["weight"]})
    ref_output = ref(x).detach()

    _assert_close(ref_output, max_output)


@pytest.mark.parametrize(
    "dim,seq_len",
    [
        (TEXT_HIDDEN_SIZE, 1),
        (TEXT_HIDDEN_SIZE, 8),
        (TEXT_HEAD_DIM, 32),
    ],
    ids=["hidden_single_token", "hidden_short_seq", "head_dim_medium_seq"],
)
def test_with_weight_false_matches_reference(dim: int, seq_len: int) -> None:
    """Verify Gemma4RMSNorm with_weight=False matches the HF reference."""
    torch.manual_seed(42)

    x = torch.randn(seq_len, dim, dtype=TORCH_DTYPE)

    max_output = _build_and_run(dim, False, {}, x)

    ref = TorchGemma4RMSNorm(dim=dim, eps=TEXT_RMS_NORM_EPS, with_scale=False)
    ref_output = ref(x).detach()

    _assert_close(ref_output, max_output)


def test_with_weight_false_has_no_parameters() -> None:
    """Verify with_weight=False registers no parameter (nothing to load)."""
    with F.lazy():
        norm = Gemma4RMSNorm(dim=64, with_weight=False)
    assert [name for name, _ in norm.parameters] == []


def test_with_weight_true_has_weight_parameter() -> None:
    """Verify with_weight=True registers the weight parameter."""
    with F.lazy():
        norm = Gemma4RMSNorm(dim=64, with_weight=True)
    assert [name for name, _ in norm.parameters] == ["weight"]
