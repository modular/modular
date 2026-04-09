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
"""Tests for Gemma4 RMSNorm layer."""

from __future__ import annotations

import pytest
import torch
from conftest import (  # type: ignore[import-not-found]
    TEXT_HEAD_DIM,
    TEXT_HIDDEN_SIZE,
    TEXT_RMS_NORM_EPS,
    TorchGemma4RMSNorm,
)
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.pipelines.architectures.gemma4.layers.rms_norm import (
    Gemma4RMSNorm,
)

TORCH_DTYPE = torch.bfloat16
MAX_DTYPE = DType.bfloat16


def _build_and_run(
    norm: Gemma4RMSNorm,
    x: torch.Tensor,
) -> Buffer:
    """Build a MAX graph with Gemma4RMSNorm, execute it, and return output."""
    device = Accelerator(0)

    session = InferenceSession(devices=[device])

    with Graph(
        "gemma4_rms_norm_test",
        input_types=[
            TensorType(MAX_DTYPE, tuple(x.shape), device=DeviceRef.GPU()),
        ],
    ) as graph:
        (graph_input,) = graph.inputs
        assert isinstance(graph_input, TensorValue)
        graph.output(norm(graph_input))

    compiled = session.load(graph, weights_registry=norm.state_dict())
    x_gpu = Buffer.from_dlpack(x).to(device)
    (result,) = compiled.execute(x_gpu)
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

    # MAX implementation
    norm = Gemma4RMSNorm(
        dim=dim, dtype=MAX_DTYPE, eps=TEXT_RMS_NORM_EPS, with_weight=True
    )
    norm.load_state_dict(weights)
    max_output = _build_and_run(norm, x)

    # HF reference
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

    # MAX implementation (no weights to load for with_weight=False)
    norm = Gemma4RMSNorm(
        dim=dim, dtype=MAX_DTYPE, eps=TEXT_RMS_NORM_EPS, with_weight=False
    )
    max_output = _build_and_run(norm, x)

    # HF reference
    ref = TorchGemma4RMSNorm(dim=dim, eps=TEXT_RMS_NORM_EPS, with_scale=False)
    ref_output = ref(x).detach()

    _assert_close(ref_output, max_output)


def test_with_weight_false_has_empty_state_dict() -> None:
    """Verify with_weight=False produces an empty raw_state_dict (no weight)."""
    norm = Gemma4RMSNorm(dim=64, dtype=MAX_DTYPE, with_weight=False)
    assert norm.raw_state_dict() == {}


def test_with_weight_true_has_weight_in_state_dict() -> None:
    """Verify with_weight=True includes the weight in raw_state_dict."""
    norm = Gemma4RMSNorm(dim=64, dtype=MAX_DTYPE, with_weight=True)
    assert "weight" in norm.raw_state_dict()
