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
"""Numerical verification for :func:`kimi_k3_output_head`, single-device.

No existing V2 unit test for this layer to compare against directly (V2 has
no dedicated ``test_output_head*.py``), so this builds an independent numpy
reference instead: project the already-normed hidden state the layer
receives (``kimi_k3_attn_res_mix_norm`` upstream applied the final norm),
cast to fp32. The final norm itself is verified in
``test_attn_residual.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Accelerator
from max.dtype import DType
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.architectures.kimi_k3_modulev3.layers.output_head import (
    kimi_k3_output_head,
)
from max.tensor import Tensor, default_device


def _reference_project(
    h: np.ndarray,
    lm_head_weight: np.ndarray,
) -> np.ndarray:
    """The vocab projection of an already-normed hidden state.

    ``h`` is the bf16-rounded value actually fed to the op (see the tests
    below), so this reference's own arithmetic runs in float64 but starts
    from the SAME rounded input the op sees -- not from the pre-rounding
    float32 original, which would compare two different computations.
    """
    return h.astype(np.float64) @ lm_head_weight.astype(np.float64).T


@pytest.mark.compile_on_gpu("the interpreter has no gather with uint32 indices")
def test_last_token_matches_independent_reference() -> None:
    """``ReturnLogits.LAST_TOKEN`` (the default, and the common serving
    path): gathers each sequence's last row, projects, casts to fp32. The
    hidden state is already normed (the model-level mixture did that), so
    the raw ``h`` is what the head consumes."""
    rng = np.random.default_rng(0)
    hidden, vocab = 16, 37

    # Two sequences, ragged: lengths 3 and 5, total_seq_len = 8.
    input_row_offsets_np = np.array([0, 3, 8], dtype=np.uint32)
    total_seq_len = int(input_row_offsets_np[-1])

    h_np = rng.standard_normal((total_seq_len, hidden)).astype(np.float32)
    lm_head_np = (
        rng.standard_normal((vocab, hidden)) / np.sqrt(hidden)
    ).astype(np.float32)

    with default_device(Accelerator()):
        h = Tensor(h_np, dtype=DType.float32).cast(DType.bfloat16)
        lm_head_weight = Tensor(lm_head_np, dtype=DType.float32).cast(
            DType.bfloat16
        )
        input_row_offsets = Tensor(input_row_offsets_np, dtype=DType.uint32)
        return_n_logits = Tensor(
            np.array([1], dtype=np.int64), dtype=DType.int64
        )

        (last_logits,) = kimi_k3_output_head(
            h,
            input_row_offsets,
            return_n_logits,
            lm_head_weight,
            return_logits=ReturnLogits.LAST_TOKEN,
        )
        last_logits_np = last_logits.to_numpy()
        h_bf16_np = h.cast(DType.float32).to_numpy()
        lm_head_bf16_np = lm_head_weight.cast(DType.float32).to_numpy()

    assert last_logits.dtype == DType.float32
    assert last_logits.shape == [2, vocab]

    last_token_h = h_bf16_np[input_row_offsets_np[1:] - 1]
    expected = _reference_project(last_token_h, lm_head_bf16_np)
    rel_error = np.max(np.abs(last_logits_np - expected)) / max(
        np.max(np.abs(expected)), 1e-12
    )
    assert rel_error < 1e-2, f"last-token logits diverge: rel_error={rel_error}"


@pytest.mark.compile_on_gpu("the interpreter has no gather with uint32 indices")
def test_all_matches_independent_reference() -> None:
    """``ReturnLogits.ALL``: every token projected, not just the last one
    per sequence."""
    rng = np.random.default_rng(1)
    hidden, vocab = 12, 29
    total_seq_len = 6
    input_row_offsets_np = np.array([0, 2, 6], dtype=np.uint32)

    h_np = rng.standard_normal((total_seq_len, hidden)).astype(np.float32)
    lm_head_np = (
        rng.standard_normal((vocab, hidden)) / np.sqrt(hidden)
    ).astype(np.float32)

    with default_device(Accelerator()):
        h = Tensor(h_np, dtype=DType.float32).cast(DType.bfloat16)
        lm_head_weight = Tensor(lm_head_np, dtype=DType.float32).cast(
            DType.bfloat16
        )
        input_row_offsets = Tensor(input_row_offsets_np, dtype=DType.uint32)
        return_n_logits = Tensor(
            np.array([1], dtype=np.int64), dtype=DType.int64
        )

        _, logits, offsets = kimi_k3_output_head(
            h,
            input_row_offsets,
            return_n_logits,
            lm_head_weight,
            return_logits=ReturnLogits.ALL,
        )
        logits_np = logits.to_numpy()
        offsets_np = offsets.to_numpy()
        h_bf16_np = h.cast(DType.float32).to_numpy()
        lm_head_bf16_np = lm_head_weight.cast(DType.float32).to_numpy()

    assert logits.dtype == DType.float32
    assert logits.shape == [total_seq_len, vocab]
    np.testing.assert_array_equal(offsets_np, input_row_offsets_np)

    expected_all = _reference_project(h_bf16_np, lm_head_bf16_np)
    rel_error = np.max(np.abs(logits_np - expected_all)) / max(
        np.max(np.abs(expected_all)), 1e-12
    )
    assert rel_error < 1e-2, f"ALL-mode logits diverge: rel_error={rel_error}"
