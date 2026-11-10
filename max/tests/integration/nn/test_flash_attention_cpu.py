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
"""Checks `flash_attention` on CPU against a PyTorch reference."""

import numpy as np
import pytest
import torch
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn._graph_ops import flash_attention

BATCH, Q_LEN, K_LEN, HEADS, HEAD_DIM = 2, 7, 11, 3, 32
SCALE = HEAD_DIM**-0.5


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    # BSHD in and out; the mask broadcasts against [batch, heads, q, k].
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * SCALE
    if mask is not None:
        scores = scores + mask
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probs, v)


@pytest.fixture(scope="module")
def outputs() -> tuple[list[np.ndarray], dict[str, torch.Tensor]]:
    def tensor(*shape: int) -> TensorType:
        return TensorType(DType.float32, shape, device=DeviceRef.CPU())

    with Graph(
        "flash_attention_cpu",
        input_types=[
            tensor(BATCH, Q_LEN, HEADS, HEAD_DIM),
            tensor(BATCH, K_LEN, HEADS, HEAD_DIM),
            tensor(BATCH, K_LEN, HEADS, HEAD_DIM),
            tensor(BATCH, 1, K_LEN),
            tensor(BATCH, HEADS, Q_LEN, K_LEN),
        ],
    ) as graph:
        q, k, v, key_mask, head_mask = (x.tensor for x in graph.inputs)
        graph.output(
            flash_attention(q, k, v, SCALE),
            # A broadcast mask, as BERT builds it from its padding mask.
            flash_attention(
                q,
                k,
                v,
                SCALE,
                mask=ops.broadcast_to(key_mask, [BATCH, Q_LEN, K_LEN]),
            ),
            flash_attention(q, k, v, SCALE, mask=head_mask),
        )

    torch.manual_seed(0)
    inputs = {
        "q": torch.randn(BATCH, Q_LEN, HEADS, HEAD_DIM),
        "k": torch.randn(BATCH, K_LEN, HEADS, HEAD_DIM),
        "v": torch.randn(BATCH, K_LEN, HEADS, HEAD_DIM),
        "key_mask": torch.where(
            torch.rand(BATCH, 1, K_LEN) < 0.3, -10000.0, 0.0
        ),
        "head_mask": torch.randn(BATCH, HEADS, Q_LEN, K_LEN),
    }
    model = InferenceSession(devices=[CPU()]).load(graph)
    results = model.execute(
        *(Buffer.from_numpy(x.numpy()) for x in inputs.values())
    )
    return [r.to_numpy() for r in results], inputs


def test_no_mask(
    outputs: tuple[list[np.ndarray], dict[str, torch.Tensor]],
) -> None:
    results, x = outputs
    expected = _reference(x["q"], x["k"], x["v"], None)
    np.testing.assert_allclose(results[0], expected, rtol=1e-4, atol=1e-5)


def test_broadcast_mask(
    outputs: tuple[list[np.ndarray], dict[str, torch.Tensor]],
) -> None:
    results, x = outputs
    mask = x["key_mask"].unsqueeze(1)
    expected = _reference(x["q"], x["k"], x["v"], mask)
    np.testing.assert_allclose(results[1], expected, rtol=1e-4, atol=1e-5)


def test_per_head_mask(
    outputs: tuple[list[np.ndarray], dict[str, torch.Tensor]],
) -> None:
    results, x = outputs
    expected = _reference(x["q"], x["k"], x["v"], x["head_mask"])
    np.testing.assert_allclose(results[2], expected, rtol=1e-4, atol=1e-5)
