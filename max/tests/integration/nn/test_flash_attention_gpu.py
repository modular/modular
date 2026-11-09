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
"""Checks `flash_attention` on GPU against a PyTorch reference.

Head dim 64 runs the attention kernel; head dim 32 runs separate ops. Each runs
in bfloat16 and, on the same values cast in the graph, in float32.
"""

import pytest
import torch
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn._graph_ops import flash_attention

BATCH, Q_LEN, K_LEN, HEADS = 2, 7, 11, 3
HEAD_DIMS = (32, 64)
DTYPES = (DType.bfloat16, DType.float32)

Inputs = dict[int, dict[str, torch.Tensor]]


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    q, k, v = (x.double() for x in (q, k, v))
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * q.shape[-1] ** -0.5
    if mask is not None:
        scores = scores + mask.double().unsqueeze(1)
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probs, v)


@pytest.fixture(scope="module")
def outputs() -> tuple[list[torch.Tensor], Inputs]:
    def tensor(dtype: DType, *shape: int) -> TensorType:
        return TensorType(dtype, shape, device=DeviceRef.GPU())

    input_types = []
    for d in HEAD_DIMS:
        input_types += [
            tensor(DType.bfloat16, BATCH, Q_LEN, HEADS, d),
            tensor(DType.bfloat16, BATCH, K_LEN, HEADS, d),
            tensor(DType.bfloat16, BATCH, K_LEN, HEADS, d),
        ]
    input_types.append(tensor(DType.float32, BATCH, 1, K_LEN))

    with Graph("flash_attention_gpu", input_types=input_types) as graph:
        values = [x.tensor for x in graph.inputs]
        key_mask = ops.broadcast_to(values[-1], [BATCH, Q_LEN, K_LEN])
        results = []
        for dtype in DTYPES:
            for i, d in enumerate(HEAD_DIMS):
                q, k, v = (x.cast(dtype) for x in values[3 * i : 3 * i + 3])
                results += [
                    flash_attention(q, k, v, d**-0.5),
                    flash_attention(q, k, v, d**-0.5, mask=key_mask),
                ]
        graph.output(*(r.cast(DType.float32) for r in results))

    torch.manual_seed(0)
    inputs: Inputs = {
        d: {
            "q": torch.randn(BATCH, Q_LEN, HEADS, d, dtype=torch.bfloat16),
            "k": torch.randn(BATCH, K_LEN, HEADS, d, dtype=torch.bfloat16),
            "v": torch.randn(BATCH, K_LEN, HEADS, d, dtype=torch.bfloat16),
        }
        for d in HEAD_DIMS
    }
    key_mask = torch.where(torch.rand(BATCH, 1, K_LEN) < 0.3, -10000.0, 0.0)
    for d in HEAD_DIMS:
        inputs[d]["mask"] = key_mask.expand(BATCH, Q_LEN, K_LEN)

    gpu = Accelerator()
    args = [
        Buffer.from_dlpack(inputs[d][name]).to(gpu)
        for d in HEAD_DIMS
        for name in ("q", "k", "v")
    ]
    args.append(Buffer.from_dlpack(key_mask).to(gpu))
    model = InferenceSession(devices=[gpu]).load(graph)
    return [
        torch.from_dlpack(r.to(CPU())) for r in model.execute(*args)
    ], inputs


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_matches_reference(
    outputs: tuple[list[torch.Tensor], Inputs],
    dtype: DType,
    head_dim: int,
    masked: bool,
) -> None:
    results, inputs = outputs
    x = inputs[head_dim]
    expected = _reference(
        x["q"], x["k"], x["v"], x["mask"] if masked else None
    ).float()
    index = DTYPES.index(dtype) * len(HEAD_DIMS) + HEAD_DIMS.index(head_dim)
    actual = results[2 * index + int(masked)]
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
