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
"""``mo.mhc.split_sinkhorn`` against a numpy transcription of the reference.

The reference is ``hc_split_sinkhorn_kernel`` in DeepSeek-V4's
``inference/kernel.py``: sigmoid pre/post, then a row softmax and alternating
column/row normalization of the ``hc x hc`` block.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, Device, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kernels import mhc_split_sinkhorn
from numpy.typing import NDArray

HC = 4
ITERS = 20
EPS = 1e-6
WIDTH = (2 + HC) * HC


def _sigmoid(x: NDArray[np.float32]) -> NDArray[np.float32]:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def reference(
    mixes: NDArray[np.float32],
    scale: NDArray[np.float32],
    base: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    pre = _sigmoid(mixes[:, :HC] * scale[0] + base[:HC]) + np.float32(EPS)
    post = 2.0 * _sigmoid(mixes[:, HC : 2 * HC] * scale[1] + base[HC : 2 * HC])
    comb = (mixes[:, 2 * HC :] * scale[2] + base[2 * HC :]).reshape(-1, HC, HC)
    comb = np.exp(comb - comb.max(axis=-1, keepdims=True))
    comb = comb / comb.sum(axis=-1, keepdims=True) + np.float32(EPS)
    comb = comb / (comb.sum(axis=-2, keepdims=True) + np.float32(EPS))
    for _ in range(ITERS - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + np.float32(EPS))
        comb = comb / (comb.sum(axis=-2, keepdims=True) + np.float32(EPS))
    return pre, post.astype(np.float32), comb.astype(np.float32)


def _devices() -> list[str]:
    return ["cpu", "gpu"] if accelerator_count() > 0 else ["cpu"]


@pytest.fixture(scope="module", params=_devices())
def compiled(request: pytest.FixtureRequest) -> tuple[Device, Model]:
    device: Device = CPU() if request.param == "cpu" else Accelerator()
    ref = DeviceRef.from_device(device)
    graph = Graph(
        "mhc_split_sinkhorn",
        input_types=[
            TensorType(DType.float32, ["tokens", WIDTH], ref),
            TensorType(DType.float32, [3], ref),
            TensorType(DType.float32, [WIDTH], ref),
        ],
    )
    with graph:
        mixes, scale, base = (v.tensor for v in graph.inputs)
        graph.output(
            *mhc_split_sinkhorn(
                mixes,
                scale,
                base,
                hc_mult=HC,
                sinkhorn_iters=ITERS,
                eps=EPS,
            )
        )
    session = InferenceSession(devices=[device])
    return device, session.load(graph)


@pytest.mark.parametrize("tokens", [1, 7, 300])
def test_matches_reference(compiled: tuple[Device, Model], tokens: int) -> None:
    device, model = compiled
    rng = np.random.default_rng(tokens)
    mixes = rng.normal(0.0, 2.0, (tokens, WIDTH)).astype(np.float32)
    scale = rng.uniform(0.5, 2.0, 3).astype(np.float32)
    base = rng.normal(0.0, 1.0, WIDTH).astype(np.float32)

    outs = model.execute(
        *(Buffer.from_numpy(a).to(device) for a in (mixes, scale, base))
    )
    got = [o.to(CPU()).to_numpy() for o in outs]
    want = reference(mixes, scale, base)

    for name, g, w in zip(("pre", "post", "comb"), got, want, strict=True):
        assert g.shape == w.shape, name
        np.testing.assert_allclose(g, w, rtol=2e-5, atol=1e-6, err_msg=name)
    # Sinkhorn's fixed point: columns of the last (column) pass sum to ~1.
    np.testing.assert_allclose(got[2].sum(axis=-2), 1.0, atol=1e-4)
