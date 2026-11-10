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
"""A split-K bf16 matmul captured into more device graphs than the split-K
reduction has counter slots, replayed after every capture, matches eager."""

from __future__ import annotations

import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

# On B200 this (N, K) has parallel split-K tuning rows for M in [1, 65).
N, K = 2304, 8192
# More captures than the 4 split-K counter slots per device; the first
# split-K launch of the process happens inside a capture.
MS = [2, 3, 8, 16, 33, 64]


def test_splitk_matmul_capture_replay_matches_eager() -> None:
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    device = Accelerator()
    session = InferenceSession(devices=[device])
    with Graph(
        "splitk_matmul_capture",
        input_types=[
            TensorType(DType.bfloat16, ["m", K], device=DeviceRef.GPU(0)),
            TensorType(DType.bfloat16, [N, K], device=DeviceRef.GPU(0)),
        ],
    ) as graph:
        x, w = (v.tensor for v in graph.inputs)
        graph.output(x @ ops.transpose(w, 0, 1))
    model = session.load(graph)

    gen = torch.Generator().manual_seed(0)
    w_host = (torch.randn(N, K, generator=gen) * 0.02).bfloat16()
    w_buf = Buffer.from_dlpack(w_host).to(device)
    xs = {m: torch.randn(m, K, generator=gen).bfloat16() for m in MS}
    x_bufs = {m: Buffer.from_dlpack(x).to(device) for m, x in xs.items()}

    captured = {m: model.capture(m, x_bufs[m], w_buf)[0] for m in MS}
    # Eager outputs are allocated after every capture and stay on the device
    # while the graphs replay, so a replay that writes memory its capture
    # released corrupts them.
    eager = {m: model.execute(x_bufs[m], w_buf)[0] for m in MS}
    # Reverse order: a graph whose workspace a later capture reused for its
    # output would overwrite that output after it was written.
    for m in reversed(MS):
        model.replay(m, x_bufs[m], w_buf)
    for m in MS:
        e = torch.from_dlpack(eager[m]).cpu()
        ref = (xs[m].float() @ w_host.float().T).bfloat16()
        torch.testing.assert_close(e, ref, atol=2e-2, rtol=2e-2)
        r = torch.from_dlpack(captured[m]).cpu()
        assert torch.equal(r, e), f"M={m}: replay != eager"
