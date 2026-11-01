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
"""A captured device graph's memory must stay reserved while it can replay.

Replays reuse the device pointers recorded at capture, so no later
allocation may be handed that memory, and the VMM allocator must never
reclaim (unmap) it. The side-stream case covers ``ops.side_stream`` bodies,
which run on a separate stream view of the device context.

Runs under a small VMM budget (the BUILD target's env) so filling it is cheap.
After capture the test fills the free budget with sentinel buffers, replays,
and checks both the replay output and every sentinel. Memory the graph still
uses either lands under a sentinel (the replay overwrites it) or is reclaimed
for one (the replay faults with an illegal address).
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

N = 2048
SENTINEL_BYTES = 4 << 20
MAX_SENTINELS = 1024


@pytest.mark.parametrize("side_stream", [False, True])
def test_captured_graph_memory_survives_later_allocations(
    side_stream: bool,
) -> None:
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    device = Accelerator()
    with Graph(
        "captured_memory",
        input_types=[TensorType(DType.float32, [N, N], DeviceRef.GPU(0))],
    ) as graph:
        x = graph.inputs[0].tensor
        if side_stream:
            (side,) = ops.side_stream(
                [x],
                lambda xx: [((xx @ xx) @ xx) @ xx],
                result_types=[x.type],
                stream_id=1,
            )
            graph.output((x @ x) @ x + side)
        else:
            graph.output(((x @ x) @ x) @ x + 1.0)
    model = InferenceSession(devices=[device]).load(graph)

    host = np.full((N, N), 1.0 / N, dtype=np.float32)
    x_dev = Buffer.from_numpy(host).to(device)
    four = host @ host @ host @ host
    expected = host @ host @ host + four if side_stream else four + 1.0

    (captured,) = model.capture(1, x_dev)
    model.replay(1, x_dev)
    np.testing.assert_allclose(captured.to_numpy(), expected, rtol=1e-5)

    pattern = np.full(SENTINEL_BYTES, 0x5A, dtype=np.uint8)
    sentinels: list[Buffer] = []
    for _ in range(MAX_SENTINELS):
        try:
            sentinels.append(Buffer.from_numpy(pattern).to(device))
        except Exception:
            # Out of budget: every free byte now holds a sentinel.
            break
    device.synchronize()
    assert sentinels, "could not allocate any sentinel buffer"

    model.replay(1, x_dev)
    device.synchronize()

    np.testing.assert_allclose(captured.to_numpy(), expected, rtol=1e-5)
    clobbered = [
        i
        for i, sentinel in enumerate(sentinels)
        if not np.array_equal(sentinel.to_numpy(), pattern)
    ]
    assert not clobbered, (
        f"replay overwrote {len(clobbered)} of {len(sentinels)} buffers "
        "allocated after capture"
    )
