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

Runs under a small VMM budget (the BUILD target's env) so filling it is cheap.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops

ROWS = 256
WIDTH = 1024
WORKSPACE_MIB = 4
SENTINEL_BYTES = 4 << 20
MAX_SENTINELS = 1024


def _stream_workspace(x: TensorValue) -> TensorValue:
    """Allocates a workspace through the context of the stream it runs on."""
    return ops.custom(
        "stream_workspace",
        device=x.device,
        values=[x],
        out_types=[TensorType(DType.float32, x.shape, device=x.device)],
        parameters={"mib": WORKSPACE_MIB},
    )[0].tensor


@pytest.mark.parametrize("side_stream", [False, True])
def test_captured_graph_memory_survives_later_allocations(
    side_stream: bool,
) -> None:
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    device = Accelerator()
    with Graph(
        "captured_memory",
        input_types=[
            TensorType(DType.float32, [ROWS, WIDTH], DeviceRef.GPU(0))
        ],
        custom_extensions=[
            Path(os.environ["MODULAR_STREAM_WORKSPACE_OPS_PATH"])
        ],
    ) as graph:
        x = graph.inputs[0].tensor
        # On a side stream the capture allocates through that stream's view of
        # the device context, not through the base context.
        if side_stream:
            (staged,) = ops.side_stream(
                [x],
                lambda xx: [_stream_workspace(xx)],
                result_types=[x.type],
                stream_id=1,
            )
        else:
            staged = _stream_workspace(x)
        graph.output(x + staged)
    model = InferenceSession(devices=[device]).load(graph)

    host = np.random.default_rng(0).standard_normal(
        (ROWS, WIDTH), dtype=np.float32
    )
    x_dev = Buffer.from_numpy(host).to(device)
    expected = 2 * host

    (captured,) = model.capture(1, x_dev)
    model.replay(1, x_dev)
    np.testing.assert_allclose(captured.to_numpy(), expected, rtol=1e-6)

    # Replays reuse the pointers recorded at capture. Memory the graph still
    # uses either lands under a sentinel (the replay overwrites it) or is
    # reclaimed for one (the replay faults with an illegal address).
    pattern = np.full(SENTINEL_BYTES, 0x5A, dtype=np.uint8)
    host_pattern = Buffer.from_numpy(pattern)
    sentinels: list[Buffer] = []
    for _ in range(MAX_SENTINELS):
        try:
            sentinel = Buffer(DType.uint8, [SENTINEL_BYTES], device=device)
        except Exception:
            # Out of budget: every free byte now holds a sentinel.
            break
        sentinel.inplace_copy_from(host_pattern)
        sentinels.append(sentinel)
    device.synchronize()
    assert sentinels, "could not allocate any sentinel buffer"

    model.replay(1, x_dev)
    device.synchronize()

    np.testing.assert_allclose(captured.to_numpy(), expected, rtol=1e-6)
    clobbered = [
        i
        for i, sentinel in enumerate(sentinels)
        if not np.array_equal(sentinel.to_numpy(), pattern)
    ]
    assert not clobbered, (
        f"replay overwrote {len(clobbered)} of {len(sentinels)} buffers "
        "allocated after capture"
    )

    # Return this case's memory to the pool before the next case captures: a
    # capture cannot wait for frees still pending on the device.
    del sentinels, captured, model, x_dev
    device.synchronize()
