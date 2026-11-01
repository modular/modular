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
"""A host-gated copy on a side stream, replayed from a captured device graph.

The ``ops.side_stream`` body waits on a host flag and copies a pinned buffer
into a device scratch, while the main stream runs a long matmul chain. Nothing
in the body reads its inputs; its ops are ordered only by the graph's chain.
The captured graph must record the wait, keep the copy behind it, and put both
on a branch beside the matmuls rather than in line with them.

The capture writes its DOT dump to a temporary directory set here before any
device stream exists, so this file has its own test target.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from max.driver import Accelerator, Buffer, CompletionFlag, Usage
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    BufferType,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    ops,
)
from max.nn.legacy.kernels import inplace_memcpy, wait_host_value_with_dep

_DOT_DIR = Path(tempfile.mkdtemp(dir=os.environ.get("TEST_TMPDIR")))
os.environ["MODULAR_DEVICE_CONTEXT_LOG_GRAPH_CAPTURE"] = str(_DOT_DIR)

N = 2048
CHAIN = 24
# The M3.1 decode grammar bitmask: 576 rows of 6252 packed int32 words, 14 MB.
MASK_SHAPE = [576, 6252]
STREAM_ID = 2


def _build(device: DeviceRef) -> Graph:
    with Graph(
        "side_stream_host_wait",
        input_types=[
            TensorType(DType.float32, [N, N], device=device),
            TensorType(DType.int32, MASK_SHAPE, device=DeviceRef.CPU()),
            BufferType(DType.int64, [2], device=DeviceRef.CPU()),
            BufferType(DType.int32, MASK_SHAPE, device=device),
        ],
    ) as graph:
        x = graph.inputs[0].tensor
        pinned = graph.inputs[1].tensor
        payload = graph.inputs[2].buffer
        scratch = graph.inputs[3].buffer

        def copy_mask() -> list[TensorValue]:
            wait_host_value_with_dep(payload, scratch, device=device)
            inplace_memcpy(scratch, pinned)
            return [scratch[0:1, 0:1].cast(DType.float32)]

        (mask,) = ops.side_stream(
            [],
            copy_mask,
            result_types=[TensorType(DType.float32, [1, 1], device=device)],
            stream_id=STREAM_ID,
        )
        y = x
        for _ in range(CHAIN):
            y = y @ x
        # The mask's first consumer also reads the chain, so the main stream
        # waits for the side stream only once the chain is enqueued.
        graph.output(y[0:1, 0:1] + mask)
    return graph


def _node_types(dot: str) -> dict[str, str]:
    """Maps each node of a CUDA graph DOT dump to its type, such as KERNEL."""
    return dict(
        re.findall(r'"(graph_\d+_node_\d+)"\[[^\n]*label="\{\s*(\w+)', dot)
    )


def _unordered_with(dot: str, types: dict[str, str], node: str) -> int:
    """Counts the nodes that run neither before nor after `node`."""
    succ: dict[str, list[str]] = {node: [] for node in types}
    pred: dict[str, list[str]] = {node: [] for node in types}
    for src, dst in re.findall(
        r'"(graph_\d+_node_\d+)"\s*->\s*"(graph_\d+_node_\d+)"', dot
    ):
        succ[src].append(dst)
        pred[dst].append(src)

    def reach(start: str, edges: dict[str, list[str]]) -> set[str]:
        seen: set[str] = set()
        stack = [start]
        while stack:
            for nxt in edges[stack.pop()]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    ordered = reach(node, succ) | reach(node, pred) | {node}
    return len(types) - len(ordered)


def test_side_stream_host_wait_under_capture() -> None:
    accelerator = Accelerator()
    if accelerator.api != "cuda":
        pytest.skip("the host wait is recorded on CUDA only")
    device = DeviceRef.from_device(accelerator)
    model = InferenceSession(devices=[accelerator]).load(_build(device))

    flag = CompletionFlag(accelerator)
    payload = Buffer(
        dtype=DType.int64,
        shape=(2,),
        device=accelerator,
        usage=Usage.STAGING | Usage.UNTRACKED,
    )
    payload.to_numpy()[:] = [flag._unsafe_ptr, 1]
    pinned = Buffer(
        dtype=DType.int32,
        shape=MASK_SHAPE,
        device=accelerator,
        usage=Usage.STAGING | Usage.UNTRACKED,
    )
    scratch = Buffer(dtype=DType.int32, shape=MASK_SHAPE, device=accelerator)
    x = Buffer.from_numpy(np.full((N, N), 1.0 / N, dtype=np.float32)).to(
        accelerator
    )
    # Every entry of a product of constant 1 / N matrices is 1 / N.
    chain_value = 1.0 / N

    pinned.to_numpy()[...] = 0
    pinned.to_numpy()[0, 0] = 7
    flag.signal(1)
    (out,) = model.execute(x, pinned, payload, scratch)
    assert float(out.to_numpy()[0, 0]) == pytest.approx(7 + chain_value)

    (captured,) = model.capture(1, x, pinned, payload, scratch)
    accelerator.synchronize()
    times = []
    for _ in range(20):
        start = time.perf_counter()
        model.replay(1, x, pinned, payload, scratch)
        accelerator.synchronize()
        times.append(time.perf_counter() - start)
    assert float(captured.to_numpy()[0, 0]) == pytest.approx(7 + chain_value)
    print(f"replay median {sorted(times)[len(times) // 2] * 1e3:.3f} ms")

    # The replay may not copy the mask before the host signals the flag.
    flag.signal(0)
    model.replay(1, x, pinned, payload, scratch)
    time.sleep(0.05)
    pinned.to_numpy()[0, 0] = 12
    flag.signal(1)
    accelerator.synchronize()
    assert float(captured.to_numpy()[0, 0]) == pytest.approx(12 + chain_value)

    (dot_file,) = sorted(_DOT_DIR.glob("cuda_graph_*.dot"))
    dot = dot_file.read_text()
    types = _node_types(dot)
    kinds = Counter(types.values())
    print(f"captured nodes: {dict(kinds)}")
    copies = [node for node, kind in types.items() if kind == "MEMCPY"]
    assert len(copies) == 1, f"expected one copy node, got {dict(kinds)}"
    unordered = _unordered_with(dot, types, copies[0])
    # Every matmul runs beside the copy, not before or after it.
    assert unordered >= CHAIN, (
        f"only {unordered} nodes run beside the copy; the captured graph "
        f"serializes it with the matmul chain ({dot_file})"
    )
