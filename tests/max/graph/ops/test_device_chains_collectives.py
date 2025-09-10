# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Tests for per-device chaining of collectives and transfers."""

from __future__ import annotations

import re

from max.dtype import DType
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops


def test_allreduce_per_device_counts_without_merge() -> None:
    t0 = TensorType(DType.float32, [4], device=DeviceRef.GPU(0))
    t1 = TensorType(DType.float32, [4], device=DeviceRef.GPU(1))
    sb0 = BufferType(DType.int64, [1], device=DeviceRef.GPU(0))
    sb1 = BufferType(DType.int64, [1], device=DeviceRef.GPU(1))

    with Graph(
        "allreduce_per_device",
        input_types=[t0, t1, sb0, sb1],
    ) as graph:
        x0, x1, s0, s1 = graph.inputs
        x0_t = x0.tensor
        x1_t = x1.tensor
        s0_b = s0.buffer
        s1_b = s1.buffer

        outs = ops.allreduce.sum([x0_t, x1_t], [s0_b, s1_b])
        graph.output(*outs)

    ir = str(graph)

    # Exactly one allreduce per device.
    allreduces = re.findall(r"mo\.distributed\.allreduce\.sum\(", ir)
    assert len(allreduces) == 2, ir

    # No multi-operand chain merges (device chains must remain independent).
    assert not re.search(r"mo\.chain\.create\([^)]*,[^)]*\)", ir), ir


def test_transfer_h2d_uses_root_chain_not_device_chain() -> None:
    # Setup per-device inputs and signals.
    t0 = TensorType(DType.float32, [4], device=DeviceRef.GPU(0))
    t1 = TensorType(DType.float32, [4], device=DeviceRef.GPU(1))
    sb0 = BufferType(DType.int64, [1], device=DeviceRef.GPU(0))
    sb1 = BufferType(DType.int64, [1], device=DeviceRef.GPU(1))

    with Graph(
        "allreduce_then_transfer",
        input_types=[t0, t1, sb0, sb1],
    ) as graph:
        x0, x1, s0, s1 = graph.inputs
        x0_t = x0.tensor
        x1_t = x1.tensor
        s0_b = s0.buffer
        s1_b = s1.buffer

        # Stage allreduce: one per device, each with its own chain.
        outs = ops.allreduce.sum([x0_t, x1_t], [s0_b, s1_b])

        # Stage H2D transfers that should use ONLY the graph root chain (not the
        # per-device chains produced by the allreduces).
        c = ops.constant(0, DType.float32, device=DeviceRef.CPU())
        t1_gpu = c.to(DeviceRef.GPU(1))
        t0_gpu = c.to(DeviceRef.GPU(0))

        graph.output(outs[0], outs[1], t0_gpu, t1_gpu)

    ir = str(graph)

    # Collect all allreduce out-chain SSA ids (second result of the op).
    # Pattern: %res, %ch = mo.distributed_allreduce_sum(...)
    allreduce_chain_ids = re.findall(
        r"%[A-Za-z0-9_]+,\s*(%[A-Za-z0-9_]+)\s*=\s*mo\.distributed\.allreduce\.sum\(",
        ir,
    )
    assert len(allreduce_chain_ids) == 2, ir

    # Capture transfer chains and destination device ids.
    # Pattern examples:
    #   rmo.mo.transfer[%ch1] %arg : ... to <"gpu", 0>
    transfer_info = re.findall(
        r"(?:rmo\.mo|mo)\.transfer\[(%[A-Za-z0-9_]+)\][^\n]*to\s+<\"gpu\",\s*([0-9]+)>",
        ir,
    )
    # Expect two transfers to GPU:1 then GPU:0 in this graph.
    assert len(transfer_info) >= 2, ir

    # Extract the two chain ids used by the transfers.
    transfer_chain_ids = [c for c, _ in transfer_info[:2]]

    # Each H2D transfer must NOT consume a chain produced by allreduce; it
    # should use the graph's root chain instead.
    for c in transfer_chain_ids:
        assert c not in allreduce_chain_ids, (c, allreduce_chain_ids, ir)


def test_transfer_d2d_merges_all_device_chains() -> None:
    # Two devices with per-device chains advanced by allreduce.
    devices = [DeviceRef.GPU(0), DeviceRef.GPU(1)]
    sb0 = BufferType(DType.int64, [1], device=devices[0])
    sb1 = BufferType(DType.int64, [1], device=devices[1])

    with Graph(
        "allreduce_then_d2d_transfer",
        input_types=[
            TensorType(DType.float32, [4], device=devices[0]),
            TensorType(DType.float32, [4], device=devices[1]),
            sb0,
            sb1,
        ],
    ) as graph:
        x0, x1, s0, s1 = graph.inputs
        x0_t = x0.tensor
        x1_t = x1.tensor
        s0_b = s0.buffer
        s1_b = s1.buffer

        outs = ops.allreduce.sum([x0_t, x1_t], [s0_b, s1_b])

        # Trigger a D2D transfer from GPU0 to GPU1, which should be chained
        # on the merge of both device chains.
        _ = outs[0].to(DeviceRef.GPU(1))
        graph.output(*outs)

    ir = str(graph)

    # Find a chain.create with multiple operands and verify a transfer uses it.
    match = re.search(
        r"(%[A-Za-z0-9_]+)\s*=\s*mo\.chain\.create\([^)]*,[^)]*\).*?mo\.transfer\[\1\]",
        ir,
        flags=re.DOTALL,
    )
    assert match, ir


def test_transfer_d2d() -> None:
    # Setup per-device inputs and signals.
    t0 = TensorType(DType.float32, [4], device=DeviceRef.GPU(0))
    t1 = TensorType(DType.float32, [4], device=DeviceRef.GPU(1))
    sb0 = BufferType(DType.uint8, [1], device=DeviceRef.GPU(0))
    sb1 = BufferType(DType.uint8, [1], device=DeviceRef.GPU(1))

    with Graph(
        "allreduce_then_transfer",
        input_types=[t0, t1, sb0, sb1],
    ) as graph:
        x0, x1, s0, s1 = graph.inputs
        x0_t = x0.tensor
        x1_t = x1.tensor
        s0_b = s0.buffer
        s1_b = s1.buffer

        # Stage allreduce: one per device, each with its own chain.
        outs = ops.allreduce.sum([x0_t, x1_t], [s0_b, s1_b])

        # Stage H2D transfers that should use ONLY the graph root chain (not the
        # per-device chains produced by the allreduces).
        c = ops.constant(0, DType.float32, device=DeviceRef.CPU())
        t0_gpu = c.to(DeviceRef.GPU(0))
        t1_gpu = t0_gpu.to(DeviceRef.GPU(1))

        graph.output(outs[0], outs[1], t0_gpu, t1_gpu)
