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

"""Reduce-scatter -> allgather round trips, the shape TP+EP decoder blocks run."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, Type, ops
from max.nn import Signals

NUM_GPUS = 4
HIDDEN = 8


def _expected(inputs: list[np.ndarray]) -> np.ndarray:
    return np.sum(inputs, axis=0).astype(np.float32)


def _inputs(seq_len: int) -> list[np.ndarray]:
    """Device ``d`` contributes ``10**d * (row + 1)``.

    Every device holds a distinct power of ten and every row a distinct digit
    position, so any permutation of ranks or rows in the round trip lands on
    values the correct answer never takes.
    """
    return [
        np.tile((np.arange(seq_len) + 1)[:, None], (1, HIDDEN)) * (10**i)
        for i in range(NUM_GPUS)
    ]


@pytest.mark.parametrize("seq_len", [7, 9, 13, 8])
def test_round_trip_preserves_row_order(seq_len: int) -> None:
    """The round trip rebuilds the rows in their original order.

    Reduce-scatter hands rank ``r`` the ``r``-th contiguous chunk and allgather
    concatenates the group's shards in rank order, so gathering a scattered
    tensor is the identity on row order -- including when the split is uneven
    and the last ranks carry one row fewer.
    """
    if NUM_GPUS > accelerator_count():
        pytest.skip(f"needs {NUM_GPUS} GPUs")

    device_refs = [DeviceRef.GPU(id=i) for i in range(NUM_GPUS)]
    signals = Signals(device_refs)
    host = CPU()
    devices = [Accelerator(i) for i in range(NUM_GPUS)]

    with Graph(
        f"rs_ag_{seq_len}",
        # https://github.com/python/mypy/issues/19413
        input_types=cast(
            list[Type[Any]],
            [
                TensorType(
                    dtype=DType.float32, shape=[seq_len, HIDDEN], device=device
                )
                for device in device_refs
            ]
            + list(signals.input_types()),
        ),
    ) as graph:
        hidden = [v.tensor for v in graph.inputs[:NUM_GPUS]]
        buffers = [v.buffer for v in graph.inputs[NUM_GPUS:]]
        shards = ops.reducescatter.sum(
            hidden, buffers, axis=0, group_size=NUM_GPUS
        )
        graph.output(
            *ops.allgather(shards, buffers, axis=0, group_size=NUM_GPUS)
        )

    compiled = InferenceSession(devices=[host, *devices]).load(graph)
    inputs = _inputs(seq_len)
    expected = _expected(inputs)
    tensors = [
        Buffer.from_numpy(a.astype(np.float32)).to(device)
        for a, device in zip(inputs, devices, strict=True)
    ]

    outputs = compiled.execute(*tensors, *signals.buffers())
    for output in outputs:
        assert isinstance(output, Buffer)
        assert np.array_equal(output.to(host).to_numpy(), expected)


def test_round_trip_with_symbolic_uneven_split() -> None:
    """A symbolic dim that reduce-scatter bins unevenly compiles and executes.

    Concat's shape inference used to mint fresh graph parameters while summing
    the gathered axis dims, and KGEN orders a commutative operand list by
    parameter name, so the term order of the sum drifted with the graph's
    parameter counter. Inference runs again when the op is verified and again
    when the graph is compiled, so a drifting order surfaced as ``'rmo.concat'
    op inferred type(s) ... are incompatible with return type(s)`` -- a build or
    compile failure, never wrong data. Stacked rounds are what exposed it.
    """
    if NUM_GPUS > accelerator_count():
        pytest.skip(f"needs {NUM_GPUS} GPUs")

    device_refs = [DeviceRef.GPU(id=i) for i in range(NUM_GPUS)]
    signals = Signals(device_refs)
    host = CPU()
    devices = [Accelerator(i) for i in range(NUM_GPUS)]

    with Graph(
        "rs_ag_symbolic",
        # https://github.com/python/mypy/issues/19413
        input_types=cast(
            list[Type[Any]],
            [
                TensorType(
                    dtype=DType.float32,
                    shape=["seq_len", HIDDEN],
                    device=device,
                )
                for device in device_refs
            ]
            + list(signals.input_types()),
        ),
    ) as graph:
        hidden = [v.tensor for v in graph.inputs[:NUM_GPUS]]
        buffers = [v.buffer for v in graph.inputs[NUM_GPUS:]]
        for _ in range(3):
            shards = ops.reducescatter.sum(
                hidden, buffers, axis=0, group_size=NUM_GPUS
            )
            gathered = ops.allgather(
                shards, buffers, axis=0, group_size=NUM_GPUS
            )
            hidden = [ops.rebind(g, ["seq_len", HIDDEN]) for g in gathered]
        graph.output(*hidden)

    compiled = InferenceSession(devices=[host, *devices]).load(graph)
    # 7 rows over 4 ranks bins as 2/2/2/1.
    inputs = _inputs(7)
    expected = _expected(inputs) * NUM_GPUS**2
    tensors = [
        Buffer.from_numpy(a.astype(np.float32)).to(device)
        for a, device in zip(inputs, devices, strict=True)
    ]

    outputs = compiled.execute(*tensors, *signals.buffers())
    for output in outputs:
        assert isinstance(output, Buffer)
        assert np.array_equal(output.to(host).to_numpy(), expected)
