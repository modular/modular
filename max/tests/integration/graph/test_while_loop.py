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
"""Test while loop."""

import os
from pathlib import Path

import numpy as np
import pytest
from max.driver import Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    ops,
)

device_ref = DeviceRef.GPU() if accelerator_count() > 0 else DeviceRef.CPU()


def test_while_loop(session: InferenceSession) -> None:
    with Graph(
        "while_loop",
        input_types=[TensorType(DType.int32, [], device=device_ref)],
    ) as graph:
        x = graph.inputs[0]

        def pred_fn(x: TensorValue) -> TensorValue:
            return ops.transfer_to(x < 10, DeviceRef.CPU())

        def body_fn(x: TensorValue) -> TensorValue:
            return x + 1

        results = ops.while_loop(x, pred_fn, body_fn)
        graph.output(results[0])

    compiled = session.load(graph)
    result = compiled.execute(0)
    assert isinstance(result[0], Buffer)
    assert result[0].to_numpy() == 10


def test_while_loop_lambda(session: InferenceSession) -> None:
    with Graph(
        "while_loop_lambda",
        input_types=[TensorType(DType.int32, [], device=device_ref)],
    ) as graph:
        x = graph.inputs[0]
        results = ops.while_loop(
            x,
            lambda x: ops.transfer_to(x < 10, DeviceRef.CPU()),
            lambda x: x + 1,
        )
        graph.output(results[0])

    compiled = session.load(graph)
    result = compiled.execute(0)
    assert isinstance(result[0], Buffer)
    assert result[0].to_numpy() == 10


def test_while_loop_body_with_multiple_args(session: InferenceSession) -> None:
    with Graph(
        "while_loop_lambda_with_multiple_args",
        input_types=[
            TensorType(DType.int32, [], device=device_ref),
            TensorType(DType.int32, [], device=device_ref),
        ],
    ) as graph:
        x, y = graph.inputs
        results = ops.while_loop(
            (x, y),
            lambda x, y: ops.transfer_to(x < 10 and y < 10, DeviceRef.CPU()),
            lambda x, y: [x + 1, y + 1],
        )
        graph.output(results[0], results[1])

    compiled = session.load(graph)
    result = compiled.execute(0, 0)
    assert isinstance(result[0], Buffer)
    assert result[0].to_numpy() == 10
    assert isinstance(result[1], Buffer)
    assert result[1].to_numpy() == 10


@pytest.mark.skipif(
    accelerator_count() == 0, reason="requires a GPU to test device check"
)
def test_while_loop_raises_on_gpu_pred() -> None:
    """ops.while_loop raises ValueError when the pred_fn returns a GPU tensor."""
    with pytest.raises(ValueError, match=r"ops\.while_loop"):
        with Graph(
            "while_loop_gpu_pred",
            input_types=[TensorType(DType.int32, [], device=DeviceRef.GPU())],
        ):
            x = Graph.current.inputs[0]
            ops.while_loop(
                x,
                lambda x: x < 10,  # returns GPU tensor, no transfer_to
                lambda x: x + 1,
            )


@pytest.fixture
def custom_ops_path() -> Path:
    return Path(os.environ["CUSTOM_OPS_PATH"])


@pytest.mark.skip(
    reason="Buffer operations are currently not supported in while loops"
)
def test_while_loop_inplace_user_supplied(
    custom_ops_path: Path, session: InferenceSession
) -> None:
    bt = BufferType(DType.float32, [2, 2], DeviceRef.CPU())

    with Graph("basic", input_types=[bt]) as graph:
        buffer: BufferValue = graph.inputs[0].buffer

        def pred_fn(_x: TensorValue) -> TensorValue:
            return buffer[0, 0] < 10

        def body_fn(_x: TensorValue) -> TensorValue:
            ops.inplace_custom(
                "mutable_test_op", device=buffer.device, values=[buffer]
            )
            return buffer[0, 0]

        _ = ops.while_loop(buffer[0, 0], pred_fn, body_fn)
        graph.output()

    compiled = session.load(graph, custom_extensions=[custom_ops_path])
    rawbuffer = np.ones((2, 2), dtype=np.float32)
    compiled.execute(Buffer.from_dlpack(rawbuffer))
    actual = np.array([[10, 1], [1, 1]], dtype=np.float32)
    np.testing.assert_equal(rawbuffer, actual)
