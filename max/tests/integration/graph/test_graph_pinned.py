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

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, DevicePinnedBuffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType


def test_add_one_cpu_with_pinned_input() -> None:
    # We can pass a pinned tensor in place of a CPU graph input.
    with Graph(
        "add_one_cpu",
        input_types=[TensorType(DType.float32, [4], device=DeviceRef.CPU())],
    ) as graph:
        x = graph.inputs[0].tensor
        graph.output(x + 1)

    cpu = CPU()
    acc = Accelerator()
    session = InferenceSession(devices=[cpu, acc])
    model = session.load(graph)

    a = DevicePinnedBuffer(shape=(4,), dtype=DType.float32, device=acc)
    a.to_numpy().fill(42)

    (actual,) = model.execute(a)
    assert isinstance(actual, Buffer)
    assert not actual.pinned
    assert actual.device == CPU()

    expected = np.full((4,), 43, dtype=np.float32)
    assert np.allclose(actual.to_numpy(), expected)


def test_add_one_gpu_with_pinned_input_raises() -> None:
    # We cannot pass a pinned tensor in place of a GPU graph input.
    with Graph(
        "add_one_gpu",
        input_types=[TensorType(DType.float32, [4], device=DeviceRef.GPU())],
    ) as graph:
        x = graph.inputs[0].tensor
        graph.output(x + 1)

    cpu = CPU()
    acc = Accelerator()
    session = InferenceSession(devices=[cpu, acc])
    model = session.load(graph)

    a = DevicePinnedBuffer(shape=(4,), dtype=DType.float32, device=acc)
    a.to_numpy().fill(42)

    with pytest.raises(
        TypeError,
        match=(
            r"expected argument 0 to be a device tensor on Device\(type=gpu,id=0\) "
            r"but found pinned tensor on Device\(type=gpu,id=0\) instead\. "
            r"Pinned tensors can only be used in place of CPU graph inputs\."
        ),
    ):
        _ = model.execute(a)
