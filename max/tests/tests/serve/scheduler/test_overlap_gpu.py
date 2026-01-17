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


import numpy as np
from max.driver import Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, SymbolicDim, TensorType, ops


def alloc_pinned(
    device: Device, size: int, init_value: int | None = None
) -> Buffer:
    t = Buffer(dtype=DType.int8, shape=[size], device=device, pinned=True)
    if init_value is not None:
        t.to_numpy().fill(init_value)
    return t


def build_graph(device_ref: DeviceRef) -> Model:
    with Graph(
        "my_add_graph",
        input_types=[
            TensorType(DType.int8, [SymbolicDim("size")], device=device_ref),
            TensorType(DType.int8, [SymbolicDim("size")], device=device_ref),
        ],
    ) as graph:
        x, y = graph.inputs
        z = ops.add(x, y)
        graph.output(z)
    device = device_ref.to_device()
    session = InferenceSession(devices=[device])
    model = session.load(graph)
    return model


def test_overlap() -> None:
    """Test overlap of GPU and Python host code.

    Note that this test does not actually achieve overlap today since the driver
    does not yet expose events. Additionally, .to_numpy() implicitly synchronizes
    with the stream.

    As we enable more driver features, we should update this test to use them.
    """
    device = Accelerator()
    device_ref = DeviceRef.from_device(device)
    size = 10 * 1024 * 1024

    # Build and load simple graph
    model = build_graph(device_ref=device_ref)

    # Allocate pinned input tensors and initialize contents
    a_pinned = alloc_pinned(device=device, size=size, init_value=1)
    b_pinned = alloc_pinned(device=device, size=size, init_value=2)

    # Allocate empty output tensors
    c_pinned = alloc_pinned(device=device, size=size)
    d_pinned = alloc_pinned(device=device, size=size)

    # Run batch 1
    (c,) = model.execute(a_pinned.to(device), b_pinned.to(device))
    c_pinned.inplace_copy_from(c)
    # TODO: record event 1

    # Run batch 2
    (d,) = model.execute(c, c)
    d_pinned.inplace_copy_from(d)
    # TODO: record event 2

    # TODO: wait for event 1
    print("Contents of c:", c_pinned.to_numpy())
    assert np.all(c_pinned.to_numpy() == 3)

    # TODO: wait for event 2
    print("Contents of d:", d_pinned.to_numpy())
    assert np.all(d_pinned.to_numpy() == 6)
