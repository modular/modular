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
"""A chain-only `buffer_store` must not be dropped by advanced fusion."""

import max.driver as md
import numpy as np
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops


def test_chain_only_store_survives() -> None:
    """The value stored by one call must be the value the next call loads.

    The graph outputs the load taken *before* the store, so no graph result
    consumes the stored value -- the store is reachable only through the
    chain. Returning 0 on the second call means it was dropped.
    """
    device = DeviceRef.CPU()
    with Graph(
        "chain_only_store",
        input_types=[TensorType(DType.int32, [1], device=device)],
    ) as graph:
        buffer = ops.buffer_create(
            BufferType(DType.int32, [1], device=device), init_value=0
        )
        previous = ops.buffer_load(buffer)
        ops.buffer_store(buffer, graph.inputs[0].tensor)
        graph.output(previous)

    model = InferenceSession(devices=[md.CPU()]).load(graph)
    seven = md.Buffer.from_numpy(np.array([7], dtype=np.int32)).to(
        model.input_devices[0]
    )
    got = [
        model.execute(seven)[0].to(md.CPU()).to_numpy().item() for _ in range(2)
    ]
    assert got == [0, 7], f"store dropped: {got}"
