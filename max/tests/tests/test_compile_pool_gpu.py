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
"""``ProcessCompilePool`` workers compile for a GPU without opening it."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from max.driver import CPU, Accelerator, Buffer, DeviceSpec
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental.compile_pool import ProcessCompilePool
from max.graph import DeviceRef, Graph, TensorType

_GPU_DEVICE_NODES = ("/dev/nvidia", "/dev/kfd", "/dev/dri")


def _gpu_device_fds(pid: int) -> list[str]:
    fd_dir = Path(f"/proc/{pid}/fd")
    targets = (os.readlink(fd) for fd in fd_dir.iterdir())
    return [t for t in targets if t.startswith(_GPU_DEVICE_NODES)]


def test_workers_compile_without_opening_the_gpu() -> None:
    session = InferenceSession(devices=[Accelerator()])
    with Graph(
        "scale_3",
        input_types=[TensorType(DType.float32, [4], device=DeviceRef.GPU())],
    ) as graph:
        (x,) = graph.inputs
        graph.output(x.tensor * 3.0)

    with ProcessCompilePool(
        device_specs=[DeviceSpec.accelerator(), DeviceSpec.cpu()]
    ) as pool:
        compiled = pool.compile(graph).result(timeout=240)
        worker_fds = {
            pid: _gpu_device_fds(pid) for pid in pool._executor._processes
        }

    assert worker_fds
    assert all(not fds for fds in worker_fds.values()), worker_fds

    model = session.init(compiled)
    gpu_input = Buffer.from_numpy(np.ones(4, dtype=np.float32)).to(
        Accelerator()
    )
    (out,) = model(gpu_input)
    np.testing.assert_array_equal(
        out.to(CPU()).to_numpy(), np.full(4, 3.0, dtype=np.float32)
    )
