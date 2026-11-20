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
"""The data-parallel variable-row gather stages on virtual SM100 devices.

A data-parallel verify scores each replica's rows with a vocab-sharded head
that needs the whole batch on every device. Each replica gathers from its
own rows, in its local frame, and one allgather joins them.
"""

from __future__ import annotations

from typing import Any

from max.driver import (
    CPU,
    Accelerator,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.pipelines.speculative.driver import gather_variable_rows
from max.tensor import Tensor, compilation
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Replicated,
    Sharded,
    TensorLayout,
)

_DP = 2


def test_each_replicas_rows_join_the_whole_batch() -> None:
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(_DP)
    devices = tuple(Accelerator(i) for i in range(_DP))
    mesh = DeviceMesh(devices, (_DP,), ("dp",))
    replica_rows = DeviceMapping(mesh, (Sharded(0),))
    seen: dict[str, Any] = {}

    def gather(
        h: Tensor, input_row_offsets: Tensor, return_n_logits: Tensor
    ) -> tuple[Tensor, Tensor]:
        rows, offsets = gather_variable_rows(
            h,
            input_row_offsets,
            return_n_logits,
            mesh,
            data_parallel_degree=_DP,
        )
        seen["rows"] = rows.mapping
        seen["offsets_device"] = offsets.device
        return rows, offsets

    staged = compilation.stage(gather, signal_devices=devices)(
        TensorLayout(DType.bfloat16, ["rows", 8], replica_rows),
        TensorLayout(DType.uint32, ["offsets_len"], replica_rows),
        TensorLayout(DType.int64, [1], CPU()),
    )

    assert staged is not None
    assert all(isinstance(p, Replicated) for p in seen["rows"].placements)
    # The acceptance sampler reads the offsets beside the logits, on the
    # first device.
    assert seen["offsets_device"] == devices[0]
