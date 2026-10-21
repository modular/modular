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
"""Regression test for a per-device result landing under the wrong device.

Needs 2 real, distinct devices -- on one device there's nothing to
mismatch, so the bug can't reproduce. See ``_join`` in ``dispatch.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Accelerator
from max.experimental import functional as F
from max.experimental.functional import transfer_to
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    Replicated,
    ShardingError,
)
from max.experimental.tensor import Tensor

MESH_2 = DeviceMesh(
    devices=(Accelerator(0), Accelerator(1)),
    mesh_shape=(2,),
    axis_names=("tp",),
)


def test_rejects_a_result_relocated_off_its_dispatched_device() -> None:
    x = transfer_to(
        Tensor(np.ones(4, dtype=np.float32)),
        DeviceMapping(MESH_2, (Replicated(),)),
    )

    def misplace(value: Tensor) -> Tensor:
        # Every call reports device 0, no matter which device ran it.
        return value.to(MESH_2.devices[0])

    with pytest.raises(ShardingError, match="landed on"):
        F.functional(misplace)(x)
