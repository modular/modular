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
"""Loads stacked routed-expert arrays into ``KimiK3LatentMoE``'s experts."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from max.driver import Device
from max.pipelines.architectures.kimi_k3_modulev3.layers.moe_latent import (
    KimiK3LatentMoE,
)
from max.tensor import Tensor


def assign_experts(
    layer: KimiK3LatentMoE,
    gate_up: np.ndarray,
    down: np.ndarray,
    make: Callable[[np.ndarray, Device], Tensor],
    leaf: str = "weight",
) -> None:
    """Splits stacked arrays across ``layer``'s per-expert ``leaf`` tensors.

    Args:
        layer: The block whose experts receive the arrays.
        gate_up: ``[E, 2 * I, K]``, each expert's gate rows then its up rows.
        down: ``[E, N, K]``.
        make: Builds one expert's tensor on the device its parameter lives
            on, the host under a multi-device mesh.
        leaf: ``weight`` or ``weight_scale``.
    """
    gate, up = np.split(gate_up, 2, axis=1)
    for e, expert in enumerate(layer.experts):
        for w, array in (("w1", gate[e]), ("w3", up[e]), ("w2", down[e])):
            projection = getattr(expert, w)
            device = getattr(projection, leaf).device
            setattr(projection, leaf, make(np.ascontiguousarray(array), device))
