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
"""Converts the speculative framework's values between graph values and tensors.

Temporary compatibility layer between eager Tensor and graph TensorValue,
as we migrate to using Tensor and ModuleV3 everywhere.
"""

from __future__ import annotations

from typing import Any

from max import tree
from max.experimental.tensor import Tensor
from max.graph import BufferValue, TensorValue


def _to_graph_value(value: Any) -> Any:
    if not isinstance(value, Tensor):
        return value
    graph_values = value.graph_values
    if len(graph_values) != 1:
        raise ValueError(
            "the speculative framework takes one single-device Tensor per"
            f" device, but got one distributed over {value.mapping}; pass its"
            " local_shards"
        )
    return graph_values[0]


def _to_tensor(value: Any) -> Any:
    if isinstance(value, (TensorValue, BufferValue)):
        return Tensor.from_graph_value(value)
    return value


def as_graph_values(value: Any) -> Any:
    """Replaces every ``Tensor`` in ``value`` with its graph value.

    A buffer stays a buffer, since kernels write the KV blocks in place.
    """
    return tree.map(
        _to_graph_value,
        value,
        leaf=lambda v: isinstance(v, Tensor) or not tree.is_node(v),
        shared=True,
    )


def as_tensors(value: Any) -> Any:
    """Wraps every graph value in ``value`` as a ``Tensor``.

    Must run inside the realization context of the graph being built.
    """
    return tree.map(_to_tensor, value, shared=True)
