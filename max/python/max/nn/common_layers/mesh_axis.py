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

"""Device mesh axis names to use with specific type of parallelisms.

Used by layers in :mod:`max.nn.common_layers` to place weight
tensors in the mesh (set by :func:`~max.tensor.default_device`).

It imports nothing from :mod:`max.nn`, so the kernel wrappers in
:mod:`max.nn.kernels` can use it without a cycle through the layers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from max.tensor.sharding import DeviceMesh

#: Mesh axis name for tensor parallelism.
TP = "tp"

#: Mesh axis name for data parallelism.
DP = "dp"

#: Mesh axis name for expert parallelism.
EP = "ep"


def dcp_axis(mesh: DeviceMesh) -> int:
    """Returns the index of the mesh axis a DCP group spans.

    Decode context parallelism (DCP, see
    :mod:`max.nn.common_layers.context_parallel`) spans the ``tp`` axis, or
    the only axis of a one-axis mesh.

    Raises:
        NotImplementedError: If a mesh of several axes has no ``tp`` axis.
    """
    if TP in mesh.axis_names:
        return mesh.axis_names.index(TP)
    if mesh.ndim == 1:
        return 0
    raise NotImplementedError(
        f"decode context parallelism needs a {TP!r} mesh axis, got {mesh}."
    )


__all__ = ["DP", "EP", "TP", "dcp_axis"]
