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

"""Distributed tensors and the eager runtime that realizes them.

:class:`Tensor` is the user-facing value type. Ops on tensors live in
:mod:`max.functional`; placements and the device mesh live in
:mod:`max.tensor.sharding`; ahead-of-time compilation of tensor programs lives
in :mod:`max.tensor.compilation`.

The following example calls ``Tensor.ones()`` to build a 4x8 tensor and
passes it to ``F.matmul`` along with its transpose:

.. code-block:: python

    from max import functional as F
    from max.tensor import Tensor

    x = Tensor.ones((4, 8))
    y = F.matmul(x, x.T)
"""

# TODO(MXF-897): Remove once device discovery stops depending on import
# order. A plugin-backed accelerator only registers if max.engine loaded
# before this package first touched the runtime. Until the driver loads what
# it needs itself, import max.engine first to keep that order.
from max import engine as _engine

from .tensor import (
    Tensor,
    default_device,
    default_dtype,
    defaults,
    defaults_like,
)

__all__ = [
    "Tensor",
    "default_device",
    "default_dtype",
    "defaults",
    "defaults_like",
]
