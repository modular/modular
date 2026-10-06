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
"""Tensor-parallel placement helpers for Nemotron-H layers."""

from __future__ import annotations

from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.sharding import NamedMapping
from max.experimental.tensor import Tensor


def shard_dim0(t: Tensor) -> Tensor:
    """Shards ``t`` on its first axis across its mesh's tensor-parallel axis."""
    return t.to(NamedMapping(t.mesh, (TP,) + (None,) * (t.rank - 1)))
