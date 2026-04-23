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

Used by layers in :mod:`max.experimental.nn.common_layers` to tag weight tensors
with device placements.

The placements are used to automatically shard the weights when
Module.to(DeviceMesh(...)) is called, as long as the mesh has the appropriate
axis name(s).
"""

TP = "tp"  # Tensor parallelism.
DP = "dp"  # Data parallelism.

__all__ = ["DP", "TP"]
