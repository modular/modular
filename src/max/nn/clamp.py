# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from max.graph import DeviceRef, TensorValue, ops


def clamp(x: TensorValue, min: float, max: float) -> TensorValue:
    """Clamps values in `x` to `[min, max]`"""
    return ops.min(
        ops.max(x, ops.constant(min, x.dtype, device=DeviceRef.CPU())),
        ops.constant(max, x.dtype, device=DeviceRef.CPU()),
    )
