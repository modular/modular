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
"""Model inputs for Kimi K3, on top of the generic Llama3 ones.

They add the EP exchange's real buffers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from max.driver import Buffer
from max.pipelines.architectures.llama3_modulev3.inputs import Llama3Inputs


@dataclass
class KimiK3Inputs(Llama3Inputs):
    """The generic Llama3 inputs plus the EP exchange's buffers."""

    ep_inputs: tuple[Buffer, ...] = field(default=(), kw_only=True)

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        """The step's buffers, plus the EP exchange's."""
        return (*super().buffers, *self.ep_inputs)
