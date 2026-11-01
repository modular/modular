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

They add the host MLA prefill length and the EP exchange's real buffers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from max.driver import Buffer
from max.pipelines.architectures.llama3_modulev3.inputs import Llama3Inputs


@dataclass
class KimiK3Inputs(Llama3Inputs):
    """The ragged inputs, host MLA prefill length, and EP exchange buffers."""

    batch_context_length: Buffer | None = field(default=None, kw_only=True)
    ep_inputs: tuple[Buffer, ...] = field(default=(), kw_only=True)

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        """Returns buffers in the model's compile order."""
        prefill_length = (
            ()
            if self.batch_context_length is None
            else (self.batch_context_length,)
        )
        return (*super().buffers, *prefill_length, *self.ep_inputs)
