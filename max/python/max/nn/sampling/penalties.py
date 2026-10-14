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
"""OpenAI presence/frequency penalties over a CSR of generated-token counts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType, TensorValue, ops
from max.nn.kernels import (
    apply_packed_bitmask_with_penalties,
    apply_penalties_to_logits,
)


@dataclass(frozen=True)
class LogitPenalties:
    """Presence/frequency penalties, one CSR row per logit row.

    Logit row ``i`` owns ``data[offsets[i]:offsets[i+1]]``, ``[token, count]``
    pairs counted over the tokens generated before that row's position. A
    negative token is padding.
    """

    data: TensorValue
    offsets: TensorValue
    frequency: TensorValue
    presence: TensorValue

    @staticmethod
    def input_types(device: DeviceRef) -> list[TensorType]:
        """Declares the four inputs, in the order :meth:`from_inputs` reads."""
        return [
            TensorType(DType.int32, ["penalty_entries", 2], device=device),
            TensorType(DType.uint32, ["penalty_offsets"], device=device),
            TensorType(DType.float32, ["penalty_rows"], device=device),
            TensorType(DType.float32, ["penalty_rows"], device=device),
        ]

    @classmethod
    def from_inputs(cls, values: Sequence[TensorValue]) -> LogitPenalties:
        """Binds the four values :meth:`input_types` declared."""
        data, offsets, frequency, presence = values
        return cls(data, offsets, frequency, presence)

    def apply(self, logits: TensorValue) -> TensorValue:
        """Returns 2-D ``logits`` with the penalties subtracted.

        Costs a copy of the logits; prefer :meth:`apply_with_bitmask` wherever
        a grammar mask is applied anyway.
        """
        rows = logits.shape[0]
        buffer = ops.buffer_create(
            BufferType(logits.dtype, logits.shape, logits.device)
        )
        ops.buffer_store(buffer, logits)
        apply_penalties_to_logits(
            buffer,
            self.data,
            self.offsets,
            frequency_penalty=ops.rebind(self.frequency, [rows]),
            presence_penalty=ops.rebind(self.presence, [rows]),
        )
        return ops.buffer_load(buffer)

    def apply_with_bitmask(
        self, logits: TensorValue, packed: TensorValue, fill_val: float
    ) -> TensorValue:
        """Masks ``logits`` like ``apply_packed_bitmask`` and subtracts the
        penalties from the tokens the mask keeps, in the mask's own pass.
        """
        return apply_packed_bitmask_with_penalties(
            logits,
            packed,
            fill_val,
            self.data,
            self.offsets,
            self.frequency,
            self.presence,
        )
