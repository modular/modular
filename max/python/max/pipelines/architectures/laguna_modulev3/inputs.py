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
"""Model inputs for the Laguna ModuleV3 pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from max import tree
from max.driver import Buffer
from max.pipelines.lib import ModelInputs


@dataclass
class LagunaInputs(ModelInputs):
    """A class representing inputs for the Laguna model.

    This class encapsulates the input tensors required for the Laguna model
    execution.
    """

    tokens: Buffer
    """Buffer containing the input token IDs."""

    input_row_offsets: Buffer
    """Buffer containing the offsets for each row in the ragged input sequence.
    """

    return_n_logits: Buffer
    """Number of logits to return."""

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        """Returns the flat graph inputs in compile ABI order.

        The order is the graph's, not this dataclass's field order, and device
        graph capture reads it through this property rather than through
        ``execute``.
        """
        if isinstance(self.input_row_offsets, np.ndarray):
            input_row_offsets = Buffer.from_numpy(self.input_row_offsets).to(
                self.tokens.device
            )
        else:
            input_row_offsets = self.input_row_offsets
        return (
            self.tokens,
            self.return_n_logits,
            input_row_offsets,
            *(
                tree.leaves(self.kv_cache_inputs)
                if self.kv_cache_inputs is not None
                else ()
            ),
        )
