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
"""Model inputs for the Hy4 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from max import tree
from max.driver import Buffer
from max.pipelines.lib import ModelInputs


@dataclass
class HYV4Inputs(ModelInputs):
    """Ragged tokens, row offsets, the host prefill buffer length and the MLA
    plus indexer cache inputs."""

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer = field(kw_only=True)
    batch_context_length: Buffer = field(kw_only=True)

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        """Flat graph inputs in compile ABI order."""
        assert self.kv_cache_inputs is not None
        return (
            self.tokens,
            self.return_n_logits,
            self.input_row_offsets,
            self.batch_context_length,
            *tree.leaves(self.kv_cache_inputs),
        )
