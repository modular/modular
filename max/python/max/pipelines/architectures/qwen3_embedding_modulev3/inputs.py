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
"""Model inputs for the Qwen3 Embedding ModuleV3 pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from max.driver import Buffer
from max.pipelines.lib import ModelInputs


@dataclass
class Qwen3EmbeddingInputs(ModelInputs):
    """Input structure for Qwen3 embedding models."""

    tokens: Buffer
    """Input token IDs [total_seq_len]"""

    input_row_offsets: Buffer
    """Row offsets for ragged tensors [batch_size + 1]"""

    return_n_logits: Buffer
    """Number of logits to return (kept for interface compatibility)"""
