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
"""The inputs and outputs of each role a registered module fills.

An architecture registers a :class:`~max.nn.Module` class per role in
:attr:`~max.pipelines.lib.arch_lookup.SupportedArchitecture.modules`. The
framework builds the role's graph and batches from the contract here, so the
module's ``forward`` takes the role's inputs and returns its outputs.
"""

from __future__ import annotations

from typing import Any

from max import tree
from max.tensor import Tensor

from .interfaces.pipeline_model import ModuleV3Outputs

__all__ = [
    "DecoderInputs",
    "DecoderOutputs",
]


@tree.dataclass(frozen=True)
class DecoderInputs:
    """The inputs of a ``DECODER`` module's ``forward``.

    The decoder graph takes the fields in this order.
    """

    tokens: Tensor
    """The batch's ragged tokens, shape ``[total_seq_len]``."""

    input_row_offsets: Tensor
    """Each sequence's start in :attr:`tokens`, shape ``[batch_size + 1]``."""

    return_n_logits: Tensor
    """How many logits per sequence to return, a CPU tensor of shape ``[1]``.
    """

    kv: Any
    """The KV cache inputs, in the tree the config's KV params declare."""


DecoderOutputs = ModuleV3Outputs
"""The outputs of a ``DECODER`` module's ``forward``."""
