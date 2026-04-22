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
"""The transformer mechanism used within the model."""

from .distributed_transformer import (
    DistributedLogitsPostprocessMixin,
    DistributedTransformer,
    DistributedTransformerBlock,
    distributed_logits_postprocess,
)
from .transformer import (
    LogitsPostprocessMixin,
    ReturnHiddenStates,
    ReturnLogits,
    Transformer,
    TransformerBlock,
    logits_postprocess,
)

__all__ = [
    "DistributedLogitsPostprocessMixin",
    "DistributedTransformer",
    "DistributedTransformerBlock",
    "LogitsPostprocessMixin",
    "ReturnHiddenStates",
    "ReturnLogits",
    "Transformer",
    "TransformerBlock",
    "distributed_logits_postprocess",
    "logits_postprocess",
]
