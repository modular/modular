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
"""Weight adapters for BERT models."""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.weight_adapters import TransformerWeightsAdapter


class BertWeightsAdapter(TransformerWeightsAdapter):
    """
    Adapter for BERT model weights from HuggingFace format.

    Handles transformations:
    - Removes 'bert.' prefix
    - Converts LayerNorm gamma/beta to weight/bias
    """

    def get_transform_rules(self) -> list[tuple[str, str]]:
        """BERT-specific transformation rules."""
        # Start with common transformer rules
        rules = super().get_transform_rules()

        # Add BERT-specific rules
        bert_rules = [
            # Remove 'bert.' prefix if present
            (r"^bert\.", ""),
        ]

        return bert_rules + rules
