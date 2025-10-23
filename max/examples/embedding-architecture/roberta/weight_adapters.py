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
"""Weight adapters for RoBERTa models."""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.weight_adapters import TransformerWeightsAdapter


class RobertaWeightsAdapter(TransformerWeightsAdapter):
    """
    Adapter for RoBERTa model weights from HuggingFace format.
    """

    def get_transform_rules(self) -> list[tuple[str, str]]:
        """RoBERTa-specific transformation rules."""
        # Start with common transformer rules
        rules = super().get_transform_rules()

        # Add RoBERTa-specific rules
        roberta_rules = [
            # Remove 'roberta.' prefix if present
            (r"^roberta\.", ""),
            # Remove sentence-transformers prefix
            (r"^0_Transformer\.auto_model\.", ""),
        ]

        return roberta_rules + rules
