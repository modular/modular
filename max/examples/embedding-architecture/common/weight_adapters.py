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
"""Base weight adapter with pattern-based transformations."""

import re

from max.graph.weights import WeightsAdapter


class PatternBasedWeightsAdapter(WeightsAdapter):
    """
    Base adapter that uses regex patterns for weight name transformations.

    This provides a clean, declarative way to specify weight name mappings
    without creating massive dictionaries. Subclasses just need to define
    their transformation rules.
    """

    def __init__(self):
        super().__init__()
        self.transform_rules = self.get_transform_rules()
        self.compiled_rules = self._compile_patterns()

    def get_transform_rules(self) -> list[tuple[str, str]]:
        """
        Define transformation rules as (pattern, replacement) tuples.
        Subclasses should override this method.

        Returns:
            List of (regex_pattern, replacement_string) tuples
        """
        return []

    def _compile_patterns(self) -> list[tuple[re.Pattern, str]]:
        """Compile regex patterns for efficient weight name transformation."""
        return [
            (re.compile(pattern), replacement)
            for pattern, replacement in self.transform_rules
        ]

    def __call__(self, name: str) -> str:
        """Transform a weight name using the compiled patterns."""
        result = name
        for pattern, replacement in self.compiled_rules:
            result = pattern.sub(replacement, result)
        return result


class TransformerWeightsAdapter(PatternBasedWeightsAdapter):
    """
    Generic transformer weight adapter with common transformations.
    Handles LayerNorm naming variations that are common across models.
    """

    def get_transform_rules(self) -> list[tuple[str, str]]:
        """Common transformation rules for transformer models."""
        return [
            # Handle LayerNorm parameter naming variations
            (r"\.LayerNorm\.gamma$", ".LayerNorm.weight"),
            (r"\.LayerNorm\.beta$", ".LayerNorm.bias"),
            # Handle layer_norm (underscore) variations
            (r"\.layer_norm\.gamma$", ".layer_norm.weight"),
            (r"\.layer_norm\.beta$", ".layer_norm.bias"),
        ]
