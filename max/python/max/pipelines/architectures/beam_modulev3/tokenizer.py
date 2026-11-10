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
"""Beam's tokenizer."""

from __future__ import annotations

from typing import ClassVar

from max.pipelines.lib.tokenizer import ReasoningTextTokenizer

from . import harmony


class BeamTokenizer(ReasoningTextTokenizer):
    """Text tokenizer for Beam.

    The overlap pipeline requires a reasoning model's tokenizer to name one
    token that opens reasoning and one that closes it, and uses them only to
    pick ``thinking_temperature`` under speculative decoding. Harmony has no
    such pair, because ``analysis`` and ``final`` messages open with the same
    tokens: ``<|message|>`` and ``<|end|>`` also mark the final answer, which
    closes on ``<|return|>``, as reasoning.
    """

    reasoning_delimiters: ClassVar[tuple[str, str]] = (
        harmony.MESSAGE,
        harmony.END,
    )
