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
"""Implements the Gemma3 model."""

from __future__ import annotations

from collections.abc import Sequence

from max.graph import TensorValue
from max.nn import Module

from .model_config import Gemma3Config


class Gemma3TextModel(Module):
    """The Gemma language model."""

    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config

    def __call__(
        self,
        tokens: TensorValue,
        input_row_offsets: TensorValue,
        kv_cache_inputs: Sequence[TensorValue],
    ) -> tuple[TensorValue, ...]:
        raise NotImplementedError


class Gemma3(Module):
    """The Gemma model (currently text-only)."""

    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.language_model = Gemma3TextModel(config)

    def __call__(
        self,
        tokens: TensorValue,
        input_row_offsets: TensorValue,
        kv_cache_inputs: Sequence[TensorValue],
    ) -> tuple[TensorValue, ...]:
        return self.language_model(
            tokens,
            input_row_offsets,
            kv_cache_inputs,
        )
