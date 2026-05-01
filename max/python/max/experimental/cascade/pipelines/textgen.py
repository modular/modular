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
"""Shared types for cascade text-generation pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TypeAlias

from pydantic import BaseModel


class GenerateRequest(BaseModel):
    """Generation parameters for a single text-generation request."""

    num_tokens: int = 10
    ignore_eos: bool = False


ChatMessages: TypeAlias = list[dict[str, str]]


class TextGenInterface(ABC):
    """Standard interface for a text-generation model."""

    @abstractmethod
    def generate(
        self,
        req: GenerateRequest,
        prompt: str | ChatMessages,
    ) -> AsyncIterator[str]:
        """Generate a streaming text response to a text prompt."""
