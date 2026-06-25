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
"""Shared types for cascade image-generation pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from pydantic import BaseModel


class ImageGenRequest(BaseModel):
    """Generation parameters for a single image-generation request."""

    height: int = 1024
    width: int = 1024
    num_steps: int = 28
    guidance_scale: float = 3.5
    seed: int = 42
    output_format: str = "PNG"


class ImageGenInterface(ABC):
    """Standard interface for an image-generation model."""

    @abstractmethod
    async def generate_image(self, req: ImageGenRequest, prompt: str) -> bytes:
        """Generate a serialized image from a text prompt."""

    async def generate_image_streaming(
        self, req: ImageGenRequest, prompt: str
    ) -> AsyncIterator[bytes]:
        """Stream serialized images for a text prompt.

        Override to emit intermediate frames (e.g., one per denoising step).
        The default implementation defers to :py:meth:`generate_image` and
        yields the final image as a single frame.
        """
        yield await self.generate_image(req, prompt)
