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
"""Pixel generation interface definitions for Modular's MAX API.

This module provides data structures and interfaces for handling pixel generation
responses, including status tracking and pixel data encapsulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, runtime_checkable

import msgspec
import numpy as np
import numpy.typing as npt
from max.interfaces.pipeline import PipelineInputs, PipelineOutput
from max.interfaces.request import Request, RequestID
from max.interfaces.status import GenerationStatus
from typing_extensions import TypeVar


@dataclass(frozen=True)
class PixelGenerationRequest(Request):
    model_name: str = field()
    """
    The name of the model to be used for generating pixels. This should match
    the available models on the server and determines the behavior and
    capabilities of the response generation.
    """
    prompt: str = ""
    """
    The text prompt to generate pixels for.
    """
    prompt_2: str | None = None
    """
    The second text prompt to generate pixels for.
    """
    negative_prompt: str | None = None
    """
    The negative prompt to guide what NOT to generate.
    """
    negative_prompt_2: str | None = None
    """
    The second negative prompt to guide what NOT to generate.
    """
    guidance_scale: float = 3.5
    """
    Guidance scale for classifier-free guidance. Set to 1.0 to disable CFG.
    """
    true_cfg_scale: float = 1.0
    """
    True classifier-free guidance is enabled when true_cfg_scale > 1.0 and negative_prompt is provided.
    """
    height: int | None = None
    """
    Height of generated output in pixels. None uses model's native resolution.
    """
    width: int | None = None
    """
    Width of generated output in pixels. None uses model's native resolution.
    """
    num_inference_steps: int = 50
    """
    Number of denoising steps. More steps = higher quality but slower.
    """
    num_images_per_prompt: int = 1
    """
    Number of images/videos to generate per prompt.
    """
    seed: int | None = None
    """
    Optional random number generator seed for reproducible generation.
    """

    def __post_init__(self) -> None:
        if self.prompt == "":
            raise ValueError("Prompt must be provided.")


@runtime_checkable
class PixelGenerationContext(BaseContext, Protocol):
    """Protocol defining the interface for pixel generation contexts.

    A ``PixelContext`` represents model inputs for pixel generation pipelines,
    managing the state and parameters needed for generating images or videos.
    """

    @property
    def tokens(self) -> TokenBuffer:
        """The token buffer for the context."""
        ...

    @property
    def model_name(self) -> str:
        """The name of the model being used."""
        ...

    def update(self, latents: npt.NDArray[Any]) -> None:
        """Update the context with newly generated latents/image data."""
        ...

    def to_generation_output(self) -> PixelGenerationOutput:
        """Convert this context to a PixelGenerationOutput object."""
        ...


PixelGenerationContextType = TypeVar(
    "PixelGenerationContextType", bound=PixelGenerationContext
)
"""Type variable for pixel generation context types, constrained to PixelContext.

This allows generic typing of pixel generation pipeline components to accept any
context type that inherits from PixelContext.
"""


@dataclass(frozen=True)
class PixelGenerationInputs(
    PipelineInputs, Generic[PixelGenerationContextType]
):
    """
    Input data structure for pixel generation pipelines.

    This class represents the input data required for pixel generation operations
    within the pipeline framework. It extends PipelineInputs and provides type-safe
    generic support for different pixel generation context types.
    """

    batch: dict[RequestID, PixelGenerationContextType]
    """A dictionary mapping RequestID to PixelGenerationContextType instances.
    This batch structure allows for processing multiple pixel generation
    requests simultaneously while maintaining request-specific context
    and configuration data.
    """


class PixelGenerationOutput(msgspec.Struct, tag=True, omit_defaults=True):
    """
    Represents a response from the pixel generation API.

    This class encapsulates the result of a pixel generation request, including
    the request ID, final status, and generated pixel data.
    """

    request_id: RequestID
    """The unique identifier for the generation request."""

    final_status: GenerationStatus
    """The final status of the generation process."""

    pixel_data: npt.NDArray[np.float32] = msgspec.field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """The generated pixel data, if available."""

    @property
    def is_done(self) -> bool:
        """
        Indicates whether the pixel generation process is complete.

        Returns:
            bool: True if the generation is done, False otherwise.
        """
        return self.final_status.is_done


def _check_pixel_generator_output_implements_pipeline_output(
    x: PixelGenerationOutput,
) -> PipelineOutput:
    return x
