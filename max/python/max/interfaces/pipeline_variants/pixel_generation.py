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
from max.interfaces.context import BaseContext
from max.interfaces.pipeline import PipelineInputs, PipelineOutput
from max.interfaces.request import Request, RequestID
from max.interfaces.status import GenerationStatus
from typing_extensions import TypeVar

from .text_generation import (
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
)


@dataclass(frozen=True)
class PixelGenerationRequest(Request):
    model: str = field()
    """The name of the model to be used for generating pixels. This should match
    the available models on the server and determines the behavior and
    capabilities of the response generation.
    """
    prompt: str | None = None
    """
    The text to generate pixels for. The maximum length is 4096 characters.
    """
    negative_prompt: str | None = None
    """
    Negative prompt to guide what NOT to generate.
    """
    messages: list[TextGenerationRequestMessage] | None = None
    """
    A list of messages for chat-based interactions. This is used in chat
    completion APIs, where each message represents a turn in the conversation.
    If provided, the model will generate responses based on these messages.
    """
    guidance_scale: float = 7.5
    """
    Guidance scale for classifier-free guidance. Set to 1 to disable CFG.
    """
    height: int | None = 1024
    """
    Height of generated image/frame in pixels. Defaults to model's default (typically 1024).
    """
    width: int | None = 1024
    """
    Width of generated image/frame in pixels. Defaults to model's default (typically 1024).
    """
    num_inference_steps: int = 50
    """
    Number of denoising steps. More steps = higher quality but slower.
    """
    num_images_per_prompt: int = 1
    """
    Number of images/videos to generate per prompt.
    """
    chat_template_options: dict[str, Any] | None = None
    """
    Optional dictionary of options to pass when applying the chat template.
    """
    tools: list[TextGenerationRequestTool] | None = None
    """
    A list of tools that can be invoked during the generation process. This
    allows the model to utilize external functionalities or APIs to enhance its
    responses.
    """

    def __post_init__(self) -> None:
        if self.prompt is None and self.messages is None:
            raise RuntimeError("Either prompt or messages must be provided.")



@runtime_checkable
class PixelGenerationContext(BaseContext, Protocol):
    """Protocol for pixel generation contexts.

    This protocol defines the interface for diffusion model pipelines,
    ensuring compatibility with the scheduler and serving infrastructure.
    """

    @property
    def prompt(self) -> str | None:
        """Text prompt for pixel generation."""
        ...

    @property
    def negative_prompt(self) -> str | None:
        """Negative prompt for what NOT to generate."""
        ...

    @property
    def messages(self) -> list[TextGenerationRequestMessage] | None:
        """Chat messages for generation."""
        ...

    @property
    def max_length(self) -> int:
        """Maximum sequence length for text encoder."""
        ...

    @property
    def height(self) -> int:
        """Height of generated output in pixels."""
        ...

    @property
    def width(self) -> int:
        """Width of generated output in pixels."""
        ...

    @property
    def num_inference_steps(self) -> int:
        """Number of denoising steps."""
        ...

    @property
    def guidance_scale(self) -> float:
        """Classifier-free guidance scale (1 to disable CFG)."""
        ...

    @property
    def num_images_per_prompt(self) -> int:
        """Number of images to generate."""
        ...

    @property
    def model_name(self) -> str:
        """Name of the model (for scheduler compatibility)."""
        ...

    @property
    def needs_ce(self) -> bool:
        """Whether this context needs context encoding."""
        return False

    @property
    def active_length(self) -> int:
        """Current sequence length for batch constructor compatibility."""
        return 1

    @property
    def current_length(self) -> int:
        """Current length for batch constructor compatibility."""
        return 1

    @property
    def processed_length(self) -> int:
        """Processed length for batch constructor compatibility."""
        return 0

    def compute_num_available_steps(self, max_seq_len: int) -> int:
        """Compute number of available steps for scheduler compatibility."""
        ...


PixelGenerationContextType = TypeVar(
    "PixelGenerationContextType", bound=PixelGenerationContext
)
"""Type variable for pixel generation context types, constrained to PixelGenerationContext.

This allows generic typing of pixel generation pipeline components to accept any
context type that implements the PixelGenerationContext protocol.
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
    request ID, the final status, generated pixel data.
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
