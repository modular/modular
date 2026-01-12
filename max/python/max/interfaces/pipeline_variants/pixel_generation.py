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
from typing import Any, Generic

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
    """
    The name of the model to be used for generating pixels. This should match
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
    image: bytes | None = None
    """
    The conditioning image to use for image-to-image/video generation.
    """
    video: bytes | None = None
    """
    The conditioning video for video-to-video generation.
    """
    mask: bytes | None = None
    """
    Mask image for inpainting/outpainting operations.
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
    num_frames: int = 1
    """
    Number of frames for video generation. 1 = single image, >1 = video.
    """
    fps: int = 24
    """
    Frames per second for video output.
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


class PixelGenerationMetadata(
    msgspec.Struct, tag=True, omit_defaults=True, kw_only=True
):
    """
    Represents metadata associated with pixel generation.

    This class will eventually replace the metadata dictionary used throughout
    the PixelGenerationOutput object, providing a structured and type-safe
    alternative for pixel generation metadata.

    Configuration:
        model_name: Name of the model used for generation.
        request_id: Unique identifier for the generation request.
    """

    model_name: str | None = None
    request_id: RequestID | None = None

    def to_dict(self) -> dict[str, int | float | str | bool]:
        """
        Convert the metadata to a dictionary format.

        Returns:
            dict[str, any]: Dictionary representation of the metadata.
        """
        result = {}
        for attr in self.__annotations__:
            if value := getattr(self, attr, None):
                result[attr] = value
        return result


PixelGenerationContextType = TypeVar(
    "PixelGenerationContextType", bound=BaseContext
)
"""Type variable for pixel generation context types.

This type variable is bound to BaseContext and represents the specific context
type used in pixel generation pipelines. It allows for type-safe generic
programming while ensuring that all context types inherit from BaseContext
and maintain the required interface for pixel generation operations.
"""


@dataclass
class PixelGenerationContext:
    """Context for pixel generation requests.

    This is a simple context that implements BaseContext protocol for diffusion
    model pipelines. It includes fields for both pixel generation parameters
    and compatibility with the text generation scheduler.

    Attributes:
        request_id: Unique identifier for this request.
        prompt: Text prompt for pixel generation.
        max_length: Maximum sequence length for tokenization (required for msgspec).
        height: Height of generated pixel in pixels.
        width: Width of generated pixel in pixels.
        num_inference_steps: Number of denoising steps.
        guidance_scale: Classifier-free guidance scale (0 to disable CFG).
        negative_prompt: Negative prompt for what NOT to generate.
        num_images_per_prompt: Number of images to generate.
        model_name: Name of the model (for scheduler compatibility).
        status: Current generation status.
    """

    request_id: RequestID
    prompt: str | None = None
    negative_prompt: str | None = None
    messages: list[TextGenerationRequestMessage] | None = None
    max_length: int = 4096  # Default max sequence length for text encoder
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 0.0
    num_images_per_prompt: int = 1
    num_videos_per_prompt: int = 1
    model_name: str = ""  # For scheduler compatibility
    _status: GenerationStatus = field(default=GenerationStatus.ACTIVE)

    @property
    def status(self) -> GenerationStatus:
        """Current generation status of the request."""
        return self._status

    @status.setter
    def status(self, value: GenerationStatus) -> None:
        """Update the generation status."""
        self._status = value

    @property
    def is_done(self) -> bool:
        """Whether the request has completed generation."""
        return self._status.is_done

    @property
    def needs_ce(self) -> bool:
        """Whether this context needs context encoding.

        For image generation, we never need context encoding since
        we process the full prompt at once through the text encoder.
        """
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
        """Compute number of available steps for scheduler compatibility.

        For image generation, this returns the number of inference steps.
        """
        return self.num_inference_steps


@dataclass(frozen=True)
class PixelGenerationInputs(
    PipelineInputs, Generic[PixelGenerationContextType]
):
    """Input data structure for image generation pipelines.

    This class represents the input data required for image generation operations
    within the pipeline framework. It extends PipelineInputs and provides type-safe
    generic support for different image generation context types.
    """

    batch: dict[RequestID, PixelGenerationContextType]
    """A dictionary mapping RequestID to PixelGenerationContextType instances.
    This batch structure allows for processing multiple image generation
    requests simultaneously while maintaining request-specific context
    and configuration data.
    """


class PixelGenerationOutput(msgspec.Struct, tag=True, omit_defaults=True):
    """Represents a response from the image/video generation API.

    This class encapsulates the result of a pixel generation request, including
    the final status, generated image/video data, and metadata.
    """

    final_status: GenerationStatus
    """The final status of the generation process."""
    steps_executed: int
    """The number of steps previously executed."""
    pixel_data: npt.NDArray[np.float32] = msgspec.field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """The generated pixel data (image/video frames), if available."""
    metadata: PixelGenerationMetadata = msgspec.field(
        default_factory=PixelGenerationMetadata
    )
    """Metadata associated with the image generation, such as chunk information, prompt details, or other relevant context."""

    @property
    def is_done(self) -> bool:
        """Indicates whether the audio generation process is complete.

        Returns:
            :class:`bool`: ``True`` if generation is done, ``False`` otherwise.
        """
        return self.final_status.is_done


def _check_pixel_generator_output_implements_pipeline_output(
    x: PixelGenerationOutput,
) -> PipelineOutput:
    return x
