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
from max.interfaces.tokens import TokenBuffer
from typing_extensions import TypeVar


@dataclass(frozen=True)
class PixelGenerationRequest(Request):
    model_name: str = field()
    """
    The name of the model to be used for generating pixels. This should match
    the available models on the server and determines the behavior and
    capabilities of the response generation.
    """
    prompt: str | None = None
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
    max_sequence_length: int = 512
    """
    Maximum sequence length for text encoder.
    """

    def __post_init__(self) -> None:
        if self.prompt is None:
            raise ValueError("Prompt must be provided.")


@dataclass(kw_only=True)
class PixelContext(BaseContext):
    """A model-ready context for image/video generation requests.

    Per the design doc, this class contains only numeric data that the model
    will execute against. User-facing strings (prompt, negative_prompt) are
    consumed during tokenization and do not appear here.

    All preprocessing is performed by PixelGenerationTokenizer.new_context():
    - Prompt tokenization -> tokens field
    - Negative prompt tokenization -> negative_tokens field
    - Timestep schedule computation -> timesteps field
    - Initial noise generation -> initial_noise field

    Configuration:
        request_id: A unique identifier for this generation request.
        max_sequence_length: Maximum sequence length for text encoder.
        tokens: Tokenized prompt IDs (TokenBuffer).
        negative_tokens: Tokenized negative prompt IDs (TokenBuffer).
        timesteps: Precomputed timestep schedule for denoising.
        initial_noise: Precomputed initial noise (latents).
        height: Height of the generated image/video in pixels.
        width: Width of the generated image/video in pixels.
        num_inference_steps: Number of denoising steps.
        guidance_scale: Guidance scale for classifier-free guidance.
        num_images_per_prompt: Number of images/videos to generate per prompt.
        model_name: Name of the model being used.
    """

    # Request identification (required)
    request_id: RequestID = field(default_factory=RequestID)

    max_sequence_length: int = field(default=512)
    """Max sequence length for text encoder. Default 512 is sufficient for most prompts."""

    model_name: str = field(default="")

    # Tokenized prompts
    tokens: TokenBuffer
    """Primary encoder tokens."""

    mask: TokenBuffer | None = field(default=None)
    """Mask for text encoder's attention."""

    tokens_2: TokenBuffer | None = field(default=None)
    """Secondary encoder tokens. None for single-encoder models."""

    negative_tokens: TokenBuffer = field(
        default_factory=lambda: TokenBuffer(np.array([], dtype=np.int64))
    )
    """Negative tokens for primary encoder."""

    negative_tokens_2: TokenBuffer | None = field(default=None)
    """Negative tokens for secondary encoder. None for single-encoder models."""

    extra_params: dict[str, npt.NDArray[Any]] = field(default_factory=dict)
    """Model-specific numeric parameters (e.g., cfg_normalization values)."""

    # Precomputed tensors
    timesteps: npt.NDArray[Any] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """Precomputed timesteps schedule for denoising."""

    sigmas: npt.NDArray[Any] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """Precomputed sigmas schedule for denoising."""

    latents: npt.NDArray[Any] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """Precomputed initial noise (latents) for generation."""

    latent_image_ids: npt.NDArray[Any] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    """Precomputed latent image IDs for generation."""

    # Image generation parameters
    height: int = field(default=1024)
    width: int = field(default=1024)
    num_inference_steps: int = field(default=50)
    guidance_scale: float = field(default=3.5)
    guidance: npt.NDArray[Any] = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    true_cfg_scale: float = field(default=1.0)
    num_images_per_prompt: int = field(default=1)

    # Generation status
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

    def reset(self) -> None:
        """Resets the context's state."""
        self.status = GenerationStatus.ACTIVE

    def update(self, latents: npt.NDArray[Any]) -> None:
        """Update the context with newly generated latents/image data."""
        self.latents = latents

    def to_generation_output(self) -> PixelGenerationOutput:
        """Convert this context to a PixelGenerationOutput object."""
        return PixelGenerationOutput(
            request_id=self.request_id,
            final_status=self.status,
            pixel_data=self.latents,
        )


PixelGenerationContextType = TypeVar(
    "PixelGenerationContextType", bound=PixelContext
)
"""Type variable for pixel generation context types, constrained to PixelContext.

This allows generic typing of pixel generation pipeline components to accept any
context type that implements the PixelContext protocol.
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
