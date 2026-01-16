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

from dataclasses import dataclass

from max.interfaces.pipeline import PipelineInputs
from PIL.Image import Image


@dataclass(eq=True)
class ImageGenerationInputs(PipelineInputs):
    """Inputs for image-generation pipelines."""

    # NOTE: Current implementation only considers offline generation without
    # request scheduling. `ImageGenerationContext` should be used once
    # request scheduling is implemented.
    prompt: str
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    num_images_per_prompt: int


@dataclass(kw_only=True)
class ImageGenerationOutput:
    """Output container for generated images."""

    images: list[Image]
    """List of generated images."""
