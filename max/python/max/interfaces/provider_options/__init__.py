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
"""Provider-specific options for MAX platform and modalities."""

from pydantic import BaseModel, ConfigDict, Field

from .max import MaxProviderOptions
from .modality import PixelModalityOptions


class ProviderOptions(BaseModel):
    """Container for all provider-specific options.

    Includes both universal MAX options and modality-specific options.
    All options are validated at the API layer.

    Example:
        {
            "max": {"target_endpoint": "instance-123"},
            "pixel": {"type": "pixel", "dummy_param": "value"}
        }
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max: MaxProviderOptions | None = Field(
        None,
        description="Universal MAX platform options.",
    )

    pixel: PixelModalityOptions | None = Field(
        None,
        description="Pixel-based (vision) modality options.",
    )

    # Add more modality fields here as needed:
    # text: TextModalityOptions | None = None
    # tts: TTSModalityOptions | None = None


__all__ = [
    "MaxProviderOptions",
    "PixelModalityOptions",
    "ProviderOptions",
]
