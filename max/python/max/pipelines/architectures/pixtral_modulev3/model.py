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
"""Compatibility names for the role-registered Pixtral pipeline."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from max.driver import Buffer
from max.pipelines.lib import ModelInputs


@dataclass
class PixtralInputs(ModelInputs):
    """Holds inputs for the Pixtral model."""

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer

    # Vision inputs - ragged tensor of pre-extracted patches from all images
    pixel_patches: Buffer | None = None
    vision_attention_mask: Buffer | None = None
    vision_position_ids: Buffer | None = None
    image_token_indices: Buffer | None = None

    @property
    def has_vision_inputs(self) -> bool:
        """Returns true iff this includes vision model inputs."""
        return self.pixel_patches is not None


if TYPE_CHECKING:
    from max.pipelines.lib.module_registration import (
        ModuleV3DecoderPipelineModel as PixtralModel,
    )
else:
    from .arch import pixtral_modulev3_arch

    PixtralModel = pixtral_modulev3_arch.pipeline_model
    assert PixtralModel is not None

__all__ = ["PixtralInputs", "PixtralModel"]
