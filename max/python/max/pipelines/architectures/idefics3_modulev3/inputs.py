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
"""Model inputs for the Idefics3 ModuleV3 pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from max.driver import Buffer
from max.pipelines.lib import ModelInputs


@dataclass
class Idefics3Inputs(ModelInputs):
    """Inputs for the Idefics3 model."""

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer

    # Vision inputs
    pixel_values: Buffer | None = None
    image_token_indices: Buffer | None = None

    @property
    def has_vision_inputs(self) -> bool:
        return self.pixel_values is not None
