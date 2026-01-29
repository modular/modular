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
"""Pixel-based (vision) modality provider options."""

from pydantic import BaseModel, ConfigDict, Field


class PixelModalityOptions(BaseModel):
    """Options specific to pixel-based (vision) pipelines."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dummy_param: str | None = Field(
        None,
        description="Dummy parameter for initial implementation.",
    )
