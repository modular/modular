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

"""Muse Spark vision encoder (ModuleV3 API)."""

from max.pipelines.architectures.muse_glimmer.vision import vision_input_types

from .data_processing import vision_inputs
from .vision_model import MuseSparkVisionModel, parse_vision_config

__all__ = [
    "MuseSparkVisionModel",
    "parse_vision_config",
    "vision_input_types",
    "vision_inputs",
]
