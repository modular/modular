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

"""Muse Spark vision-language MoE architecture (ModuleV3 API)."""

from .arch import muse_spark_arch
from .model import MuseSparkModel
from .model_config import (
    MuseSparkConfig,
    MuseSparkNVFP4,
    MuseSparkTextConfig,
)

__all__ = [
    "MuseSparkConfig",
    "MuseSparkModel",
    "MuseSparkNVFP4",
    "MuseSparkTextConfig",
    "muse_spark_arch",
]
