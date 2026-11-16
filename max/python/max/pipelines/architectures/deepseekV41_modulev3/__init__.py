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
"""The DeepSeek-V4.1-Flash architecture, in the ModuleV3 API."""

from .arch import deepseekV41_modulev3_arch
from .model import DeepseekV41Model
from .model_config import (
    DeepseekV41Config,
    EngramTableFormat,
    RoutedExpertFormat,
)

__all__ = [
    "DeepseekV41Config",
    "DeepseekV41Model",
    "EngramTableFormat",
    "RoutedExpertFormat",
    "deepseekV41_modulev3_arch",
]
