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
"""DeepSeek-V4.1-Flash speculative decoding with its built-in DSpark drafter."""

from .arch import unified_dspark_deepseekV41_speculator
from .batch_processor import (
    UnifiedDSparkDeepseekV41BatchProcessor,
    UnifiedDSparkDeepseekV41Inputs,
)
from .model import UnifiedDSparkDeepseekV41Model
from .unified_dspark_deepseekV41 import UnifiedDSparkDeepseekV41

__all__ = [
    "UnifiedDSparkDeepseekV41",
    "UnifiedDSparkDeepseekV41BatchProcessor",
    "UnifiedDSparkDeepseekV41Inputs",
    "UnifiedDSparkDeepseekV41Model",
    "unified_dspark_deepseekV41_speculator",
]
