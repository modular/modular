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
"""DeepSeek-V4.1-Flash mixture-of-experts architecture for text generation.

V4.1 changes the decoder block, hyper-connections, output head, config,
attention modes and weight layout relative to V4, so it is a separate
package that reuses V4's unchanged layers.
"""

from .arch import deepseekV41_arch
from .model import DeepseekV41Model
from .model_config import DeepseekV41Config

__all__ = [
    "DeepseekV41Config",
    "DeepseekV41Model",
    "deepseekV41_arch",
]
