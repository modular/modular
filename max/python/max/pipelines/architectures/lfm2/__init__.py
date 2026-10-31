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

"""The LFM2 architecture.

A hybrid model combining full attention with short convolutions.
"""

from .arch import lfm2_arch
from .conv_state import ConvStateStore
from .lfm2 import LFM2
from .model_config import LFM2Config

__all__ = [
    "LFM2",
    "ConvStateStore",
    "LFM2Config",
    "lfm2_arch",
]
