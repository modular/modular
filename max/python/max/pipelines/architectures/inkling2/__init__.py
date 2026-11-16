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

"""The Inkling2 architecture on ModuleV3."""

from .arch import inkling2_arch
from .model_config import (
    Inkling2Config,
    Inkling2TextConfig,
    Inkling2VisionConfig,
)

__all__ = [
    "Inkling2Config",
    "Inkling2TextConfig",
    "Inkling2VisionConfig",
    "inkling2_arch",
]
