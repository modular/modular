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
"""Lfm2VlForConditionalGeneration port — ModuleV3.

Registered in-tree; the arch lookup matches the package's `lfm2_vl_arch`
on the checkpoint's `architectures[0]` tag.
"""

from .arch import lfm2_vl_arch
from .model_config import Lfm2VlConfig

__all__ = ["Lfm2VlConfig", "lfm2_vl_arch"]

# The registry contract for custom architectures.
ARCHITECTURES = [lfm2_vl_arch]
