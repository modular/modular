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
"""Laguna mixture-of-experts architecture on ModuleV3."""

from .arch import laguna_modulev3_arch
from .inputs import LagunaInputs
from .model import LagunaModel

__all__ = [
    "LagunaInputs",
    "LagunaModel",
    "laguna_modulev3_arch",
]
