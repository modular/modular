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
"""Nemotron-H hybrid Mamba-2 and attention architecture (ModuleV3)."""

from .arch import nemotron_h_modulev3_arch
from .model import NemotronHModel
from .model_config import NemotronHConfig

__all__ = [
    "NemotronHConfig",
    "NemotronHModel",
    "nemotron_h_modulev3_arch",
]
