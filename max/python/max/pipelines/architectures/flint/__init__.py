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

"""FlintForCausalLM architecture."""

from .arch import flint_arch
from .inputs import FlintInputs
from .model import FlintModel
from .model_config import FlintConfig

__all__ = [
    "FlintConfig",
    "FlintInputs",
    "FlintModel",
    "flint_arch",
]
