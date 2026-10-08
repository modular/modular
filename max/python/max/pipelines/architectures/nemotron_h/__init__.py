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

"""Nemotron-H architecture: a hybrid decoder interleaving Mamba-2
mixers, NoPE GQA attention, relu2 MLP blocks, and, in the Nemotron 3 MoE
variants, mixture-of-experts blocks.
"""

from .arch import nemotron_h_arch
from .model import NemotronHInputs, NemotronHModel
from .model_config import NemotronHConfig

__all__ = [
    "NemotronHConfig",
    "NemotronHInputs",
    "NemotronHModel",
    "nemotron_h_arch",
]
