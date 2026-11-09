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
"""The Kimi-K3 DSpark draft, a ModuleV3 block drafter."""

from .arch import k3_dspark_draft_arch
from .config import K3DSparkDraftConfig, parse_k3_dspark_config
from .dspark import K3DSpark, tensor_parallel_k3_dspark
from .proposer import K3DSparkProposer, k3_dspark_block_proposer

__all__ = [
    "K3DSpark",
    "K3DSparkDraftConfig",
    "K3DSparkProposer",
    "k3_dspark_block_proposer",
    "k3_dspark_draft_arch",
    "parse_k3_dspark_config",
    "tensor_parallel_k3_dspark",
]
