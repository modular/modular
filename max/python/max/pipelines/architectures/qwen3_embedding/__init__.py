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

from .arch import qwen3_embedding_arch
from .model import Qwen3EmbeddingInputs, Qwen3EmbeddingModel

# Expose ARCHITECTURES list for custom architecture loading
ARCHITECTURES = [qwen3_embedding_arch]

__all__ = [
    "Qwen3EmbeddingInputs",
    "Qwen3EmbeddingModel",
    "qwen3_embedding_arch",
]
