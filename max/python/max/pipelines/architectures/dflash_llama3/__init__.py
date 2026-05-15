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
"""DFlash draft model for Llama3-family targets.

The draft is a Qwen3-style transformer (per-head Q/K RMSNorm, non-causal
attention) that fuses concatenated target hidden states into its KV cache
via :meth:`AttentionWithRope.materialize_kv_from_hidden` and runs a single
non-causal block forward over [verified_id, MASK, MASK, ...] per
iteration.
"""

from .arch import dflash_llama_arch
from .dflash_llama3 import DFlashLlama3
from .model import DFlashLlama3Model

__all__ = [
    "DFlashLlama3",
    "DFlashLlama3Model",
    "dflash_llama_arch",
]
