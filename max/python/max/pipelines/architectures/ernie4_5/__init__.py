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

"""ERNIE-4.5 (Ernie4_5ForCausalLM) transformer architecture for MAX.

Decoder-only transformer (Llama-style) with actual config defaults:
  - 18 layers, hidden_size=1024, intermediate_size=3072
  - GQA: 16 Q heads, 2 KV heads (head_dim=128) → group=8 ✓ GPU OK
  - SwiGLU MLP (use_bias: false)
  - RMSNorm (rms_norm_eps=1e-5)
  - RoPE theta from rope_parameters dict (new-style transformers config)
  - tie_word_embeddings: true
  - vocab_size: 103424, max_position_embeddings: 131072
  - Apache 2.0 license

CLI usage:
    max serve \\
        --model-path baidu/ERNIE-4.5-0.3B-PT \\
        --custom-architectures /path/to/architectures/ernie45 \\
        --max-batch-size 1 \\
        --max-length 512 \\
        --quantization-encoding bfloat16
"""

from .arch import ernie45_arch
from .model import ERNIE45Inputs, ERNIE45Model
from .model_config import ERNIE45Config

ARCHITECTURES = [ernie45_arch]

__all__ = [
    "ARCHITECTURES",
    "ERNIE45Config",
    "ERNIE45Inputs",
    "ERNIE45Model",
    "ernie45_arch",
]
