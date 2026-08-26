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

"""Mamba2-specific tokenizer with default chat template support.

Mamba2 base models (like ``state-spaces/mamba2-130m``) ship without a chat
template in their tokenizer configuration. This tokenizer reuses the same
passthrough chat-template behavior as the Mamba1
:class:`~max.pipelines.architectures.mamba.tokenizer.MambaTokenizer` so the
OpenAI-compatible ``/v1/chat/completions`` endpoint can run against base
Mamba2 checkpoints out of the box.

For chat applications, consider an instruction-tuned Mamba2 variant or pass
``--chat-template`` to override the default.
"""

from __future__ import annotations

from ..mamba.tokenizer import MambaTokenizer


class Mamba2Tokenizer(MambaTokenizer):
    """Mamba2 tokenizer wrapper.

    Identical behavior to :class:`MambaTokenizer`: a thin
    :class:`~max.pipelines.lib.TextTokenizer` adapter that installs a simple
    passthrough chat template when the underlying HF tokenizer does not
    define one. Kept as a distinct class so the architecture registry binds
    the Mamba2 entry to a Mamba2-named tokenizer (matching the rest of the
    package), and so future Mamba2-specific behavior can be added without
    touching the Mamba1 path.
    """
