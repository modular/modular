# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""MiniLM sentence transformer architecture for MAX.

This module implements the all-MiniLM-L6-v2 architecture, a BERT-based sentence transformer model
that maps sentences to 384-dimensional dense vectors
"""

from .arch import minilm_arch

ARCHITECTURES = [minilm_arch]

__all__ = ["minilm_arch", "ARCHITECTURES"]
