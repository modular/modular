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
"""Tests the ModuleV3 Laguna model against the graph architecture, in NVFP4.

S-2.1 packs its routed experts NVFP4 in most layers and leaves them bfloat16 in
the rest, so the same model mixes the quantized and unquantized expert paths.
The NVFP4 grouped matmul is SM100-only.
"""

from __future__ import annotations

from laguna_parity import assert_logits_match


def test_mixed_precision_logits_match_the_graph_architecture() -> None:
    assert_logits_match("s21_like_nvfp4")
