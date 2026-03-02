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
"""Tests for max.nn.rope."""

from __future__ import annotations

from max.nn.module_v3.rope import yarn


def test_yarn() -> None:
    dim = 64
    max_sequence_length = 4096 * 32
    embedding = yarn.positional_embedding(
        dim=dim,
        base=150000,
        alpha=32,
        beta=1,
        max_sequence_length=max_sequence_length,
        original_max_sequence_length=4096,
    )
    assert embedding.shape == [max_sequence_length, dim // 2, 2]
