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
"""Accelerator tests for common_layers.rotary_embedding."""

from __future__ import annotations

import pytest
from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.experimental.nn.common_layers.rotary_embedding import (
    RotaryEmbedding,
)
from max.experimental.tensor import TensorType


@pytest.mark.skipif(not accelerator_count(), reason="requires accelerator")
def test_forward_on_accelerator() -> None:
    gpu = Accelerator()
    rope = RotaryEmbedding(
        dim=2, n_heads=1, theta=1e4, max_seq_len=1, device=gpu
    )

    # The table is built on the host, and forward copies all of it to the
    # input's device, so the copy can be hoisted.
    assert rope.freqs_cis.device == CPU()
    staged = rope.trace(TensorType(DType.float32, [1, "seqlen", 1, 2], gpu))
    transfers = [
        line for line in str(staged).splitlines() if "mo.transfer" in line
    ]
    rows, cols = rope.freqs_cis.shape
    [transfer] = transfers
    assert f'!mo.tensor<[{rows}, {cols}], f32> to <"gpu", 0>' in transfer
