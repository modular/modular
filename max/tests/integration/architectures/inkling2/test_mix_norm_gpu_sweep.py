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
"""The full shape sweep of the fused attn_res mixture + sublayer norm.

Postsubmit only: every shape compiles the op on the GPU. test_mix_norm_gpu
runs the two presubmit cases.
"""

from __future__ import annotations

import pytest
from _mix_norm_check import run_case
from max.dtype import DType


@pytest.mark.compile_on_gpu("inkling2_mix_norm has no custom.declare")
@pytest.mark.parametrize(
    ("hidden", "tokens", "num_blocks", "partial_dtype"),
    [
        (4608, 5, 4, DType.float32),
        (4608, 1, 1, DType.bfloat16),
        (4608, 1, 4, DType.float32),
        (100, 5, 4, DType.bfloat16),
    ],
)
def test_mix_norm_matches_unfused(
    hidden: int, tokens: int, num_blocks: int, partial_dtype: DType
) -> None:
    run_case(hidden, tokens, num_blocks, partial_dtype, num_queries=1)


@pytest.mark.compile_on_gpu("inkling2_mix_norm has no custom.declare")
def test_mix_norm_global_layer_rows() -> None:
    """Each of a global layer's three query rows matches its own mixture."""
    run_case(4608, 3, 3, DType.bfloat16, num_queries=3)
