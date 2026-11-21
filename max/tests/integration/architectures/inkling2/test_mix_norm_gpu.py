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
"""The fused attn_res mixture + sublayer norm matches the unfused math.

Each shape compiles the op on the GPU (it has no custom.declare), about a
minute and a half apiece on CI, so presubmit runs two shapes that between them
reach every axis: the global layer's last query row (Inkling 2's hidden,
several tokens and blocks, a bfloat16 partial, a nonzero row) and a ragged
single-token float32 case. test_mix_norm_gpu_sweep runs the rest in
postsubmit.
"""

from __future__ import annotations

import pytest
from _mix_norm_check import run_case
from max.dtype import DType


@pytest.mark.compile_on_gpu("inkling2_mix_norm has no custom.declare")
def test_mix_norm_ragged_single_token() -> None:
    run_case(100, 1, 1, DType.float32, num_queries=1)


@pytest.mark.compile_on_gpu("inkling2_mix_norm has no custom.declare")
def test_mix_norm_global_layer_last_row() -> None:
    """A global layer's last query row, the one an offset bug would miss."""
    run_case(4608, 3, 3, DType.bfloat16, num_queries=3, rows=(2,))
