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
"""MLA (Multi-Latent Attention) decode kernel for gfx950.

Thin wrapper that delegates to `mha_decode`. `Self.mla_mode=True`
produces MLA-style coords (kv_head_idx=0, q_tile_idx=block_idx.y) via
`AMDStructuredConfig`.
"""

from std.utils.numerics import get_accum_type

from .attention import Attention

# Activate the `__extension Attention` block that defines `mha_decode`
# (delegate target below).
from .mha_decode import Attention


__extension Attention:
    @always_inline
    def mla_decode(
        mut self,
        exp_sum_ptr: UnsafePointer[
            Scalar[get_accum_type[Self.q_type]()], MutAnyOrigin
        ],
        qk_max_ptr: UnsafePointer[
            Scalar[get_accum_type[Self.q_type]()], MutAnyOrigin
        ],
        num_partitions: Int,
    ):
        """MLA decode — delegates to MHA decode."""
        self.mha_decode(exp_sum_ptr, qk_max_ptr, num_partitions)
