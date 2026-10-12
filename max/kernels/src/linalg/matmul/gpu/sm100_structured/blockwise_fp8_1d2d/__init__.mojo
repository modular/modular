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
"""Blockwise FP8 1D2D grouped matmul kernel for SM100.

This module provides a structured kernel implementation for grouped blockwise
FP8 GEMM using the 1D-1D tensor layout with offset-based addressing.

It combines:
- Accumulation pattern from blockwise_fp8/ (register-based per-K scaling)
- 1D2D work distribution from grouped_block_scaled_1d1d/ (offset-based A
  tensor addressing, bounds-checked output, 3-warp specialization)
"""

from .blockwise_fp8_1d2d_matmul import (
    grouped_matmul_1d2d_blockwise_fp8,
    grouped_matmul_dynamic_scaled_fp8_1d2d,
)
