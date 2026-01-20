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
"""SM100 Structured Kernels - self-contained Blackwell matmul implementation.

This module provides the canonical implementation for SM100 (Blackwell) GPU
matmul operations using the structured kernels architecture.
"""

from .matmul import (
    blackwell_matmul_tma_umma_warp_specialized,
    matmul_sm100_fallback,
)
from .block_scaled_matmul import (
    blackwell_block_scaled_matmul_tma_umma_warp_specialized,
)
from .config import MatmulConfig, BlockScaledMatmulConfig, choose_config
from .dispatch import matmul_dispatch_sm100
from .pipeline import ProducerConsumerPipeline, MbarPtr
