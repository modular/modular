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
"""Provides the AMD GPU backend implementations for matmuls."""

from .mxfp4_dequant_matmul_amd import mxfp4_dequant_matmul_amd
from .mxfp4_grouped_matmul_amd import mxfp4_grouped_matmul_amd
from .mxfp4_matmul_amd import mxfp4_block_scaled_matmul_amd, MXFP4MatmulAMD
from .amd_matmul import AMDMatmul
from .amd_ping_pong_matmul import (
    AMDPingPongMatmul,
    KernelConfig,
    structured_ping_pong_matmul,
)
