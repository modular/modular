# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Qwerky AI Inc. All rights reserved.
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
"""Mamba-specific kernel API package.

This package provides custom operations required for Mamba models:
- causal_conv1d: Causal 1D convolution with optional SiLU activation
- selective_scan_fwd: Selective scan forward pass for Mamba SSM
- rms_norm: Root Mean Square normalization for Mamba blocks
"""

from .MambaKernelAPI import (
    CausalConv1D,
    CausalConv1DUpdate,
    RMSNorm,
    RMSNormFusedResidual,
    RMSNormFusedResidualAdd,
    SelectiveScanFwd,
    SelectiveScanUpdate,
)

