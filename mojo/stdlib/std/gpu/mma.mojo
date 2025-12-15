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
"""Matrix multiply-accumulate operations (deprecated - use `gpu.compute.mma`).

This module is deprecated. For new code, import from the `gpu.compute.mma`
module instead:

```mojo
# Deprecated:
from gpu.mma import mma, ld_matrix, st_matrix

# Recommended:
from gpu.compute.mma import mma, ld_matrix, st_matrix
```

This module provides warp-matrix-matrix-multiplication (WMMA) and tensor core
operations for GPU programming.
"""

# Re-export all public symbols from compute.mma for backward compatibility
from .compute.mma import (
    MMAOperandDescriptor,
    WGMMADescriptor,
    get_amd_bf8_dtype,
    get_amd_fp8_dtype,
    ld_matrix,
    mma,
    st_matrix,
    wgmma_async,
    wgmma_commit_group_sync,
    wgmma_fence_aligned,
    wgmma_wait_group_sync,
)
