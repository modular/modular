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
"""SM100 (Blackwell) matrix multiply operations (deprecated - use `gpu.compute.arch.mma_nvidia_sm100`).

This module is deprecated. For new code, import from the `gpu.compute.arch.mma_nvidia_sm100`
module instead:

```mojo
# Deprecated:
from gpu.mma_sm100 import mma, UMMAKind, UMMAInsDescriptor

# Recommended:
from gpu.compute.arch.mma_nvidia_sm100 import mma, UMMAKind, UMMAInsDescriptor
```

This module provides NVIDIA Blackwell (SM100) specific tensor core operations
including unified MMA (UMMA) instructions.
"""

# Re-export all public symbols from compute.arch.mma_nvidia_sm100 for backward compatibility
from .compute.arch.mma_nvidia_sm100 import (
    MMASmemDescriptor,
    UMMAInsDescriptor,
    UMMAKind,
    mma,
    mma_arrive,
    mma_arrive_multicast,
)
