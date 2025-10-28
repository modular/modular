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
"""Tensor core generation 05 operations (deprecated - use `gpu.compute.tcgen05`).

This module is deprecated. For new code, import from the `gpu.compute.tcgen05`
module instead:

```mojo
# Deprecated:
from gpu.tcgen05 import tcgen05_alloc, tcgen05_ld, tcgen05_st

# Recommended:
from gpu.compute.tcgen05 import tcgen05_alloc, tcgen05_ld, tcgen05_st
```

This module provides NVIDIA Blackwell (SM100) tensor memory operations for
the fifth generation tensor cores (TCGEN05).
"""

# Re-export all public symbols from compute.tcgen05 for backward compatibility
from .compute.tcgen05 import (
    TensorMemory,
    check_blackwell_constraint,
    tcgen05_alloc,
    tcgen05_cp,
    tcgen05_dealloc,
    tcgen05_fence_after,
    tcgen05_fence_before,
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
    tcgen05_st,
    tcgen05_store_wait,
)
