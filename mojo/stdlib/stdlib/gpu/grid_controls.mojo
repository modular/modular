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
"""GPU grid dependency control (deprecated - use `gpu.primitives.grid_controls` or `gpu`).

This module is deprecated. For new code, import grid control operations from
the `gpu` package or `gpu.primitives.grid_controls` module:

```mojo
# Deprecated:
from gpu.grid_controls import PDL, PDLLevel, launch_dependent_grids

# Recommended (import from top-level gpu package):
from gpu import PDL, PDLLevel, launch_dependent_grids

# Or import the module:
from gpu.primitives import grid_controls
```

This module provides Hopper PDL (Programmable Distributed Launch) operations
for controlling grid dependencies on NVIDIA GPUs.
"""

# Re-export all symbols from primitives.grid_controls for backward compatibility
from .primitives.grid_controls import (
    PDL,
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
