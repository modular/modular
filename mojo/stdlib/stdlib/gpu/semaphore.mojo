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
"""GPU semaphore operations (deprecated - use `gpu.sync.semaphore`).

This module is deprecated. For new code, import from the `gpu.sync.semaphore`
module instead:

```mojo
# Deprecated:
from gpu.semaphore import Semaphore, NamedBarrierSemaphore

# Recommended:
from gpu.sync.semaphore import Semaphore, NamedBarrierSemaphore
```

This module provides semaphore synchronization primitives for GPU programming.
"""

# Re-export all public symbols from sync.semaphore for backward compatibility
from .sync.semaphore import NamedBarrierSemaphore, Semaphore
