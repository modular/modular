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

# ===----------------------------------------------------------------------=== #
# Package file
# ===----------------------------------------------------------------------=== #
"""Basic mathematical utilities.

This package defines a collection of utility functions for manipulating
integer values.

You can import these APIs from the `my_math` package. For example:

```mojo
from my_math import dec, inc
```

The `inc()` function performs a simple increment:

```mojo
%# from testing import assert_equal
from my_math import inc
a = 1
b = inc(a)  # b = 2
%# assert_equal(b, 2)
```

However, `inc()` raises an error if it would result in integer overflow:

```mojo
c = 0
try:
    c = inc(Int.MAX)
except e:
    print(e)
%#     assert_equal("inc overflow", String(e))
```

"""

from .utils import dec, inc
