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
"""Contains a bounds check which is on by default for CPU and off by default for GPU.
"""

from std.reflection import SourceLocation
from std.sys.info import is_gpu
from std.reflection import call_location


@always_inline
def check_bounds[
    cpu_default: Bool = True,
](idx: Some[Indexer], size: Int, location: OptionalReg[SourceLocation] = None):
    """Bounds check which is on by default for CPU, and off by default for GPU.

    You can turn off CPU bounds checks for a specific collection by setting
    `check_bounds[cpu_default=False](idx, size, loc)`, but turn them on for
    tests with:

    ```bash
    mojo build -D ASSERT=all main.mojo
    ```

    The defaults are optimal for most use cases, where CPU bounds checks are
    cheap and valuable, but GPU bounds checks are too expensive due to branching
    costs. For maximum performance you can turn off all asserts regardless of
    defaults with:

    ```bash
    mojo build -D ASSERT=none main.mojo
    ```

    Parameters:
        cpu_default: If the bounds check is on by default on CPU.

    Args:
        idx: The index for the bounds check.
        size: The size of the container, and first index that would be out of range.
        location: `SourceLocation` shown on assert error. Defaults to showing the callsite
            two levels of function calls above this one. So if `check_bounds` is called
            inside a `__getitem__` method, it will show the source location where
            the incorrect index was provided.
    """
    debug_assert[
        assert_mode="safe" if cpu_default and not is_gpu() else "none",
    ](
        UInt(index(idx)) < UInt(size),
        "index ",
        index(idx),
        " is out of bounds, valid range is 0 to ",
        size - 1,
        location=location.value() if location else call_location[
            inline_count=2
        ](),
    )
