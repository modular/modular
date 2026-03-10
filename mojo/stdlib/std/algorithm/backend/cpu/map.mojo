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
"""CPU implementation of the map function."""


# ===-----------------------------------------------------------------------===#
# Map
# ===-----------------------------------------------------------------------===#


@always_inline
fn map[
    origins: OriginSet, //, func: fn(Int) capturing[origins] -> None
](size: Int):
    """Maps a function over the integer range [0, size).
    This lets you apply an integer index-based operation across data
    captured by the mapped function (for example, an indexed buffer).

    Parameters:
        origins: Capture origins for mapped function.
        func: Parameterized function applied at each index.

    Args:
        size: Number of elements in the index range.

    For example:

    ```mojo
    from std.algorithm import map

    def main() raises:
        # Create list with initial values to act on
        var list: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0]

        # Function applied to the value at each index
        @parameter
        fn exponent_2(idx: Int):
            list[idx] = 2.0 ** list[idx]

        # Apply the mapped function across the index range
        map[exponent_2](len(list))

        # Show results
        for idx in range(len(list)):
            print(list[idx])
    ```

    Example output:

    ```output
    2.0
    4.0
    8.0
    16.0
    32.0
    ```

    :::note
    Don't confuse `algorithm.map` (this eager, index-based helper) with
    [`iter.map`](https://docs.modular.com/mojo/std/iter/map/),
    which returns a lazy iterator that applies a function to each element.
    :::

    """
    for i in range(size):
        func(i)
