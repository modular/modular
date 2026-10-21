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
"""CPU-only release checks of the grouped token-row metadata.

The static extents are `comptime assert`s in `_launch_block_scaled_grouped`;
the token rows are the only dynamic ones. This test creates no GPU context and
launches no device work.
"""

from std.testing import assert_true
from linalg.matmul.gpu.amd.block_scaled_grouped_matmul_amd import (
    _validate_block_scaled_grouped_rows,
)


def _bad(c_rows: Int, a_rows: Int, a_scale_rows: Int) raises:
    var rejected = False
    try:
        _validate_block_scaled_grouped_rows(c_rows, a_rows, a_scale_rows)
    except:
        rejected = True
    assert_true(rejected, "malformed row metadata must be rejected in release")


def main() raises:
    _bad(2, 3, 3)
    _bad(3, 3, 2)
    _bad(-1, -1, -1)
    _validate_block_scaled_grouped_rows(3, 3, 3)
    _validate_block_scaled_grouped_rows(0, 0, 0)
    print("row metadata: 3 malformed cases rejected, 2 valid cases pass")
