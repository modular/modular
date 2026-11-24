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

# RUN: not %mojo-no-debug %s 2>&1 | FileCheck %s
#
# Dependent images send two points to one offset, so BitTensorLayout
# rejects them at compile time.

from layout import row_major
from layout.bit_layout import BitLayout
from layout.two_level_layout import bit_tensor_layout


def main():
    # 1 ^ 2 = 3: points 3 and 4 both land at offset 0.
    comptime images = BitLayout().put(0, 1).put(1, 2).put(2, 3)
    # CHECK: BitTensorLayout images must be nonzero and linearly independent
    _ = bit_tensor_layout[images, type_of(row_major[8]())]()
