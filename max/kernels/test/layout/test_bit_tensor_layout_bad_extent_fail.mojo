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
# A mode whose extent is not a power of two has no whole number of
# coordinate bits, so BitTensorLayout rejects it at compile time.

from layout import row_major
from layout.bit_layout import BitLayout
from layout.two_level_layout import bit_tensor_layout


def main():
    comptime images = BitLayout().run(0, 2, 0)
    # CHECK: BitTensorLayout mode 0 needs a static power-of-two extent, not 6
    _ = bit_tensor_layout[images, type_of(row_major[6]())]()
