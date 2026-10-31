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

from layout import Coord, Idx
from layout.owned_tensor import OwnedGPUTensor
from layout.tile_layout import Layout


def main():
    # A row stride of 6 pads each 4-element row, so the layout spans more
    # slots than it has elements.
    var padded = Layout(Coord(Idx[2], Idx[4]), Coord(Idx[6], Idx[1]))

    # CHECK: GPUStorage requires a compact layout
    _ = OwnedGPUTensor[.float32, .SHARED](padded)
