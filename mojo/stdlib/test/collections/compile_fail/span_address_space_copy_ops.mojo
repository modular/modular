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

# RUN: not %mojo %s 2>&1 | FileCheck %s

# Element-copying operations (`fill`, iteration, writing) must be rejected on
# spans viewing a non-default address space; only address-only operations are
# address-space-generic.

from std.memory import unsafe_stack_allocation


def _shared_tile() -> (
    Span[
        mut=True,
        Float32,
        MutUntrackedOrigin,
        address_space=AddressSpace.SHARED,
    ]
):
    var smem = unsafe_stack_allocation[
        4, Float32, address_space=AddressSpace.SHARED
    ]()
    return {unsafe_ptr = smem, length = 4}


def main():
    var tile = _shared_tile()

    # CHECK: error: invalid call to 'fill'
    tile.fill(0.0)

    # CHECK: error: no matching method in call to '__iter__'
    for x in tile:
        print(x)

    # CHECK: error: no matching function in initialization
    print(String(tile))
