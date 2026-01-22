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

from sys import align_of
from memory import UnsafePointer


@fieldwise_init
@align(64)
struct CacheAligned:
    var data: Int


fn use_aligned():
    # Stack allocation
    var _ = CacheAligned(42)  # `stack_value`

    # Heap allocation
    var heap_ptr = alloc[CacheAligned](1)
    heap_ptr.free()


fn main():
    use_aligned()
