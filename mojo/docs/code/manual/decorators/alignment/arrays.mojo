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

from sys import align_of, size_of
from memory import UnsafePointer
from testing import *


@align(64)
struct CacheAligned:
    var data: Int  # 8 bytes


fn demonstrate_array_stride() raises:
    var arr = alloc[CacheAligned](4)

    print(align_of[CacheAligned]())  # 64
    print(size_of[CacheAligned]())  # 8

    assert_equal(64, align_of[CacheAligned](), "align should be 64")
    assert_equal(8, size_of[CacheAligned](), "size_of should be 8")

    # Only arr[0] is guaranteed to be 64-byte aligned
    # Subsequent elements follow at 8-byte intervals

    arr.free()


fn main() raises:
    demonstrate_array_stride()
