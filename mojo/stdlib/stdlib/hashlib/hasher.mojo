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


trait Hasher:
    fn __init__(out self):
        ...

    fn _update_with_bytes(
        mut self,
        data: UnsafePointer[
            UInt8, address_space = AddressSpace.GENERIC, mut=False, **_
        ],
        length: Int,
    ):
        ...

    fn _update_with_simd(mut self, value: SIMD[_, _]):
        ...

    fn update[T: Hashable](mut self, value: T):
        ...

    fn finish(owned self) -> UInt64:
        ...
