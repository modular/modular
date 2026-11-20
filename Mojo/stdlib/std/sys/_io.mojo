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
"""IO constants and functions."""
comptime stdin = BorrowedFileDescriptor[ImmStaticOrigin](
    unsafe_from_raw=RawFileDescriptor(unsafe_unchecked_value=0)
)
comptime stdout = BorrowedFileDescriptor[ImmStaticOrigin](
    unsafe_from_raw=RawFileDescriptor(unsafe_unchecked_value=1)
)
comptime stderr = BorrowedFileDescriptor[ImmStaticOrigin](
    unsafe_from_raw=RawFileDescriptor(unsafe_unchecked_value=2)
)
