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
from sys.compile import SanitizeAddress, is_compile_time
from ffi import external_call


@always_inline
fn __sanitizer_annotate_contiguous_container(
    beg: UnsafePointer[NoneType],
    end: UnsafePointer[NoneType],
    old_mid: UnsafePointer[NoneType],
    new_mid: UnsafePointer[NoneType],
):
    # follows __annotate_contiguous_container from __debug_utils
    # https://github.com/llvm/llvm-project/blob/main/libcxx/include/__debug_utils/sanitizers.h
    comptime if SanitizeAddress:
        if not is_compile_time() and beg:
            external_call[
                "__sanitizer_annotate_contiguous_container", NoneType
            ](beg, end, old_mid, new_mid)
