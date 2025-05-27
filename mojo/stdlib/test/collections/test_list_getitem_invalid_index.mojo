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
# REQUIRES: has_not
# RUN: not --crash %bare-mojo -D BUILD_TYPE=debug %s 2>&1 | FileCheck %s


# CHECK-LABEL: test_fail_list_index
fn main():
    print("== test_fail_list_index")
    # CHECK: index: 4 is out of bounds for `List` of length: 3
    nums = [1, 2, 3]
    print(nums[4])

    # CHECK-NOT: is never reached
    print("is never reached")
