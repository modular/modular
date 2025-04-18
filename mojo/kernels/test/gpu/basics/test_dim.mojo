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
# RUN: %mojo-no-debug %s | FileCheck %s


from gpu.host import Dim


# CHECK-LABEL: test_dim
fn test_dim():
    print("== test_dim")

    # CHECK: (x=4, y=1, z=2)
    print(String(Dim(4, 1, 2)))
    # CHECK: (x=4, y=2)
    print(String(Dim(4, 2)))
    # CHECK: (x=4, )
    print(String(Dim(4)))

    # CHECK: (x=4, y=5)
    print(String(Dim((4, 5))))

    # CHECK: (x=4, y=2, z=3)
    print(String(Dim((4, 2, 3))))


fn main():
    test_dim()
