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
# RUN: kgen -emit=llvm %s -o - | FileCheck %s


# CHECK: define internal void @[[KERNEL2:.*llvm_metadata::nested_kernel2.*]]()
# CHECK: define internal void @[[KERNEL1:.*llvm_metadata::kernel1.*]]()
# CHECK: define internal void @[[KERNEL:.*llvm_metadata::kernel.*]]()


@no_inline
@__llvm_metadata(
    `nvvm.maxntid`=__mlir_attr.`#pop.array<8, 8, 16> : !pop.array<3, i32>`
)
def kernel[x: Int, y: Int, z: Int]():
    pass


comptime mname = "nvvm.maxntid".value


@no_inline
@__llvm_metadata(mname=__mlir_attr.`#pop.array<8, 8, 16> : !pop.array<3, i32>`)
def kernel1[x: Int, y: Int, z: Int]():
    pass


@no_inline
def kernel2[x: Int, y: Int, z: Int]():
    @no_inline
    @__llvm_metadata(
        mname=__mlir_attr.`#pop.array<8, 8, 16> : !pop.array<3, i32>`
    )
    def nested_kernel2[x: Int, y: Int, z: Int]():
        pass

    nested_kernel2[8, 8, 16]()
    pass


@export
def top() abi("Mojo"):
    kernel[8, 8, 16]()
    kernel1[8, 8, 16]()
    kernel2[8, 8, 16]()


# CHECK attributes #1 - { {{.*}} "nvvm.maxntid"=8,8,16" }
