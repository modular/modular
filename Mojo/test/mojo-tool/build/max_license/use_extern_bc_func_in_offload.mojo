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
#
# Test linking an external bitcode file.
#
# ===----------------------------------------------------------------------=== #

# Can only run with asan version of amd drivers installed.
# UNSUPPORTED: asan

# RUN: %mojo %S/inputs/define_extern_bc_func_offload.mojo > %t.ll
# RUN: llvm-as %t.ll -o %t.bc
# RUN: %mojo-build --bitcode-libs=%t.bc %s -o %t
# RUN: %t | FileCheck %s

from std.compile import compile_info
from max.gpu.host.info import _mi355x_target


@extern("my_add_one")
def my_add_one(x: Pointer[Int32, MutAnyOrigin]) abi("Mojo"):
    ...


def my_add_two(y: Pointer[Int32, MutAnyOrigin]):
    my_add_one(y)
    my_add_one(y)


def main():
    print(
        compile_info[
            my_add_two,
            emission_kind="llvm",
            target=_mi355x_target,
        ]()
    )


# CHECK: call void @my_add_one
# CHECK: call void @my_add_one

# CHECK-LABEL: define {{.*}} @my_add_one
