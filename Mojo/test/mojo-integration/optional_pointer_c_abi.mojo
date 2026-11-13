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
# Tests that an `Optional` of a pointer crosses `abi("C")` in one pointer-sized
# integer register, so that C can declare it as a nullable pointer. `None` is
# the null pointer, so the `Optional` has the pointer's size and alignment.

# RUN: mojo build --target-triple x86_64-unknown-linux-gnu --target-cpu=x86-64 --emit=llvm -o - %s | FileCheck %s
# RUN: mojo build --target-triple aarch64-unknown-linux-gnu --target-cpu=generic --emit=llvm -o - %s | FileCheck %s
# RUN: mojo build --target-triple arm64-apple-macosx --target-cpu=apple-m1 --emit=llvm -o - %s | FileCheck %s

comptime NullablePointer = Optional[MutPointer[NoneType, MutUntrackedOrigin]]


# CHECK: declare i64 @echo_nullable_pointer(i64)
@extern("echo_nullable_pointer")
def echo(value: NullablePointer) abi("C") -> NullablePointer:
    ...


# CHECK: define {{.*}}i64 @forward_nullable_pointer(i64 %0)
@export
def forward_nullable_pointer(
    value: NullablePointer,
) abi("C") -> NullablePointer:
    return echo(value)


# The value is the pointer alone, in no array and with no tag beside it.
# CHECK: define {{.*}} @pass_nullable_pointer({{(\{ )+}}ptr{{( \})+}} noundef %0)
@export
def pass_nullable_pointer(
    value: NullablePointer,
) abi("Mojo") -> NullablePointer:
    return value
