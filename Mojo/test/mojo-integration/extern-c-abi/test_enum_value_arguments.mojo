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
# RUN: %mojo-build -I %S/.. --target-triple x86_64-unknown-linux-gnu --target-cpu x86-64 --emit llvm %s -o - | FileCheck %s --check-prefix=EXT
# RUN: %mojo-build -I %S/.. --target-triple x86_64-unknown-linux-gnu --target-cpu x86-64 -O3 --emit llvm %s -o - | FileCheck %s --check-prefix=EXT
# RUN: %mojo-build -I %S/.. --target-triple arm64-apple-macosx --emit llvm %s -o - | FileCheck %s --check-prefix=EXT
# RUN: %mojo-build -I %S/.. --target-triple arm64-apple-macosx -O3 --emit llvm %s -o - | FileCheck %s --check-prefix=EXT
# RUN: %mojo-build -I %S/.. --target-triple aarch64-unknown-linux-gnu --emit llvm %s -o - | FileCheck %s --check-prefix=NOEXT
# RUN: %mojo-build -I %S/.. --target-triple aarch64-unknown-linux-gnu -O3 --emit llvm %s -o - | FileCheck %s --check-prefix=NOEXT
# RUN: mkdir -p %t.dir
# RUN: %mojo-build -I %S/.. -Xlinker %S/libc_abi_reference.lo %s -o %t.dir/enum-values
# RUN: %t.dir/enum-values | FileCheck %s --check-prefix=RUNTIME
# RUN: %mojo-build -I %S/.. -O3 -Xlinker %S/libc_abi_reference.lo %s -o %t.dir/enum-values-O3
# RUN: %t.dir/enum-values-O3 | FileCheck %s --check-prefix=RUNTIME

# Scalar enum arguments are noundef. Narrow returns are extended on x86-64
# and Darwin; Linux AArch64 leaves them unextended.
# EXT-DAG: declare {{(!dbg ![0-9]+ )?}}zeroext i8 @c_enum_two(i8 noundef)
# EXT-DAG: define dso_local zeroext i8 @forward_enum_two(i8 noundef %{{[0-9]+}})
# NOEXT-DAG: declare {{(!dbg ![0-9]+ )?}}i8 @c_enum_two(i8 noundef)
# NOEXT-DAG: define dso_local i8 @forward_enum_two(i8 noundef %{{[0-9]+}})
# EXT-DAG: declare {{(!dbg ![0-9]+ )?}}zeroext i16 @c_enum_wide(i16 noundef)
# EXT-DAG: define dso_local zeroext i16 @forward_enum_wide(i16 noundef %{{[0-9]+}})
# NOEXT-DAG: declare {{(!dbg ![0-9]+ )?}}i16 @c_enum_wide(i16 noundef)
# NOEXT-DAG: define dso_local i16 @forward_enum_wide(i16 noundef %{{[0-9]+}})
# RUNTIME: enum values: C calls and callbacks passed

from enum_many_cases import Many
from std.ffi import external_call
from std.testing import assert_equal, assert_true


__enum Two:
    case c0
    case c1


@extern("c_enum_two")
def c_enum_two(value: Two) abi("C") -> Two:
    ...


@export
def forward_enum_two(value: Two) abi("C") -> Two:
    return c_enum_two(value)


@extern("c_enum_wide")
def c_enum_wide(value: Many) abi("C") -> Many:
    ...


@export
def forward_enum_wide(value: Many) abi("C") -> Many:
    return c_enum_wide(value)


def main() raises:
    assert_true(forward_enum_two(.c0) == .c1)
    assert_true(forward_enum_two(.c1) == .c0)
    assert_true(forward_enum_wide(.c0) == .c256)
    assert_true(forward_enum_wide(.c256) == .c0)
    assert_equal(
        external_call["c_enum_callbacks", Int64](
            forward_enum_two,
            forward_enum_wide,
        ),
        1,
    )
    print("enum values: C calls and callbacks passed")
