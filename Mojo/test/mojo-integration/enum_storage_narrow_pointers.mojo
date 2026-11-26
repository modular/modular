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
# RUN: mojo build --target-triple i686-unknown-linux-gnu --emit=llvm -o - %s | FileCheck %s

# On a target with 4-byte pointers, a compile-time union's payload holds a
# 4-byte string pointer followed by the string's length. Matching the union
# reads the union field by the memory's marks, at the target's pointer size.
# CHECK: target datalayout = "{{.*}}-p:32:32
# CHECK-LABEL: define {{.*}} @main(
# CHECK: call void @"std::io::io::print{{.*}}"(i32 4,


__union Kind:
    case named(StaticString)
    case other()


def name_bytes[k: Kind]() -> Int:
    comptime if case .named(s) = k:
        return s.byte_length()
    return 0


def main():
    comptime k = Kind.named("host")
    print(name_bytes[k]())
