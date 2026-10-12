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
# Checks that a package extending a struct loaded from another precompiled
# package (here std's `String`) precompiles, and that the extension works
# through the result.
#
# ===----------------------------------------------------------------------=== #

# RUN: rm -rf %t.precompiled-struct-ext && mkdir -p %t.precompiled-struct-ext
# RUN: mojo precompile %S/inputs/precompiled_struct_extension_package -o %t.precompiled-struct-ext/precompiled_struct_extension_package.mojoc
# RUN: %mojo -I %t.precompiled-struct-ext %s | FileCheck %s

from precompiled_struct_extension_package import Tagged


def tag_of[T: Tagged](value: T) -> Int:
    return value.tag()


def main():
    # CHECK: 42
    print(tag_of(String("a")))
