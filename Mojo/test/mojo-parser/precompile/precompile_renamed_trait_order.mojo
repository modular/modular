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

# A trait composition orders its symbols by name. A renamed precompiled
# package keeps that order for its output name, also when the rename moves
# the package to the other side of `std`.

# RUN: rm -rf %t.dir && mkdir -p %t.dir
# RUN: %precompile-mojo-isolated %S/inputs/renamed_trait_order/alib -o %t.dir/zlib.mojoc
# RUN: kgen-opt %t.dir/zlib.mojoc | FileCheck %s --check-prefix=ARTIFACT
# RUN: %parse-mojo-isolated -I %t.dir %s | FileCheck %s

# ARTIFACT: !lit.trait<@std::@builtin::@stubs::@AnyType, @std::@builtin::@stubs::@Copyable, @std::@builtin::@stubs::@Movable, @zlib::@__init__::@Bound>

# CHECK: lit.call @zlib::@__init__::@"split[{{.*}}]($0)"{{.*}}<:!AnyType_Copyable_Movable_Bound !MyBound>

from zlib import Bound, split


@fieldwise_init
struct MyBound(Bound):
    pass


def main():
    split(MyBound())
