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

# RUN: mkdir -p %t.cross-package-extension
# RUN: mojo precompile %S/inputs/simple_struct_package -o %t.cross-package-extension/simple_struct_package.mojoc
# RUN: mojo precompile -I %t.cross-package-extension %S/inputs/extension_only_package -o %t.cross-package-extension/extension_only_package.mojoc
# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo -I %t.cross-package-extension %s --kgen-print-inline-type-values | kgen-opt -lower-semantic-cf -check-lifetimes -lower-lit | FileCheck %s

from simple_struct_package import MyStruct
from extension_only_package import MyStruct


# CHECK-LABEL: kgen.generator @"struct_extensions_cross_package::test
def test():
    var car = MyStruct(0)

    # Call method from the original struct
    car.accelerate()
    # CHECK: kgen.call @"simple_struct_package::simple::MyStruct::accelerate

    # Call method from the extension
    var current_speed = car.get_speed()
    # CHECK: kgen.call @"extension_only_package::my_extension::extension:MyStruct::get_speed
