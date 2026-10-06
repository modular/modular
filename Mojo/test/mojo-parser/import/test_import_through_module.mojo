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

# Tests that a user sees a useful error message if they use the wrong import
# syntax, trying to import through a *module*. Only packages can have modules or
# packages imported from them.

# RUN: %parse-mojo-isolated -split-input-file -I=%S/inputs -verify-diagnostics %s

# expected-error @+1 {{'module' is a module, not a package; it has no nested module or package 'bar'}}
import test_package.test_nested_package.module.bar

# // -----

# expected-error @+1 {{'module' is a module, not a package; it has no nested module or package 'module2'}}
import test_package.test_nested_package.module.module2.bar

# // -----

# expected-error @+1 {{'module' is a module, not a package; it has no nested module or package 'module2'}}
from test_package.test_nested_package.module.module2 import bar

# // -----

# expected-error @+1 {{'module' is a module, not a package; it has no nested module or package 'bar'}}
from test_package.test_nested_package.module.bar import woof

# // -----

# expected-error @+1 {{'module' is a module, not a package; it has no nested module or package 'module2'}}
from test_package.test_nested_package.module.module2.bar import woof
