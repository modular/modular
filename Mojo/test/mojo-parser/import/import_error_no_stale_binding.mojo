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

# RUN: %parse-mojo-isolated -split-input-file -I=%S/inputs -verify-diagnostics %s

# A structural import failure records its diagnostic keyed by the full dotted
# path, but must not bind that name in the importing scope. An
# escaped-identifier reference with the same spelling is an unrelated name: it
# must not silently resolve to the stale failure record.

# expected-error @+1 {{'module' is a module, not a package; it has no nested module or package 'bar'}}
import test_package.test_nested_package.module.bar


def use_stale() -> Int:
    # expected-error @+1 {{use of unknown declaration 'test_package.test_nested_package.module.bar'}}
    return `test_package.test_nested_package.module.bar`()


# // -----

# Likewise, a real binding of the same spelling must not collide with the
# failure record.

# expected-error @+1 {{'module' is a module, not a package; it has no nested module or package 'bar'}}
import test_package.test_nested_package.module.bar


def `test_package.test_nested_package.module.bar`() -> Int:
    return 7


def use_real() -> Int:
    return `test_package.test_nested_package.module.bar`()
