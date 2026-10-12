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

# RUN: %parse-mojo-isolated -I=%S/inputs %s | FileCheck %s

# An import also brings in every extension visible in the source module,
# including ones that module imported itself. Thing's extension reaches the file
# scope from dup_scope_ext_base, and reaches main's scope a second time through
# the local import from dup_scope_ext_reexport (whose own import of Thing is
# already resolved by then, via make_thing). Member lookup walks both scopes and
# must count the extension once, or `extended` is ambiguous with itself.
from dup_scope_ext_base import Thing
from dup_scope_ext_reexport import make_thing


# CHECK-LABEL: lit.fn @"main
def main():
    from dup_scope_ext_reexport import Other

    _ = Other()
    var t = make_thing()
    # CHECK: lit.call {{.*}}extended
    _ = t.extended()
