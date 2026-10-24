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

# A package's children are reachable only through an import, or a re-export
# from its __init__, whether the package was built from source or loaded from a
# precompiled file. Each split runs against both, and they must agree.

# RUN: %parse-mojo-isolated -split-input-file -verify-diagnostics -I=%S/inputs %s
# RUN: rm -rf %t && mkdir -p %t
# RUN: %precompile-mojo-isolated %S/inputs/hidden_children_pkg -o %t/hidden_children_pkg.mojoc
# RUN: %parse-mojo-isolated -split-input-file -verify-diagnostics -I=%t %s

# A bare import does not expose a submodule that __init__ doesn't re-export.
import hidden_children_pkg


def main():
    # expected-error @+1 {{package 'hidden_children_pkg' has no declaration 'sub'}}
    _ = hidden_children_pkg.sub


# // -----

# A wildcard from a sub-package without an __init__ brings in none of that
# sub-package's children.
import hidden_children_pkg


def main():
    # expected-error @+1 {{package 'hidden_children_pkg' has no declaration 'child'}}
    _ = hidden_children_pkg.child


# // -----

# Explicitly imported children are reachable, including through a sub-package
# without an __init__.
import hidden_children_pkg.sub
from hidden_children_pkg.nsdir.child import child_fn


def main():
    hidden_children_pkg.sub.sub_fn()
    child_fn()
