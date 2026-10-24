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

# `dotted_user`'s __init__ runs `import dotted_dep.inner`, which re-exports
# `dotted_dep` with `inner` reachable through it. `dotted_dep` doesn't re-export
# `inner` itself, so that import is the only way to reach it. It must work the
# same whether the packages were built from source or precompiled.

# RUN: %parse-mojo-isolated -verify-diagnostics -I=%S/inputs %s
# RUN: rm -rf %t && mkdir -p %t
# RUN: %precompile-mojo-isolated %S/inputs/dotted_dep -o %t/dotted_dep.mojoc
# RUN: %precompile-mojo-isolated -I %t %S/inputs/dotted_user -o %t/dotted_user.mojoc
# RUN: %parse-mojo-isolated -verify-diagnostics -I=%t %s

import dotted_user


def main():
    dotted_user.dotted_dep.inner.inner_fn()
