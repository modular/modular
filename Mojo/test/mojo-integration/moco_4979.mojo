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

# Verify that a bytecode reference to the universal closure trait resolves even
# when this compilation has not synthesized that trait yet. Before the fix, the
# lookup failed and interrupted the reference walk, so decls reachable only
# through the trait's parameters (`Shape` here) were never loaded, and the
# KGEN verifier reported "does not reference a KGEN type declaration".

# RUN: mkdir -p %t.moco-4979
# RUN: mojo precompile %S/inputs/moco_4979_package -o %t.moco-4979/moco_4979_package.mojoc
# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo -I %t.moco-4979 %s | FileCheck %s

from moco_4979_package import use_hooks


def test():
    use_hooks()


# CHECK: lit.struct.decl @Shape
