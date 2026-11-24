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
# Checks that an extension conformance which reaches the trait's file only
# through a wildcard import is still visible to `conforms_to` when the generic
# testing it is instantiated through a second precompiled package. The
# forwarder's import of the trait's module was resolved when it was
# precompiled, so nothing in this compile imports from that module by name.
#
# ===----------------------------------------------------------------------=== #

# RUN: rm -rf %t.wildcard-ext && mkdir -p %t.wildcard-ext
# RUN: mojo precompile %S/inputs/wildcard_extension_package -o %t.wildcard-ext/wildcard_extension_package.mojoc
# RUN: mojo precompile -I %t.wildcard-ext %S/inputs/wildcard_extension_forwarder -o %t.wildcard-ext/wildcard_extension_forwarder.mojoc
# RUN: %mojo -I %t.wildcard-ext %s | FileCheck %s

from wildcard_extension_forwarder import forward


def main():
    # CHECK: {{^conforms$}}
    forward[Float64]()
