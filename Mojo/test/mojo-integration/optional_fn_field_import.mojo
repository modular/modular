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

# The layout of `Optional[Item]` checks whether `Tuple[Item]` conforms to
# `UnsafeNicheable`. Imported from a package, `Backend` lowers after the std
# struct generators, so that check reaches an already lowered `Tuple` with LIT
# parameter values, which crashed the compiler (issue #7687).

# RUN: %mojo -I %S/inputs %s | FileCheck %s

from optional_fn_field_package.backend import backend_is_none


def main():
    # CHECK: none True
    print("none", backend_is_none())
