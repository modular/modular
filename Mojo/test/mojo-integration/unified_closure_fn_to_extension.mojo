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
# RUN: %mojo %s | FileCheck %s


trait HasE:
    comptime E: Movable


struct Base(HasE):
    comptime E = Int


struct Fwd[T: HasE](HasE):
    comptime E = Self.T.E


def ten(x: Int) -> Int:
    return x * 10


def apply[T: HasE, F: def(T.E) -> Int](f: F):
    print(f(rebind[T.E](10)))


def main():
    # CHECK: 100
    apply[Fwd[Base]](ten)
