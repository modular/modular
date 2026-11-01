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
# Trait-body decls alias the parent trait's FnOps, so the unparsed-decl erase in
# DeclResolver::resolveAllReferencedFrom sees several decls per operation.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics %s


trait Greeter:
    def greet(self) -> Int:
        ...

    # Unreferenced, so these stay unparsed and are the ones erased.
    def unused_one(self) -> Int:
        ...

    def unused_two(self) -> Int:
        ...


trait Describer(Greeter):
    def describe(self) -> Int:
        ...


@fieldwise_init
struct Person(Describer):
    var tag: Int

    def greet(self) -> Int:
        return self.tag

    def unused_one(self) -> Int:
        return 1

    def unused_two(self) -> Int:
        return 2

    def describe(self) -> Int:
        return self.greet()


def use_describer[T: Describer](value: T) -> Int:
    return value.describe()
