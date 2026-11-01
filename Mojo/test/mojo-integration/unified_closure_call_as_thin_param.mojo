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

# COM: A closure's `__call__` is a `thin` function value, so it is accepted by
# COM: a parameter declared as `def(*args: *ArgTypes) thin -> None`.


def take_thin[
    ArgTypes: TypeList[Trait=AnyType, ...],
    //,
    func: def(*args: *ArgTypes) thin -> None,
]():
    print("accepted")


def forward[F: def() -> None](f: F):
    take_thin[F.__call__]()


# CHECK-LABEL: == test_closure_call_as_thin_param
def test_closure_call_as_thin_param():
    print("== test_closure_call_as_thin_param")
    var a = 1

    def cl() {var a}:
        print(a)

    # CHECK: accepted
    take_thin[type_of(cl).__call__]()
    # CHECK: accepted
    forward(cl)


def main():
    test_closure_call_as_thin_param()
