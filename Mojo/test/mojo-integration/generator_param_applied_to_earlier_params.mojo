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

# A parameter whose type applies an earlier generator parameter to other
# earlier parameters (`G: GT[F, T]`) must keep that type applied when the
# function is referenced with its parameters still unbound.

# RUN: %mojo %s | FileCheck %s

comptime GeneratorTypeGeneratorType[
    From: type_of(AnyType), To: type_of(AnyType)
] = __generator_type[From: From] To

comptime Generator[From: Copyable] = Int


def takeGenerator[
    F: type_of(AnyType),
    T: type_of(AnyType),
    //,
    GT: __generator_type[From: type_of(AnyType), To: type_of(AnyType)] type_of(
        __generator_type[From: From] To
    ),
    G: GT[F, T],
]():
    pass


def main():
    takeGenerator[GeneratorTypeGeneratorType, Generator]()
    # CHECK: done
    print("done")
