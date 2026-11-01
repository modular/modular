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

# The SIMD conformance to `Tr` reaches this file only through the wildcard.
from .impls import *


trait Tr:
    def tag(self) -> Int:
        ...


def check[T: AnyType]():
    comptime if conforms_to(T, Tr):
        print("conforms")
    else:
        print("does not conform")
