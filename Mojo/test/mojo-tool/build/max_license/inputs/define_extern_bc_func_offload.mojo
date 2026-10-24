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

from std.compile import compile_info
from max.gpu.host.info import _mi355x_target


@export("my_add_one")
def my_add_one(x: Pointer[Int32, MutAnyOrigin]) abi("C"):
    x[] += 1


def main():
    var llvm_text = compile_info[
        my_add_one,
        emission_kind="llvm",
        target=_mi355x_target,
    ]()
    print(llvm_text)
