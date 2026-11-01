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

from max.gpu.host.compile import _compile_code
from max.gpu.host import get_gpu_target
from std.sys._assembly import inlined_assembly

# RUN: %mojo -debug-level full %s

# COM: MOCO-1881 This file tests having one function to be both kernel function
# and callee in another kernel.


def alloc[p: UInt32](i: UInt32, j: Pointer[UInt32, MutAnyOrigin]):
    j[unsafe_offset=0] = p + i


def test_fn(v: Pointer[UInt32, MutAnyOrigin]):
    alloc[1](1, v)


def main() raises:
    # use alloc[1] as kernel entry
    print(
        _compile_code[
            alloc[1],
            target=get_gpu_target["sm_100a"](),
        ]()
    )

    # alloc[1] is a callee in kernel entry test_fn
    print(
        _compile_code[
            test_fn,
            target=get_gpu_target["sm_100a"](),
        ]()
    )
