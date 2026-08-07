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


@no_inline
def takeIt[F: def[width: Int](idx: Int) -> Scalar[DType.int]](impl: F):
    print(impl.__call__[1](0))


def emitLoad(x: SIMD[DType.int, 1]):
    var ptr = alloc[SIMD[DType.int, 1]]({count = 1}).unsafe_leak()
    ptr.store(x)
    var count = Scalar[DType.int](0)

    @no_inline
    def foo[width: Int](idx: Int) {mut count, read ptr} -> Scalar[DType.int]:
        var vec = ptr.load[width=width](idx).cast[DType.int]()
        count = count + rebind[type_of(count)](vec)
        return count

    takeIt(foo)
