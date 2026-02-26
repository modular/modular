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


fn append():
    list = [1, 2, 3, 4, 5]
    list.append(6)
    print(list)  # [1, 2, 3, 4, 5, 6]


fn insert():
    list = ["one", "three"]
    list.insert(1, "two")
    print(list)  # ['one', 'two', 'three']


fn extend():
    list = ["one", "two", "three"]
    more = ["four", "five"]
    list.extend(more^)  # more's values are consumed
    # print(more)      # Error: use of initialized value
    print(list)  # ['one', 'two', 'three', 'four', 'five']


fn extend_span():
    numbers = [1, 2, 3]
    more = [4, 5, 6]
    numbers.extend(Span(more))
    print(numbers)  # [1, 2, 3, 4, 5, 6]


fn extend_dtype():
    from collections import List

    numbers: List[Int64] = [1, 2]
    more = SIMD[DType.int64, 2](3, 4)
    numbers.extend(more)
    print(numbers)  # [SIMD[DType.int64, 1](1), SIMD[DType.int64, 1](2),
    #  SIMD[DType.int64, 1](3), SIMD[DType.int64, 1](4)]


fn extend_dtype_count():
    from collections import List

    numbers: List[Int64] = [1, 2]
    more = SIMD[DType.int64, 4](3, 4, 5, 6)
    numbers.extend(more, count=2)
    print(numbers)  # [SIMD[DType.int64, 1](1), SIMD[DType.int64, 1](2),
    #  SIMD[DType.int64, 1](3), SIMD[DType.int64, 1](4)]


fn pop():
    numbers = ["1", "2", "3", "4", "5"]
    value = numbers.pop()
    print(value)  # 5
    print("length", len(numbers))  # length 4
    value = numbers.pop(2)
    print(value)  # 3
    print(numbers)  # ['1', '2', '4']
    value = numbers.pop(-2)
    print(value)  # 2, negative index


fn resize():
    list = ["z", "y", "x", "w"]
    list.resize(3, "v")
    print(list)  # ['z', 'y', 'x']
    list.resize(6, "v")
    print(list)  # ['z', 'y', 'x', 'v', 'v', 'v']


fn unsafe_resize():
    list = [1, 2, 3]
    list.resize(
        unsafe_uninit_length=5
    )  # Indices 3 and 4 are uninitialized memory
    print(len(list))  # 5
    list[3] = 10
    list[4] = 20
    print(list)  # [1, 2, 3, 10, 20]


fn shrink():
    numbers = [1, 2, 3, 4, 5, 6]
    numbers.shrink(2)
    print(numbers)  # [1, 2]
    # numbers.shrink(8)               # Error: new size is bigger than current


fn reverse():
    list = ["o", "l", "l", "e", "H"]
    list.reverse()
    print("".join(list))  # Hello


fn clear():
    list = ["o", "l", "l", "e", "H"]
    print(len(list))  # 5
    list.clear()
    print(len(list))  # 0


fn steal():
    from collections import List
    from memory import ArcPointer

    list: List[Int64] = [1, 2, 3, 4]
    ptr = list.steal_data()
    for idx in range(4):
        print(ptr[idx], end=" ")
    print()  # Output: 1 2 3 4
    for idx in range(4):
        (ptr + idx).destroy_pointee()
    ptr.free()


fn count():
    list = ["a", "b", "c", "b", "b", "a", "c"]
    print(list.count("b"))  # 3


fn main():
    append()
    insert()
    extend()
    extend_span()
    extend_dtype()
    extend_dtype_count()
    pop()
    resize()
    unsafe_resize()
    shrink()
    reverse()
    clear()
    steal()
    count()
