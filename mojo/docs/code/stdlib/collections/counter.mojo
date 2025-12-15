# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from collections import Counter


fn init1():
    var counter = Counter[String]("a", "a", "a", "b", "b", "c", "d", "c", "c")
    print(counter["a"])  # prints 3
    print(counter["b"])  # prints 2


fn init2():
    var counter = Counter[String]("a", "a", "a", "b", "b", "c", "d", "c", "c")
    print(counter["a"])  # print 3
    print(counter["b"])  # print 2


fn fromkeys():
    counter = Counter[String].fromkeys(["a", "b", "c"], 1)
    print(counter["a"])  # output: 1


fn is_le():
    counter = Counter[Int]([1, 2, 1, 2, 3, 3, 3])
    other = Counter[Int].fromkeys([1, 2, 3], 10)
    print(counter.le(other))  # output: True
    counter[3] += 20
    print(counter.le(other))  # output: False


fn is_lt():
    counter = Counter[Int]([1, 2, 1, 2, 3, 3])
    other = Counter[Int].fromkeys([1, 2, 3], 3)
    print(counter.lt(other))  # output: True
    counter[1] += 1
    print(counter.lt(other))  # output: False


fn is_gt():
    counter = Counter[Int]([1, 2, 1, 2, 3, 3])
    other = Counter[Int].fromkeys([1, 2, 3], 3)
    print(other.gt(counter))  # output: True
    counter[1] += 1
    print(other.gt(counter))  # output: False


fn is_ge():
    counter = Counter[Int]([1, 2, 1, 2, 3, 3, 3])
    other = Counter[Int].fromkeys([1, 2, 3], 10)
    print(other.ge(counter))  # output: True
    counter[3] += 20
    print(other.ge(counter))  # output: False


fn get():
    counter = Counter[String].fromkeys(["a", "b", "c"], 1)
    print(counter.get("a").or_else(0))  # output: 1
    print(counter.get("d").or_else(0))  # output: 0


fn get_default():
    from collections import Counter

    counter = Counter[String].fromkeys(["a", "b", "c"], 1)
    print(counter.get("a", default=0))  # output: 1
    print(counter.get("d", default=0))  # output: 0


fn pop():
    from collections import Counter

    counter = Counter[String].fromkeys(["a", "b", "c"], 1)
    print(counter.get("b").or_else(0))  # output: 1
    try:
        count = counter.pop("b")
        print(count)  # output: 1
        print(counter.get("b").or_else(0))  # output: 0
    except e:
        print(e)  # KeyError if the key was not in the counter


fn pop_default():
    counter = Counter[String].fromkeys(["a", "b", "c"], 1)
    count = counter.pop("b", default=100)
    print(count)  # output: 1
    count = counter.pop("not-a-key", default=0)
    print(count)  # output 0


fn keys():
    counter = Counter[String].fromkeys(["d", "b", "a", "c"], 1)
    var key_list = List[String]()
    for key in counter.keys():
        key_list.append(key)
    sort(key_list[:])
    print(key_list)  # output: ['a', 'b', 'c', 'd']


fn values():
    # Construct `counter`
    counter = Counter[Int]([1, 2, 3, 1, 2, 1, 1, 1, 2, 5, 2, 9])

    # Find most populous key
    max_count: Int = Int.MIN
    for count in counter.values():
        if count > max_count:
            max_count = count

    # Max count is the five ones
    print(max_count)  # output: 5


fn items():
    counter = Counter[Int]([1, 2, 1, 2, 1, 1, 1, 2, 2])
    for count in counter.items():
        print(count.key, count.value)
    # output: 1 5
    # output: 2 4


fn clear():
    counter = Counter[Int]([1, 2, 1, 2, 1, 1, 1, 2, 2])
    print(counter.total())  # output: 9 (5 ones + 4 twos)
    counter.clear()  # Removes both entries
    print(counter.total())  # output: 0


fn popitem():
    counter = Counter[String].fromkeys(["a", "b", "c"], 5)
    try:
        tuple = counter.popitem()
        print(tuple._value, tuple._count)
        # output: probably c 5 since that was last in
    except e:
        print(e)  # KeyError if the key was not in the counter


fn total():
    counter = Counter[Int]([1, 2, 1, 2, 1, 1, 1, 2, 2])
    print(counter.total())  # output: 9 (5 ones + 4 twos)
    counter.clear()  # Removes both entries
    print(counter.total())  # output: 0


fn most_common():
    counter = Counter[Int]([1, 2, 1, 2, 3, 3, 3, 1, 1, 1, 6, 6, 2, 2, 7])
    for tuple in counter.most_common(2):
        print(tuple._value, tuple._count)
        # output: 1 5
        # output: 2 4


fn elements():
    counter = Counter[Int]([1, 2, 1, 2, 3, 3, 3, 1, 1, 1, 6, 6, 2, 2, 7])
    print(counter.elements())
    # output: [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 6, 6, 7]


fn update():
    counter = Counter[Int]([1, 2, 1, 2, 3, 3, 3])
    other = Counter[Int].fromkeys([1, 2, 3], 10)
    print(counter[1])  # output: 2
    counter.update(other)
    print(counter[1])  # output: 12


fn subtract():
    counter = Counter[Int]([1, 2, 1, 2, 3, 3, 3])
    other = Counter[Int].fromkeys([1, 2, 3], 10)
    print(counter[1])  # output: 2
    counter.subtract(other)
    print(counter[1])  # output: -8


fn main():
    init1()
    init2()
    fromkeys()
    is_le()
    is_lt()
    is_gt()
    is_ge()
    get()
    get_default()
    pop()
    pop_default()
    keys()
    values()
    items()
    clear()
    popitem()
    total()
    most_common()
    elements()
    update()
    subtract()
