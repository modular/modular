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

from collections import Counter
from testing import *


fn init1() raises:
    from collections import Counter

    var counter = Counter[String]("a", "a", "a", "b", "b", "c", "d", "c", "c")
    assert_true(counter["a"] == 3)
    assert_true(counter["b"] == 2)


fn fromkeys() raises:
    from collections import Counter

    var counter = Counter[String].fromkeys(["a", "b", "c"], 1)
    assert_true(counter["a"] == 1)


fn string() raises:
    from collections import Counter

    var c = Counter[String]("a", "a", "a", "b", "b", "c", "d", "c", "c")
    var _ = String(c)
    # Would be "Counter({'a': 3, 'c': 3, 'b': 2, 'd': 1})"
    # No test because exact output formatting may change over time


fn is_lt() raises:
    from collections import Counter

    var counter = Counter[Int]([1, 2, 1, 2, 3, 3])
    var other = Counter[Int].fromkeys([1, 2, 3], 3)
    assert_true(counter.lt(other))
    counter[1] += 1
    assert_false(counter.lt(other))


fn is_le() raises:
    from collections import Counter

    var counter = Counter[Int]([1, 2, 1, 2, 3, 3, 3])
    var other = Counter[Int].fromkeys([1, 2, 3], 10)
    assert_true(counter.le(other))
    counter[3] += 20
    assert_false(counter.le(other))


fn is_gt() raises:
    from collections import Counter

    var counter = Counter[Int]([1, 2, 1, 2, 3, 3])
    var other = Counter[Int].fromkeys([1, 2, 3], 3)
    assert_true(other.gt(counter))
    counter[1] += 1
    assert_false(other.gt(counter))


fn is_ge() raises:
    from collections import Counter

    var counter = Counter[Int]([1, 2, 1, 2, 3, 3, 3])
    var other = Counter[Int].fromkeys([1, 2, 3], 10)
    assert_true(other.ge(counter))
    counter[3] += 20
    assert_false(other.ge(counter))


fn lt() raises:
    from collections import Counter

    var counter = Counter[Int]([1, 2, 1, 2, 3, 3])
    var other = Counter[Int].fromkeys([1, 2, 3], 3)
    assert_true(counter.lt(other))
    counter[1] += 1
    assert_false(counter.lt(other))


fn gt() raises:
    from collections import Counter

    var counter = Counter[Int]([1, 2, 1, 2, 3, 3])
    var other = Counter[Int].fromkeys([1, 2, 3], 3)
    assert_true(other.gt(counter))
    counter[1] += 1
    assert_false(other.gt(counter))


fn ge() raises:
    from collections import Counter

    var counter = Counter[Int]([1, 2, 1, 2, 3, 3, 3])
    var other = Counter[Int].fromkeys([1, 2, 3], 10)
    assert_true(other.ge(counter))
    counter[3] += 20
    assert_false(other.ge(counter))


fn get() raises:
    from collections import Counter

    var counter = Counter[String].fromkeys(["a", "b", "c"], 1)
    assert_true(counter.get("a").or_else(0) == 1)
    assert_true(counter.get("d").or_else(0) == 0)


fn get_default() raises:
    from collections import Counter

    var counter = Counter[String].fromkeys(["a", "b", "c"], 1)
    assert_true(counter.get("a", default=0) == 1)
    assert_true(counter.get("d", default=0) == 0)


fn pop() raises:
    from collections import Counter

    var counter = Counter[String].fromkeys(["a", "b", "c"], 1)
    assert_true(counter.get("b").or_else(0) == 1)
    count = counter.pop("b")
    assert_true(count == 1)
    assert_true(counter.get("b").or_else(0) == 0)


fn pop_default() raises:
    var counter = Counter[String].fromkeys(["a", "b", "c"], 1)
    var count = counter.pop("b", default=100)
    assert_true(count == 1)
    count = counter.pop("not-a-key", default=0)
    assert_true(count == 0)


fn keys() raises:
    from collections import Counter

    var counter = Counter[String].fromkeys(["d", "b", "a", "c"], 1)
    var key_list = List[String]()
    for key in counter.keys():
        key_list.append(key)
    sort(key_list[:])
    assert_true(key_list == ["a", "b", "c", "d"])


fn values() raises:
    from collections import Counter

    # Construct `counter`
    var counter = Counter[Int]([1, 2, 3, 1, 2, 1, 1, 1, 2, 5, 2, 9])

    # Find most populous key
    max_count: Int = Int.MIN
    for count in counter.values():
        if count > max_count:
            max_count = count

    # Max count is the five ones
    assert_true(max_count == 5)


fn items() raises:
    from collections import Counter

    var counter = Counter[Int]([1, 2, 1, 2, 1, 1, 1, 2, 2])
    for count in counter.items():
        assert_true(
            (count.key == 1 and count.value == 5)
            or (count.key == 2 and count.value == 4)
        )
    # output: 1 5
    # output: 2 4


fn clear() raises:
    from collections import Counter

    var counter = Counter[Int]([1, 2, 1, 2, 1, 1, 1, 2, 2])
    assert_true(counter.total() == 9)
    counter.clear()  # Removes both entries
    assert_true(counter.total() == 0)


fn popitem() raises:
    from collections import Counter

    var counter = Counter[String].fromkeys(["a", "b", "c"], 5)
    var tuple = counter.popitem()
    assert_true(tuple._count == 5)
    # output: probably c 5 since that was last in
    # output's value may vary since order is not guaranteed


fn total() raises:
    from collections import Counter

    var counter = Counter[Int]([1, 2, 1, 2, 1, 1, 1, 2, 2])
    assert_true(counter.total() == 9)
    counter.clear()  # Removes both entries
    assert_true(counter.total() == 0)


fn most_common() raises:
    from collections import Counter

    var counter = Counter[Int]([1, 2, 1, 2, 3, 3, 3, 1, 1, 1, 6, 6, 2, 2, 7])
    for tuple in counter.most_common(2):
        assert_true(
            (tuple._value == 1 and tuple._count == 5)
            or (tuple._value == 2 and tuple._count == 4)
        )
        # output: 1 5
        # output: 2 4


fn elements() raises:
    from collections import Counter

    var counter = Counter[Int]([1, 2, 1, 2, 3, 3, 3, 1, 1, 1, 6, 6, 2, 2, 7])
    # output: [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 6, 6, 7]
    assert_true(
        counter.elements() == [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 6, 6, 7]
    )


fn update() raises:
    from collections import Counter

    var counter = Counter[Int]([1, 2, 1, 2, 3, 3, 3])
    var other = Counter[Int].fromkeys([1, 2, 3], 10)
    assert_true(counter[1] == 2)
    counter.update(other)
    assert_true(counter[1] == 12)


fn subtract() raises:
    from collections import Counter

    var counter = Counter[Int]([1, 2, 1, 2, 3, 3, 3])
    var other = Counter[Int].fromkeys([1, 2, 3], 10)
    assert_true(counter[1] == 2)
    counter.subtract(other)
    assert_true(counter[1] == -8)


fn main() raises:
    init1()
    fromkeys()
    is_le()
    is_lt()
    is_gt()
    is_ge()
    lt()
    gt()
    ge()
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
