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
# RUN: %mojo %s

from collections import Dict
from memory import UnsafePointer
from collections.dict import ParamKVCache
from testing import assert_equal, assert_raises


def test_basic():
    var dict = Dict[String, Int]()
    dict["a"] = 1
    dict["b"] = 2
    dict["c"] = 3
    var dict_cache = ParamKVCache[List[String]("a", "b")](dict)

    dict_cache.get_ref["a"]() *= 10
    assert_equal(10, dict["a"])
    dict["a"] = 1
    assert_equal(1, dict_cache.get_ref["a"]())

    # insert 4 new value (dict resize)
    dict["A"] = 4
    dict["B"] = 5
    dict["C"] = 6
    dict["D"] = 7
    assert_equal(len(dict._entries), 16)

    # update cache
    dict_cache.update_cache()
    dict["b"] = 20
    assert_equal(20, dict_cache.get_ref["b"]())

    # pop "b"
    var b_ptr = UnsafePointer.address_of(dict["b"])
    dict.pop("b")
    assert_equal(False, "b" in dict)

    # compact the dict
    dict._compact()

    # reinsert "b"
    dict["b"] = 2
    var b_ptr_after = UnsafePointer.address_of(dict["b"])
    assert_equal(b_ptr_after == b_ptr, False)

    # update cache
    dict_cache.update_cache()
    assert_equal(2, dict["b"])
    assert_equal(
        UnsafePointer.address_of(dict_cache.get_ref["b"]()).origin_cast[
            origin = MutableOrigin.empty
        ](),
        b_ptr_after,
    )

    # clear dict
    dict.clear()

    # update cache
    with assert_raises(contains="KeyError"):
        dict_cache.update_cache()

    # get_ref to K not in dict_cache.Keys
    with assert_raises(contains="KeyError"):
        dict_cache.get_ref["ABC"]()

    # init with 2 keys not in dict
    with assert_raises(contains="KeyError"):
        var dict_cache2 = ParamKVCache[List[String]("a", "b")](dict)

    # init with 1 key not in dict
    dict["a"] = 1
    with assert_raises(contains="KeyError"):
        var dict_cache2 = ParamKVCache[List[String]("a", "b")](dict)
    dict["b"] = 2
    dict.pop("a")
    with assert_raises(contains="KeyError"):
        var dict_cache2 = ParamKVCache[List[String]("a", "b")](dict)

    # test pointers
    dict.clear()
    dict["a"] = 1
    dict["b"] = 2
    var dict_cache2 = ParamKVCache[List[String]("a", "b")](dict)
    assert_equal(
        UnsafePointer.address_of(dict["a"]),
        UnsafePointer.address_of(dict_cache2.get_ref["a"]()),
    )
    assert_equal(
        UnsafePointer.address_of(dict["b"]),
        UnsafePointer.address_of(dict_cache2.get_ref["b"]()),
    )

    assert_equal(True, Bool(dict._entries[0]))
    assert_equal(True, Bool(dict._entries[1]))
    dict.pop("a")
    dict.pop("b")
    dict["a"] = 3
    dict["b"] = 4
    assert_equal(False, Bool(dict._entries[0]))
    assert_equal(False, Bool(dict._entries[1]))
    dict_cache2.update_cache()
    assert_equal(
        UnsafePointer.address_of(dict["a"]),
        UnsafePointer.address_of(dict_cache2.get_ref["a"]()),
    )
    assert_equal(
        UnsafePointer.address_of(dict["b"]),
        UnsafePointer.address_of(dict_cache2.get_ref["b"]()),
    )


def test_list_sizes():
    @parameter
    fn create_list[size: Int]() -> List[Int]:
        var tmp = List[Int](capacity=size)
        for i in range(size):
            tmp.append(i)
        return tmp

    @parameter
    fn test_sizes[size: Int]() -> Tuple[Int, Int]:
        alias keys = create_list[size]()
        var dict = Dict[Int, Int]()
        var expected_result = 0
        var result = 0
        for i in range(size):
            dict[i] = i
            expected_result += i
        try:
            var dict_cache = ParamKVCache[keys](dict)

            @parameter
            for i in range(size):
                assert_equal(dict_cache.get_ref[i](), dict[i])
                result += dict_cache.get_ref[i]()
        except:
            result = 0
        return (result, expected_result)

    var a: Int
    var b: Int
    a, b = test_sizes[4]()
    assert_equal(a, b)
    assert_equal(a, 6)

    a, b = test_sizes[6]()
    assert_equal(a, b)
    assert_equal(a, 15)

    a, b = test_sizes[8]()
    assert_equal(a, b)
    assert_equal(a, 28)

    a, b = test_sizes[15]()
    assert_equal(a, b)
    assert_equal(a, 105)

    a, b = test_sizes[128]()
    assert_equal(a, b)
    var expected_result = 0
    for i in range(128):
        expected_result += i
    assert_equal(a, expected_result)


def test_kwargs():
    def kwargs_function(**kwargs: Int) -> Int:
        var kwargs_cache = ParamKVCache[List[String]("a", "b")](kwargs._dict)
        return kwargs_cache.get_ref["a"]() + kwargs_cache.get_ref["b"]()

    var res = kwargs_function(a=10, b=4)
    assert_equal(res, 14)

    with assert_raises(contains="KeyError"):
        res = kwargs_function(c=10, d=4)


def main():
    test_basic()
    test_list_sizes()
    test_kwargs()
