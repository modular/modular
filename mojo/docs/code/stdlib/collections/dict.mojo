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
from testing import *


fn test_dict() raises:
    # Empty dictionary
    var empty_dict = Dict[String, Int]()
    assert_true(len(empty_dict) == 0)

    # Dictionary literal syntax
    var scores = {"Alice": 95, "Bob": 87, "Charlie": 92}
    assert_true(len(scores) == 3)
    assert_true(scores["Alice"] == 95)
    assert_true(scores["Bob"] == 87)
    assert_true(scores["Charlie"] == 92)

    # Pre-allocated capacity
    var large_dict = Dict[String, Int](capacity=64)
    assert_true(len(large_dict) == 0)

    # From separate key and value lists
    var keys = ["red", "green", "blue"]
    var values = [255, 128, 64]
    var colors = Dict[String, Int]()
    for key, value in zip(keys, values):
        colors[String(key)] = value  # cast list iterator to key-type

    assert_true(len(colors) == 3)
    assert_true(colors["red"] == 255)
    assert_true(colors["green"] == 128)
    assert_true(colors["blue"] == 64)

    var dict1 = {"a": 1, "b": 2}
    # var dict2 = dict1  # Error: Dict is not implicitly copyable
    var dict2 = dict1.copy()  # Deep copy
    dict2["c"] = 3
    # print(dict1.__str__())   # => {"a": 1, "b": 2}
    # print(dict2.__str__())   # => {"a": 1, "b": 2, "c": 3}

    assert_true(len(dict1) == 2)
    assert_true(len(dict2) == 3)
    assert_true(dict1["a"] == 1)
    assert_true(dict1["b"] == 2)
    assert_true(dict2["a"] == 1)
    assert_true(dict2["b"] == 2)
    assert_true(dict2["c"] == 3)

    var inventory = {"apples": 10, "bananas": 5}

    # Default behavior creates immutable (read-only) references
    # for value in inventory.values():
    #     value += 1  # error: expression must be mutable

    # Using `ref` gets mutable (read-write) references
    for ref value in inventory.values():
        value += 1  # Modify inventory values in-place
    # print(inventory.__str__())  # => {"apples": 11, "bananas": 6}
    assert_true(inventory["apples"] == 11)
    assert_true(inventory["bananas"] == 6)

    # Using `var` gets an owned copy of the value
    for var key in inventory.keys():
        inventory[key] += 1  # Modify inventory values in-place
    # print(inventory.__str__())  # => {"apples": 12, "bananas": 7}
    assert_true(inventory["apples"] == 12)
    assert_true(inventory["bananas"] == 7)

    var phonebook = {"Alice": "555-0101", "Bob": "555-0102"}
    var phone = phonebook.get("Charlie")
    # print(phone.__str__()) if phone else print('phone not found')
    assert_true(phone == None)
    assert_true(phonebook.get("Alice") == "555-0101")
    assert_true(phonebook.get("Bob") == "555-0102")
    assert_true(
        phone.or_else("555-1212") == "555-1212"
    )  # Accessing missing key returns default value


fn test_itemstring() raises:
    var my_dict = Dict[Int, Float64]()
    my_dict[1] = 1.1
    my_dict[2] = 2.2
    dict_as_string = String(my_dict)
    # print(dict_as_string)
    # prints {1: 1.1, 2: 2.2}

    # This is vulnerable to format output changes
    # assert_true(dict_as_string == "{1: 1.1, 2: 2.2}" or dict_as_string == "{2: 2.2, 1: 1.1}")
    # Instead, will check for the presence of the keys and values in the string representation

    assert_true(dict_as_string.__contains__("1:"))
    assert_true(dict_as_string.__contains__("1.1"))
    assert_true(dict_as_string.__contains__("2:"))
    assert_true(dict_as_string.__contains__("2.2"))


fn test_fromkeys() raises:
    var keys = ["a", "b", "c"]
    var dict = Dict.fromkeys(keys, 0)
    # print(dict.__str__())  # => {"a": 0, "b": 0, "c": 0}
    assert_true(len(dict) == 3)
    assert_true(dict["a"] == 0)
    assert_true(dict["b"] == 0)
    assert_true(dict["c"] == 0)


fn test_find() raises:
    var my_dict = Dict[String, Int]()
    my_dict["a"] = 1
    my_dict["b"] = 2
    var value = my_dict.find("a")
    # print(value.__str__())  # => 1
    assert_true(value.or_else(Int.MAX) == 1)
    var missing_value = my_dict.find("c")
    # print(missing_value.__str__())  # => None
    assert_true(missing_value == None)


fn test_get() raises:
    var my_dict = Dict[String, Int]()
    my_dict["a"] = 1
    my_dict["b"] = 2
    var value = my_dict.get("a")
    # print(value.__str__())  # => 1
    assert_true(value.or_else(Int.MAX) == 1)
    var missing_value = my_dict.get("c")
    # print(missing_value.__str__())  # => -1
    assert_true(missing_value == None)
    assert_true(my_dict["a"] == my_dict.get("a").or_else(Int.MAX))


fn test_get_with_default() raises:
    var my_dict = Dict[String, Int]()
    my_dict["a"] = 1
    my_dict["b"] = 2

    value = my_dict.get("a", Int.MAX)
    # print(value.__str__())  # => 1
    assert_true(value == 1)

    missing_value = my_dict.get("c", -1)
    # print(missing_value.__str__())  # => -1
    assert_true(missing_value == -1)

    from testing import assert_true

    assert_true(my_dict["a"] == my_dict.get("a", Int.MAX))


fn test_pop() raises:
    var my_dict = Dict[String, Int]()
    my_dict["a"] = 1
    my_dict["b"] = 2
    var value = my_dict.pop("a", 99)
    # print(value.__str__())  # => 1
    assert_true(value == 1)
    var missing_value = my_dict.pop("c", 99)
    # print(missing_value.__str__())  # => 99
    assert_true(missing_value == 99)


fn test_pop_with_default() raises:
    var my_dict = Dict[String, Int]()
    my_dict["a"] = 1
    my_dict["b"] = 2

    var value = my_dict.pop("a", 99)
    # print(value.__str__())  # => 1
    assert_true(value == 1)

    missing_value = my_dict.pop("c", 99)
    # print(missing_value.__str__())  # => 99
    assert_true(missing_value == 99)


fn test_popitem() raises:
    var my_dict = Dict[String, Int]()
    my_dict["a"] = 1
    my_dict["b"] = 2
    # print(len(my_dict))  # => 2
    assert_true(len(my_dict) == 2)

    var item = my_dict.popitem()
    # print(item.key, item.value)  # => ("b", 2) or ("a", 1)
    # print(item.__str__())  # => ("b", 2) or ("a", 1)
    assert_true(item.key == "a" or item.key == "b")
    assert_true(item.value == 1 or item.value == 2)
    assert_true(len(my_dict) == 1)


fn test_keys() raises:
    var my_dict = Dict[String, Int]()
    my_dict["a"] = 1
    my_dict["b"] = 2
    var keys = my_dict.keys()
    # print(keys.__str__())  # => ["a", "b"]
    var keylist = List[String](keys)
    assert_true(len(keylist) == 2)
    assert_true(keylist[0] == "a" or keylist[0] == "b")
    assert_true(keylist[1] == "a" or keylist[1] == "b")


fn test_values() raises:
    var my_dict = Dict[String, Int]()
    my_dict["a"] = 1
    my_dict["b"] = 2
    var values = my_dict.values()
    # for value in values:
    #     print(value.__str__())  # => 1 or 2
    var valuelist = List[Int](values)
    assert_true(len(valuelist) == 2)
    assert_true(valuelist[0] == 1 or valuelist[0] == 2)
    assert_true(valuelist[1] == 1 or valuelist[1] == 2)


fn test_items() raises:
    var my_dict = Dict[String, Int]()
    my_dict["a"] = 1
    my_dict["b"] = 2

    # for item in my_dict.items():
    # print(item.key, item.value) # prints a 1 then b 2 or b 2 then a 1
    # All entries will be printed, but order is not guaranteed

    for item in my_dict.items():
        assert_true(
            (item.key == "a" and item.value == 1)
            or (item.key == "b" and item.value == 2)
        )


fn test_takeitems() raises:
    var my_dict = Dict[String, Int]()
    my_dict["a"] = 1
    my_dict["b"] = 2

    # for item in my_dict.take_items():
    # print(item.key, item.value) # prints a 1 then b 2 or b 2 then a 1
    # All entries will be printed, but order is not guaranteed

    for item in my_dict.take_items():
        assert_true(
            (item.key == "a" and item.value == 1)
            or (item.key == "b" and item.value == 2)
        )

    assert_true(len(my_dict) == 0)


fn test_update() raises:
    var dict1 = Dict[String, Int]()
    dict1["a"] = 1
    dict1["b"] = 2

    var dict2 = Dict[String, Int]()
    dict2["b"] = 3
    dict2["c"] = 4

    dict1.update(dict2)

    assert_true(len(dict1) == 3)
    assert_true(dict1["a"] == 1)
    assert_true(
        dict1["b"] == 3
    )  # value from dict2 should overwrite value from dict1
    assert_true(dict1["c"] == 4)


fn test_clear() raises:
    var my_dict = Dict[String, Int]()
    my_dict["a"] = 1
    my_dict["b"] = 2
    assert_true(len(my_dict) == 2)

    my_dict.clear()

    assert_true(len(my_dict) == 0)


fn test_setdefault() raises:
    var my_dict = Dict[String, Int]()
    my_dict["a"] = 1
    my_dict["b"] = 2

    var value = my_dict.setdefault("a", 99)
    # print(value.__str__())  # => 1
    assert_true(value == 1)

    missing_value = my_dict.setdefault("c", 99)
    # print(missing_value.__str__())  # => 99
    assert_true(missing_value == 99)
    assert_true(my_dict["c"] == 99)


fn main() raises:
    test_dict()
    test_itemstring()
    test_fromkeys()
    test_find()
    test_get()
    test_get_with_default()
    test_pop()
    test_pop_with_default()
    test_popitem()
    test_keys()
    test_values()
    test_items()
    test_takeitems()
    test_update()
    test_clear()
    test_setdefault()
