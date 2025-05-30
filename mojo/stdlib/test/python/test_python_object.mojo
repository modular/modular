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
# XFAIL: asan && !system-darwin
# RUN: %mojo %s


from python import Python, PythonObject
from python.bindings import PythonTypeBuilder, PythonModuleBuilder
from testing import assert_equal, assert_false, assert_raises, assert_true


def test_dunder_methods(mut python: Python):
    var a = PythonObject(34)
    var b = PythonObject(10)

    # __add__
    var c = a + b
    assert_equal(c, 44)

    # __add__
    c = a + 100
    assert_equal(c, 134)

    # __iadd__
    c += 100
    assert_equal(c, 234)

    # __radd__
    c = 100 + a
    assert_equal(c, 134)

    # __sub__
    c = a - b
    assert_equal(c, 24)

    # __isub__
    c -= 100
    assert_equal(c, -76)

    # __sub__
    c = a - 100
    assert_equal(c, -66)

    # __rsub__
    c = 100 - a
    assert_equal(c, 66)

    # __mul__
    c = a * b
    assert_equal(c, 340)

    # __imul__
    c *= 10
    assert_equal(c, 3400)

    # __mul__
    c = a * 10
    assert_equal(c, 340)

    # __rmul__
    c = 34 * b
    assert_equal(c, 340)

    # __floordiv__
    c = a // b
    assert_equal(c, 3)

    # __ifloordiv__
    c //= 2
    assert_equal(c, 1)

    # __floordiv__
    c = a // 10
    assert_equal(c, 3)

    # __rfloordiv__
    c = 34 // b
    assert_equal(c, 3)

    # __truediv__
    c = a / b
    assert_equal(c, 3.4)

    # __itruediv__
    c /= 2
    assert_equal(c, 1.7)

    # __truediv__
    c = a / 10
    assert_equal(c, 3.4)

    # __rtruediv__
    c = 34 / b
    assert_equal(c, 3.4)

    # __mod__
    c = a % b
    assert_equal(c, 4)

    # __imod__
    c %= 3
    assert_equal(c, 1)

    # __mod__
    c = a % 10
    assert_equal(c, 4)

    # __rmod__
    c = 34 % b
    assert_equal(c, 4)

    # __xor__
    c = a ^ b
    assert_equal(c, 40)

    # __ixor__
    c ^= 15
    assert_equal(c, 39)

    # __xor__
    c = a ^ 10
    assert_equal(c, 40)

    # __rxor__
    c = 34 ^ b
    assert_equal(c, 40)

    # __or__
    c = a | b
    assert_equal(c, 42)

    # __ior__
    c |= 9
    assert_equal(c, 43)

    # __or__
    c = a | 10
    assert_equal(c, 42)

    # __ror__
    c = 34 | b
    assert_equal(c, 42)

    # __and__
    c = a & b
    assert_equal(c, 2)

    # __iand__
    c &= 6
    assert_equal(c, 2)

    # __and__
    c = a & 10
    assert_equal(c, 2)

    # __rand__
    c = 34 & b
    assert_equal(c, 2)

    # __rshift__
    var d = PythonObject(2)
    c = a >> d
    assert_equal(c, 8)

    # __irshift__
    c >>= 2
    assert_equal(c, 2)

    # __rshift__
    c = a >> 2
    assert_equal(c, 8)

    # __rrshift__
    c = 34 >> d
    assert_equal(c, 8)

    # __lshift__
    c = a << d
    assert_equal(c, 136)

    # __ilshift__
    c <<= 1
    assert_equal(c, 272)

    # __lshift__
    c = a << 2
    assert_equal(c, 136)

    # __rlshift__
    c = 34 << d
    assert_equal(c, 136)

    # __pow__
    c = a**d
    assert_equal(c, 1156)

    # __ipow__
    c = 3
    c **= 4
    assert_equal(c, 81)

    # __pow__
    c = a**2
    assert_equal(c, 1156)

    # __rpow__
    c = 34**d
    assert_equal(c, 1156)

    # __lt__
    c = a < b
    assert_false(c)

    # __le__
    c = a <= b
    assert_false(c)

    # __gt__
    c = a > b
    assert_true(c)

    # __ge__
    c = a >= b
    assert_true(c)

    # __eq__
    c = a == b
    assert_false(c)

    # __ne__
    c = a != b
    assert_true(c)

    # __pos__
    c = +a
    assert_equal(c, 34)

    # __neg__
    c = -a
    assert_equal(c, -34)

    # __invert__
    c = ~a
    assert_equal(c, -35)


def test_inplace_dunder_methods(mut python: Python):
    # test dunder methods that don't fall back to their non-inplace counterparts
    var list_obj: PythonObject = [1, 2]

    list_obj += [3, 4]
    assert_equal(String(list_obj), "[1, 2, 3, 4]")

    list_obj *= 2
    assert_equal(String(list_obj), "[1, 2, 3, 4, 1, 2, 3, 4]")

    _ = python.eval("class A:\n  def __iadd__(self, other):\n    return 1")
    var a = python.evaluate("A()")
    a += 1
    assert_equal(a, 1)


def test_num_conversion():
    alias n = UInt64(0xFEDC_BA09_8765_4321)
    alias n_str = String(n)
    assert_equal(n_str, String(PythonObject(n)))


def test_bool_conversion():
    var x: PythonObject = 1
    assert_true(x == 1)
    assert_false(x == 0)
    assert_true(x == 0 or x == 1)


fn test_string_conversions() raises -> None:
    fn test_string_literal() -> None:
        try:
            var mojo_str: StaticString = "mojo"
            var py_str = PythonObject(mojo_str)
            var py_capitalized = py_str.capitalize()
            var py = Python()
            var mojo_capitalized = py.as_string_slice(py_capitalized)
            assert_true(mojo_capitalized == "Mojo")
        except e:
            print("Error occurred")

    fn test_string() -> None:
        try:
            var mo_str = String("mo")
            var jo_str = String("jo")
            var mojo_str = mo_str + jo_str
            var py_str = PythonObject(mojo_str)
            var py_capitalized = py_str.capitalize()
            var py = Python()
            var mojo_capitalized = py.as_string_slice(py_capitalized)
            assert_true(mojo_capitalized == "Mojo")
        except e:
            print("Error occurred")

    fn test_type_object() raises -> None:
        var py_float = PythonObject(3.14)
        var type_obj = Python.type(py_float)
        assert_equal(String(type_obj), "<class 'float'>")

    test_string_literal()
    test_string()
    test_type_object()


def test_len():
    var empty_list: PythonObject = []
    assert_equal(len(empty_list), 0)

    var l1 = Python.evaluate("[1,2,3]")
    assert_equal(len(l1), 3)

    var l2 = Python.evaluate("[42,42.0]")
    assert_equal(len(l2), 2)

    var x = PythonObject(42)
    with assert_raises(contains="object of type 'int' has no len()"):
        _ = len(x)


def test_is():
    var x = PythonObject(500)
    var y = PythonObject(500)
    assert_false(x is y)
    assert_true(x is not y)

    # Assign to a new variable but this still holds
    # the same object and same memory location
    var z = x
    assert_true(z is x)
    assert_false(z is not x)

    # Two separate lists/objects, and therefore are not the "same".
    # as told by the `__is__` function. They point to different addresses.
    var l1 = Python.evaluate("[1,2,3]")
    var l2 = Python.evaluate("[1,2,3]")
    assert_false(l1 is l2)
    assert_true(l1 is not l2)


def test_nested_object():
    var a: PythonObject = [1, 2, 3]
    var b: PythonObject = [4, 5, 6]
    var nested_list: PythonObject = [a, b]
    var nested_tuple = Python.tuple(a, b)

    assert_equal(String(nested_list), "[[1, 2, 3], [4, 5, 6]]")
    assert_equal(String(nested_tuple), "([1, 2, 3], [4, 5, 6])")


fn test_iter() raises:
    var list_obj: PythonObject = ["apple", "orange", "banana"]
    var i = 0
    for fruit in list_obj:
        if i == 0:
            assert_equal(fruit, "apple")
        elif i == 1:
            assert_equal(fruit, "orange")
        elif i == 2:
            assert_equal(fruit, "banana")
        i += 1

    var list2: PythonObject = []
    for _ in list2:
        raise Error("This should not be reachable as the list is empty.")

    var not_iterable: PythonObject = 3
    with assert_raises():
        for _ in not_iterable:
            assert_false(
                True,
                msg=(
                    "This should not be reachable as the object is not"
                    " iterable."
                ),
            )


fn test_setitem() raises:
    var ll: PythonObject = [1, 2, 3, "food"]
    assert_equal(String(ll), "[1, 2, 3, 'food']")
    ll[1] = "nomnomnom"
    assert_equal(String(ll), "[1, 'nomnomnom', 3, 'food']")


fn test_dict() raises:
    # Test Python.dict from keyword arguments.
    var dd = Python.dict(food=123, fries="yes")
    assert_equal(String(dd), "{'food': 123, 'fries': 'yes'}")

    var dd2: PythonObject = {"food": 123, "fries": "yes"}
    assert_equal(String(dd2), "{'food': 123, 'fries': 'yes'}")

    dd["food"] = "salad"
    dd[42] = Python.list(4, 2)
    assert_equal(String(dd), "{'food': 'salad', 'fries': 'yes', 42: [4, 2]}")

    # Test Python.dict from a Span of tuples.
    var tuples = [(123, PythonObject("food")), (42, PythonObject("42"))]
    dd = Python.dict(tuples)
    assert_equal(String(dd), "{123: 'food', 42: '42'}")

    # Also test that Python.dict() creates the right object.
    var empty = Python.dict()
    assert_equal(String(empty), "{}")

    var empty2: PythonObject = {}
    assert_equal(String(empty2), "{}")


fn test_set() raises:
    # Test Python set literals.
    var dd: PythonObject = {123, "yes"}
    var dd2 = Python.evaluate("{123, 'yes'}")
    # Be care about instability of set ordering across platforms.
    assert_equal(String(dd), String(dd2))

    assert_true(123 in dd)
    assert_true("yes" in dd)
    assert_false(42 in dd)


fn test_none() raises:
    var n = Python.none()
    assert_equal(String(n), "None")
    assert_true(n is None)


fn test_getitem_raises() raises:
    custom_indexable = Python.import_module("custom_indexable")

    var a = PythonObject(2)
    with assert_raises(contains="'int' object is not subscriptable"):
        _ = a[0]
    with assert_raises(contains="'int' object is not subscriptable"):
        _ = a[0, 0]

    var b = PythonObject(2.2)
    with assert_raises(contains="'float' object is not subscriptable"):
        _ = b[0]
    with assert_raises(contains="'float' object is not subscriptable"):
        _ = b[0, 0]

    var c = PythonObject(True)
    with assert_raises(contains="'bool' object is not subscriptable"):
        _ = c[0]
    with assert_raises(contains="'bool' object is not subscriptable"):
        _ = c[0, 0]

    var d = PythonObject(None)
    with assert_raises(contains="'NoneType' object is not subscriptable"):
        _ = d[0]
    with assert_raises(contains="'NoneType' object is not subscriptable"):
        _ = d[0, 0]

    with_get = custom_indexable.WithGetItem()
    assert_equal("Key: 0", String(with_get[0]))
    assert_equal("Keys: 0, 0", String(with_get[0, 0]))
    assert_equal("Keys: 0, 0, 0", String(with_get[0, 0, 0]))

    var without_get = custom_indexable.Simple()
    with assert_raises(contains="'Simple' object is not subscriptable"):
        _ = without_get[0]

    with assert_raises(contains="'Simple' object is not subscriptable"):
        _ = without_get[0, 0]

    var with_get_exception = custom_indexable.WithGetItemException()
    with assert_raises(contains="Custom error"):
        _ = with_get_exception[1]

    with_2d = custom_indexable.With2DGetItem()
    assert_equal("[1, 2, 3]", String(with_2d[0]))
    assert_equal(2, Int(with_2d[0, 1]))
    assert_equal(6, Int(with_2d[1, 2]))

    with assert_raises(contains="list index out of range"):
        _ = with_2d[0, 4]

    with assert_raises(contains="list index out of range"):
        _ = with_2d[3, 0]

    with assert_raises(contains="list index out of range"):
        _ = with_2d[3]


def test_setitem_raises():
    custom_indexable = Python.import_module("custom_indexable")
    t = Python.evaluate("(1,2,3)")
    with assert_raises(
        contains="'tuple' object does not support item assignment"
    ):
        t[0] = 0

    lst = Python.evaluate("[1, 2, 3]")
    with assert_raises(contains="list assignment index out of range"):
        lst[10] = 4

    s = Python.evaluate('"hello"')
    with assert_raises(
        contains="'str' object does not support item assignment"
    ):
        s[3] = "xy"

    with_out = custom_indexable.Simple()
    with assert_raises(
        contains="'Simple' object does not support item assignment"
    ):
        with_out[0] = 0

    d = Python.evaluate("{}")
    with assert_raises(contains="unhashable type: 'list'"):
        d[[1, 2, 3]] = 5


fn test_py_slice() raises:
    custom_indexable = Python.import_module("custom_indexable")
    var a: PythonObject = [1, 2, 3, 4, 5]
    assert_equal("[2, 3]", String(a[1:3]))
    assert_equal("[1, 2, 3, 4, 5]", String(a[:]))
    assert_equal("[1, 2, 3]", String(a[:3]))
    assert_equal("[3, 4, 5]", String(a[2:]))
    assert_equal("[1, 3, 5]", String(a[::2]))
    assert_equal("[2, 4]", String(a[1::2]))
    assert_equal("[4, 5]", String(a[-2:]))
    assert_equal("[1, 2, 3]", String(a[:-2]))
    assert_equal("[5, 4, 3, 2, 1]", String(a[::-1]))
    assert_equal("[1, 2, 3, 4, 5]", String(a[-10:10]))  # out of bounds
    assert_equal("[1, 2, 3, 4, 5]", String(a[::]))
    assert_equal("[1, 2, 3, 4, 5]", String(a[:100]))
    assert_equal("[]", String(a[5:]))
    assert_equal("[5, 4, 3, 2]", String(a[:-5:-1]))

    var b = Python.evaluate("[i for i in range(1000)]")
    assert_equal("[0, 250, 500, 750]", String(b[::250]))
    with assert_raises(contains="slice step cannot be zero"):
        _ = b[::0]
    # Negative cases such as `b[1.3:10]` or `b["1":10]` are handled by parser
    # which would normally throw a TypeError in Python

    var s = PythonObject("Hello, World!")
    assert_equal("Hello", String(s[:5]))
    assert_equal("World!", String(s[7:]))
    assert_equal("!dlroW ,olleH", String(s[::-1]))
    assert_equal("Hello, World!", String(s[:]))
    assert_equal("Hlo ol!", String(s[::2]))
    assert_equal("Hlo ol!", String(s[None:None:2]))

    var t = Python.tuple(1, 2, 3, 4, 5)
    assert_equal("(2, 3, 4)", String(t[1:4]))
    assert_equal("(4, 3, 2)", String(t[3:0:-1]))

    var empty: PythonObject = []
    assert_equal("[]", String(empty[:]))
    assert_equal("[]", String(empty[1:2:3]))

    # TODO: enable this test.  Currently it fails with error: unhashable type: 'slice'
    # var d = Python.dict()
    # d["a"] = 1
    # d["b"] = 2
    # with assert_raises(contains="slice(1, 3, None)"):
    #     _ = d[1:3]

    var custom = custom_indexable.Sliceable()
    assert_equal("slice(1, 3, None)", String(custom[1:3]))

    var i = PythonObject(1)
    with assert_raises(contains="'int' object is not subscriptable"):
        _ = i[0:1]

    with_2d = custom_indexable.With2DGetItem()
    assert_equal("[1, 2]", String(with_2d[0, PythonObject(Slice(0, 2))]))
    assert_equal("[1, 2]", String(with_2d[0][0:2]))

    assert_equal("[4, 5, 6]", String(with_2d[PythonObject(Slice(0, 2)), 1]))
    assert_equal("[4, 5, 6]", String(with_2d[0:2][1]))

    assert_equal(
        "[[1, 2, 3], [4, 5, 6]]", String(with_2d[PythonObject(Slice(0, 2))])
    )
    assert_equal("[[1, 2, 3], [4, 5, 6]]", String(with_2d[0:2]))
    assert_equal("[[1, 3], [4, 6]]", String(with_2d[0:2, ::2]))

    assert_equal(
        "[6, 5, 4]", String(with_2d[1, PythonObject(Slice(None, None, -1))])
    )
    assert_equal("[6, 5, 4]", String(with_2d[1][::-1]))

    assert_equal("[7, 9]", String(with_2d[2][::2]))

    with assert_raises(contains="list index out of range"):
        _ = with_2d[0:1][4]


def test_contains_dunder():
    with assert_raises(contains="'int' object is not iterable"):
        var z = PythonObject(0)
        _ = 5 in z

    var x: PythonObject = [1.1, 2.2]
    assert_true(1.1 in x)
    assert_false(3.3 in x)

    x = Python.list("Hello", "World")
    assert_true("World" in x)

    x = Python.tuple(1.5, 2)
    assert_true(1.5 in x)
    assert_false(3.5 in x)

    var y = Python.dict(A="A", B=5)
    assert_true("A" in y)
    assert_false("C" in y)
    assert_true("B" in y)


@fieldwise_init
struct Person(Movable, Defaultable, Representable):
    var name: String
    var age: Int

    fn __init__(out self):
        self.name = ""
        self.age = 0

    fn __repr__(self) -> String:
        return String("Person(", self.name, ", ", self.age, ")")


def test_python_mojo_object_operations():
    # Type registration
    var b = PythonModuleBuilder("fake_module")
    _ = b.add_type[Person]("Person")
    _ = b.finalize()

    # Alloc
    var person_obj = PythonObject(alloc=Person("John Smith", 42))

    # Downcast
    var person_ptr = person_obj.downcast_value_ptr[Person]()

    assert_equal(person_ptr[].name, "John Smith")


def main():
    # initializing Python instance calls init_python
    var python = Python()

    test_dunder_methods(python)
    test_inplace_dunder_methods(python)
    test_num_conversion()
    test_bool_conversion()
    test_string_conversions()
    test_len()
    test_is()
    test_iter()
    test_setitem()
    test_dict()
    test_set()
    test_none()
    test_nested_object()
    test_getitem_raises()
    test_setitem_raises()
    test_py_slice()
    test_contains_dunder()
    test_python_mojo_object_operations()
