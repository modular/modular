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
"""Implements PythonObject.

You can import these APIs from the `python` package. For example:

```mojo
from python import PythonObject
```
"""

from memory import LegacyUnsafePointer as UnsafePointer
from os import abort
from sys import bit_width_of
from sys.ffi import c_double, c_long, c_size_t, c_ssize_t
from sys.intrinsics import _unsafe_aliasing_address_to_pointer

from compile.reflection import get_type_name

from ._cpython import CPython, GILAcquired, PyObject, PyObjectPtr, PyTypeObject
from .bindings import PyMojoObject, _get_type_name, lookup_py_type_object
from .python import Python
from .conversions import ConvertibleToPython


struct _PyIter(ImplicitlyCopyable, Iterable, Iterator):
    """A Python iterator."""

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    alias IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[iterable_mut]
    ]: Iterator = Self
    alias Element = PythonObject

    var iterator: PythonObject
    """The iterator object that stores location."""
    var next_item: PyObjectPtr
    """The next item to vend or zero if there are no items."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(out self, iter: PythonObject):
        """Initialize an iterator.

        Args:
            iter: A Python iterator instance.
        """
        ref cpy = Python().cpython()
        self.iterator = iter
        self.next_item = cpy.PyIter_Next(iter._obj_ptr)

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __has_next__(self) -> Bool:
        return Bool(self.next_item)

    fn __next__(mut self) -> PythonObject:
        """Return the next item and update to point to subsequent item.

        Returns:
            The next item in the traversable object that this iterator
            points to.
        """
        ref cpy = Python().cpython()
        var curr_item = self.next_item
        self.next_item = cpy.PyIter_Next(self.iterator._obj_ptr)
        return PythonObject(from_owned=curr_item)

    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self


@register_passable
struct PythonObject(
    Boolable,
    ConvertibleToPython,
    Defaultable,
    Identifiable,
    ImplicitlyCopyable,
    Movable,
    SizedRaising,
    Writable,
):
    """A Python object."""

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var _obj_ptr: PyObjectPtr
    """A pointer to the underlying Python object."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(out self):
        """Initialize the object with a `None` value."""
        self = Self(None)

    fn __init__(out self, *, from_owned: PyObjectPtr):
        """Initialize this object from an owned reference-counted Python object
        pointer.

        For example, this function should be used to construct a `PythonObject`
        from the pointer returned by "New reference"-type objects from the
        CPython API.

        Args:
            from_owned: An owned pointer to a Python object.

        References:
        - https://docs.python.org/3/glossary.html#term-strong-reference
        """
        self._obj_ptr = from_owned

    fn __init__(out self, *, from_borrowed: PyObjectPtr):
        """Initialize this object from a borrowed reference-counted Python
        object pointer.

        For example, this function should be used to construct a `PythonObject`
        from the pointer returned by "Borrowed reference"-type objects from the
        CPython API.

        Args:
            from_borrowed: A borrowed pointer to a Python object.

        References:
        - https://docs.python.org/3/glossary.html#term-borrowed-reference
        """
        ref cpy = Python().cpython()
        # SAFETY:
        #   We were passed a Python "borrowed reference", so for it to be
        #   safe to store this reference, we must increment the reference
        #   count to convert this to a "strong reference".
        cpy.Py_IncRef(from_borrowed)
        self._obj_ptr = from_borrowed

    @always_inline
    fn __init__[T: Movable](out self, *, var alloc: T) raises:
        """Allocate a new `PythonObject` and store a Mojo value in it.

        The newly allocated Python object will contain the provided Mojo `T`
        instance directly, without attempting conversion to an equivalent Python
        builtin type.

        Only Mojo types that have a registered Python 'type' object can be stored
        as a Python object. Mojo types are registered using a
        `PythonTypeBuilder`.

        Parameters:
            T: The Mojo type of the value that the resulting Python object
              holds.

        Args:
            alloc: The Mojo value to store in the new Python object.

        Raises:
            If no Python type object has been registered for `T` by a
            `PythonTypeBuilder`.
        """
        # NOTE:
        #   We can't use PythonTypeBuilder.bind[T]() because that constructs a
        #   _new_ PyTypeObject. We want to reference the existing _singleton_
        #   PyTypeObject that represents a given Mojo type.
        var type_obj = lookup_py_type_object[T]()
        var type_obj_ptr = type_obj._obj_ptr.bitcast[PyTypeObject]()
        return _unsafe_alloc_init(type_obj_ptr, alloc^)

    # TODO(MSTDL-715):
    #   This initializer should not be necessary, we should need
    #   only the initializer from a `NoneType`.
    @doc_private
    @implicit
    fn __init__(out self, none: NoneType._mlir_type):
        """Initialize a none value object from a `None` literal.

        Args:
            none: None.
        """
        self = Self(none=NoneType())

    @implicit
    fn __init__(out self, none: NoneType):
        """Initialize a none value object from a `None` literal.

        Args:
            none: None.
        """
        ref cpy = Python().cpython()
        self = Self(from_borrowed=cpy.Py_None())

    @implicit
    fn __init__(out self, value: Bool):
        """Initialize the object from a bool.

        Args:
            value: The boolean value.
        """
        ref cpy = Python().cpython()
        self = Self(from_owned=cpy.PyBool_FromLong(c_long(Int(value))))

    @implicit
    fn __init__(out self, value: Int):
        """Initialize the object with an integer value.

        Args:
            value: The integer value.
        """
        ref cpy = Python().cpython()
        self = Self(from_owned=cpy.PyLong_FromSsize_t(c_ssize_t(value)))

    @implicit
    fn __init__[dtype: DType](out self, value: Scalar[dtype]):
        """Initialize the object with a generic scalar value. If the scalar
        value type is bool, it is converted to a boolean. Otherwise, it is
        converted to the appropriate integer or floating point type.

        Parameters:
            dtype: The scalar value type.

        Args:
            value: The scalar value.
        """
        ref cpy = Python().cpython()

        @parameter
        if dtype is DType.bool:
            var val = c_long(Int(value))
            self = Self(from_owned=cpy.PyBool_FromLong(val))
        elif dtype.is_unsigned():
            var val = c_size_t(mlir_value=value.cast[DType.int]()._mlir_value)
            self = Self(from_owned=cpy.PyLong_FromSize_t(val))
        elif dtype.is_integral():
            var val = c_ssize_t(value.cast[DType.int]()._mlir_value)
            self = Self(from_owned=cpy.PyLong_FromSsize_t(val))
        else:
            var val = c_double(value.cast[DType.float64]())
            self = Self(from_owned=cpy.PyFloat_FromDouble(val))

    @implicit
    fn __init__(out self, string: StringSlice) raises:
        """Initialize the object from a string.

        Args:
            string: The string value.

        Raises:
            If the string is not valid UTF-8.
        """
        ref cpy = Python().cpython()
        var unicode = cpy.PyUnicode_DecodeUTF8(string)
        if not unicode:
            raise cpy.unsafe_get_error()
        self = Self(from_owned=unicode)

    @implicit
    fn __init__(out self, value: StringLiteral) raises:
        """Initialize the object from a string literal.

        Args:
            value: The string literal value.

        Raises:
            If the string is not valid UTF-8.
        """
        self = Self(value.as_string_slice())

    @implicit
    fn __init__(out self, value: String) raises:
        """Initialize the object from a string.

        Args:
            value: The string value.

        Raises:
            If the string is not valid UTF-8.
        """
        self = Self(value.as_string_slice())

    @implicit
    fn __init__(out self, slice: Slice):
        """Initialize the object from a Mojo Slice.

        Args:
            slice: The dictionary value.
        """
        self = Self(from_owned=_slice_to_py_object_ptr(slice))

    @always_inline
    fn __init__[
        *Ts: ConvertibleToPython & Copyable
    ](out self, var *values: *Ts, __list_literal__: ()) raises:
        """Construct an Python list of objects.

        Parameters:
            Ts: The types of the input values.

        Args:
            values: The values to initialize the list with.
            __list_literal__: Tell Mojo to use this method for list literals.

        Returns:
            The constructed Python list.

        Raises:
            If the list construction fails.
        """
        self = Python._list(values)

    @always_inline
    fn __init__[
        *Ts: ConvertibleToPython & Copyable
    ](out self, var *values: *Ts, __set_literal__: ()) raises:
        """Construct an Python set of objects.

        Parameters:
            Ts: The types of the input values.

        Args:
            values: The values to initialize the set with.
            __set_literal__: Tell Mojo to use this method for set literals.

        Returns:
            The constructed Python set.

        Raises:
            If adding an element to the set fails.
        """
        ref cpy = Python().cpython()
        var set_ptr = cpy.PySet_New({})

        @parameter
        for i in range(len(VariadicList(Ts))):
            var obj = values[i].copy().to_python_object()
            var errno = cpy.PySet_Add(set_ptr, obj.steal_data())
            if errno == -1:
                raise cpy.unsafe_get_error()
        return PythonObject(from_owned=set_ptr)

    fn __init__(
        out self,
        var keys: List[PythonObject],
        var values: List[PythonObject],
        __dict_literal__: (),
    ) raises:
        """Construct a Python dictionary from a list of keys and a list of values.

        Args:
            keys: The keys of the dictionary.
            values: The values of the dictionary.
            __dict_literal__: Tell Mojo to use this method for dict literals.

        Raises:
            If setting a dictionary item fails.
        """
        ref cpy = Python().cpython()
        var dict_ptr = cpy.PyDict_New()
        for key, val in zip(keys, values):
            var errno = cpy.PyDict_SetItem(dict_ptr, key._obj_ptr, val._obj_ptr)
            if errno == -1:
                raise cpy.unsafe_get_error()
        return PythonObject(from_owned=dict_ptr)

    fn __copyinit__(out self, existing: Self):
        """Copy the object.

        This increments the underlying refcount of the existing object.

        Args:
            existing: The value to copy.
        """
        self = Self(from_borrowed=existing._obj_ptr)

    fn __del__(deinit self):
        """Destroy the object.

        This decrements the underlying refcount of the pointed-to object.
        """
        ref cpy = Python().cpython()
        # Acquire GIL such that __del__ can be called safely for cases where the
        # PyObject is handled in non-python contexts.
        with GILAcquired(Python(cpy)):
            cpy.Py_DecRef(self._obj_ptr)

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    fn __iter__(self) raises -> _PyIter:
        """Iterate over the object.

        Returns:
            An iterator object.

        Raises:
            If the object is not iterable.
        """
        ref cpy = Python().cpython()
        var iter_ptr = cpy.PyObject_GetIter(self._obj_ptr)
        if not iter_ptr:
            raise cpy.unsafe_get_error()
        return _PyIter(PythonObject(from_owned=iter_ptr))

    fn __getattr__(self, var name: String) raises -> PythonObject:
        """Return the value of the object attribute with the given name.

        Args:
            name: The name of the object attribute to return.

        Returns:
            The value of the object attribute with the given name.

        Raises:
            If the attribute does not exist.
        """
        ref cpy = Python().cpython()
        var attr_ptr = cpy.PyObject_GetAttrString(self._obj_ptr, name^)
        if not attr_ptr:
            raise cpy.unsafe_get_error()
        return PythonObject(from_owned=attr_ptr)

    fn __setattr__(self, var name: String, value: PythonObject) raises:
        """Set the given value for the object attribute with the given name.

        Args:
            name: The name of the object attribute to set.
            value: The new value to be set for that attribute.

        Raises:
            If setting the attribute fails.
        """
        ref cpy = Python().cpython()
        var errno = cpy.PyObject_SetAttrString(
            self._obj_ptr, name^, value._obj_ptr
        )
        if errno == -1:
            raise cpy.unsafe_get_error()

    fn __bool__(self) -> Bool:
        """Evaluate the boolean value of the object.

        Returns:
            Whether the object evaluates as true.
        """
        try:
            return Python().is_true(self)
        except Error:
            # TODO: make this function raise when we can raise parametrically.
            return abort[Bool]("object cannot be converted to bool")

    fn __is__(self, other: PythonObject) -> Bool:
        """Test if the PythonObject is the `other` PythonObject, the same as `x is y` in
        Python.

        Args:
            other: The right-hand-side value in the comparison.

        Returns:
            True if they are the same object and False otherwise.
        """
        ref cpy = Python().cpython()
        return cpy.Py_Is(self._obj_ptr, other._obj_ptr) != 0

    fn __getitem__(self, *args: PythonObject) raises -> PythonObject:
        """Return the value for the given key or keys.

        Args:
            args: The key or keys to access on this object.

        Returns:
            The value corresponding to the given key for this object.

        Raises:
            If the index is out of bounds or the key does not exist.
        """
        ref cpy = Python().cpython()
        var size = len(args)
        var key_ptr: PyObjectPtr
        if size == 1:
            key_ptr = cpy.Py_NewRef(args[0]._obj_ptr)
        else:
            key_ptr = cpy.PyTuple_New(size)
            for i in range(size):
                _ = cpy.PyTuple_SetItem(
                    key_ptr, i, cpy.Py_NewRef(args[i]._obj_ptr)
                )
        var res_ptr = cpy.PyObject_GetItem(self._obj_ptr, key_ptr)
        cpy.Py_DecRef(key_ptr)
        if not res_ptr:
            raise cpy.unsafe_get_error()
        return PythonObject(from_owned=res_ptr)

    fn __getitem__(self, *args: Slice) raises -> PythonObject:
        """Return the sliced value for the given Slice or Slices.

        Args:
            args: The Slice or Slices to apply to this object.

        Returns:
            The sliced value corresponding to the given Slice(s) for this object.

        Raises:
            If the index is out of bounds or the key does not exist.
        """
        ref cpy = Python().cpython()
        var size = len(args)
        var key_ptr: PyObjectPtr
        if size == 1:
            key_ptr = _slice_to_py_object_ptr(args[0])
        else:
            key_ptr = cpy.PyTuple_New(size)
            for i in range(size):
                var slice_ptr = _slice_to_py_object_ptr(args[i])
                _ = cpy.PyTuple_SetItem(key_ptr, i, slice_ptr)
        var res_ptr = cpy.PyObject_GetItem(self._obj_ptr, key_ptr)
        cpy.Py_DecRef(key_ptr)
        if not res_ptr:
            raise cpy.unsafe_get_error()
        return PythonObject(from_owned=res_ptr)

    fn __setitem__(self, *args: PythonObject, value: PythonObject) raises:
        """Set the value with the given key or keys.

        Args:
            args: The key or keys to set on this object.
            value: The value to set.

        Raises:
            If setting the item fails.
        """
        ref cpy = Python().cpython()
        var size = len(args)
        var key_ptr: PyObjectPtr
        if size == 1:
            key_ptr = cpy.Py_NewRef(args[0]._obj_ptr)
        else:
            key_ptr = cpy.PyTuple_New(size)
            for i in range(size):
                _ = cpy.PyTuple_SetItem(
                    key_ptr, i, cpy.Py_NewRef(args[i]._obj_ptr)
                )
        var errno = cpy.PyObject_SetItem(self._obj_ptr, key_ptr, value._obj_ptr)
        cpy.Py_DecRef(key_ptr)
        if errno == -1:
            raise cpy.unsafe_get_error()

    @doc_private
    fn __call_single_arg_inplace_method__(
        mut self, var method_name: String, rhs: PythonObject
    ) raises:
        var callable_obj: PythonObject
        try:
            callable_obj = self.__getattr__(String("__i", method_name[2:]))
        except:
            self = self.__getattr__(method_name^)(rhs)
        else:
            self = callable_obj(rhs)

    fn __mul__(self, rhs: PythonObject) raises -> PythonObject:
        """Multiplication.

        Calls the underlying object's `__mul__` method.

        Args:
            rhs: Right hand value.

        Returns:
            The product.

        Raises:
            If the operation is not supported.
        """
        return self.__getattr__("__mul__")(rhs)

    fn __rmul__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse multiplication.

        Calls the underlying object's `__rmul__` method.

        Args:
            lhs: The left-hand-side value that is multiplied by this object.

        Returns:
            The product of the multiplication.

        Raises:
            If the operation is not supported.
        """
        return self.__getattr__("__rmul__")(lhs)

    fn __imul__(mut self, rhs: PythonObject) raises:
        """In-place multiplication.

        Calls the underlying object's `__imul__` method.

        Args:
            rhs: The right-hand-side value by which this object is multiplied.

        Raises:
            If the operation is not supported.
        """
        return self.__call_single_arg_inplace_method__("__mul__", rhs)

    fn __add__(self, rhs: PythonObject) raises -> PythonObject:
        """Addition and concatenation.

        Calls the underlying object's `__add__` method.

        Args:
            rhs: Right hand value.

        Returns:
            The sum or concatenated values.

        Raises:
            If the operation is not supported.
        """
        return self.__getattr__("__add__")(rhs)

    fn __radd__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse addition and concatenation.

        Calls the underlying object's `__radd__` method.

        Args:
            lhs: The left-hand-side value to which this object is added or
                 concatenated.

        Returns:
            The sum.

        Raises:
            If the operation is not supported.
        """
        return self.__getattr__("__radd__")(lhs)

    fn __iadd__(mut self, rhs: PythonObject) raises:
        """Immediate addition and concatenation.

        Args:
            rhs: The right-hand-side value that is added to this object.

        Raises:
            If the operation is not supported.
        """
        return self.__call_single_arg_inplace_method__("__add__", rhs)

    fn __sub__(self, rhs: PythonObject) raises -> PythonObject:
        """Subtraction.

        Calls the underlying object's `__sub__` method.

        Args:
            rhs: Right hand value.

        Returns:
            The difference.

        Raises:
            If the operation is not supported.
        """
        return self.__getattr__("__sub__")(rhs)

    fn __rsub__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse subtraction.

        Calls the underlying object's `__rsub__` method.

        Args:
            lhs: The left-hand-side value from which this object is subtracted.

        Returns:
            The result of subtracting this from the given value.

        Raises:
            If the operation is not supported.
        """
        return self.__getattr__("__rsub__")(lhs)

    fn __isub__(mut self, rhs: PythonObject) raises:
        """Immediate subtraction.

        Args:
            rhs: The right-hand-side value that is subtracted from this object.

        Raises:
            If the operation is not supported.
        """
        return self.__call_single_arg_inplace_method__("__sub__", rhs)

    fn __floordiv__(self, rhs: PythonObject) raises -> PythonObject:
        """Return the division of self and rhs rounded down to the nearest
        integer.

        Calls the underlying object's `__floordiv__` method.

        Args:
            rhs: The right-hand-side value by which this object is divided.

        Returns:
            The result of dividing this by the right-hand-side value, modulo any
            remainder.

        Raises:
            If the operation fails.
        """
        return self.__getattr__("__floordiv__")(rhs)

    fn __rfloordiv__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse floor division.

        Calls the underlying object's `__rfloordiv__` method.

        Args:
            lhs: The left-hand-side value that is divided by this object.

        Returns:
            The result of dividing the given value by this, modulo any
            remainder.

        Raises:
            If the operation fails.
        """
        return self.__getattr__("__rfloordiv__")(lhs)

    fn __ifloordiv__(mut self, rhs: PythonObject) raises:
        """Immediate floor division.

        Args:
            rhs: The value by which this object is divided.

        Raises:
            If the operation fails.
        """
        return self.__call_single_arg_inplace_method__("__floordiv__", rhs)

    fn __truediv__(self, rhs: PythonObject) raises -> PythonObject:
        """Division.

        Calls the underlying object's `__truediv__` method.

        Args:
            rhs: The right-hand-side value by which this object is divided.

        Returns:
            The result of dividing the right-hand-side value by this.

        Raises:
            If the operation fails.
        """
        return self.__getattr__("__truediv__")(rhs)

    fn __rtruediv__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse division.

        Calls the underlying object's `__rtruediv__` method.

        Args:
            lhs: The left-hand-side value that is divided by this object.

        Returns:
            The result of dividing the given value by this.

        Raises:
            If the operation fails.
        """
        return self.__getattr__("__rtruediv__")(lhs)

    fn __itruediv__(mut self, rhs: PythonObject) raises:
        """Immediate division.

        Args:
            rhs: The value by which this object is divided.

        Raises:
            If the operation fails.
        """
        return self.__call_single_arg_inplace_method__("__truediv__", rhs)

    fn __mod__(self, rhs: PythonObject) raises -> PythonObject:
        """Return the remainder of self divided by rhs.

        Calls the underlying object's `__mod__` method.

        Args:
            rhs: The value to divide on.

        Returns:
            The remainder of dividing self by rhs.

        Raises:
            If the operation fails.
        """
        return self.__getattr__("__mod__")(rhs)

    fn __rmod__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse modulo.

        Calls the underlying object's `__rmod__` method.

        Args:
            lhs: The left-hand-side value that is divided by this object.

        Returns:
            The remainder from dividing the given value by this.

        Raises:
            If the operation fails.
        """
        return self.__getattr__("__rmod__")(lhs)

    fn __imod__(mut self, rhs: PythonObject) raises:
        """Immediate modulo.

        Args:
            rhs: The right-hand-side value that is used to divide this object.

        Raises:
            If the operation fails.
        """
        return self.__call_single_arg_inplace_method__("__mod__", rhs)

    fn __xor__(self, rhs: PythonObject) raises -> PythonObject:
        """Exclusive OR.

        Args:
            rhs: The right-hand-side value with which this object is exclusive
                 OR'ed.

        Returns:
            The exclusive OR result of this and the given value.

        Raises:
            If the operation fails.
        """
        return self.__getattr__("__xor__")(rhs)

    fn __rxor__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse exclusive OR.

        Args:
            lhs: The left-hand-side value that is exclusive OR'ed with this
                 object.

        Returns:
            The exclusive OR result of the given value and this.

        Raises:
            If the operation fails.
        """
        return self.__getattr__("__rxor__")(lhs)

    fn __ixor__(mut self, rhs: PythonObject) raises:
        """Immediate exclusive OR.

        Args:
            rhs: The right-hand-side value with which this object is
                 exclusive OR'ed.

        Raises:
            If the operation fails.
        """
        return self.__call_single_arg_inplace_method__("__xor__", rhs)

    fn __or__(self, rhs: PythonObject) raises -> PythonObject:
        """Bitwise OR.

        Args:
            rhs: The right-hand-side value with which this object is bitwise
                 OR'ed.

        Returns:
            The bitwise OR result of this and the given value.

        Raises:
            If the operation fails.
        """
        return self.__getattr__("__or__")(rhs)

    fn __ror__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse bitwise OR.

        Args:
            lhs: The left-hand-side value that is bitwise OR'ed with this
                 object.

        Returns:
            The bitwise OR result of the given value and this.

        Raises:
            If the operation fails.
        """
        return self.__getattr__("__ror__")(lhs)

    fn __ior__(mut self, rhs: PythonObject) raises:
        """Immediate bitwise OR.

        Args:
            rhs: The right-hand-side value with which this object is bitwise
                 OR'ed.

        Raises:
            If the operation fails.
        """
        return self.__call_single_arg_inplace_method__("__or__", rhs)

    fn __and__(self, rhs: PythonObject) raises -> PythonObject:
        """Bitwise AND.

        Args:
            rhs: The right-hand-side value with which this object is bitwise
                 AND'ed.

        Returns:
            The bitwise AND result of this and the given value.

        Raises:
            If the operation fails.
        """
        return self.__getattr__("__and__")(rhs)

    fn __rand__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse bitwise and.

        Args:
            lhs: The left-hand-side value that is bitwise AND'ed with this
                 object.

        Returns:
            The bitwise AND result of the given value and this.

        Raises:
            If the operation fails.
        """
        return self.__getattr__("__rand__")(lhs)

    fn __iand__(mut self, rhs: PythonObject) raises:
        """Immediate bitwise AND.

        Args:
            rhs: The right-hand-side value with which this object is bitwise
                 AND'ed.

        Raises:
            If the operation fails.
        """
        return self.__call_single_arg_inplace_method__("__and__", rhs)

    fn __rshift__(self, rhs: PythonObject) raises -> PythonObject:
        """Bitwise right shift.

        Args:
            rhs: The right-hand-side value by which this object is bitwise
                 shifted to the right.

        Returns:
            This value, shifted right by the given value.

        Raises:
            If the operation fails.
        """
        return self.__getattr__("__rshift__")(rhs)

    fn __rrshift__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse bitwise right shift.

        Args:
            lhs: The left-hand-side value that is bitwise shifted to the right
                 by this object.

        Returns:
            The given value, shifted right by this.

        Raises:
            If the operation fails.
        """
        return self.__getattr__("__rrshift__")(lhs)

    fn __irshift__(mut self, rhs: PythonObject) raises:
        """Immediate bitwise right shift.

        Args:
            rhs: The right-hand-side value by which this object is bitwise
                 shifted to the right.

        Raises:
            If the operation fails.
        """
        return self.__call_single_arg_inplace_method__("__rshift__", rhs)

    fn __lshift__(self, rhs: PythonObject) raises -> PythonObject:
        """Bitwise left shift.

        Args:
            rhs: The right-hand-side value by which this object is bitwise
                 shifted to the left.

        Returns:
            This value, shifted left by the given value.

        Raises:
            If the operation fails.
        """
        return self.__getattr__("__lshift__")(rhs)

    fn __rlshift__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse bitwise left shift.

        Args:
            lhs: The left-hand-side value that is bitwise shifted to the left
                 by this object.

        Returns:
            The given value, shifted left by this.

        Raises:
            If the operation fails.
        """
        return self.__getattr__("__rlshift__")(lhs)

    fn __ilshift__(mut self, rhs: PythonObject) raises:
        """Immediate bitwise left shift.

        Args:
            rhs: The right-hand-side value by which this object is bitwise
                 shifted to the left.

        Raises:
            If the operation fails.
        """
        return self.__call_single_arg_inplace_method__("__lshift__", rhs)

    fn __pow__(self, exp: PythonObject) raises -> PythonObject:
        """Raises this object to the power of the given value.

        Args:
            exp: The exponent.

        Returns:
            The result of raising this by the given exponent.

        Raises:
            If the operation fails.
        """
        return self.__getattr__("__pow__")(exp)

    fn __rpow__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse power of.

        Args:
            lhs: The number that is raised to the power of this object.

        Returns:
            The result of raising the given value by this exponent.

        Raises:
            If the operation fails.
        """
        return self.__getattr__("__rpow__")(lhs)

    fn __ipow__(mut self, rhs: PythonObject) raises:
        """Immediate power of.

        Args:
            rhs: The exponent.

        Raises:
            If the operation fails.
        """
        return self.__call_single_arg_inplace_method__("__pow__", rhs)

    fn __lt__(self, rhs: PythonObject) raises -> PythonObject:
        """Less than (rich) comparison operator.

        Args:
            rhs: The value of the right hand side of the comparison.

        Returns:
            The result of the comparison, not necessarily a boolean.

        Raises:
            If the object doesn't implement the `__lt__` method, or if it fails.
        """
        return self.__getattr__("__lt__")(rhs)

    fn __le__(self, rhs: PythonObject) raises -> PythonObject:
        """Less than or equal (rich) comparison operator.

        Args:
            rhs: The value of the right hand side of the comparison.

        Returns:
            The result of the comparison, not necessarily a boolean.

        Raises:
            If the object doesn't implement the `__le__` method, or if it fails.
        """
        return self.__getattr__("__le__")(rhs)

    fn __gt__(self, rhs: PythonObject) raises -> PythonObject:
        """Greater than (rich) comparison operator.

        Args:
            rhs: The value of the right hand side of the comparison.

        Returns:
            The result of the comparison, not necessarily a boolean.

        Raises:
            If the object doesn't implement the `__gt__` method, or if it fails.
        """
        return self.__getattr__("__gt__")(rhs)

    fn __ge__(self, rhs: PythonObject) raises -> PythonObject:
        """Greater than or equal (rich) comparison operator.

        Args:
            rhs: The value of the right hand side of the comparison.

        Returns:
            The result of the comparison, not necessarily a boolean.

        Raises:
            If the object doesn't implement the `__ge__` method, or if it fails.
        """
        return self.__getattr__("__ge__")(rhs)

    fn __eq__(self, rhs: PythonObject) raises -> PythonObject:
        """Equality (rich) comparison operator.

        Args:
            rhs: The value of the right hand side of the comparison.

        Returns:
            The result of the comparison, not necessarily a boolean.

        Raises:
            If the object doesn't implement the `__eq__` method, or if it fails.
        """
        return self.__getattr__("__eq__")(rhs)

    fn __ne__(self, rhs: PythonObject) raises -> PythonObject:
        """Inequality (rich) comparison operator.

        Args:
            rhs: The value of the right hand side of the comparison.

        Returns:
            The result of the comparison, not necessarily a boolean.

        Raises:
            If the object doesn't implement the `__ne__` method, or if it fails.
        """
        return self.__getattr__("__ne__")(rhs)

    fn __pos__(self) raises -> PythonObject:
        """Positive.

        Calls the underlying object's `__pos__` method.

        Returns:
            The result of prefixing this object with a `+` operator. For most
            numerical objects, this does nothing.

        Raises:
            If the operation fails.
        """
        return self.__getattr__("__pos__")()

    fn __neg__(self) raises -> PythonObject:
        """Negative.

        Calls the underlying object's `__neg__` method.

        Returns:
            The result of prefixing this object with a `-` operator. For most
            numerical objects, this returns the negative.

        Raises:
            If the call fails.
        """
        return self.__getattr__("__neg__")()

    fn __invert__(self) raises -> PythonObject:
        """Inversion.

        Calls the underlying object's `__invert__` method.

        Returns:
            The logical inverse of this object: a bitwise representation where
            all bits are flipped, from zero to one, and from one to zero.

        Raises:
            If the call fails.
        """
        return self.__getattr__("__invert__")()

    fn __contains__(self, rhs: PythonObject) raises -> Bool:
        """Contains dunder.

        Calls the underlying object's `__contains__` method.

        Args:
            rhs: Right hand value.

        Returns:
            True if rhs is in self.

        Raises:
            If the operation fails.
        """
        # TODO: replace/optimize with c-python function.
        # TODO: implement __getitem__ step for cpython membership test operator.
        ref cpy = Python().cpython()
        if cpy.PyObject_HasAttrString(self._obj_ptr, "__contains__"):
            return self.__getattr__("__contains__")(rhs).__bool__()
        for v in self:
            if v == rhs:
                return True
        return False

    # see https://github.com/python/cpython/blob/main/Objects/call.c
    # for decrement rules
    fn __call__(
        self, *args: PythonObject, **kwargs: PythonObject
    ) raises -> PythonObject:
        """Call the underlying object as if it were a function.

        Args:
            args: Positional arguments to the function.
            kwargs: Keyword arguments to the function.

        Raises:
            If the function cannot be called for any reason.

        Returns:
            The return value from the called object.
        """
        ref cpy = Python().cpython()
        var size = len(args)
        var args_ptr = cpy.PyTuple_New(size)
        for i in range(size):
            _ = cpy.PyTuple_SetItem(
                args_ptr, i, cpy.Py_NewRef(args[i]._obj_ptr)
            )
        var kwargs_ptr = Python._dict(kwargs)
        var res_ptr = cpy.PyObject_Call(self._obj_ptr, args_ptr, kwargs_ptr)
        cpy.Py_DecRef(args_ptr)
        cpy.Py_DecRef(kwargs_ptr)
        if not res_ptr:
            raise cpy.unsafe_get_error()
        return PythonObject(from_owned=res_ptr)

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    fn __len__(self) raises -> Int:
        """Returns the length of the object.

        Returns:
            The length of the object.

        Raises:
            If the operation fails.
        """
        ref cpy = Python().cpython()
        var length = Int(cpy.PyObject_Length(self._obj_ptr))
        if length == -1 and cpy.PyErr_Occurred():
            # Custom python types may return -1 even in non-error cases.
            raise cpy.unsafe_get_error()
        return length

    fn __hash__(self) raises -> Int:
        """Returns the hash value of the object.

        Returns:
            The hash value of the object.

        Raises:
            If the operation fails.
        """
        ref cpy = Python().cpython()
        var res = Int(cpy.PyObject_Hash(self._obj_ptr))
        if res == -1 and cpy.PyErr_Occurred():
            # Custom python types may return -1 even in non-error cases.
            raise cpy.unsafe_get_error()
        return res

    fn __int__(self) raises -> PythonObject:
        """Convert the PythonObject to a Python `int` (i.e. arbitrary precision
        integer).

        Returns:
            A Python `int` object.

        Raises:
            An error if the conversion failed.
        """
        return Python.int(self)

    fn __float__(self) raises -> PythonObject:
        """Convert the PythonObject to a Python `float` object.

        Returns:
            A Python `float` object.

        Raises:
            If the conversion fails.
        """
        return Python.float(self)

    @no_inline
    fn __str__(self) raises -> PythonObject:
        """Convert the PythonObject to a Python `str`.

        Returns:
            A Python `str` object.

        Raises:
            An error if the conversion failed.
        """
        return Python.str(self)

    fn write_to(self, mut writer: Some[Writer]):
        """
        Formats this Python object to the provided Writer.

        Args:
            writer: The object to write to.
        """

        try:
            # TODO: Avoid this intermediate String allocation, if possible.
            writer.write(String(self))
        except:
            # TODO: make this method raising when we can raise parametrically.
            return abort("failed to write PythonObject to writer")

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn to_python_object(var self) raises -> PythonObject:
        """Convert this value to a PythonObject.

        Returns:
            A PythonObject representing the value.

        Raises:
            If the conversion to Python object fails.
        """
        return self^

    fn steal_data(var self) -> PyObjectPtr:
        """Take ownership of the underlying pointer from the Python object.

        Returns:
            The underlying data.
        """
        var ptr = self._obj_ptr
        self._obj_ptr = {}
        return ptr

    fn unsafe_get_as_pointer[
        dtype: DType
    ](self) raises -> UnsafePointer[Scalar[dtype]]:
        """Reinterpret a Python integer as a Mojo pointer.

        Warning: converting from an integer to a pointer is unsafe! The
        compiler assumes the resulting pointer DOES NOT alias any Mojo-derived
        pointer. This is OK if the pointer originates from and is owned by
        Python, e.g. the data underpinning a torch tensor.

        Parameters:
            dtype: The desired DType of the pointer.

        Returns:
            An `UnsafePointer` for the underlying Python data.

        Raises:
            If the operation fails.
        """
        return _unsafe_aliasing_address_to_pointer[Scalar[dtype]](Int(self))

    fn downcast_value_ptr[
        T: AnyType
    ](self, *, func: Optional[StaticString] = None) raises -> UnsafePointer[T]:
        """Get a pointer to the expected contained Mojo value of type `T`.

        This method validates that this object actually contains an instance of
        `T`, and will raise an error if it does not.

        Mojo values are stored as Python objects backed by the `PyMojoObject[T]`
        struct.

        Args:
            func: Optional name of bound Mojo function that the raised
              TypeError should reference if downcasting fails.

        Parameters:
            T: The type of the Mojo value that this Python object is expected
              to contain.

        Returns:
            A pointer to the inner Mojo value.

        Raises:
            If the Python object does not contain an instance of the Mojo `T`
            type.
        """
        if opt := self._try_downcast_value[T]():
            return opt.unsafe_take()

        if func:
            raise Error(
                String.format(
                    (
                        "TypeError: {}() expected Mojo '{}' type argument, got"
                        " '{}'"
                    ),
                    func[],
                    get_type_name[T](),
                    _get_type_name(self),
                )
            )
        else:
            raise Error(
                String.format(
                    "TypeError: expected Mojo '{}' type value, got '{}'",
                    get_type_name[T](),
                    _get_type_name(self),
                )
            )

    fn _try_downcast_value[
        T: AnyType
    ](var self) raises -> Optional[UnsafePointer[T]]:
        """Try to get a pointer to the expected contained Mojo value of type `T`.

        None will be returned if the type of this object does not match the
        bound Python type of `T`, or if the Mojo value has not been initialized.

        This function will raise if the provided Mojo type `T` has not been
        bound to a Python type using a `PythonTypeBuilder`.

        Parameters:
            T: The type of the Mojo value that this Python object is expected
              to contain.

        Raises:
            If `T` has not been bound to a Python type object.
        """
        ref cpy = Python().cpython()
        var type = PyObjectPtr(upcast_from=cpy.Py_TYPE(self._obj_ptr))
        var expected_type = lookup_py_type_object[T]()._obj_ptr
        if type == expected_type:
            ref mojo_obj = self._obj_ptr.bitcast[PyMojoObject[T]]()[]
            if mojo_obj.is_initialized:
                return UnsafePointer(to=mojo_obj.mojo_value)
        return None

    fn unchecked_downcast_value_ptr[
        mut: Bool, origin: Origin[mut], //, T: AnyType
    ](ref [origin]self) -> UnsafePointer[T, mut=mut, origin=origin]:
        """Get a pointer to the expected Mojo value of type `T`.

        This function assumes that this Python object was allocated as an
        instance of `PyMojoObject[T]` and that the Mojo value has been
        initialized.

        Parameters:
            mut: The mutability of self.
            origin: The origin of self.
            T: The type of the Mojo value stored in this object.

        Returns:
            A pointer to the inner Mojo value.

        Safety:

        The user must be certain that this Python object type matches the bound
        Python type object for `T`.
        """
        ref mojo_obj = self._obj_ptr.bitcast[PyMojoObject[T]]()[]
        # TODO(MSTDL-950): Should use something like `addr_of!`
        # Safety: The mutability matches that of `self`.
        return (
            UnsafePointer(to=mojo_obj.mojo_value)
            .unsafe_mut_cast[mut]()
            .unsafe_origin_cast[origin]()
        )


# ===-----------------------------------------------------------------------===#
# Factory functions for PythonObject
# ===-----------------------------------------------------------------------===#


fn _unsafe_alloc[
    T: AnyType
](type_obj_ptr: UnsafePointer[PyTypeObject]) raises -> PyObjectPtr:
    """Allocate an uninitialized Python object for storing a Mojo value.

    Parameters:
        T: The Mojo type of the value that will be stored in the Python object.

    Args:
        type_obj_ptr: Pointer to the Python type object describing the layout.

    Returns:
        A new Python object pointer with uninitialized storage.

    Raises:
        If the Python object allocation fails.
    """
    ref cpy = Python().cpython()
    var obj_ptr = cpy.PyType_GenericAlloc(type_obj_ptr, 0)
    if not obj_ptr:
        raise Error("Allocation of Python object failed.")
    return obj_ptr


fn _unsafe_init[
    T: Movable, //,
](obj_ptr: PyObjectPtr, var mojo_value: T) raises:
    """Initialize a Python object pointer with a Mojo value.

    Parameters:
        T: The Mojo type of the value that the resulting Python object holds.

    Args:
        obj_ptr: The Python object pointer to initialize.
            The pointer must have been allocated using the correct type object.
        mojo_value: The Mojo value to store in the Python object.

    # Safety
     `obj_ptr` must be a Python object pointer allocated using the correct
     type object. Use of any other pointer is invalid.
    """
    ref mojo_obj = obj_ptr.bitcast[PyMojoObject[T]]()[]
    UnsafePointer(to=mojo_obj.mojo_value).init_pointee_move(mojo_value^)
    mojo_obj.is_initialized = True


fn _unsafe_alloc_init[
    T: Movable, //,
](
    type_obj_ptr: UnsafePointer[PyTypeObject], var mojo_value: T
) raises -> PythonObject:
    """Allocate a Python object pointer and initialize it with a Mojo value.

    Parameters:
        T: The Mojo type of the value that the resulting Python object holds.

    Args:
        type_obj_ptr: Must be the Python type object describing `PyTypeObject[T]`.
        mojo_value: The Mojo value to store in the new Python object.

    Returns:
        A new PythonObject containing the Mojo value.

    # Safety
    `type_obj_ptr` must be a Python type object created by `PythonTypeBuilder`,
    whose underlying storage type is the `PyMojoObject` struct. Use of any other
    type object is invalid.
    Raises:
        If the Python object allocation fails.
    """
    var obj_ptr = _unsafe_alloc[T](type_obj_ptr)
    _unsafe_init(obj_ptr, mojo_value^)
    return PythonObject(from_owned=obj_ptr)


# ===-----------------------------------------------------------------------===#
# Helper functions
# ===-----------------------------------------------------------------------===#


fn _slice_to_py_object_ptr(slice: Slice) -> PyObjectPtr:
    """Convert Mojo Slice to Python slice parameters.

    Deliberately avoids using `span.indices()` here and instead passes
    the Slice parameters directly to Python. Python's C implementation
    already handles such conditions, allowing Python to apply its own slice
    handling and error handling.


    Args:
        slice: A Mojo slice object to be converted.

    Returns:
        PyObjectPtr: The pointer to the Python slice.

    """
    ref cpy = Python().cpython()
    var start = cpy.PyLong_FromSsize_t(
        c_ssize_t(slice.start.value())
    ) if slice.start else cpy.Py_None()
    var stop = cpy.PyLong_FromSsize_t(
        c_ssize_t(slice.end.value())
    ) if slice.end else cpy.Py_None()
    var step = cpy.PyLong_FromSsize_t(
        c_ssize_t(slice.step.value())
    ) if slice.step else cpy.Py_None()
    var res = cpy.PySlice_New(start, stop, step)
    cpy.Py_DecRef(start)
    cpy.Py_DecRef(stop)
    cpy.Py_DecRef(step)
    return res


__extension SIMD:
    # TODO(MSTDL-1587): Remove the dummy parameter.
    @always_inline
    fn __init__[
        *, `_`: NoneType = None
    ](out self: Scalar[dtype], obj: PythonObject, /) raises:
        """Initialize a SIMD value from a PythonObject.

        Parameters:
            _: A dummy parameter to ensure this overload has lower priority than
                the others. Its value is ignored.

        Args:
            obj: The PythonObject to convert.

        Raises:
            If the conversion to double fails.
        """

        @parameter
        if dtype.is_floating_point():
            ref cpy = Python().cpython()
            var float_value = cpy.PyFloat_AsDouble(obj._obj_ptr)
            if float_value == -1.0 and cpy.PyErr_Occurred():
                # Note that -1.0 does not guarantee an error, it just means we
                # need to check if there was an exception.
                raise cpy.unsafe_get_error()
            # NOTE: if dtype is not float64, we truncate.
            self = Scalar[dtype](float_value)
        elif dtype.is_integral() and bit_width_of[dtype]() <= 64:
            self = Int(obj)
        else:
            self = Scalar[dtype]()
            constrained[False, "unsupported dtype"]()
