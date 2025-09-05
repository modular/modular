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

from os import abort
from sys.ffi import c_double, c_long, c_size_t, c_ssize_t
from sys.intrinsics import _unsafe_aliasing_address_to_pointer
from compile.reflection import get_type_name

from ._cpython import CPython, PyObjectPtr, PyObject, PyTypeObject, GILAcquired
from .python import Python
from .bindings import _get_type_name, lookup_py_type_object, PyMojoObject


trait ConvertibleToPython:
    """A trait that indicates a type can be converted to a PythonObject, and
    that specifies the behavior with a `to_python_object` method."""

    fn to_python_object(var self) raises -> PythonObject:
        """Convert a value to a PythonObject.

        Returns:
            A PythonObject representing the value.

        Raises:
            If the conversion to a PythonObject failed.
        """
        ...


trait ConvertibleFromPython(Copyable, Movable):
    """Denotes a type that can attempt construction from a read-only Python
    object.
    """

    fn __init__(out self, obj: PythonObject) raises:
        """Attempt to construct an instance of this object from a read-only
        Python value.

        Args:
            obj: The Python object to convert from.

        Raises:
            If conversion was not successful.
        """
        ...


struct _PyIter(ImplicitlyCopyable):
    """A Python iterator."""

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

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
            var val = c_size_t(mlir_value=value.cast[DType.index]()._mlir_value)
            self = Self(from_owned=cpy.PyLong_FromSize_t(val))
        elif dtype.is_integral():
            var val = c_ssize_t(value.cast[DType.index]()._mlir_value)
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
            raise cpy.get_error()
        self = Self(from_owned=unicode)

    @implicit
    fn __init__(out self, value: StringLiteral) raises:
        """Initialize the object from a string literal.

        Args:
            value: The string literal value.
        """
        self = Self(value.as_string_slice())

    @implicit
    fn __init__(out self, value: String) raises:
        """Initialize the object from a string.

        Args:
            value: The string value.
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
        """
        ref cpython = Python().cpython()
        var obj_ptr = cpython.PySet_New({})

        if not obj_ptr:
            raise cpython.get_error()

        @parameter
        for i in range(len(VariadicList(Ts))):
            var obj = values[i].copy().to_python_object()
            cpython.Py_IncRef(obj._obj_ptr)
            var result = cpython.PySet_Add(obj_ptr, obj._obj_ptr)
            if result == -1:
                raise cpython.get_error()

        return PythonObject(from_owned=obj_ptr)

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
        """
        ref cpython = Python().cpython()
        var dict_obj_ptr = cpython.PyDict_New()
        if not dict_obj_ptr:
            raise Error("internal error: PyDict_New failed")

        for i in range(len(keys)):
            var key_obj = keys[i].copy().to_python_object()
            var val_obj = values[i].copy().to_python_object()
            var result = cpython.PyDict_SetItem(
                dict_obj_ptr, key_obj._obj_ptr, val_obj._obj_ptr
            )
            if result != 0:
                raise Error("internal error: PyDict_SetItem failed")

        return PythonObject(from_owned=dict_obj_ptr)

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
        ref cpython = Python().cpython()
        var iter_ptr = cpython.PyObject_GetIter(self._obj_ptr)
        if not iter_ptr:
            raise cpython.get_error()
        return _PyIter(PythonObject(from_owned=iter_ptr))

    fn __getattr__(self, var name: String) raises -> PythonObject:
        """Return the value of the object attribute with the given name.

        Args:
            name: The name of the object attribute to return.

        Returns:
            The value of the object attribute with the given name.
        """
        ref cpython = Python().cpython()
        var result = cpython.PyObject_GetAttrString(self._obj_ptr, name^)
        if not result:
            raise cpython.get_error()
        return PythonObject(from_owned=result)

    fn __setattr__(self, var name: String, new_value: PythonObject) raises:
        """Set the given value for the object attribute with the given name.

        Args:
            name: The name of the object attribute to set.
            new_value: The new value to be set for that attribute.
        """
        ref cpython = Python().cpython()
        var result = cpython.PyObject_SetAttrString(
            self._obj_ptr, name^, new_value._obj_ptr
        )
        if result != 0:
            raise cpython.get_error()

    fn __bool__(self) -> Bool:
        """Evaluate the boolean value of the object.

        Returns:
            Whether the object evaluates as true.
        """
        try:
            return Python().is_true(self)
        except Error:
            # TODO: make this function raise when we can raise parametrically.
            debug_assert(False, "object cannot be converted to a bool")
            return False

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
        """
        ref cpython = Python().cpython()
        var size = len(args)
        var key_obj: PyObjectPtr
        if size == 1:
            key_obj = args[0]._obj_ptr
        else:
            key_obj = cpython.PyTuple_New(size)
            for i in range(size):
                var arg_value = args[i]._obj_ptr
                cpython.Py_IncRef(arg_value)
                var result = cpython.PyTuple_SetItem(key_obj, i, arg_value)
                if result != 0:
                    raise Error("internal error: PyTuple_SetItem failed")

        cpython.Py_IncRef(key_obj)
        var result = cpython.PyObject_GetItem(self._obj_ptr, key_obj)
        cpython.Py_DecRef(key_obj)
        if not result:
            raise cpython.get_error()
        return PythonObject(from_owned=result)

    fn __getitem__(self, *args: Slice) raises -> PythonObject:
        """Return the sliced value for the given Slice or Slices.

        Args:
            args: The Slice or Slices to apply to this object.

        Returns:
            The sliced value corresponding to the given Slice(s) for this object.
        """
        ref cpython = Python().cpython()
        var size = len(args)
        var key_obj: PyObjectPtr

        if size == 1:
            key_obj = _slice_to_py_object_ptr(args[0])
        else:
            key_obj = cpython.PyTuple_New(size)
            for i in range(size):
                var slice_obj = _slice_to_py_object_ptr(args[i])
                var result = cpython.PyTuple_SetItem(key_obj, i, slice_obj)
                if result != 0:
                    raise Error("internal error: PyTuple_SetItem failed")

        cpython.Py_IncRef(key_obj)
        var result = cpython.PyObject_GetItem(self._obj_ptr, key_obj)
        cpython.Py_DecRef(key_obj)
        if not result:
            raise cpython.get_error()
        return PythonObject(from_owned=result)

    fn __setitem__(self, *args: PythonObject, value: PythonObject) raises:
        """Set the value with the given key or keys.

        Args:
            args: The key or keys to set on this object.
            value: The value to set.
        """
        ref cpython = Python().cpython()
        var size = len(args)
        var key_obj: PyObjectPtr

        if size == 1:
            key_obj = args[0]._obj_ptr
        else:
            key_obj = cpython.PyTuple_New(size)
            for i in range(size):
                var arg_value = args[i]._obj_ptr
                cpython.Py_IncRef(arg_value)
                var result = cpython.PyTuple_SetItem(key_obj, i, arg_value)
                if result != 0:
                    raise Error("internal error: PyTuple_SetItem failed")

        cpython.Py_IncRef(key_obj)
        cpython.Py_IncRef(value._obj_ptr)
        var result = cpython.PyObject_SetItem(
            self._obj_ptr, key_obj, value._obj_ptr
        )
        if result != 0:
            raise cpython.get_error()
        cpython.Py_DecRef(key_obj)
        cpython.Py_DecRef(value._obj_ptr)

    @doc_private
    fn __call_single_arg_inplace_method__(
        mut self, var method_name: String, rhs: PythonObject
    ) raises:
        var callable_obj: PythonObject
        try:
            callable_obj = self.__getattr__(String("__i", method_name[2:]))
        except:
            self = self.__getattr__(method_name^)(rhs)
            return

        self = callable_obj(rhs)

    fn __mul__(self, rhs: PythonObject) raises -> PythonObject:
        """Multiplication.

        Calls the underlying object's `__mul__` method.

        Args:
            rhs: Right hand value.

        Returns:
            The product.
        """
        return self.__getattr__("__mul__")(rhs)

    fn __rmul__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse multiplication.

        Calls the underlying object's `__rmul__` method.

        Args:
            lhs: The left-hand-side value that is multiplied by this object.

        Returns:
            The product of the multiplication.
        """
        return self.__getattr__("__rmul__")(lhs)

    fn __imul__(mut self, rhs: PythonObject) raises:
        """In-place multiplication.

        Calls the underlying object's `__imul__` method.

        Args:
            rhs: The right-hand-side value by which this object is multiplied.
        """
        return self.__call_single_arg_inplace_method__("__mul__", rhs)

    fn __add__(self, rhs: PythonObject) raises -> PythonObject:
        """Addition and concatenation.

        Calls the underlying object's `__add__` method.

        Args:
            rhs: Right hand value.

        Returns:
            The sum or concatenated values.
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
        """
        return self.__getattr__("__radd__")(lhs)

    fn __iadd__(mut self, rhs: PythonObject) raises:
        """Immediate addition and concatenation.

        Args:
            rhs: The right-hand-side value that is added to this object.
        """
        return self.__call_single_arg_inplace_method__("__add__", rhs)

    fn __sub__(self, rhs: PythonObject) raises -> PythonObject:
        """Subtraction.

        Calls the underlying object's `__sub__` method.

        Args:
            rhs: Right hand value.

        Returns:
            The difference.
        """
        return self.__getattr__("__sub__")(rhs)

    fn __rsub__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse subtraction.

        Calls the underlying object's `__rsub__` method.

        Args:
            lhs: The left-hand-side value from which this object is subtracted.

        Returns:
            The result of subtracting this from the given value.
        """
        return self.__getattr__("__rsub__")(lhs)

    fn __isub__(mut self, rhs: PythonObject) raises:
        """Immediate subtraction.

        Args:
            rhs: The right-hand-side value that is subtracted from this object.
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
        """
        return self.__getattr__("__rfloordiv__")(lhs)

    fn __ifloordiv__(mut self, rhs: PythonObject) raises:
        """Immediate floor division.

        Args:
            rhs: The value by which this object is divided.
        """
        return self.__call_single_arg_inplace_method__("__floordiv__", rhs)

    fn __truediv__(self, rhs: PythonObject) raises -> PythonObject:
        """Division.

        Calls the underlying object's `__truediv__` method.

        Args:
            rhs: The right-hand-side value by which this object is divided.

        Returns:
            The result of dividing the right-hand-side value by this.
        """
        return self.__getattr__("__truediv__")(rhs)

    fn __rtruediv__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse division.

        Calls the underlying object's `__rtruediv__` method.

        Args:
            lhs: The left-hand-side value that is divided by this object.

        Returns:
            The result of dividing the given value by this.
        """
        return self.__getattr__("__rtruediv__")(lhs)

    fn __itruediv__(mut self, rhs: PythonObject) raises:
        """Immediate division.

        Args:
            rhs: The value by which this object is divided.
        """
        return self.__call_single_arg_inplace_method__("__truediv__", rhs)

    fn __mod__(self, rhs: PythonObject) raises -> PythonObject:
        """Return the remainder of self divided by rhs.

        Calls the underlying object's `__mod__` method.

        Args:
            rhs: The value to divide on.

        Returns:
            The remainder of dividing self by rhs.
        """
        return self.__getattr__("__mod__")(rhs)

    fn __rmod__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse modulo.

        Calls the underlying object's `__rmod__` method.

        Args:
            lhs: The left-hand-side value that is divided by this object.

        Returns:
            The remainder from dividing the given value by this.
        """
        return self.__getattr__("__rmod__")(lhs)

    fn __imod__(mut self, rhs: PythonObject) raises:
        """Immediate modulo.

        Args:
            rhs: The right-hand-side value that is used to divide this object.
        """
        return self.__call_single_arg_inplace_method__("__mod__", rhs)

    fn __xor__(self, rhs: PythonObject) raises -> PythonObject:
        """Exclusive OR.

        Args:
            rhs: The right-hand-side value with which this object is exclusive
                 OR'ed.

        Returns:
            The exclusive OR result of this and the given value.
        """
        return self.__getattr__("__xor__")(rhs)

    fn __rxor__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse exclusive OR.

        Args:
            lhs: The left-hand-side value that is exclusive OR'ed with this
                 object.

        Returns:
            The exclusive OR result of the given value and this.
        """
        return self.__getattr__("__rxor__")(lhs)

    fn __ixor__(mut self, rhs: PythonObject) raises:
        """Immediate exclusive OR.

        Args:
            rhs: The right-hand-side value with which this object is
                 exclusive OR'ed.
        """
        return self.__call_single_arg_inplace_method__("__xor__", rhs)

    fn __or__(self, rhs: PythonObject) raises -> PythonObject:
        """Bitwise OR.

        Args:
            rhs: The right-hand-side value with which this object is bitwise
                 OR'ed.

        Returns:
            The bitwise OR result of this and the given value.
        """
        return self.__getattr__("__or__")(rhs)

    fn __ror__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse bitwise OR.

        Args:
            lhs: The left-hand-side value that is bitwise OR'ed with this
                 object.

        Returns:
            The bitwise OR result of the given value and this.
        """
        return self.__getattr__("__ror__")(lhs)

    fn __ior__(mut self, rhs: PythonObject) raises:
        """Immediate bitwise OR.

        Args:
            rhs: The right-hand-side value with which this object is bitwise
                 OR'ed.
        """
        return self.__call_single_arg_inplace_method__("__or__", rhs)

    fn __and__(self, rhs: PythonObject) raises -> PythonObject:
        """Bitwise AND.

        Args:
            rhs: The right-hand-side value with which this object is bitwise
                 AND'ed.

        Returns:
            The bitwise AND result of this and the given value.
        """
        return self.__getattr__("__and__")(rhs)

    fn __rand__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse bitwise and.

        Args:
            lhs: The left-hand-side value that is bitwise AND'ed with this
                 object.

        Returns:
            The bitwise AND result of the given value and this.
        """
        return self.__getattr__("__rand__")(lhs)

    fn __iand__(mut self, rhs: PythonObject) raises:
        """Immediate bitwise AND.

        Args:
            rhs: The right-hand-side value with which this object is bitwise
                 AND'ed.
        """
        return self.__call_single_arg_inplace_method__("__and__", rhs)

    fn __rshift__(self, rhs: PythonObject) raises -> PythonObject:
        """Bitwise right shift.

        Args:
            rhs: The right-hand-side value by which this object is bitwise
                 shifted to the right.

        Returns:
            This value, shifted right by the given value.
        """
        return self.__getattr__("__rshift__")(rhs)

    fn __rrshift__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse bitwise right shift.

        Args:
            lhs: The left-hand-side value that is bitwise shifted to the right
                 by this object.

        Returns:
            The given value, shifted right by this.
        """
        return self.__getattr__("__rrshift__")(lhs)

    fn __irshift__(mut self, rhs: PythonObject) raises:
        """Immediate bitwise right shift.

        Args:
            rhs: The right-hand-side value by which this object is bitwise
                 shifted to the right.
        """
        return self.__call_single_arg_inplace_method__("__rshift__", rhs)

    fn __lshift__(self, rhs: PythonObject) raises -> PythonObject:
        """Bitwise left shift.

        Args:
            rhs: The right-hand-side value by which this object is bitwise
                 shifted to the left.

        Returns:
            This value, shifted left by the given value.
        """
        return self.__getattr__("__lshift__")(rhs)

    fn __rlshift__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse bitwise left shift.

        Args:
            lhs: The left-hand-side value that is bitwise shifted to the left
                 by this object.

        Returns:
            The given value, shifted left by this.
        """
        return self.__getattr__("__rlshift__")(lhs)

    fn __ilshift__(mut self, rhs: PythonObject) raises:
        """Immediate bitwise left shift.

        Args:
            rhs: The right-hand-side value by which this object is bitwise
                 shifted to the left.
        """
        return self.__call_single_arg_inplace_method__("__lshift__", rhs)

    fn __pow__(self, exp: PythonObject) raises -> PythonObject:
        """Raises this object to the power of the given value.

        Args:
            exp: The exponent.

        Returns:
            The result of raising this by the given exponent.
        """
        return self.__getattr__("__pow__")(exp)

    fn __rpow__(self, lhs: PythonObject) raises -> PythonObject:
        """Reverse power of.

        Args:
            lhs: The number that is raised to the power of this object.

        Returns:
            The result of raising the given value by this exponent.
        """
        return self.__getattr__("__rpow__")(lhs)

    fn __ipow__(mut self, rhs: PythonObject) raises:
        """Immediate power of.

        Args:
            rhs: The exponent.
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
        """
        return self.__getattr__("__pos__")()

    fn __neg__(self) raises -> PythonObject:
        """Negative.

        Calls the underlying object's `__neg__` method.

        Returns:
            The result of prefixing this object with a `-` operator. For most
            numerical objects, this returns the negative.
        """
        return self.__getattr__("__neg__")()

    fn __invert__(self) raises -> PythonObject:
        """Inversion.

        Calls the underlying object's `__invert__` method.

        Returns:
            The logical inverse of this object: a bitwise representation where
            all bits are flipped, from zero to one, and from one to zero.
        """
        return self.__getattr__("__invert__")()

    fn __contains__(self, rhs: PythonObject) raises -> Bool:
        """Contains dunder.

        Calls the underlying object's `__contains__` method.

        Args:
            rhs: Right hand value.

        Returns:
            True if rhs is in self.
        """
        # TODO: replace/optimize with c-python function.
        # TODO: implement __getitem__ step for cpython membership test operator.
        ref cpython = Python().cpython()
        if cpython.PyObject_HasAttrString(self._obj_ptr, "__contains__"):
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

        var num_pos_args = len(args)
        var args_ = cpy.PyTuple_New(num_pos_args)
        for i in range(num_pos_args):
            var arg = args[i]._obj_ptr
            # increment the refcount for `PyTuple_SetItem` steals the reference
            # to `arg`
            cpy.Py_IncRef(arg)
            _ = cpy.PyTuple_SetItem(args_, i, arg)
        var kwargs_ = Python._dict(kwargs)
        var result = cpy.PyObject_Call(self._obj_ptr, args_, kwargs_)
        cpy.Py_DecRef(args_)
        cpy.Py_DecRef(kwargs_)
        if not result:
            raise cpy.get_error()
        return PythonObject(from_owned=result)

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    fn __len__(self) raises -> Int:
        """Returns the length of the object.

        Returns:
            The length of the object.
        """
        ref cpython = Python().cpython()
        var result = cpython.PyObject_Length(self._obj_ptr)
        if result == -1 and cpython.PyErr_Occurred():
            # Custom python types may return -1 even in non-error cases.
            raise cpython.unsafe_get_error()
        return result

    fn __hash__(self) raises -> Int:
        """Returns the hash value of the object.

        Returns:
            The hash value of the object.
        """
        ref cpython = Python().cpython()
        var result = cpython.PyObject_Hash(self._obj_ptr)
        if result == -1 and cpython.PyErr_Occurred():
            # Custom python types may return -1 even in non-error cases.
            raise cpython.unsafe_get_error()
        return result

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
        """
        return self^

    fn steal_data(var self) -> PyObjectPtr:
        """Take ownership of the underlying pointer from the Python object.

        Returns:
            The underlying data.
        """
        var ptr = self._obj_ptr
        self._obj_ptr = PyObjectPtr()

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
        """
        var tmp = Int(self)
        var result = _unsafe_aliasing_address_to_pointer[dtype](tmp)
        _ = tmp
        return result

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
        var opt: Optional[UnsafePointer[T]] = self._try_downcast_value[T]()

        if not opt:
            if func:
                raise Error(
                    String.format(
                        (
                            "TypeError: {}() expected Mojo '{}' type argument,"
                            " got '{}'"
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

        # SAFETY: We just validated that this Optional is not empty.
        return opt.unsafe_take()

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
        ref cpython = Python().cpython()

        var type = PyObjectPtr(upcast_from=cpython.Py_TYPE(self._obj_ptr))
        var expected_type = lookup_py_type_object[T]()._obj_ptr

        if type == expected_type:
            ref obj = self._obj_ptr.bitcast[PyMojoObject[T]]()[]
            if obj.is_initialized:
                return UnsafePointer(to=obj.mojo_value)
        return None

    fn unchecked_downcast_value_ptr[T: AnyType](self) -> UnsafePointer[T]:
        """Get a pointer to the expected Mojo value of type `T`.

        This function assumes that this Python object was allocated as an
        instance of `PyMojoObject[T]` and that the Mojo value has been
        initialized.

        Parameters:
            T: The type of the Mojo value stored in this object.

        Returns:
            A pointer to the inner Mojo value.

        # Safety

        The user must be certain that this Python object type matches the bound
        Python type object for `T`.
        """
        ref obj = self._obj_ptr.bitcast[PyMojoObject[T]]()[]
        # TODO(MSTDL-950): Should use something like `addr_of!`
        return UnsafePointer(to=obj.mojo_value)


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
    ref cpython = Python().cpython()
    var obj_py_ptr = cpython.PyType_GenericAlloc(type_obj_ptr, 0)
    if not obj_py_ptr:
        raise Error("Allocation of Python object failed.")
    return obj_py_ptr


fn _unsafe_init[
    T: Movable, //,
](obj_py_ptr: PyObjectPtr, var mojo_value: T) raises:
    """Initialize a Python object pointer with a Mojo value.

    Parameters:
        T: The Mojo type of the value that the resulting Python object holds.

    Args:
        obj_py_ptr: The Python object pointer to initialize.
            The pointer must have been allocated using the correct type object.
        mojo_value: The Mojo value to store in the Python object.

    # Safety
     `obj_py_ptr` must be a Python object pointer allocated using the correct
     type object. Use of any other pointer is invalid.
    """
    ref obj = obj_py_ptr.bitcast[PyMojoObject[T]]()[]
    UnsafePointer(to=obj.mojo_value).init_pointee_move(mojo_value^)
    obj.is_initialized = True


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

    Raises:
        If the Python object allocation fails.

    # Safety
    `type_obj_ptr` must be a Python type object created by `PythonTypeBuilder`,
    whose underlying storage type is the `PyMojoObject` struct. Use of any other
    type object is invalid.
    """
    var obj_py_ptr = _unsafe_alloc[T](type_obj_ptr)
    _unsafe_init(obj_py_ptr, mojo_value^)
    return PythonObject(from_owned=obj_py_ptr)


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
    ref cpython = Python().cpython()
    var py_start = cpython.Py_None()
    var py_stop = cpython.Py_None()
    var py_step = cpython.Py_None()

    if slice.start:
        py_start = cpython.PyLong_FromSsize_t(c_ssize_t(slice.start.value()))
    if slice.end:
        py_stop = cpython.PyLong_FromSsize_t(c_ssize_t(slice.end.value()))
    if slice.step:
        py_step = cpython.PyLong_FromSsize_t(c_ssize_t(slice.step.value()))

    var py_slice = cpython.PySlice_New(py_start, py_stop, py_step)

    if py_start != cpython.Py_None():
        cpython.Py_DecRef(py_start)
    if py_stop != cpython.Py_None():
        cpython.Py_DecRef(py_stop)
    cpython.Py_DecRef(py_step)

    return py_slice
