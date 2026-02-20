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
"""Implements PythonObject.

You can import these APIs from the `python` package. For example:

```mojo
from python import PythonObject
```
"""

from os import abort
from sys import bit_width_of
from ffi import c_double, c_long, c_size_t, c_ssize_t

from reflection import get_type_name

from ._cpython import CPython, GILAcquired, PyObject, PyObjectPtr, PyTypeObject
from .bindings import PyMojoObject, _get_type_name, lookup_py_type_object
from .python import Python
from .conversions import ConvertibleToPython


struct _PyIter(ImplicitlyCopyable, Iterable, Iterator):
    """A Python iterator."""

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime Element = PythonObject

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
    fn __next__(mut self) raises StopIteration -> PythonObject:
        """Return the next item and update to point to subsequent item.

        Returns:
            The next item in the traversable object that this iterator
            points to.
        """
        if not self.next_item:
            raise StopIteration()
        ref cpy = Python().cpython()
        var curr_item = self.next_item
        self.next_item = cpy.PyIter_Next(self.iterator._obj_ptr)
        return PythonObject(from_owned=curr_item)

    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self


struct PythonObject(
    Boolable,
    ConvertibleToPython,
    Defaultable,
    Identifiable,
    ImplicitlyCopyable,
    Movable,
    RegisterPassable,
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
    fn __init__[
        T: Movable & ImplicitlyDestructible
    ](out self, *, var alloc: T) raises:
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

        comptime if dtype == DType.bool:
            var val = c_long(Int(value))
            self = Self(from_owned=cpy.PyBool_FromLong(val))
        elif dtype.is_unsigned():
            var val = c_size_t(value.cast[DType.uint]())
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
        # TODO: This should not be necessary, as `StringSlice` is guaranteed to
        # be valid UTF-8.
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
        self = Self(StringSlice(value))

    @implicit
    fn __init__(out self, value: String) raises:
        """Initialize the object from a string.

        Args:
            value: The string value.

        Raises:
            If the string is not valid UTF-8.
        """
        self = Self(StringSlice(value))

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

        comptime for i in range(Variadic.size(Ts)):
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

    fn __copyinit__(out self, copy: Self):
        """Copy the object.

        This increments the underlying refcount of the existing object.

        Args:
            copy: The value to copy.
        """
        self = Self(from_borrowed=copy._obj_ptr)

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

    fn __setattr__[
        V: ConvertibleToPython, //
    ](self, var name: String, var value: V) raises:
        """Set the given value for the object attribute with the given name.

        Parameters:
            V: Attribute value that can be converted to a `PythonObject`.

        Args:
            name: The name of the object attribute to set.
            value: The new value to be set for that attribute.

        Raises:
            If setting the attribute fails.
        """
        var value_obj = value^.to_python_object()
        ref cpy = Python().cpython()
        var errno = cpy.PyObject_SetAttrString(
            self._obj_ptr, name^, value_obj._obj_ptr
        )
        _ = value_obj^
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
            abort("object cannot be converted to bool")

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

    # TODO(MOCO-2924): This should take a `*Ts: ConvertibleToPython` like other
    #   methods, however this currently runs into a spurious inference warning.
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

    fn __setitem__[
        *Ks: ConvertibleToPython & Copyable,
        V: ConvertibleToPython,
    ](self, *args: *Ks, var value: V) raises:
        """Set the value with the given key or keys.

        Parameters:
            Ks: Index types that can be converted to `PythonObject`s.
            V: Element value that can be converted to a `PythonObject`

        Args:
            args: The key or keys to set on this object.
            value: The value to set.

        Raises:
            If setting the item fails.
        """
        ref cpy = Python().cpython()
        comptime size = Variadic.size(Ks)
        var key_ptr: PyObjectPtr
        if size == 1:
            var single = args[0].copy().to_python_object()
            key_ptr = cpy.Py_NewRef(single._obj_ptr)
            _ = single^
        else:
            key_ptr = cpy.PyTuple_New(size)

            comptime for i in range(size):
                var arg = args[i].copy().to_python_object()
                _ = cpy.PyTuple_SetItem(key_ptr, i, cpy.Py_NewRef(arg._obj_ptr))
                _ = arg^

        var value_obj = value^.to_python_object()
        var errno = cpy.PyObject_SetItem(
            self._obj_ptr, key_ptr, value_obj._obj_ptr
        )
        _ = value_obj^
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

    fn __mul__[
        T: ConvertibleToPython, //
    ](self, var rhs: T) raises -> PythonObject:
        """Multiplication.

        Calls the underlying object's `__mul__` method.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            rhs: Right hand value.

        Returns:
            The product.

        Raises:
            If the operation is not supported.
        """
        var rhs_obj = rhs^.to_python_object()
        return self.__getattr__("__mul__")(rhs_obj)

    fn __rmul__[
        T: ConvertibleToPython, //
    ](self, var lhs: T) raises -> PythonObject:
        """Reverse multiplication.

        Calls the underlying object's `__rmul__` method.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            lhs: The left-hand-side value that is multiplied by this object.

        Returns:
            The product of the multiplication.

        Raises:
            If the operation is not supported.
        """
        var lhs_obj = lhs^.to_python_object()
        return self.__getattr__("__rmul__")(lhs_obj)

    fn __imul__[T: ConvertibleToPython, //](mut self, var rhs: T) raises:
        """In-place multiplication.

        Calls the underlying object's `__imul__` method.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            rhs: The right-hand-side value by which this object is multiplied.

        Raises:
            If the operation is not supported.
        """
        var rhs_obj = rhs^.to_python_object()
        return self.__call_single_arg_inplace_method__("__mul__", rhs_obj)

    fn __add__[
        T: ConvertibleToPython, //
    ](self, var rhs: T) raises -> PythonObject:
        """Addition and concatenation.

        Calls the underlying object's `__add__` method.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            rhs: Right hand value.

        Returns:
            The sum or concatenated values.

        Raises:
            If the operation is not supported.
        """
        var rhs_obj = rhs^.to_python_object()
        return self.__getattr__("__add__")(rhs_obj)

    fn __radd__[
        T: ConvertibleToPython, //
    ](self, var lhs: T) raises -> PythonObject:
        """Reverse addition and concatenation.

        Calls the underlying object's `__radd__` method.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            lhs: The left-hand-side value to which this object is added or
                 concatenated.

        Returns:
            The sum.

        Raises:
            If the operation is not supported.
        """
        var lhs_obj = lhs^.to_python_object()
        return self.__getattr__("__radd__")(lhs_obj)

    # Note: `T = PythonObject` default helps `foo += [1, 2]` succeed, by helping
    #   the compiler see that `[1, 2]` can instantiate a PythonObject.
    fn __iadd__[
        T: ConvertibleToPython = PythonObject
    ](mut self, var rhs: T) raises:
        """Immediate addition and concatenation.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            rhs: The right-hand-side value that is added to this object.

        Raises:
            If the operation is not supported.
        """
        var rhs_obj = rhs^.to_python_object()
        return self.__call_single_arg_inplace_method__("__add__", rhs_obj)

    fn __sub__[
        T: ConvertibleToPython, //
    ](self, var rhs: T) raises -> PythonObject:
        """Subtraction.

        Calls the underlying object's `__sub__` method.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            rhs: Right hand value.

        Returns:
            The difference.

        Raises:
            If the operation is not supported.
        """
        var rhs_obj = rhs^.to_python_object()
        return self.__getattr__("__sub__")(rhs_obj)

    fn __rsub__[
        T: ConvertibleToPython, //
    ](self, var lhs: T) raises -> PythonObject:
        """Reverse subtraction.

        Calls the underlying object's `__rsub__` method.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            lhs: The left-hand-side value from which this object is subtracted.

        Returns:
            The result of subtracting this from the given value.

        Raises:
            If the operation is not supported.
        """
        var lhs_obj = lhs^.to_python_object()
        return self.__getattr__("__rsub__")(lhs_obj)

    fn __isub__[T: ConvertibleToPython, //](mut self, var rhs: T) raises:
        """Immediate subtraction.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            rhs: The right-hand-side value that is subtracted from this object.

        Raises:
            If the operation is not supported.
        """
        var rhs_obj = rhs^.to_python_object()
        return self.__call_single_arg_inplace_method__("__sub__", rhs_obj)

    fn __floordiv__[
        T: ConvertibleToPython, //
    ](self, var rhs: T) raises -> PythonObject:
        """Return the division of self and rhs rounded down to the nearest
        integer.

        Calls the underlying object's `__floordiv__` method.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            rhs: The right-hand-side value by which this object is divided.

        Returns:
            The result of dividing this by the right-hand-side value, modulo any
            remainder.

        Raises:
            If the operation fails.
        """
        var rhs_obj = rhs^.to_python_object()
        return self.__getattr__("__floordiv__")(rhs_obj)

    fn __rfloordiv__[
        T: ConvertibleToPython, //
    ](self, var lhs: T) raises -> PythonObject:
        """Reverse floor division.

        Calls the underlying object's `__rfloordiv__` method.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            lhs: The left-hand-side value that is divided by this object.

        Returns:
            The result of dividing the given value by this, modulo any
            remainder.

        Raises:
            If the operation fails.
        """
        var lhs_obj = lhs^.to_python_object()
        return self.__getattr__("__rfloordiv__")(lhs_obj)

    fn __ifloordiv__[T: ConvertibleToPython, //](mut self, var rhs: T) raises:
        """Immediate floor division.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            rhs: The value by which this object is divided.

        Raises:
            If the operation fails.
        """
        var rhs_obj = rhs^.to_python_object()
        return self.__call_single_arg_inplace_method__("__floordiv__", rhs_obj)

    fn __truediv__[
        T: ConvertibleToPython, //
    ](self, var rhs: T) raises -> PythonObject:
        """Division.

        Calls the underlying object's `__truediv__` method.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            rhs: The right-hand-side value by which this object is divided.

        Returns:
            The result of dividing the right-hand-side value by this.

        Raises:
            If the operation fails.
        """
        var rhs_obj = rhs^.to_python_object()
        return self.__getattr__("__truediv__")(rhs_obj)

    fn __rtruediv__[
        T: ConvertibleToPython, //
    ](self, var lhs: T) raises -> PythonObject:
        """Reverse division.

        Calls the underlying object's `__rtruediv__` method.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            lhs: The left-hand-side value that is divided by this object.

        Returns:
            The result of dividing the given value by this.

        Raises:
            If the operation fails.
        """
        var lhs_obj = lhs^.to_python_object()
        return self.__getattr__("__rtruediv__")(lhs_obj)

    fn __itruediv__[T: ConvertibleToPython, //](mut self, var rhs: T) raises:
        """Immediate division.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            rhs: The value by which this object is divided.

        Raises:
            If the operation fails.
        """
        var rhs_obj = rhs^.to_python_object()
        return self.__call_single_arg_inplace_method__("__truediv__", rhs_obj)

    fn __mod__[
        T: ConvertibleToPython, //
    ](self, var rhs: T) raises -> PythonObject:
        """Return the remainder of self divided by rhs.

        Calls the underlying object's `__mod__` method.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            rhs: The value to divide on.

        Returns:
            The remainder of dividing self by rhs.

        Raises:
            If the operation fails.
        """
        var rhs_obj = rhs^.to_python_object()
        return self.__getattr__("__mod__")(rhs_obj)

    fn __rmod__[
        T: ConvertibleToPython, //
    ](self, var lhs: T) raises -> PythonObject:
        """Reverse modulo.

        Calls the underlying object's `__rmod__` method.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            lhs: The left-hand-side value that is divided by this object.

        Returns:
            The remainder from dividing the given value by this.

        Raises:
            If the operation fails.
        """
        var lhs_obj = lhs^.to_python_object()
        return self.__getattr__("__rmod__")(lhs_obj)

    fn __imod__[T: ConvertibleToPython, //](mut self, var rhs: T) raises:
        """Immediate modulo.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            rhs: The right-hand-side value that is used to divide this object.

        Raises:
            If the operation fails.
        """
        var rhs_obj = rhs^.to_python_object()
        return self.__call_single_arg_inplace_method__("__mod__", rhs_obj)

    fn __xor__[
        T: ConvertibleToPython, //
    ](self, var rhs: T) raises -> PythonObject:
        """Exclusive OR.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            rhs: The right-hand-side value with which this object is exclusive
                 OR'ed.

        Returns:
            The exclusive OR result of this and the given value.

        Raises:
            If the operation fails.
        """
        var rhs_obj = rhs^.to_python_object()
        return self.__getattr__("__xor__")(rhs_obj)

    fn __rxor__[
        T: ConvertibleToPython, //
    ](self, var lhs: T) raises -> PythonObject:
        """Reverse exclusive OR.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            lhs: The left-hand-side value that is exclusive OR'ed with this
                 object.

        Returns:
            The exclusive OR result of the given value and this.

        Raises:
            If the operation fails.
        """
        var lhs_obj = lhs^.to_python_object()
        return self.__getattr__("__rxor__")(lhs_obj)

    fn __ixor__[T: ConvertibleToPython, //](mut self, var rhs: T) raises:
        """Immediate exclusive OR.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            rhs: The right-hand-side value with which this object is
                 exclusive OR'ed.

        Raises:
            If the operation fails.
        """
        var rhs_obj = rhs^.to_python_object()
        return self.__call_single_arg_inplace_method__("__xor__", rhs_obj)

    fn __or__[
        T: ConvertibleToPython, //
    ](self, var rhs: T) raises -> PythonObject:
        """Bitwise OR.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            rhs: The right-hand-side value with which this object is bitwise
                 OR'ed.

        Returns:
            The bitwise OR result of this and the given value.

        Raises:
            If the operation fails.
        """
        var rhs_obj = rhs^.to_python_object()
        return self.__getattr__("__or__")(rhs_obj)

    fn __ror__[
        T: ConvertibleToPython, //
    ](self, var lhs: T) raises -> PythonObject:
        """Reverse bitwise OR.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            lhs: The left-hand-side value that is bitwise OR'ed with this
                 object.

        Returns:
            The bitwise OR result of the given value and this.

        Raises:
            If the operation fails.
        """
        var lhs_obj = lhs^.to_python_object()
        return self.__getattr__("__ror__")(lhs_obj)

    fn __ior__[T: ConvertibleToPython, //](mut self, var rhs: T) raises:
        """Immediate bitwise OR.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            rhs: The right-hand-side value with which this object is bitwise
                 OR'ed.

        Raises:
            If the operation fails.
        """
        var rhs_obj = rhs^.to_python_object()
        return self.__call_single_arg_inplace_method__("__or__", rhs_obj)

    fn __and__[
        T: ConvertibleToPython, //
    ](self, var rhs: T) raises -> PythonObject:
        """Bitwise AND.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            rhs: The right-hand-side value with which this object is bitwise
                 AND'ed.

        Returns:
            The bitwise AND result of this and the given value.

        Raises:
            If the operation fails.
        """
        var rhs_obj = rhs^.to_python_object()
        return self.__getattr__("__and__")(rhs_obj)

    fn __rand__[
        T: ConvertibleToPython, //
    ](self, var lhs: T) raises -> PythonObject:
        """Reverse bitwise and.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            lhs: The left-hand-side value that is bitwise AND'ed with this
                 object.

        Returns:
            The bitwise AND result of the given value and this.

        Raises:
            If the operation fails.
        """
        var lhs_obj = lhs^.to_python_object()
        return self.__getattr__("__rand__")(lhs_obj)

    fn __iand__[T: ConvertibleToPython, //](mut self, var rhs: T) raises:
        """Immediate bitwise AND.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            rhs: The right-hand-side value with which this object is bitwise
                 AND'ed.

        Raises:
            If the operation fails.
        """
        var rhs_obj = rhs^.to_python_object()
        return self.__call_single_arg_inplace_method__("__and__", rhs_obj)

    fn __rshift__[
        T: ConvertibleToPython, //
    ](self, var rhs: T) raises -> PythonObject:
        """Bitwise right shift.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            rhs: The right-hand-side value by which this object is bitwise
                 shifted to the right.

        Returns:
            This value, shifted right by the given value.

        Raises:
            If the operation fails.
        """
        var rhs_obj = rhs^.to_python_object()
        return self.__getattr__("__rshift__")(rhs_obj)

    fn __rrshift__[
        T: ConvertibleToPython, //
    ](self, var lhs: T) raises -> PythonObject:
        """Reverse bitwise right shift.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            lhs: The left-hand-side value that is bitwise shifted to the right
                 by this object.

        Returns:
            The given value, shifted right by this.

        Raises:
            If the operation fails.
        """
        var lhs_obj = lhs^.to_python_object()
        return self.__getattr__("__rrshift__")(lhs_obj)

    fn __irshift__[T: ConvertibleToPython, //](mut self, var rhs: T) raises:
        """Immediate bitwise right shift.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            rhs: The right-hand-side value by which this object is bitwise
                 shifted to the right.

        Raises:
            If the operation fails.
        """
        var rhs_obj = rhs^.to_python_object()
        return self.__call_single_arg_inplace_method__("__rshift__", rhs_obj)

    fn __lshift__[
        T: ConvertibleToPython, //
    ](self, var rhs: T) raises -> PythonObject:
        """Bitwise left shift.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            rhs: The right-hand-side value by which this object is bitwise
                 shifted to the left.

        Returns:
            This value, shifted left by the given value.

        Raises:
            If the operation fails.
        """
        var rhs_obj = rhs^.to_python_object()
        return self.__getattr__("__lshift__")(rhs_obj)

    fn __rlshift__[
        T: ConvertibleToPython, //
    ](self, var lhs: T) raises -> PythonObject:
        """Reverse bitwise left shift.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            lhs: The left-hand-side value that is bitwise shifted to the left
                 by this object.

        Returns:
            The given value, shifted left by this.

        Raises:
            If the operation fails.
        """
        var lhs_obj = lhs^.to_python_object()
        return self.__getattr__("__rlshift__")(lhs_obj)

    fn __ilshift__[T: ConvertibleToPython, //](mut self, var rhs: T) raises:
        """Immediate bitwise left shift.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            rhs: The right-hand-side value by which this object is bitwise
                 shifted to the left.

        Raises:
            If the operation fails.
        """
        var rhs_obj = rhs^.to_python_object()
        return self.__call_single_arg_inplace_method__("__lshift__", rhs_obj)

    fn __pow__[
        T: ConvertibleToPython, //
    ](self, var exp: T) raises -> PythonObject:
        """Raises this object to the power of the given value.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            exp: The exponent.

        Returns:
            The result of raising this by the given exponent.

        Raises:
            If the operation fails.
        """
        var exp_obj = exp^.to_python_object()
        return self.__getattr__("__pow__")(exp_obj)

    fn __rpow__[
        T: ConvertibleToPython, //
    ](self, var lhs: T) raises -> PythonObject:
        """Reverse power of.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            lhs: The number that is raised to the power of this object.

        Returns:
            The result of raising the given value by this exponent.

        Raises:
            If the operation fails.
        """
        var lhs_obj = lhs^.to_python_object()
        return self.__getattr__("__rpow__")(lhs_obj)

    fn __ipow__[T: ConvertibleToPython, //](mut self, var rhs: T) raises:
        """Immediate power of.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            rhs: The exponent.

        Raises:
            If the operation fails.
        """
        var rhs_obj = rhs^.to_python_object()
        return self.__call_single_arg_inplace_method__("__pow__", rhs_obj)

    fn __lt__[
        T: ConvertibleToPython, //
    ](self, var rhs: T) raises -> PythonObject:
        """Less than (rich) comparison operator.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            rhs: The value of the right hand side of the comparison.

        Returns:
            The result of the comparison, not necessarily a boolean.

        Raises:
            If the object doesn't implement the `__lt__` method, or if it fails.
        """
        var rhs_obj = rhs^.to_python_object()
        return self.__getattr__("__lt__")(rhs_obj)

    fn __le__[
        T: ConvertibleToPython, //
    ](self, var rhs: T) raises -> PythonObject:
        """Less than or equal (rich) comparison operator.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            rhs: The value of the right hand side of the comparison.

        Returns:
            The result of the comparison, not necessarily a boolean.

        Raises:
            If the object doesn't implement the `__le__` method, or if it fails.
        """
        var rhs_obj = rhs^.to_python_object()
        return self.__getattr__("__le__")(rhs_obj)

    fn __gt__[
        T: ConvertibleToPython, //
    ](self, var rhs: T) raises -> PythonObject:
        """Greater than (rich) comparison operator.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            rhs: The value of the right hand side of the comparison.

        Returns:
            The result of the comparison, not necessarily a boolean.

        Raises:
            If the object doesn't implement the `__gt__` method, or if it fails.
        """
        var rhs_obj = rhs^.to_python_object()
        return self.__getattr__("__gt__")(rhs_obj)

    fn __ge__[
        T: ConvertibleToPython, //
    ](self, var rhs: T) raises -> PythonObject:
        """Greater than or equal (rich) comparison operator.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            rhs: The value of the right hand side of the comparison.

        Returns:
            The result of the comparison, not necessarily a boolean.

        Raises:
            If the object doesn't implement the `__ge__` method, or if it fails.
        """
        var rhs_obj = rhs^.to_python_object()
        return self.__getattr__("__ge__")(rhs_obj)

    fn __eq__[
        T: ConvertibleToPython, //
    ](self, var rhs: T) raises -> PythonObject:
        """Equality (rich) comparison operator.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            rhs: The value of the right hand side of the comparison.

        Returns:
            The result of the comparison, not necessarily a boolean.

        Raises:
            If the object doesn't implement the `__eq__` method, or if it fails.
        """
        var rhs_obj = rhs^.to_python_object()
        return self.__getattr__("__eq__")(rhs_obj)

    fn __ne__[
        T: ConvertibleToPython, //
    ](self, var rhs: T) raises -> PythonObject:
        """Inequality (rich) comparison operator.

        Parameters:
            T: Argument type that can be converted to `PythonObject`.

        Args:
            rhs: The value of the right hand side of the comparison.

        Returns:
            The result of the comparison, not necessarily a boolean.

        Raises:
            If the object doesn't implement the `__ne__` method, or if it fails.
        """
        var rhs_obj = rhs^.to_python_object()
        return self.__getattr__("__ne__")(rhs_obj)

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

    fn __contains__[
        RHS: ConvertibleToPython, //
    ](self, var rhs: RHS) raises -> Bool:
        """Contains dunder.

        Calls the underlying object's `__contains__` method.

        Parameters:
            RHS: Type of value that can be converted to `PythonObject` to check
              for membership.

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
        var rhs_obj = rhs^.to_python_object()
        if cpy.PyObject_HasAttrString(self._obj_ptr, "__contains__"):
            return self.__getattr__("__contains__")(rhs_obj).__bool__()
        for v in self:
            if v == rhs_obj:
                return True
        return False

    # see https://github.com/python/cpython/blob/main/Objects/call.c
    # for decrement rules
    fn __call__[
        *Ts: ConvertibleToPython & Copyable,
    ](self, *args: *Ts, **kwargs: PythonObject) raises -> PythonObject:
        """Call the underlying object as if it were a function.

        Parameters:
            Ts: Types of the positional arguments.

        Args:
            args: Positional arguments to the function.
            kwargs: Keyword arguments to the function.

        Raises:
            If the function cannot be called for any reason.

        Returns:
            The return value from the called object.
        """
        comptime size = Variadic.size(Ts)

        ref cpy = Python().cpython()
        var args_ptr = cpy.PyTuple_New(size)

        comptime for i in range(size):
            var arg = args[i].copy().to_python_object()

            _ = cpy.PyTuple_SetItem(args_ptr, i, cpy.Py_NewRef(arg._obj_ptr))

            _ = arg^
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
            writer.write(String(py=self))
        except e:
            # TODO: make this method raising when we can raise parametrically.
            abort(String("failed to write PythonObject to writer: ", e))

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
    ](self) raises -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
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
        return UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=Int(py=self)
        )

    fn downcast_value_ptr[
        T: ImplicitlyDestructible
    ](self, *, func: Optional[StaticString] = None) raises -> UnsafePointer[
        T, MutAnyOrigin
    ]:
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
        T: ImplicitlyDestructible
    ](var self) raises -> Optional[UnsafePointer[T, MutAnyOrigin]]:
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
                return UnsafePointer(to=mojo_obj.mojo_value).as_any_origin()
        return None

    fn unchecked_downcast_value_ptr[
        mut: Bool, origin: Origin[mut=mut], //, T: ImplicitlyDestructible
    ](ref[origin] self) -> UnsafePointer[T, origin]:
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
        return UnsafePointer[mut=mut](
            to=mojo_obj.mojo_value
        ).unsafe_origin_cast[origin]()


# ===-----------------------------------------------------------------------===#
# Factory functions for PythonObject
# ===-----------------------------------------------------------------------===#


fn _unsafe_alloc[
    T: AnyType
](
    type_obj_ptr: UnsafePointer[PyTypeObject, MutAnyOrigin]
) raises -> PyObjectPtr:
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
    T: Movable & ImplicitlyDestructible,
    //,
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
    T: Movable & ImplicitlyDestructible,
    //,
](
    type_obj_ptr: UnsafePointer[PyTypeObject, MutAnyOrigin], var mojo_value: T
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
