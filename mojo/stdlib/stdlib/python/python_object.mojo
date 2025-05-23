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

from hashlib._hasher import _HashableWithHasher, _Hasher
from os import abort
from sys.ffi import c_ssize_t
from sys.intrinsics import _unsafe_aliasing_address_to_pointer

# This apparently redundant import is needed so PythonBindingsGen.cpp can find
# the StringLiteral declaration.
from builtin.string_literal import StringLiteral
from builtin.identifiable import TypeIdentifiable
from memory import UnsafePointer

from ._cpython import CPython, PyObjectPtr, PyObject, PyTypeObject
from .python import Python
from .bindings import _get_type_name, lookup_py_type_object, PyMojoObject


trait PythonConvertible:
    """A trait that indicates a type can be converted to a PythonObject, and
    that specifies the behavior with a `to_python_object` method."""

    fn to_python_object(self) -> PythonObject:
        """Convert a value to a PythonObject.

        Returns:
            A PythonObject representing the value.
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


struct _PyIter(Sized):
    """A Python iterator."""

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var iterator: PythonObject
    """The iterator object that stores location."""
    var prepared_next_item: PythonObject
    """The next item to vend or zero if there are no items."""
    var is_done: Bool
    """Stores True if the iterator is pointing to the last item."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __copyinit__(out self, existing: Self):
        """Copy another iterator.

        Args:
            existing: Initialized _PyIter instance.
        """
        self.iterator = existing.iterator
        self.prepared_next_item = existing.prepared_next_item
        self.is_done = existing.is_done

    @implicit
    fn __init__(out self, iter: PythonObject):
        """Initialize an iterator.

        Args:
            iter: A Python iterator instance.
        """
        var cpython = Python().cpython()
        self.iterator = iter
        var maybe_next_item = cpython.PyIter_Next(self.iterator.py_object)
        if maybe_next_item.is_null():
            self.is_done = True
            self.prepared_next_item = PythonObject(from_owned_ptr=PyObjectPtr())
        else:
            self.prepared_next_item = PythonObject(
                from_owned_ptr=maybe_next_item
            )
            self.is_done = False

    fn __init__(out self):
        """Initialize an empty iterator."""
        self.iterator = PythonObject(from_owned_ptr=PyObjectPtr())
        self.is_done = True
        self.prepared_next_item = PythonObject(from_owned_ptr=PyObjectPtr())

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    fn __next__(mut self: _PyIter) -> PythonObject:
        """Return the next item and update to point to subsequent item.

        Returns:
            The next item in the traversable object that this iterator
            points to.
        """
        if not self.iterator:
            return self.iterator
        var cpython = Python().cpython()
        var current = self.prepared_next_item
        var maybe_next_item = cpython.PyIter_Next(self.iterator.py_object)
        if maybe_next_item.is_null():
            self.is_done = True
        else:
            self.prepared_next_item = PythonObject(
                from_owned_ptr=maybe_next_item
            )
        return current

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    fn __len__(self) -> Int:
        """Return zero to halt iteration.

        Returns:
            0 if the traversal is complete and 1 otherwise.
        """
        if self.is_done:
            return 0
        else:
            return 1


alias PythonModule = TypedPythonObject["Module"]
alias PyFunction = fn (
    mut PythonObject, mut TypedPythonObject["Tuple"]
) -> PythonObject
alias PyFunctionRaising = fn (
    mut PythonObject, mut TypedPythonObject["Tuple"]
) raises -> PythonObject


@register_passable
struct TypedPythonObject[type_hint: StaticString](
    Movable,
    Copyable,
    PythonConvertible,
    SizedRaising,
):
    """A wrapper around `PythonObject` that indicates the type of the contained
    object.

    The PythonObject structure is entirely dynamically typed. This type provides
    a weak layer of optional static typing.

    Parameters:
        type_hint: The type name hint indicating the static type of this
            object.
    """

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var _obj: PythonObject

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(out self, *, owned unsafe_unchecked_from: PythonObject):
        """Construct a TypedPythonObject without any validation that the given
        object is of the specified hinted type.

        Args:
            unsafe_unchecked_from: The PythonObject to construct from. This
                will not be type checked.
        """
        self._obj = unsafe_unchecked_from^

    fn __init__(out self: PythonModule, name: StaticString) raises:
        """Construct a Python module with the given name.

        Args:
            name: The name of the module.

        Raises:
            If the module creation fails.
        """
        self = Python.create_module(name)

    fn __copyinit__(out self, other: Self):
        """Copy an instance of this type.

        Args:
            other: The value to copy.
        """
        self._obj = other._obj

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    fn __len__(self) raises -> Int:
        """Returns the length of the object.

        Returns:
            The length of the object.
        """
        return len(self._obj)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn to_python_object(self) -> PythonObject:
        """Convert the TypedPythonObject to a PythonObject.

        Returns:
            A PythonObject representing the value.
        """
        return self._obj

    # TODO:
    #   This should have origin, or we should do this with a context
    #   manager, to prevent use after ASAP destruction.
    fn unsafe_as_py_object_ptr(self) -> PyObjectPtr:
        """Get the underlying PyObject pointer.

        Returns:
            The underlying PyObject pointer.

        Safety:
            Use-after-free: The caller must take care that `self` outlives the
            usage of the pointer returned by this function.
        """
        return self._obj.unsafe_as_py_object_ptr()

    # ===-------------------------------------------------------------------===#
    # 'Tuple' Operations
    # ===-------------------------------------------------------------------===#

    fn __getitem__[
        I: Indexer
    ](self: TypedPythonObject["Tuple"], pos: I,) raises -> PythonObject:
        """Get an element from this tuple.

        Args:
            pos: The tuple element position to retrieve.

        Parameters:
            I: A type that can be used as an index.

        Returns:
            The value of the tuple element at the specified position.
        """
        var cpython = Python().cpython()

        var item: PyObjectPtr = cpython.PyTuple_GetItem(
            self.unsafe_as_py_object_ptr(),
            index(pos),
        )

        if item.is_null():
            raise cpython.get_error()

        # TODO(MSTDL-911): Avoid unnecessary owned reference counts when
        #   returning read-only PythonObject values.
        return PythonObject(from_borrowed_ptr=item)


@register_passable
struct PythonObject(
    Boolable,
    Copyable,
    Movable,
    SizedRaising,
    Writable,
    PythonConvertible,
):
    """A Python object."""

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var py_object: PyObjectPtr
    """A pointer to the underlying Python object."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(out self):
        """Initialize the object with a `None` value."""
        self = Self(None)

    fn copy(self) -> Self:
        """Copy the object.

        Returns:
            A copy of the value.
        """
        return self

    fn __init__(out self, *, from_owned_ptr: PyObjectPtr):
        """Initialize this object from an owned reference-counted Python object
        pointer.

        Ownership of the reference will be assumed by `PythonObject`.

        Args:
            from_owned_ptr: The `PyObjectPtr` to take ownership of.
        """
        self.py_object = from_owned_ptr

    fn __init__(out self, *, from_borrowed_ptr: PyObjectPtr):
        """Initialize this object from a read-only reference-counted Python
        object pointer.

        The reference count of the pointee object will be incremented, and
        ownership of the additional reference count will be assumed by the
        initialized `PythonObject`.

        The CPython API documentation indicates the ownership semantics of the
        returned object on any function that returns a `PyObject*` value. The
        two possible annotations are:

        * "Return value: New reference."
        * "Return value: Borrowed reference.

        This function should be used to construct a `PythonObject` from the
        pointer returned by 'Borrowed reference'-type objects.

        Args:
            from_borrowed_ptr: A read-only reference counted pointer to a Python
                object.

        Returns:
            An owned PythonObject pointer.
        """
        var cpython = Python().cpython()

        # SAFETY:
        #   We were passed a Python 'read-only reference', so for it to be
        #   safe to store this reference, we must increment the reference
        #   count to convert this to a 'strong reference'.
        cpython.Py_IncRef(from_borrowed_ptr)

        self = PythonObject(from_owned_ptr=from_borrowed_ptr)

    @always_inline
    fn __init__[
        T: Movable & TypeIdentifiable
    ](out self, *, owned alloc: T) raises:
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

        var type_obj_ptr = type_obj.unsafe_as_py_object_ptr().unsized_obj_ptr.bitcast[
            PyTypeObject
        ]()

        return PythonObject._unsafe_alloc(type_obj_ptr, alloc^)

    @implicit
    fn __init__(out self, owned typed_obj: TypedPythonObject[_]):
        """Construct a PythonObject from a typed object, dropping the type hint
        information.

        This is a no-op at runtime. The only information that is lost is static
        type information.

        Args:
            typed_obj: The typed python object to unwrap.
        """
        self = typed_obj._obj^

        # Mark destroyed so we can transfer out its field.
        __disable_del typed_obj

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
        cpython = Python().cpython()
        self.py_object = cpython.Py_None()
        cpython.Py_IncRef(self.py_object)

    @implicit
    fn __init__(out self, value: Bool):
        """Initialize the object from a bool.

        Args:
            value: The boolean value.
        """
        cpython = Python().cpython()
        self.py_object = cpython.PyBool_FromLong(Int(value))

    @implicit
    fn __init__(out self, integer: Int):
        """Initialize the object with an integer value.

        Args:
            integer: The integer value.
        """
        cpython = Python().cpython()
        self.py_object = cpython.PyLong_FromSsize_t(integer)

    @implicit
    fn __init__[dtype: DType](out self, value: SIMD[dtype, 1]):
        """Initialize the object with a generic scalar value. If the scalar
        value type is bool, it is converted to a boolean. Otherwise, it is
        converted to the appropriate integer or floating point type.

        Parameters:
            dtype: The scalar value type.

        Args:
            value: The scalar value.
        """
        var cpython = Python().cpython()

        @parameter
        if dtype is DType.bool:
            self.py_object = cpython.PyBool_FromLong(Int(value))
        elif dtype.is_unsigned():
            var uint_val = value.cast[DType.index]().value
            self.py_object = cpython.PyLong_FromSize_t(uint_val)
        elif dtype.is_integral():
            var int_val = value.cast[DType.index]().value
            self.py_object = cpython.PyLong_FromSsize_t(int_val)
        else:
            var fp_val = value.cast[DType.float64]()
            self.py_object = cpython.PyFloat_FromDouble(fp_val)

    @implicit
    fn __init__(out self, value: StringLiteral):
        """Initialize the object from a string literal.

        Args:
            value: The string value.
        """
        self = PythonObject(value.as_string_slice())

    @implicit
    fn __init__(out self, value: String):
        """Initialize the object from a string.

        Args:
            value: The string value.
        """
        self = PythonObject(value.as_string_slice())

    @implicit
    fn __init__(out self, string: StringSlice):
        """Initialize the object from a string.

        Args:
            string: The string value.
        """
        cpython = Python().cpython()
        self.py_object = cpython.PyUnicode_DecodeUTF8(string)

    @implicit
    fn __init__(out self, slice: Slice):
        """Initialize the object from a Mojo Slice.

        Args:
            slice: The dictionary value.
        """
        self.py_object = _slice_to_py_object_ptr(slice)

    @always_inline
    fn __init__[
        *Ts: PythonConvertible
    ](out self, owned *values: *Ts, __list_literal__: ()):
        """Construct an Python list of objects.

        Parameters:
            Ts: The types of the input values.

        Args:
            values: The values to initialize the list with.
            __list_literal__: Tell Mojo to use this method for list literals.

        Returns:
            The constructed Python list.
        """
        return Python._list(values)

    @always_inline
    fn __init__[
        *Ts: PythonConvertible
    ](out self, owned *values: *Ts, __set_literal__: ()) raises:
        """Construct an Python set of objects.

        Parameters:
            Ts: The types of the input values.

        Args:
            values: The values to initialize the set with.
            __set_literal__: Tell Mojo to use this method for set literals.

        Returns:
            The constructed Python set.
        """
        var cpython = Python().cpython()
        var obj_ptr = cpython.PySet_New()
        if obj_ptr.is_null():
            raise cpython.get_error()

        @parameter
        for i in range(len(VariadicList(Ts))):
            var obj = values[i].to_python_object()
            cpython.Py_IncRef(obj.py_object)
            var result = cpython.PySet_Add(obj_ptr, obj.py_object)
            if result == -1:
                raise cpython.get_error()

        return PythonObject(from_owned_ptr=obj_ptr)

    fn __init__(
        out self,
        owned keys: List[PythonObject],
        owned values: List[PythonObject],
        __dict_literal__: (),
    ) raises:
        """Construct a Python dictionary from a list of keys and a list of values.

        Args:
            keys: The keys of the dictionary.
            values: The values of the dictionary.
            __dict_literal__: Tell Mojo to use this method for dict literals.
        """
        var cpython = Python().cpython()
        var dict_obj_ptr = cpython.PyDict_New()
        if dict_obj_ptr.is_null():
            raise Error("internal error: PyDict_New failed")

        for i in range(len(keys)):
            var key_obj = keys[i].to_python_object()
            var val_obj = values[i].to_python_object()
            var result = cpython.PyDict_SetItem(
                dict_obj_ptr, key_obj.py_object, val_obj.py_object
            )
            if result != 0:
                raise Error("internal error: PyDict_SetItem failed")

        return PythonObject(from_owned_ptr=dict_obj_ptr)

    fn __copyinit__(out self, existing: Self):
        """Copy the object.

        This increments the underlying refcount of the existing object.

        Args:
            existing: The value to copy.
        """
        self.py_object = existing.py_object
        var cpython = Python().cpython()
        cpython.Py_IncRef(self.py_object)

    fn __del__(owned self):
        """Destroy the object.

        This decrements the underlying refcount of the pointed-to object.
        """
        var cpython = Python().cpython()
        # Acquire GIL such that __del__ can be called safely for cases where the
        # PyObject is handled in non-python contexts.
        var state = cpython.PyGILState_Ensure()
        if not self.py_object.is_null():
            cpython.Py_DecRef(self.py_object)
        self.py_object = PyObjectPtr()
        cpython.PyGILState_Release(state)

    # ===-------------------------------------------------------------------===#
    # Factory methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    fn _unsafe_alloc[
        T: Movable
    ](
        type_obj_ptr: UnsafePointer[PyTypeObject],
        owned mojo_value: T,
    ) raises -> PythonObject:
        """Allocate a new `PythonObject` and store a Mojo value in it.

        Parameters:
            T: The Mojo type of the value that the resulting Python object
              holds.

        Args:
            type_obj_ptr: Must be the Python type object describing
              `PyTypeObject[T]`.
            mojo_value: The Mojo value to store in the new Python object.

        # Safety

        `type_obj_ptr` must be a Python type object created by
        `PythonTypeBuilder`, whose underying storage type is the `PyMojoObject`
        struct. Use of any other type object is invalid.
        """

        var cpython = Python().cpython()

        # Allocates and zero-initializes the new `PyMojoObject[T]`.
        # (For some objects, zeroed values are valid. But that isn't guaranteed
        # for any given Mojo object.)
        var obj_py_ptr: PyObjectPtr = cpython.PyType_GenericAlloc(
            type_obj_ptr,
            0,
        )

        # If we failed to allocate, raise.
        if not obj_py_ptr.unsized_obj_ptr:
            # Intentionally try to avoid allocating to raise this error.
            raise Error("Allocation of Python object failed.")

        var obj_ptr = obj_py_ptr.unsized_obj_ptr.bitcast[PyMojoObject[T]]()
        var obj_value_ptr = UnsafePointer[T](to=obj_ptr[].mojo_value)

        # Initialize the PyMojoObject[T] with the user-provided Mojo value.
        __get_address_as_uninit_lvalue(obj_value_ptr.address) = mojo_value^

        return PythonObject(from_owned_ptr=obj_py_ptr)

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
        var cpython = Python().cpython()
        var iter_ptr = cpython.PyObject_GetIter(self.py_object)
        if iter_ptr.is_null():
            raise cpython.get_error()
        return _PyIter(PythonObject(from_owned_ptr=iter_ptr))

    fn __getattr__(self, owned name: String) raises -> PythonObject:
        """Return the value of the object attribute with the given name.

        Args:
            name: The name of the object attribute to return.

        Returns:
            The value of the object attribute with the given name.
        """
        var cpython = Python().cpython()
        var result = cpython.PyObject_GetAttrString(self.py_object, name^)
        if result.is_null():
            raise cpython.get_error()
        return PythonObject(from_owned_ptr=result)

    fn __setattr__(self, owned name: String, new_value: PythonObject) raises:
        """Set the given value for the object attribute with the given name.

        Args:
            name: The name of the object attribute to set.
            new_value: The new value to be set for that attribute.
        """
        var cpython = Python().cpython()
        var result = cpython.PyObject_SetAttrString(
            self.py_object, name^, new_value.py_object
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
        var cpython = Python().cpython()
        return cpython.Py_Is(self.py_object, other.py_object)

    fn __isnot__(self, other: PythonObject) -> Bool:
        """Test if the PythonObject is not the `other` PythonObject, the same as `x is not y` in
        Python.

        Args:
            other: The right-hand-side value in the comparison.

        Returns:
            True if they are not the same object and False otherwise.
        """
        return not (self is other)

    fn __getitem__(self, *args: PythonObject) raises -> PythonObject:
        """Return the value for the given key or keys.

        Args:
            args: The key or keys to access on this object.

        Returns:
            The value corresponding to the given key for this object.
        """
        var cpython = Python().cpython()
        var size = len(args)
        var key_obj: PyObjectPtr
        if size == 1:
            key_obj = args[0].py_object
        else:
            key_obj = cpython.PyTuple_New(size)
            for i in range(size):
                var arg_value = args[i].py_object
                cpython.Py_IncRef(arg_value)
                var result = cpython.PyTuple_SetItem(key_obj, i, arg_value)
                if result != 0:
                    raise Error("internal error: PyTuple_SetItem failed")

        cpython.Py_IncRef(key_obj)
        var result = cpython.PyObject_GetItem(self.py_object, key_obj)
        cpython.Py_DecRef(key_obj)
        if result.is_null():
            raise cpython.get_error()
        return PythonObject(from_owned_ptr=result)

    fn __getitem__(self, *args: Slice) raises -> PythonObject:
        """Return the sliced value for the given Slice or Slices.

        Args:
            args: The Slice or Slices to apply to this object.

        Returns:
            The sliced value corresponding to the given Slice(s) for this object.
        """
        var cpython = Python().cpython()
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
        var result = cpython.PyObject_GetItem(self.py_object, key_obj)
        cpython.Py_DecRef(key_obj)
        if result.is_null():
            raise cpython.get_error()
        return PythonObject(from_owned_ptr=result)

    fn __setitem__(self, *args: PythonObject, value: PythonObject) raises:
        """Set the value with the given key or keys.

        Args:
            args: The key or keys to set on this object.
            value: The value to set.
        """
        var cpython = Python().cpython()
        var size = len(args)
        var key_obj: PyObjectPtr

        if size == 1:
            key_obj = args[0].py_object
        else:
            key_obj = cpython.PyTuple_New(size)
            for i in range(size):
                var arg_value = args[i].py_object
                cpython.Py_IncRef(arg_value)
                var result = cpython.PyTuple_SetItem(key_obj, i, arg_value)
                if result != 0:
                    raise Error("internal error: PyTuple_SetItem failed")

        cpython.Py_IncRef(key_obj)
        cpython.Py_IncRef(value.py_object)
        var result = cpython.PyObject_SetItem(
            self.py_object, key_obj, value.py_object
        )
        if result != 0:
            raise cpython.get_error()
        cpython.Py_DecRef(key_obj)
        cpython.Py_DecRef(value.py_object)

    @doc_private
    fn __call_single_arg_inplace_method__(
        mut self, owned method_name: String, rhs: PythonObject
    ) raises:
        var callable_obj: PythonObject
        try:
            callable_obj = self.__getattr__("__i" + method_name[2:])
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
        return self.__getattr__("__ne__")(rhs).__bool__()

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
        var cpython = Python().cpython()
        if cpython.PyObject_HasAttrString(self.py_object, "__contains__"):
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
        var cpython = Python().cpython()

        var num_pos_args = len(args)
        var tuple_obj = cpython.PyTuple_New(num_pos_args)
        for i in range(num_pos_args):
            var arg_value = args[i].py_object
            cpython.Py_IncRef(arg_value)
            var result = cpython.PyTuple_SetItem(tuple_obj, i, arg_value)
            if result != 0:
                raise Error("internal error: PyTuple_SetItem failed")

        var dict_ptr = Python._dict(kwargs)
        var callable_obj = self.py_object
        cpython.Py_IncRef(callable_obj)
        var result = cpython.PyObject_Call(callable_obj, tuple_obj, dict_ptr)
        cpython.Py_DecRef(callable_obj)
        cpython.Py_DecRef(tuple_obj)
        cpython.Py_DecRef(dict_ptr)
        if result.is_null():
            raise cpython.get_error()
        return PythonObject(from_owned_ptr=result)

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    fn __len__(self) raises -> Int:
        """Returns the length of the object.

        Returns:
            The length of the object.
        """
        var cpython = Python().cpython()
        var result = cpython.PyObject_Length(self.py_object)
        if result == -1 and cpython.PyErr_Occurred():
            # Custom python types may return -1 even in non-error cases.
            raise cpython.unsafe_get_error()
        return result

    fn __hash__(self) raises -> Int:
        """Returns the hash value of the object.

        Returns:
            The hash value of the object.
        """
        var cpython = Python().cpython()
        var result = cpython.PyObject_Hash(self.py_object)
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

    @always_inline
    fn __str__(self) raises -> PythonObject:
        """Convert the PythonObject to a Python `str`.

        Returns:
            A Python `str` object.

        Raises:
            An error if the conversion failed.
        """
        return Python.str(self)

    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this Python object to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

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

    fn to_python_object(self) -> PythonObject:
        """Convert this value to a PythonObject.

        Returns:
            A PythonObject representing the value.
        """
        return self

    fn unsafe_as_py_object_ptr(self) -> PyObjectPtr:
        """Get the underlying PyObject pointer.

        Returns:
            The underlying PyObject pointer.

        Safety:
            Use-after-free: The caller must take care that `self` outlives the
            usage of the pointer returned by this function.
        """
        return self.py_object

    fn steal_data(owned self) -> PyObjectPtr:
        """Take ownership of the underlying pointer from the Python object.

        Returns:
            The underlying data.
        """
        var ptr = self.py_object
        self.py_object = PyObjectPtr()

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
        T: TypeIdentifiable
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
                        T.TYPE_ID,
                        _get_type_name(self),
                    )
                )
            else:
                raise Error(
                    String.format(
                        "TypeError: expected Mojo '{}' type value, got '{}'",
                        T.TYPE_ID,
                        _get_type_name(self),
                    )
                )

        # SAFETY: We just validated that this Optional is not empty.
        return opt.unsafe_take()

    fn _try_downcast_value[
        T: TypeIdentifiable,
    ](owned self) raises -> Optional[UnsafePointer[T]]:
        """Try to get a pointer to the expected contained Mojo value of type `T`.

        None will be returned if the type of this object does not match the
        bound Python type of `T`.

        This function will raise if the provided Mojo type `T` has not been
        bound to a Python type using a `PythonTypeBuilder`.

        Parameters:
            T: The type of the Mojo value that this Python object is expected
              to contain.

        Raises:
            If `T` has not been bound to a Python type object.
        """
        var cpython = Python().cpython()
        var type = PyObjectPtr(
            cpython.Py_TYPE(self.unsafe_as_py_object_ptr()).bitcast[PyObject]()
        )

        var expected_type_obj = lookup_py_type_object[T]()

        if type == expected_type_obj.unsafe_as_py_object_ptr():
            return self.unchecked_downcast_value_ptr[T]()
        else:
            return None

    fn unchecked_downcast_value_ptr[T: AnyType](self) -> UnsafePointer[T]:
        """Get a pointer to the expected Mojo value of type `T`.

        This function assumes that this Python object was allocated as an
        instance of `PyMojoObject[T]`.

        Parameters:
            T: The type of the Mojo value stored in this object.

        Returns:
            A pointer to the inner Mojo value.

        # Safety

        The user must be certain that this Python object type matches the bound
        Python type object for `T`.
        """
        var obj_ptr = self._unchecked_downcast_object_ptr[PyMojoObject[T]]()

        # TODO(MSTDL-950): Should use something like `addr_of!`
        return UnsafePointer[T](to=obj_ptr[].mojo_value)

    @always_inline
    fn _unchecked_downcast_object_ptr[
        T: AnyType
    ](owned self) -> UnsafePointer[T]:
        """Assume that this Python object contains a wrapped Mojo value."""
        return self.py_object.unsized_obj_ptr.bitcast[T]()


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
    var cpython = Python().cpython()
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
