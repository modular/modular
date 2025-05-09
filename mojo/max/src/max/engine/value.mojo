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
"""Defines different value types you can be pass in and out of models."""

from sys.ffi import DLHandle, external_call

from max._utils import call_dylib_func, exchange
from max.tensor import Tensor
from memory import UnsafePointer

from ._context import CRuntimeContext
from ._tensor_impl import EngineTensor
from ._value_impl import CList, CValue


struct Value:
    """Owns a single reference to a value passed in or out of a model."""

    var _ptr: CValue
    var _lib: DLHandle
    var _session: InferenceSession

    alias _NewBorrowedTensorFnName = "M_createBorrowedTensor"
    alias _NewBoolFnName = "M_createBoolAsyncValue"
    alias _NewListFnName = "M_createListAsyncValue"

    fn __init__(
        out self,
        ptr: CValue,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        """Internal use only.

        To create values yourself, use `new_*_value` methods on
        `InferenceSession`.

        Args:
            ptr: Internal use only.
            lib: Internal use only.
            session: Internal use only.
        """
        self._ptr = ptr
        self._lib = lib
        self._session = session^

    fn __moveinit__(out self, owned existing: Self):
        """Take ownership of the value contained in another `Value`.

        Args:
            existing: The value to take ownership of.
        """
        self._ptr = exchange[CValue](existing._ptr, UnsafePointer[NoneType]())
        self._lib = existing._lib
        self._session = existing._session^

    fn __del__(owned self):
        """Dispose of this reference to this value."""
        self._ptr.free(self._lib)
        _ = self._session^

    @staticmethod
    fn _new_borrowed_tensor[
        type: DType
    ](
        ctx: CRuntimeContext,
        lib: DLHandle,
        owned session: InferenceSession,
        tensor: Tensor[type],
    ) raises -> Self:
        var spec = EngineTensorSpec("", tensor.spec(), lib, session)
        var ptr = call_dylib_func[CValue](
            lib,
            Self._NewBorrowedTensorFnName,
            tensor.unsafe_ptr(),
            spec._borrow_ptr(),
            ctx,
        )
        _ = spec^
        return Self(ptr, lib, session^)

    fn _as_engine_tensor(self) raises -> EngineTensor:
        var ptr = self._ptr.get_c_tensor(self._lib)
        if not ptr.ptr:
            raise "value is not a tensor"
        return EngineTensor(ptr, self._lib, self._session)

    fn as_tensor_copy[type: DType](self) raises -> Tensor[type]:
        """Return a copy of the tensor contained in this value.

        Parameters:
            type: The expected DType of the tensor.

        Returns:
            A copy of the tensor contained in this value.

        Raises:
            If the value is not a tensor, or the dtype of the tensor value is
            mismatched with the provided type.
        """
        return self._as_engine_tensor().tensor[type]()

    @staticmethod
    fn _new_bool(
        ctx: CRuntimeContext,
        lib: DLHandle,
        owned session: InferenceSession,
        value: Bool,
    ) raises -> Self:
        var ptr = call_dylib_func[CValue](lib, Self._NewBoolFnName, value, ctx)
        return Self(ptr, lib, session^)

    fn as_bool(self) -> Bool:
        """Get the boolean contained in this value.

        The result is undefined if this value is not a boolean.

        Returns:
            Boolean contained in this value.
        """
        return self._ptr.get_bool(self._lib)

    @staticmethod
    fn _new_list(
        ctx: CRuntimeContext, lib: DLHandle, owned session: InferenceSession
    ) raises -> Self:
        var ptr = call_dylib_func[CValue](lib, Self._NewListFnName, ctx)
        return Self(ptr, lib, session^)

    fn as_list(self) raises -> List:
        """Borrow the list contained in this value.

        Ownership of the list is not transferred.  User must ensure the value
        outlives the list.

        Returns:
            A `List` borrowing the internal storage of this value.
        """
        var ptr = self._ptr.get_list(self._lib)
        if not ptr.ptr:
            raise "value is not a list"
        return List(ptr, self._lib, self._session)

    fn _take_mojo_value[T: Movable](self) raises -> T:
        """Move the mojo object out of this value.

        Owernship of the object is transfered to the caller.

        Parameters:
            T: the type of the mojo object inside the value.

        Returns:
            The mojo object inside the value.
        """
        var ptr = self._ptr.take_mojo_value(self._lib).bitcast[T]()
        var res = ptr.take_pointee()
        external_call["KGEN_CompilerRT_MojoValueFreeBuffer", NoneType](ptr)
        return res^


struct List(Sized):
    """Uncounted reference to underlying storage of a list `Value`.

    The user must take special care not to allow the underlying `Value` from
    which this list was obtained to be destroyed prior to being done with the
    `List`.  However, items within the list are separately reference-counted,
    so it is safe to use an item retrieved from the list after the list itself
    has been destroyed, or to destroy an item that was appended to a list
    (provided, however, that if the item itself is borrowing data (e.g. a
    tensor), that that underlying data must remain present).
    """

    var _ptr: CList
    var _lib: DLHandle
    var _session: InferenceSession

    fn __init__(
        out self,
        ptr: CList,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        """Internal use only.

        To create lists yourself, use `InferenceSession.new_list_value` and
        then use `Value.as_list` on the resulting value.

        Args:
            ptr: Internal use only.
            lib: Internal use only.
            session: Internal use only.
        """
        self._ptr = ptr
        self._lib = lib
        self._session = session^

    fn __moveinit__(out self, owned existing: Self):
        """Create a new List pointing at the internals of another List.

        Lists do not own anything and are borrowed from the internal storage of
        a `Value`, so the user must continue to take care to ensure that the
        `Value` the original `List` was sourced from continues to outlive this
        new `List` object.

        Args:
            existing: The List to represent.
        """
        self._ptr = exchange[CList](existing._ptr, UnsafePointer[NoneType]())
        self._lib = existing._lib
        self._session = existing._session^

    fn __del__(owned self):
        """Release the handle to this list.

        The underlying storage remains owned by the `Value` from which this
        List was obtained.
        """
        self._ptr.free(self._lib)
        _ = self._session^

    fn __len__(self) -> Int:
        """Get the length of the list.

        Returns:
            The length of the list.
        """
        return self._ptr.get_size(self._lib)

    fn __getitem__(self, index: Int) raises -> Value:
        """Get the value at an index of the list.

        The returned `Value` owns a new reference to the underlying storage of
        the value, and so the returned item `Value` may be safely used even
        after the list (and value containing the list) are destroyed.

        Args:
            index: The index of the item to retrieve within the list.

        Returns:
            A new reference to an existing value within the list.

        Raises:
            If the index is out of bounds.
        """
        var c_value = self._ptr.get_value(self._lib, index)
        if not c_value.ptr:
            raise "list index out of range"
        return Value(c_value, self._lib, self._session)

    fn append(self, value: Value):
        """Append a Value to the list.

        The list will own a new reference to the value, so it is safe to allow
        `value` to be destroyed after this operation.

        Args:
            value: The value to append to the list.
        """
        self._ptr.append(self._lib, value._ptr)
