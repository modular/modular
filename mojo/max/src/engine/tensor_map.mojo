# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""
Holds the input and output tensors for a model.
"""
from memory.unsafe import DTypePointer
from sys.ffi import DLHandle
from memory.unsafe import bitcast
from tensor import Tensor, TensorShape
from .session import InferenceSession
from ._context import CRuntimeContext
from ._utils import call_dylib_func, exchange
from ._tensor_impl import EngineTensor
from ._tensor_map_impl import CTensorMap
from .value import Value


struct TensorMap(SizedRaising):
    """
    Maps inputs and outputs to their respective names and can
    be used to supply and receive data to Max Engine model.
    """

    var _ptr: CTensorMap
    var _lib: DLHandle
    var _session: InferenceSession

    alias _NewTensorMapFnName = "M_newAsyncTensorMap"

    fn __init__(
        inout self,
        ctx: CRuntimeContext,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        self._ptr = call_dylib_func[CTensorMap](
            lib, Self._NewTensorMapFnName, ctx
        )
        self._lib = lib
        self._session = session ^

    fn __init__(
        inout self,
        ptr: CTensorMap,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        self._ptr = ptr
        self._lib = lib
        self._session = session ^

    fn __moveinit__(inout self, owned existing: Self):
        self._ptr = exchange[CTensorMap](
            existing._ptr, DTypePointer[DType.invalid]()
        )
        self._lib = existing._lib
        self._session = existing._session ^

    fn borrow[type: DType](self, key: String, value: Tensor[type]) raises:
        """Borrow the given tensor into the map at the key location.
           User needs to make sure tensor is alive for
           the duration of map.

        Args:
            key: Name of tensor in map.
            value: Tensor to be held in map.
        """
        var spec = EngineTensorSpec(
            key._strref_dangerous(),
            value.spec(),
            self._lib,
            self._session.copy(),
        )
        self._ptr.borrow_tensor_by_name(
            bitcast[DType.invalid](value.data()), spec, self._lib
        )
        key._strref_keepalive()

    fn borrow(self, key: String, value: EngineTensorView) raises:
        """Borrow the given tensor view into the map at the key location.
           User needs to make sure tensor backing the view is alive for
           the duration of map.

        Args:
            key: Name of tensor in map.
            value: View of a tensor.
        """
        var spec = EngineTensorSpec(
            key._strref_dangerous(),
            value.spec(),
            self._lib,
            self._session.copy(),
        )
        self._ptr.borrow_tensor_by_name(value.data(), spec, self._lib)
        key._strref_keepalive()

    fn borrow(self, key: String, value: EngineNumpyView) raises:
        """Borrow the given numpy view into the map at the key location.
           User needs to make sure numpy array backing the view is alive for
           the duration of map.

        Args:
            key: Name of numpy array in map.
            value: View of a numpy array.
        """
        var spec = EngineTensorSpec(
            key._strref_dangerous(),
            value.spec(),
            self._lib,
            self._session.copy(),
        )
        self._ptr.borrow_tensor_by_name(value.data(), spec, self._lib)
        key._strref_keepalive()

    fn borrow(self, key: String, value: Value) raises:
        """Borrow the given value into the map at the key location.

        User needs to make sure value is alive for the duration of the map.

        Args:
            key: Name of value in map.
            value: Value to insert into map.
        """
        self._ptr.borrow_value_by_name(
            key._strref_dangerous(), value._ptr.ptr, self._lib
        )
        key._strref_keepalive()

    fn get[type: DType](self, key: String) raises -> Tensor[type]:
        """Gets the tensor / numpy array indicated by the key.
           The value is copied and returned to the user.

        Args:
            key: Name of tensor / numpy array in the map.
        """
        var tensor_ptr = self._ptr.get_tensor_by_name(
            key._strref_dangerous().data, self._lib
        )
        key._strref_keepalive()
        var mof_tensor = EngineTensor(
            tensor_ptr, self._lib, self._session.copy()
        )
        var tensor = mof_tensor.tensor[type]()
        return tensor ^

    fn buffer[type: DType](self, key: String) raises -> Buffer[type]:
        """Gets a buffer to the tensor pointed by the key.

        Args:
            key: Name in TensorMap.

        Returns:
            Buffer of the tensor pointed by the key.
        """
        var tensor_ptr = self._ptr.get_tensor_by_name(
            key._strref_dangerous().data, self._lib
        )
        key._strref_keepalive()
        return EngineTensor(tensor_ptr, self._lib, self._session.copy()).buffer[
            type
        ]()

    fn get_value(self, key: String) raises -> Value:
        """Gets the value pointed by the key.

        Args:
            key: Name in TensorMap.

        Returns:
            Value pointed by the key.
        """
        var value_ptr = self._ptr.get_value_by_name(
            key._strref_dangerous().data, self._lib
        )
        key._strref_keepalive()
        return Value(value_ptr, self._lib, self._session.copy())

    fn __len__(self) raises -> Int:
        return self._ptr.size(self._lib)

    fn _borrow_ptr(self) -> CTensorMap:
        return self._ptr

    fn __del__(owned self):
        self._ptr.free(self._lib)
        _ = self._session ^
