# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""
Defines the `TensorMap` type that holds input and output tensors for a model.
"""
from sys.ffi import DLHandle

from buffer import NDBuffer
from max._utils import CString, call_dylib_func, exchange
from max.tensor import Tensor, TensorSpec
from memory import UnsafePointer
from memory.unsafe import bitcast

from utils.write import _WriteBufferStack

from ._context import CRuntimeContext
from ._tensor_impl import EngineTensor
from ._tensor_map_impl import CTensorMap
from .session import InferenceSession
from .value import Value


struct TensorMap(CollectionElement, SizedRaising, Stringable):
    """
    Maps inputs and outputs to their respective names and can
    be used to supply and receive data to MAX Engine model.

    This is the data type returned by
    [`Model.execute()`](/max/api/mojo/engine/model/Model#execute), and
    you can also use this type for the inputs you pass in (although `execute()`
    also supports other formats for the input).
    """

    var _ptr: CTensorMap
    var _lib: DLHandle
    var _session: InferenceSession

    alias _NewTensorMapFnName = "M_newAsyncTensorMap"
    alias _DeleteTensorMapKeysFnName = "M_deleteTensorMapKeys"

    fn __init__(
        out self,
        ctx: CRuntimeContext,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        """Internal only. Use InferenceSession.new_tensor_map
        for external use.

        Creates a new instance of tensor map.

        Args:
            ctx: Context of API.
            lib: Handle to library.
            session: Copy of InferenceSession from which this instance
                     was created.
        """
        self._ptr = call_dylib_func[CTensorMap](
            lib, Self._NewTensorMapFnName, ctx
        )
        self._lib = lib
        self._session = session^

    fn __init__(
        out self,
        ptr: CTensorMap,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        """Internal only. Use InferenceSession.new_tensor_map
        for external use.

        Creates a new instance of tensor map.

        Args:
            ptr: C API pointer of TensorMap.
            lib: Handle to library.
            session: Copy of InferenceSession from which this instance
                     was created.
        """
        self._ptr = ptr
        self._lib = lib
        self._session = session^

    fn __moveinit__(out self, owned existing: Self):
        """Move contructor for TensorMap.

        Args:
            existing: Instance of TensorMap to move from.
        """
        self._ptr = exchange[CTensorMap](
            existing._ptr, UnsafePointer[NoneType]()
        )
        self._lib = existing._lib
        self._session = existing._session^

    fn __copyinit__(out self, existing: Self):
        """Copy contructor for TensorMap.

        Args:
            existing: Instance of TensorMap to copy from.
        """
        self._ptr = existing._ptr.copy(existing._lib)
        self._lib = existing._lib
        self._session = existing._session

    @always_inline
    fn copy(self) -> Self:
        """Explicitly construct a copy of self.

        Returns:
            A copy of this value.
        """
        return self

    fn borrow[type: DType](self, key: String, value: Tensor[type]) raises:
        """Borrow the given tensor into the map at the key location.
           User needs to make sure tensor is alive for
           the duration of map.

        Parameters:
            type: DType of tensor being borrowed.

        Args:
            key: Name of tensor in map.
            value: Tensor to be held in map.
        """
        var spec = EngineTensorSpec(
            key,
            value.spec(),
            self._lib,
            self._session,
        )
        self._ptr.borrow_tensor_by_name(
            value.unsafe_ptr().bitcast[NoneType](), spec, self._lib
        )

    fn borrow[
        type: DType
    ](
        self, key: String, spec: TensorSpec, ptr: UnsafePointer[Scalar[type]]
    ) raises:
        """Borrow the given pointer into the map at the key location.
           User needs to make sure the backing array is alive for
           the duration of map.

        Parameters:
            type: DType of tensor being borrowed.

        Args:
            key: Name of tensor in map.
            spec: The tensor spec. This is the standard library
                  [`TensorSpec`](/mojo/stdlib/tensor/tensor_spec/TensorSpec).
            ptr: The tensor pointer.
        """
        var tensor_spec = EngineTensorSpec(
            key,
            spec,
            self._lib,
            self._session,
        )
        self._ptr.borrow_tensor_by_name(
            ptr.bitcast[NoneType](),
            tensor_spec,
            self._lib,
        )

    fn borrow(self, key: String, value: EngineTensorView) raises:
        """Borrow the given tensor view into the map at the key location.
           User needs to make sure tensor backing the view is alive for
           the duration of map.

        Args:
            key: Name of tensor in map.
            value: View of a tensor.
        """
        var spec = EngineTensorSpec(
            key,
            value.spec(),
            self._lib,
            self._session,
        )
        self._ptr.borrow_tensor_by_name(value.unsafe_ptr(), spec, self._lib)

    fn borrow(self, key: String, value: EngineNumpyView) raises:
        """Borrow the given numpy view into the map at the key location.
           User needs to make sure numpy array backing the view is alive for
           the duration of map.

        Args:
            key: Name of numpy array in map.
            value: View of a numpy array.
        """
        var spec = EngineTensorSpec(
            key,
            value.spec(),
            self._lib,
            self._session,
        )
        self._ptr.borrow_tensor_by_name(value.unsafe_ptr(), spec, self._lib)

    fn borrow(self, key: String, value: Value) raises:
        """Borrow the given value into the map at the key location.

        User needs to make sure value is alive for the duration of the map.

        Args:
            key: Name of value in map.
            value: Value to insert into map.
        """
        self._ptr.borrow_value_by_name(key, value._ptr.ptr, self._lib)

    fn _move_mojo_value[T: Movable](self, key: String, owned value: T) raises:
        """Move the mojo value inside the map at the key location.

        Parameters:
            T: Type of the mojo value.

        Args:
            key: Name of value in map.
            value: Mojo Value to insert into map.
        """
        self._ptr.move_mojo_value_by_name(key, value^, self._lib)

    fn get[type: DType](self, key: String) raises -> Tensor[type]:
        """Gets the tensor / numpy array indicated by the key.
           The value is copied and returned to the user.

        Parameters:
            type: DType of tensor to be returned.

        Args:
            key: Name of tensor / numpy array in the map.

        Returns:
            A copy of the tensor held by the map.
        """
        var tensor_ptr = self._ptr.get_tensor_by_name(key, self._lib)
        var mof_tensor = EngineTensor(tensor_ptr, self._lib, self._session)
        var tensor = mof_tensor.tensor[type]()
        return tensor^

    fn _take_mojo_value[T: Movable](self, key: String) raises -> T:
        """Gets the custom mojo value indicated by the key.
           The value is moved and returned to the user.
           The same key can't be requested again.

        Parameters:
            T: Type of value of to be returned.

        Args:
            key: Name of value in the map.

        Returns:
            The mojo object held by the map.
        """
        var val = self.get_value(key)
        return val._take_mojo_value[T]()

    fn buffer[
        type: DType
    ](self, key: String) raises -> NDBuffer[type, 1, MutableAnyOrigin]:
        """Gets a buffer to the tensor pointed by the key.

        Parameters:
            type: DType of buffer to be returned.

        Args:
            key: Name in TensorMap.

        Returns:
            Buffer of the tensor pointed by the key.
        """
        var tensor_ptr = self._ptr.get_tensor_by_name(key, self._lib)
        return EngineTensor(tensor_ptr, self._lib, self._session).buffer[type]()

    fn get_spec(self, key: String) raises -> TensorSpec:
        """Gets the spec of the tensor pointed by the key.

        Args:
            key: Name in TensorMap.

        Returns:
            Buffer of the tensor pointed by the key, as a
            [`TensorSpec`](/mojo/stdlib/tensor/tensor_spec/TensorSpec).
        """
        var tensor_ptr = self._ptr.get_tensor_by_name(key, self._lib)
        var mof_tensor = EngineTensor(tensor_ptr, self._lib, self._session)
        return mof_tensor.spec()

    fn get_value(self, key: String) raises -> Value:
        """Gets the value pointed by the key.

        Args:
            key: Name in TensorMap.

        Returns:
            [`Value`](/max/api/mojo/engine/value/Value) pointed by
            the key.
        """
        var value_ptr = self._ptr.get_value_by_name(key, self._lib)
        return Value(value_ptr, self._lib, self._session)

    fn keys(self) -> List[String]:
        """Returns all held keys.

        Returns:
            A list with all contained keys.
        """
        var size: Int64 = 0
        var keys_arr = self._ptr.keys(UnsafePointer(to=size), self._lib)
        var keys = List[String](capacity=Int(size))
        for i in range(Int(size)):
            keys.append(String(keys_arr[i]))

        call_dylib_func[NoneType](
            self._lib, Self._DeleteTensorMapKeysFnName, keys_arr
        )
        return keys

    fn __len__(self) raises -> Int:
        """Gets number of elements in the map.

        Returns:
            Number of elements the map contains.
        """
        return self._ptr.size(self._lib)

    fn _borrow_ptr(self) -> CTensorMap:
        return self._ptr

    fn __del__(owned self):
        """Destructor for the tensor map."""
        self._ptr.free(self._lib)
        _ = self._session^

    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats a description of the DeviceMemory to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """
        try:
            var string = String()
            var buffer = _WriteBufferStack(string)
            buffer.write("{")
            var keys = self.keys()
            for i in range(len(keys)):
                if i > 0:
                    buffer.write(",\n")

                var key = keys[i]
                var dtype = self.get_spec(key).dtype()
                buffer.write("'", key, "' : ")
                if dtype is DType.bool:
                    buffer.write(self.get[DType.bool](key))
                elif dtype is DType.uint8:
                    buffer.write(self.get[DType.uint8](key))
                elif dtype is DType.uint16:
                    buffer.write(self.get[DType.uint16](key))
                elif dtype is DType.uint32:
                    buffer.write(self.get[DType.uint32](key))
                elif dtype is DType.uint64:
                    buffer.write(self.get[DType.uint64](key))
                elif dtype is DType.int8:
                    buffer.write(self.get[DType.int8](key))
                elif dtype is DType.int16:
                    buffer.write(self.get[DType.int16](key))
                elif dtype is DType.int32:
                    buffer.write(self.get[DType.int32](key))
                elif dtype is DType.int64:
                    buffer.write(self.get[DType.int64](key))
                elif dtype is DType.float16:
                    buffer.write(self.get[DType.float16](key))
                elif dtype is DType.float32:
                    buffer.write(self.get[DType.float32](key))
                elif dtype is DType.float64:
                    buffer.write(self.get[DType.float64](key))
                else:
                    buffer.write(self.get[DType.uint8](key))
                buffer.write("}")
                buffer.flush()

                return writer.write(string)
        except:
            writer.write("{}")

    fn __str__(self) -> String:
        """Returns a `String` representation of this `TensorMap`.

        Returns:
            A textual representation of this `TensorMap`.
        """
        return String.write(self)
