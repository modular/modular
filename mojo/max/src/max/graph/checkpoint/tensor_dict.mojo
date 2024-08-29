# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Structs used for maintaining a collection of tensors."""
from max.tensor import Tensor, TensorSpec
from collections.dict import _DictKeyIter, _DictEntryIter
from memory import memcpy
from collections import Dict


@value
struct _CheckpointTensor:
    """A wrapper around a Tensor pointer that can be saved/loaded from disk."""

    var ptr: UnsafePointer[UInt8]
    var spec: TensorSpec

    fn __init__(
        inout self,
        owned ptr: UnsafePointer[UInt8],
        owned spec: TensorSpec,
    ):
        """Creates a _CheckpointTensor.

        Args:
            ptr: UnsafePointer to tensor data.
            spec: Tensor's spec.
        """
        self.ptr = ptr
        self.spec = spec^

    fn copy_to_tensor[T: DType](owned self) -> Tensor[T]:
        """Returns a deep copy of the Tensor data."""
        var num_elements = self.spec.num_elements()
        var spec = self.spec
        var self_ptr = self.ptr.bitcast[T]()
        var ptr = UnsafePointer[Scalar[T]].alloc(num_elements)
        memcpy(ptr, self_ptr, num_elements)
        return Tensor[T](spec, ptr)

    fn to_tensor[T: DType](owned self) -> Tensor[T]:
        """Converts this object to a Tensor."""
        var spec = self.spec^
        var ptr = self.ptr.bitcast[T]()
        self.spec = TensorSpec()
        self.ptr = UnsafePointer[UInt8]()
        return Tensor[T](spec, ptr)

    @staticmethod
    fn from_tensor(owned tensor: Tensor) -> Self:
        """Creates a _CheckpointTensor from a Tensor."""
        var spec = tensor.spec()
        var ptr = tensor._steal_ptr().bitcast[DType.uint8]()
        return Self(ptr, spec)


struct TensorDict(Sized):
    """A collection of keyed `Tensor` values used with checkpoint files.

    This is the type accepted by
    [`save()`](/max/api/mojo/graph/checkpoint/save_load/save) and
    returned by
    [`load()`](/max/api/mojo/graph/checkpoint/save_load/load).

    For example:

    ```mojo
    from max.graph.checkpoint import load, save, TensorDict
    from max.tensor import Tensor, TensorShape

    def write_to_disk():
        tensors = TensorDict()
        tensors.set("x", Tensor[DType.int32](TensorShape(1, 2, 2), 1, 2, 3, 4))
        tensors.set("y", Tensor[DType.float32](TensorShape(10, 5), -1.23))
        save(tensors, "/path/to/checkpoint.maxckpt")

    def read_from_disk():
        tensors = load("/path/to/checkpoint.maxckpt")
        x = tensors.get[DType.int32]("x")
    ```
    """

    var _items: Dict[String, _CheckpointTensor]

    def __init__(inout self):
        self._items = Dict[String, _CheckpointTensor]()

    fn __setitem__[T: DType](inout self, key: String, value: Tensor[T]):
        """Supports setting items with the bracket accessor.

        For example:

        ```mojo
        tensors = TensorDict()
        tensors["x"] = Tensor[DType.int32](TensorShape(1, 2, 2), 1, 2, 3, 4)
        ```

        Args:
            key: The key to associate with the specified value.
            value: The data to store in the dictionary.
        """
        self._items._insert(key, _CheckpointTensor.from_tensor(value))

    fn set[T: DType](inout self, key: String, value: Tensor[T]):
        """Adds or updates a tensor in the dictionary.

        Args:
            key: The name of the tensor.
            value: The tensor to add.
        """
        self._items._insert(key, _CheckpointTensor.from_tensor(value))

    fn get[type: DType](self, key: String) raises -> Tensor[type]:
        """Gets a tensor from the dictionary.

        Currently, this returns a copy of the tensor. For better performance,
        use `Dict.pop()`.

        This method may change in the future to return an immutable reference
        instead of a mutable tensor copy.

        Args:
            key: The name of the tensor.

        Returns:
            A copy of the tensor.
        """
        try:
            return self._items[key].copy_to_tensor[type]()
        except e:
            raise "Error when getting key '" + key + "': " + str(e)

    fn _set(inout self, key: String, value: _CheckpointTensor):
        """Adds or updates a tensor in the dictionary.

        Args:
            key: The name of the tensor.
            value: The tensor to add.
        """
        self._items._insert(key, value)

    fn _get(self, key: String) raises -> _CheckpointTensor:
        """Gets a raw `CheckpointTensor` value from the dictionary.

        Args:
            key: The name of the tensor.
        """
        return self._items[key]

    fn pop[type: DType](inout self, key: String) raises -> Tensor[type]:
        """Removes a tensor from the dictionary.

        This function moves the Tensor pointer out of the dictionary and returns
        it to the caller.

        Args:
            key: The name of the tensor.

        Returns:
            The tensor.
        """
        try:
            return self._items.pop(key).to_tensor[type]()
        except e:
            raise "Error when getting key '" + key + "': " + str(e)

    fn __len__(self) -> Int:
        return len(self._items)

    fn items(
        ref [_]self: Self,
    ) -> _DictEntryIter[String, _CheckpointTensor, __lifetime_of(self._items)]:
        """Gets an iterable view of all elements in the dictionary."""
        return _DictEntryIter(0, 0, self._items)

    fn keys(
        ref [_]self: Self,
    ) -> _DictKeyIter[String, _CheckpointTensor, __lifetime_of(self._items)]:
        """Gets an iterable view of all keys in the dictionary."""
        return _DictKeyIter(_DictEntryIter(0, 0, self._items))

    def __iter__(
        ref [_]self: Self,
    ) -> _DictKeyIter[String, _CheckpointTensor, __lifetime_of(self._items)]:
        return _DictKeyIter(_DictEntryIter(0, 0, self._items))

    fn __copyinit__(inout self, existing: Self):
        """Copies a dictionary.

        Args:
            existing: The existing dict.
        """
        self._items = existing._items

    fn __moveinit__(inout self, owned existing: Self):
        """Moves data of an existing dictionary into a new one.

        Args:
            existing: The existing dict.
        """
        self._items = existing._items^

    fn __str__(self) -> String:
        var contents: String = ""
        var first = True
        for key_ref in self._items.keys():
            var key = key_ref[]
            if first:
                first = False
            else:
                contents += ", "
            try:
                contents += key + ": " + str(self._items[key].spec)
            except:
                # Should never happen.
                contents += key + ": " + "(contents could not be read)"
        return "TensorDict(" + contents + ")"
