# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Defines functions to save/load tensors from a checkpoint file."""
from tensor import Tensor, TensorShape, TensorSpec
from max.engine import TensorMap
from pathlib import Path

from .tensor_dict import TensorDict, _CheckpointTensor

# 0x93 🔥 + + 0x93
alias _SERIALIZATION_HEADER = InlineArray[Int8, 8](
    0x93, 0xF0, 0x9F, 0x94, 0xA5, 0x2B, 0x2B, 0x93
)

# Serialization constants
alias _SERIALIZATION_MAJOR_FORMAT: UInt32 = 110
alias _SERIALIZATION_MINOR_FORMAT: UInt32 = 1


@always_inline
fn _write_int[type: Intable](ref [_]object: type, f: FileHandle) raises:
    """Writes an int value to a file."""
    var ptr = UnsafePointer.address_of(object).bitcast[UInt8]()
    f._write(ptr, sizeof[type]())


def save(tensor_dict: TensorDict, path: Path):
    """Saves a collection of tensors to a file.

    The file is saved in a binary format that's specific to MAX. You can then
    load the checkpoint with
    [`load()`](/max/reference/mojo/graph/checkpoint/save_load/load).

    For example:

    ```mojo
    from max.graph.checkpoint import save, TensorDict
    from tensor import Tensor, TensorShape

    def write_to_disk():
        tensors = TensorDict()
        tensors.set("x", Tensor[DType.int32](TensorShape(1, 2, 2), 1, 2, 3, 4))
        tensors.set("y", Tensor[DType.float32](TensorShape(10, 5), -1.23))
        save(tensors, "/path/to/checkpoint.maxckpt")
    ```

    Args:
        tensor_dict: Tensors to save.
        path: The location to save the checkpoint file. You can use whatever
              filename and file extension you want.
    """
    # Write header and metadata
    # TODO: Using _SERIALIZATION_HEADER raises errors.
    var header_buf = List[UInt8](
        0x93, 0xF0, 0x9F, 0x94, 0xA5, 0x2B, 0x2B, 0x93, 0x0
    )
    var header = String(header_buf)
    var major_format: UInt32 = _SERIALIZATION_MAJOR_FORMAT
    var minor_format: UInt32 = _SERIALIZATION_MINOR_FORMAT

    # Compute metadata size and tensor offsets.
    # Each entry of the metadata consists of:
    #   1. Length of key (UInt32)
    #   2. String Key
    #   3. TensorSpec:
    #      - DType (UInt8)
    #      - Rank  (UInt8)
    #      - Dims (List[UInt32])
    #   5. Tensor Offset (UInt64)
    var metadata_size: UInt64 = 0

    # Use zero as the relative offset for now, this will be corrected
    # in the next loop.
    var last_tensor_offset = 0
    var tensor_offsets = List[UInt64](capacity=len(tensor_dict))
    # Keep a list of keys, since the order may not be guaranteed in TensorDict.
    var tensor_keys = List[String](capacity=len(tensor_dict))

    for key_ref in tensor_dict:
        var key = key_ref[]
        var spec = tensor_dict._get(key).spec
        tensor_keys.append(key)

        metadata_size += (
            sizeof[UInt32]()  # Length of key
            + len(key)  # String key
            + sizeof[UInt8]()  # TensorSpec dtype
            + sizeof[UInt8]()  # TensorSpec rank
            + sizeof[UInt32]() * spec.rank()  # TensorSpec dims
            + sizeof[UInt64]()  # Tensor Offset
        )
        tensor_offsets.append(last_tensor_offset)
        last_tensor_offset += tensor_dict._get(key).spec.bytecount()

    # Now that we know the metadata size, we can correct the tensor offsets.
    var first_tensor_offset = (
        sizeof[Int8]() * len(header)  # Header
        + sizeof[UInt32]()  # Major version
        + sizeof[UInt32]()  # Minor version
        + sizeof[UInt64]()  # Int containing metadata size
        + metadata_size  # Metadata bytes
    )
    for n in range(len(tensor_offsets)):
        tensor_offsets[n] += first_tensor_offset

    with open(path, "wb") as f:
        # Write header.
        f.write(header)
        _write_int(major_format, f)
        _write_int(minor_format, f)
        _write_int(metadata_size, f)

        # Write out metadata (key, spec and offset).
        for i in range(len(tensor_keys)):
            var key = tensor_keys[i]
            var key_size: UInt32 = len(key)
            _write_int(key_size, f)
            f.write(key)

            var spec = tensor_dict._get(key).spec
            var dtype: UInt8 = spec.dtype()._as_i8()
            _write_int(dtype, f)
            var rank: UInt8 = spec.rank()
            _write_int(rank, f)
            for d in range(rank):
                var dim: UInt32 = spec.shape[d]
                _write_int(dim, f)

            var tensor_offset = tensor_offsets[i]
            _write_int(tensor_offset, f)

        # Write out each tensor.
        for i in range(len(tensor_keys)):
            var key = tensor_keys[i]
            var ptr = UnsafePointer[UInt8]._from_dtype_ptr(
                tensor_dict._get(key).ptr
            )
            var tensor_size = tensor_dict._get(key).spec.bytecount()
            f._write(ptr, tensor_size)


@always_inline
fn _read_int[type: DType](f: FileHandle) raises -> Scalar[type]:
    """Reads an int value from a file."""
    var size = sizeof[type]()
    var bytes_tensor = Tensor[DType.uint8](f.read_bytes(size))
    var result = bytes_tensor.unsafe_ptr().bitcast[type]().load()
    _ = bytes_tensor^
    return result


@always_inline
fn _read_string(f: FileHandle, size: UInt32) raises -> String:
    """Reads string of the specified size from a file."""
    var string_bytes = f.read_bytes(size.cast[DType.int64]())
    string_bytes.append(0)  # Add null terminator
    return String(string_bytes)


@value
struct _KeysAndSpecs:
    var key: String
    var spec: TensorSpec


@always_inline
def _version_to_string(
    major_version: UInt32 = _SERIALIZATION_MAJOR_FORMAT,
    minor_version: UInt32 = _SERIALIZATION_MAJOR_FORMAT,
) -> String:
    return str(major_version) + "." + str(minor_version)


def load(path: Path) -> TensorDict:
    """Reads tensors from saved checkpoint file.

    This supports only MAX checkpoint files saved with
    [`save()`](/max/reference/mojo/graph/checkpoint/save_load/save).

    For example:

    ```mojo
    from max.graph.checkpoint import load, TensorDict
    from tensor import Tensor, TensorShape

    def read_from_disk():
        tensors = load("/path/to/checkpoint.maxckpt")
        x = tensors.get("x").to_tensor[DType.int32]()
    ```

    Args:
        path: Path to existing checkpoint file.

    Returns:
        TensorDict containing loaded Tensors.

    Raises:
        If the checkpoint file was saved with an older serialization format.
    """
    # TODO: Using _SERIALIZATION_HEADER raises errors.
    var header_buf = List[UInt8](
        0x93, 0xF0, 0x9F, 0x94, 0xA5, 0x2B, 0x2B, 0x93, 0x0
    )
    with open(path, "rb") as f:
        var header = f.read_bytes(8)
        if len(header) != 8:
            raise "Invalid checkpoint file: " + str(path)
        for i in range(8):
            if header[i] != header_buf[i]:
                raise "Given file is not a max.graph checkpoint."

        var major_version = _read_int[DType.uint32](f)
        var minor_version = _read_int[DType.uint32](f)

        if major_version > _SERIALIZATION_MAJOR_FORMAT:
            raise (
                "Cannot read from checkpoint version "
                + _version_to_string(major_version, minor_version)
                + ". Current version: "
                + _version_to_string()
            )
        var metadata_size = _read_int[DType.uint64](f)

        var bytes_read: UInt64 = 0

        # Read metadata.
        var keys_and_specs = List[_KeysAndSpecs]()
        var tensor_offsets = List[UInt64]()
        while bytes_read < metadata_size:
            var key_size = _read_int[DType.uint32](f)
            var key = _read_string(f, key_size)
            var dtype = _read_int[DType.uint8](f)
            var rank = _read_int[DType.uint8](f)
            var dims = List[Int]()
            for _ in range(rank):
                var d = _read_int[DType.uint32](f)
                dims.append(int(d))
            var spec = TensorSpec(DType._from_ui8(dtype.value), dims)
            var tensor_offset = _read_int[DType.uint64](f)
            tensor_offsets.append(tensor_offset)
            keys_and_specs.append(_KeysAndSpecs(key, spec))
            bytes_read += (
                sizeof[UInt32]()  # Length of key
                + int(key_size)  # String key
                + sizeof[UInt8]()  # TensorSpec dtype
                + sizeof[UInt8]()  # TensorSpec rank
                + sizeof[UInt32]() * int(rank)  # TensorSpec dims
                + sizeof[UInt64]()  # Tensor Offset
            )

        # Construct TensorDict.
        var ret = TensorDict()
        for i in range(len(keys_and_specs)):
            _ = f.seek(tensor_offsets[i])
            var ks = keys_and_specs[i]
            var bytes = f.read_bytes(ks.spec.bytecount())
            var ptr = DTypePointer(bytes.steal_data()).bitcast[DType.uint8]()
            var tensor = _CheckpointTensor(ptr, ks.spec)
            ret._set(ks.key, tensor)
        return ret
