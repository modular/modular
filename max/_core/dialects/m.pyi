# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

import enum
from collections.abc import Callable, Sequence
from typing import Protocol, overload

import max._core
import max._core.dialects.builtin

# Many of the generated overloads for constructors are more specialized in
# C++ than they are in Python. For example, `int32_t` and `int64_t` and `size_t`
# all map to `int` in Python typing. It may not always be clear which of these
# overloads will be run for a given set of inputs (though in most cases it's the first one)
# but we disable mypy errors for shadowed overloads.
#
# mypy: disable-error-code="overload-cannot-match"

# DiagnosticHandlers aren't a thing that Python can reasonably provided. In most cases
# these are automatically provided, but there are a few custom verifiers not covered yet.
# This binding prevents errors in those cases.
DiagnosticHandler = Callable

class InOutSemantics(enum.Enum):
    none = 46

    in_ = 105

    out = 111

    mut = 109

class HasAlignedBytesInterface(Protocol):
    """
    This interface allows an attribute to describe the size and
    alignment of its underlying ArrayRef<uint8_t> data as an !M.aligned_bytes.
    """

    @property
    def aligned_bytes_type(self) -> AlignedBytesType: ...

class AlignedBytesAttr(max._core.Attribute):
    """
    This attribute is the 'inline' equivalent of `dense_resource`, and describes
    a byte array with byte alignment. Nothing is implied about the internal
    structure of the bytes (rank, dimensions, underlying element type, padding,
    or element endianness).

    This attribute implements the ElementsAttrInterface as a rank-1 ShapedType
    over uint8_t elements.

    This attribute implements the HasAlignedBytesInterface by returning the
    obvious !M.aligned_bytes type.

    Generally this attribute should only be used for 'small' literals. Larger
    literals should be placed into the resources table and be referenced by the
    `dense_resources` attribute. Note that dense resources encode their
    alignment in their first four bytes using the little-endian convention,
    where as this attribute describes the alignment explicitly.

    Examples:
    ```mlir
    #M.aligned_bytes<"0x01020304", align 64> // 4 bytes, aligned on 64 bytes
    ```
    """

    def __init__(self, data: Sequence[int], align: int) -> None: ...
    @property
    def data(self) -> Sequence[int]: ...
    @property
    def align(self) -> int: ...

class ArrayElementsAttr(max._core.Attribute):
    """
    The `#M.dense_array` attribute is an elements attribute backed by a
    primitive array. The attribute supports the full API expected by
    `ElementsAttr` and users thereof, including arbitrary shaped types and
    failable value iteration.

    Importantly, this attribute can only contain scalar integer and floating
    point types, does not bit-pack elements, and does not have special handling
    for splat elements.

    This attribute implements the HasAlignedBytesInterface by returning a
    !M.aligned_bytes type capturing the byte size of the underlying element data
    array and an alignment of the next power-of-two of the byte size of the
    element type.

    Example:

    ```mlir
    // An array of integers.
    #M.dense_array<-3, 0, 1, 42> : !M.array<4xi32>

    // A float tensor.
    #M.dense_array<3.4, 2.3, 5.2, 1.9> : tensor<2x2xf32>

    // A 0D vector.
    #M.dense_array<1> : vector<ui64>
    ```
    """

    @overload
    def __init__(
        self, data: Sequence[int], type: max._core.dialects.builtin.ShapedType
    ) -> None: ...
    @overload
    def __init__(
        self,
        data: PrimitiveArrayAttr,
        type: max._core.dialects.builtin.ShapedType,
    ) -> None: ...
    @property
    def data(self) -> PrimitiveArrayAttr: ...
    @property
    def type(self) -> max._core.dialects.builtin.ShapedType: ...

class DeviceRefAttr(max._core.Attribute):
    """
    The `#M.device_ref` attribute refers to a unique `#M.device_spec` within the
    overall model `#M.device_spec_collection`. It contain a label and id.

    Example:
    ```mlir
      #M.device_ref<"gpu", 0>
    ```
    """

    def __init__(self, label: str, id: int) -> None: ...
    @property
    def label(self) -> str: ...
    @property
    def id(self) -> int: ...

class DeviceSpecAttr(max._core.Attribute):
    """
    The `#M.device_spec` attribute describes everything we need to know about a
    computational device at compile time which is expected to be present at
    runtime. This includes the device's `#M.target` configuration (to guide
    code generation), the device's id (to disambiguate the device from other
    devices with similar target configuration, default 0), and a string label
    (so model types and ops may refer to a specific device when needed, eg for
    device placement).

    Device ids need not refer to physical devices (eg a specific CUDA device
    id), they are just placeholders. It is the responsibility of the various
    setup ops in the model's 'init' block to establish the mapping from device
    ids used in the model to actual physical devices where required. This may
    require matching properties of the `#M.target` to capabilities probed at
    runtime.

    ```mlir
    #M.device_spec<ref = <"gpu", 1>,
                   target = <triple="nvptx64-nvidia-cuda", arch="sm_80">>
    ```

    Can be represented at runtime by M::DeviceSpec.
    """

    def __init__(self, ref: DeviceRefAttr, target: TargetInfoAttr) -> None: ...
    @property
    def ref(self) -> DeviceRefAttr: ...
    @property
    def target(self) -> TargetInfoAttr: ...

class DeviceSpecCollectionAttr(max._core.Attribute):
    """
    The `#M.device_spec_collection` attribute describes a collection of
    `#M.device_spec` attributes along with the device reference for the
    unique 'host' device. Each `#M.device_spec` must have a distinct device
    reference, and obviously the 'host' device reference must exist.

    ```mlir
    #M.device_spec_collection<
      host = <"cpu", 0>,
      devices = [<ref = <"cpu", 0>,
                  target = <triple="x86_64-unknown-linux-gnu", arch="znver3",
                            features="+avx2">>,
                 <ref = <"gpu", 1>,
                  target = <triple="nvptx64-nvidia-cuda", arch="sm_80">>]>
    ```

    Encoded into MEF as JSON in string form.

    Can be represented at runtime by M::DeviceSpecCollection.
    """

    def __init__(
        self, host: DeviceRefAttr, devices: Sequence[DeviceSpecAttr]
    ) -> None: ...
    @property
    def host(self) -> DeviceRefAttr: ...
    @property
    def devices(self) -> Sequence[DeviceSpecAttr]: ...

class InOutSignatureAttr(max._core.Attribute):
    """
    This attribute captures the intended semantics for each pointer-like
    operand of a primitive or kernel.
     - '.': Operand is a value.
     - 'i': Operand is only read.
     - 'o': Operand is only written.
     - 'm': Operand is mutated in-place.

    May be used to distinguish buffer-like operands on generic ops during
    compilation, in which case the attribute can be erased before conversion
    to MEF etc. May also be made available at runtime to help in runtime
    assertions and safety checks.

    Encoded into MEF as a string.

    Example:
    ```mlir
    #M.inout_sig<"ii.mo">
    ```
    """

    @overload
    def __init__(self, signature: str) -> None: ...
    @overload
    def __init__(self, signature: Sequence[InOutSemantics]) -> None: ...
    @property
    def signature(self) -> str: ...

class MultiLineStringAttr(max._core.Attribute):
    r"""
    An array of strings.

    Encoded into MEF as a multi-line null-terminated string using `\n` as
    the line terminator.

    Example: The attribute
    ```mlir
      #m<multiline["line1",
                   "line2"]>
    ```
    will appear in MEF as the the string "line1\nline2\n\0".
    """

    def __init__(
        self, value: Sequence[max._core.dialects.builtin.StringAttr]
    ) -> None: ...
    @property
    def value(self) -> Sequence[max._core.dialects.builtin.StringAttr]: ...

class PrimitiveArrayAttr(max._core.Attribute):
    """
    The `#M.primitives_array` attribute represents an array of primitive
    (boolean, integer, index, or floating point) data of equal size. The data is
    stored as a byte array with element types whose sizes are not multiples of
    bytes padded to the nearest byte. The underlying array is aligned to that
    byte size.

    Example:

    ```mlir
    // An array of integers.
    #M.primitives_array<si24: -2, 0, 2>

    // An array of floats.
    #M.primitives_array<bf16: 0.2, 1.2, 3.>

    // Boolean arrays use `i1` elements.
    #M.primitives_array<i1: true, false, true>
    ```
    """

    @overload
    def __init__(
        self, data: Sequence[int], element_type: max._core.Type
    ) -> None: ...
    @overload
    def __init__(
        self, data: Sequence[int], element_type: max._core.Type
    ) -> None: ...
    @property
    def data(self) -> Sequence[int]: ...
    @property
    def element_type(self) -> max._core.Type | None: ...

class SymbolRefArrayAttr(max._core.Attribute):
    def __init__(
        self, value: Sequence[max._core.dialects.builtin.SymbolRefAttr]
    ) -> None: ...
    @property
    def value(self) -> Sequence[max._core.dialects.builtin.SymbolRefAttr]: ...

class TargetInfoAttr(max._core.Attribute):
    """
    The `#M.target` attribute represents a compilation target configuration. It
    contains the target triple, microarchitecture, optional features, and
    derived data such as data layout and SIMD width.

    The user can also specify the default index bit width being used. This is
    important for GPUs where the hardware can support different index bit widths
    for cross compilation.

    The microarchitecture and features may encode version numbers, such as
    'sm_80' for CUDA compute capability 8.0. When deciding if a runtime host
    supports a compile time microarchitecutre or feature it may be necessary
    to compare these versions numerically rather than textually.

    Features are encoded in the form of "+feature1,+feature2".

    Example:
    ```mlir
    #M.target<triple="x86_64-unknown-linux-gnu", arch="znver3",
              features="+avx,+avx2", data_layout="p:64:64-i64:64:64",
              relocation_model="static", simd_bit_width=256, index_bit_width=64>
    #M.target<triple="nvptx64-nvidia-cuda", arch="sm_80">
    ```

    Can be (partially) represented at runtime by M::TargetInfo.

    The `accelerator_arch` contains the vendor name of the accelerator in
    lowercase (e.g. nvidia or amd) along with the compute capability (e.g. 80 or
    90). For example, a valid `accelerator_arch` is `nvidia:80` or `amdgpu:gfx942`.
    """

    def __init__(
        self,
        triple: max._core._TargetTriple,
        arch: str,
        features: str,
        data_layout: DataLayout,
        relocation_model: max._core._RelocationModel,
        simd_bit_width: int,
        index_bit_width: int,
        tune_cpu: str,
        accelerator_arch: str,
    ) -> None: ...
    @property
    def triple(self) -> max._core._TargetTriple: ...
    @property
    def arch(self) -> str: ...
    @property
    def features(self) -> str: ...
    @property
    def data_layout(self) -> DataLayout: ...
    @property
    def relocation_model(self) -> max._core._RelocationModel: ...
    @property
    def simd_bit_width(self) -> int: ...
    @property
    def index_bit_width(self) -> int | None: ...
    @property
    def tune_cpu(self) -> str: ...
    @property
    def accelerator_arch(self) -> str: ...

class TypeArrayAttr(max._core.Attribute):
    def __init__(self, value: Sequence[max._core.Type]) -> None: ...
    @property
    def value(self) -> Sequence[max._core.Type]: ...

class StringArrayAttr(max._core.Attribute):
    def __init__(
        self, value: Sequence[max._core.dialects.builtin.StringAttr]
    ) -> None: ...
    @property
    def value(self) -> Sequence[max._core.dialects.builtin.StringAttr]: ...

class AlignedBytesType(max._core.Type):
    """
    This type has no values and no runtime representation. It is intended only
    to be used as a type annotation on `dense_resource` attribute operands
    so as to convey a desired alignment. This is needed in two situations:
     - As a way to 'forward declare' the alignment for an attribute who's
       blob has not yet been parsed.
     - As a way to override the required alignment for an attribute without
       reallocating the underlying data.

    Example:
    ```mlir
    // An array of 4 uint8_ts with 16 byte alignment
    !M.aligned_bytes<4, align 16>
    ```
    """

    def __init__(self, size: int, align: int) -> None: ...
    @property
    def size(self) -> int: ...
    @property
    def align(self) -> int: ...

class ArrayType(max._core.Type):
    """
    The `!M.array` type represents one dimensional data of known length. This
    type implements `ShapedType` and can be used with `ElementsAttr`.

    Example:

    ```mlir
    // An array of integers.
    !M.array<32xi32>

    // An array of floats.
    !M.array<256xf64>
    ```
    """

    @overload
    def __init__(self, size: int, element_type: max._core.Type) -> None: ...
    @overload
    def __init__(self, size: int, element_type: max._core.Type) -> None: ...
    @property
    def size(self) -> int: ...
    @property
    def element_type(self) -> max._core.Type | None: ...

class DataLayout:
    pass

class IntArrayElementsAttr:
    pass
