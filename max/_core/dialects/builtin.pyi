# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

import enum
from collections.abc import Sequence
from typing import Callable, Protocol, overload

import max._core
from max.mlir import Location

# Many of the generated overloads for constructors are more specialized in
# C++ than they are in Python. For example, `int32_t` and `int64_t` and `size_t`
# all map to `int` in Python typing. It may not always be clear which of these
# overloads will be run for a given set of inputs (though in most cases it's the first one)
# but we disable mypy errors for shadowed overloads.
#
# mypy: disable-error-code="overload-cannot-match"

# DiagnosticHandlers aren't a thing that Python can reasonbly provided. In most cases
# these are automatically provided, but there are a few custom verifiers not covered yet.
# This binding prevents errors in those cases.
DiagnosticHandler = Callable

# This is a bug I haven't yet chased down in Nanobind's type renderings.
# - In some circumstances, `max._core.dialicts.mosh.ShapeType` is being shortened
#   to `h.ShapeType`, which obviously doesn't exist.
# - I haven't figured out a clean repro or workaround, I suspect it's some awkward case
#   with `nb_func_render_signature` because for instance adding garbage characters to the
#   `const_name` in the type caster will cause it to repro in different places.
# - For now, really hacky thing to work around.

class AffineMap:
    pass

class AffineMapAttr(max._core.Attribute):
    """
    Syntax:

    ```
    affine-map-attribute ::= `affine_map` `<` affine-map `>`
    ```

    Examples:

    ```mlir
    affine_map<(d0) -> (d0)>
    affine_map<(d0, d1, d2) -> (d0, d1)>
    ```
    """

    def __init__(self, value: AffineMap) -> None: ...
    @property
    def value(self) -> AffineMap: ...

class ArrayAttr(max._core.Attribute):
    """
    Syntax:

    ```
    array-attribute ::= `[` (attribute-value (`,` attribute-value)*)? `]`
    ```

    An array attribute is an attribute that represents a collection of attribute
    values.

    Examples:

    ```mlir
    []
    [10, i32]
    [affine_map<(d0, d1, d2) -> (d0, d1)>, i32, "string attribute"]
    ```
    """

    def __init__(self, value: Sequence[max._core.Attribute]) -> None: ...
    @property
    def value(self) -> Sequence[max._core.Attribute]: ...

class BFloat16Type(max._core.Type):
    def __init__(self) -> None: ...

class BlobAttr(Protocol):
    """
    This interface allows an attribute to expose a blob of data without more
    information. The data must be stored so that it can be accessed as a
    contiguous ArrayRef.
    """

    @property
    def data(self) -> Sequence[str]: ...

class BoolAttr(max._core.Attribute):
    def __init__(self, arg: bool, /) -> None: ...
    @property
    def value(self) -> bool: ...

class CallSiteLoc(max._core.Attribute):
    """
    Syntax:

    ```
    callsite-location ::= `callsite` `(` location `at` location `)`
    ```

    An instance of this location allows for representing a directed stack of
    location usages. This connects a location of a `callee` with the location
    of a `caller`.

    Example:

    ```mlir
    loc(callsite("foo" at "mysource.cc":10:8))
    ```
    """

    @overload
    def __init__(self, callee: Location, caller: Location) -> None: ...
    @overload
    def __init__(self, name: Location, frames: Sequence[Location]) -> None: ...
    @property
    def callee(self) -> Location: ...
    @property
    def caller(self) -> Location: ...

class ComplexType(max._core.Type):
    """
    Syntax:

    ```
    complex-type ::= `complex` `<` type `>`
    ```

    The value of `complex` type represents a complex number with a parameterized
    element type, which is composed of a real and imaginary value of that
    element type. The element must be a floating point or integer scalar type.

    #### Example:

    ```mlir
    complex<f32>
    complex<i32>
    ```
    """

    def __init__(self, element_type: max._core.Type) -> None: ...
    @property
    def element_type(self) -> max._core.Type | None: ...

class DenseArrayAttr(max._core.Attribute):
    """
    A dense array attribute is an attribute that represents a dense array of
    primitive element types. Contrary to DenseIntOrFPElementsAttr this is a
    flat unidimensional array which does not have a storage optimization for
    splat. This allows to expose the raw array through a C++ API as
    `ArrayRef<T>` for compatible types. The element type must be bool or an
    integer or float whose bitwidth is a multiple of 8. Bool elements are stored
    as bytes.

    This is the base class attribute. Access to C++ types is intended to be
    managed through the subclasses `DenseI8ArrayAttr`, `DenseI16ArrayAttr`,
    `DenseI32ArrayAttr`, `DenseI64ArrayAttr`, `DenseF32ArrayAttr`,
    and `DenseF64ArrayAttr`.

    Syntax:

    ```
    dense-array-attribute ::= `array` `<` (integer-type | float-type)
                                          (`:` tensor-literal)? `>`
    ```
    Examples:

    ```mlir
    array<i8>
    array<i32: 10, 42>
    array<f64: 42., 12.>
    ```

    When a specific subclass is used as argument of an operation, the
    declarative assembly will omit the type and print directly:

    ```mlir
    [1, 2, 3]
    ```
    """

    @overload
    def __init__(
        self, element_type: max._core.Type, size: int, raw_data: Sequence[str]
    ) -> None: ...
    @overload
    def __init__(
        self, element_type: max._core.Type, size: int, raw_data: Sequence[str]
    ) -> None: ...
    @property
    def element_type(self) -> max._core.Type | None: ...
    @property
    def size(self) -> int: ...
    @property
    def raw_data(self) -> Sequence[str]: ...

class DenseElementsAttr(max._core.Attribute):
    pass

class DenseIntOrFPElementsAttr(max._core.Attribute):
    """
    Syntax:

    ```
    tensor-literal ::= integer-literal | float-literal | bool-literal | [] | [tensor-literal (, tensor-literal)* ]
    dense-intorfloat-elements-attribute ::= `dense` `<` tensor-literal `>` `:`
                                            ( tensor-type | vector-type )
    ```

    A dense int-or-float elements attribute is an elements attribute containing
    a densely packed vector or tensor of integer or floating-point values. The
    element type of this attribute is required to be either an `IntegerType` or
    a `FloatType`.

    Examples:

    ```
    // A splat tensor of integer values.
    dense<10> : tensor<2xi32>
    // A tensor of 2 float32 elements.
    dense<[10.0, 11.0]> : tensor<2xf32>
    ```
    """

class DenseResourceElementsAttr(max._core.Attribute):
    pass

class DenseStringElementsAttr(max._core.Attribute):
    """
    Syntax:

    ```
    dense-string-elements-attribute ::= `dense` `<` attribute-value `>` `:`
                                        ( tensor-type | vector-type )
    ```

    A dense string elements attribute is an elements attribute containing a
    densely packed vector or tensor of string values. There are no restrictions
    placed on the element type of this attribute, enabling the use of dialect
    specific string types.

    Examples:

    ```
    // A splat tensor of strings.
    dense<"example"> : tensor<2x!foo.string>
    // A tensor of 2 string elements.
    dense<["example1", "example2"]> : tensor<2x!foo.string>
    ```
    """

    def __init__(self, type: ShapedType, values: Sequence[str]) -> None: ...

class DictionaryAttr(max._core.Attribute):
    """
    Syntax:

    ```
    dictionary-attribute ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
    ```

    A dictionary attribute is an attribute that represents a sorted collection of
    named attribute values. The elements are sorted by name, and each name must be
    unique within the collection.

    Examples:

    ```mlir
    {}
    {attr_name = "string attribute"}
    {int_attr = 10, "string attr name" = "string attribute"}
    ```
    """

    def __init__(
        self, value: Sequence[max._core.NamedAttribute] = []
    ) -> None: ...
    @property
    def value(self) -> Sequence[max._core.NamedAttribute]: ...

class ElementsAttr(Protocol):
    """
    This interface is used for attributes that contain the constant elements of
    a tensor or vector type. It allows for opaquely interacting with the
    elements of the underlying attribute, and most importantly allows for
    accessing the element values (including iteration) in any of the C++ data
    types supported by the underlying attribute.

    An attribute implementing this interface can expose the supported data types
    in two steps:

    * Define the set of iterable C++ data types:

    An attribute may define the set of iterable types by providing a definition
    of tuples `ContiguousIterableTypesT` and/or `NonContiguousIterableTypesT`.

    -  `ContiguousIterableTypesT` should contain types which can be iterated
       contiguously. A contiguous range is an array-like range, such as
       ArrayRef, where all of the elements are layed out sequentially in memory.

    -  `NonContiguousIterableTypesT` should contain types which can not be
       iterated contiguously. A non-contiguous range implies no contiguity,
       whose elements may even be materialized when indexing, such as the case
       for a mapped_range.

    As an example, consider an attribute that only contains i64 elements, with
    the elements being stored within an ArrayRef. This attribute could
    potentially define the iterable types as so:

    ```c++
    using ContiguousIterableTypesT = std::tuple<uint64_t>;
    using NonContiguousIterableTypesT = std::tuple<APInt, Attribute>;
    ```

    * Provide a `FailureOr<iterator> try_value_begin_impl(OverloadToken<T>) const`
      overload for each iterable type

    These overloads should return an iterator to the start of the range for the
    respective iterable type or fail if the type cannot be iterated. Consider
    the example i64 elements attribute described in the previous section. This
    attribute may define the value_begin_impl overloads like so:

    ```c++
    /// Provide begin iterators for the various iterable types.
    /// * uint64_t
    FailureOr<const uint64_t *>
    value_begin_impl(OverloadToken<uint64_t>) const {
      return getElements().begin();
    }
    /// * APInt
    auto value_begin_impl(OverloadToken<llvm::APInt>) const {
      auto it = llvm::map_range(getElements(), [=](uint64_t value) {
        return llvm::APInt(/*numBits=*/64, value);
      }).begin();
      return FailureOr<decltype(it)>(std::move(it));
    }
    /// * Attribute
    auto value_begin_impl(OverloadToken<mlir::Attribute>) const {
      mlir::Type elementType = getShapedType().getElementType();
      auto it = llvm::map_range(getElements(), [=](uint64_t value) {
        return mlir::IntegerAttr::get(elementType,
                                      llvm::APInt(/*numBits=*/64, value));
      }).begin();
      return FailureOr<decltype(it)>(std::move(it));
    }
    ```

    After the above, ElementsAttr will now be able to iterate over elements
    using each of the registered iterable data types:

    ```c++
    ElementsAttr attr = myI64ElementsAttr;

    // We can access value ranges for the data types via `getValues<T>`.
    for (uint64_t value : attr.getValues<uint64_t>())
      ...;
    for (llvm::APInt value : attr.getValues<llvm::APInt>())
      ...;
    for (mlir::IntegerAttr value : attr.getValues<mlir::IntegerAttr>())
      ...;

    // We can also access the value iterators directly.
    auto it = attr.value_begin<uint64_t>(), e = attr.value_end<uint64_t>();
    for (; it != e; ++it) {
      uint64_t value = *it;
      ...
    }
    ```

    ElementsAttr also supports failable access to iterators and ranges. This
    allows for safely checking if the attribute supports the data type, and can
    also allow for code to have fast paths for native data types.

    ```c++
    // Using `tryGetValues<T>`, we can also safely handle when the attribute
    // doesn't support the data type.
    if (auto range = attr.tryGetValues<uint64_t>()) {
      for (uint64_t value : *range)
        ...;
      return;
    }

    // We can also access the begin iterator safely, by using `try_value_begin`.
    if (auto safeIt = attr.try_value_begin<uint64_t>()) {
      auto it = *safeIt, e = attr.value_end<uint64_t>();
      for (; it != e; ++it) {
        uint64_t value = *it;
        ...
      }
      return;
    }
    ```
    """

    @property
    def type(self) -> max._core.Type | None: ...
    @property
    def splat(self) -> bool: ...
    @property
    def shaped_type(self) -> ShapedType: ...
    def get_values_impl(
        self, arg: max._core.TypeID, /
    ) -> _ElementsAttrIndexer | None: ...

class FileLineColRange(max._core.Attribute):
    """
    Syntax:

    ```
    filelinecol-location ::= string-literal `:` integer-literal `:`
                             integer-literal
                             (`to` (integer-literal ?) `:` integer-literal ?)
    ```

    An instance of this location represents a tuple of file, start and end line
    number, and start and end column number. It allows for the following
    configurations:

    *   A single file line location: `file:line`;
    *   A single file line col location: `file:line:column`;
    *   A single line range: `file:line:column to :column`;
    *   A single file range: `file:line:column to line:column`;

    Example:

    ```mlir
    loc("mysource.cc":10:8 to 12:18)
    ```
    """

    @overload
    def __init__(self, filename: StringAttr) -> None: ...
    @overload
    def __init__(self, filename: StringAttr, line: int) -> None: ...
    @overload
    def __init__(
        self, filename: StringAttr, line: int, column: int
    ) -> None: ...
    @overload
    def __init__(
        self, filename: str, start_line: int, start_column: int
    ) -> None: ...
    @overload
    def __init__(
        self,
        filename: StringAttr,
        line: int,
        start_column: int,
        end_column: int,
    ) -> None: ...
    @overload
    def __init__(
        self,
        filename: StringAttr,
        start_line: int,
        start_column: int,
        end_line: int,
        end_column: int,
    ) -> None: ...
    @overload
    def __init__(
        self,
        filename: str,
        start_line: int,
        start_column: int,
        end_line: int,
        end_column: int,
    ) -> None: ...

class FlatSymbolRefAttr(max._core.Attribute):
    pass

class Float128Type(max._core.Type):
    def __init__(self) -> None: ...

class Float16Type(max._core.Type):
    def __init__(self) -> None: ...

class Float32Type(max._core.Type):
    def __init__(self) -> None: ...

class Float4E2M1FNType(max._core.Type):
    """
    An 4-bit floating point type with 1 sign bit, 2 bits exponent and 1 bit
    mantissa. This is not a standard type as defined by IEEE-754, but it
    follows similar conventions with the following characteristics:

      * bit encoding: S1E2M1
      * exponent bias: 1
      * infinities: Not supported
      * NaNs: Not supported
      * denormals when exponent is 0

    Open Compute Project (OCP) microscaling formats (MX) specification:
    https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    """

    def __init__(self) -> None: ...

class Float64Type(max._core.Type):
    def __init__(self) -> None: ...

class Float6E2M3FNType(max._core.Type):
    """
    An 6-bit floating point type with 1 sign bit, 2 bits exponent and 3 bits
    mantissa. This is not a standard type as defined by IEEE-754, but it
    follows similar conventions with the following characteristics:

      * bit encoding: S1E2M3
      * exponent bias: 1
      * infinities: Not supported
      * NaNs: Not supported
      * denormals when exponent is 0

    Open Compute Project (OCP) microscaling formats (MX) specification:
    https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    """

    def __init__(self) -> None: ...

class Float6E3M2FNType(max._core.Type):
    """
    An 6-bit floating point type with 1 sign bit, 3 bits exponent and 2 bits
    mantissa. This is not a standard type as defined by IEEE-754, but it
    follows similar conventions with the following characteristics:

      * bit encoding: S1E3M2
      * exponent bias: 3
      * infinities: Not supported
      * NaNs: Not supported
      * denormals when exponent is 0

    Open Compute Project (OCP) microscaling formats (MX) specification:
    https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    """

    def __init__(self) -> None: ...

class Float80Type(max._core.Type):
    def __init__(self) -> None: ...

class Float8E3M4Type(max._core.Type):
    """
    An 8-bit floating point type with 1 sign bit, 3 bits exponent and 4 bits
    mantissa. This is not a standard type as defined by IEEE-754, but it
    follows similar conventions with the following characteristics:

      * bit encoding: S1E3M4
      * exponent bias: 3
      * infinities: supported with exponent set to all 1s and mantissa 0s
      * NaNs: supported with exponent bits set to all 1s and mantissa values of
        {0,1}⁴ except S.111.0000
      * denormals when exponent is 0
    """

    def __init__(self) -> None: ...

class Float8E4M3B11FNUZType(max._core.Type):
    """
    An 8-bit floating point type with 1 sign bit, 4 bits exponent and 3 bits
    mantissa. This is not a standard type as defined by IEEE-754, but it follows
    similar conventions, with the exception that there are no infinity values,
    no negative zero, and only one NaN representation. This type has the
    following characteristics:

      * bit encoding: S1E4M3
      * exponent bias: 11
      * infinities: Not supported
      * NaNs: Supported with sign bit set to 1, exponent bits and mantissa bits set to all 0s
      * denormals when exponent is 0

    Related to: https://dl.acm.org/doi/10.5555/3454287.3454728
    """

    def __init__(self) -> None: ...

class Float8E4M3FNType(max._core.Type):
    """
    An 8-bit floating point type with 1 sign bit, 4 bits exponent and 3 bits
    mantissa. This is not a standard type as defined by IEEE-754, but it follows
    similar conventions, with the exception that there are no infinity values
    and only two NaN representations. This type has the following
    characteristics:

      * bit encoding: S1E4M3
      * exponent bias: 7
      * infinities: Not supported
      * NaNs: supported with exponent bits and mantissa bits set to all 1s
      * denormals when exponent is 0

    Described in: https://arxiv.org/abs/2209.05433
    """

    def __init__(self) -> None: ...

class Float8E4M3FNUZType(max._core.Type):
    """
    An 8-bit floating point type with 1 sign bit, 4 bits exponent and 3 bits
    mantissa. This is not a standard type as defined by IEEE-754, but it follows
    similar conventions, with the exception that there are no infinity values,
    no negative zero, and only one NaN representation. This type has the
    following characteristics:

      * bit encoding: S1E4M3
      * exponent bias: 8
      * infinities: Not supported
      * NaNs: Supported with sign bit set to 1, exponent bits and mantissa bits set to all 0s
      * denormals when exponent is 0

    Described in: https://arxiv.org/abs/2209.05433
    """

    def __init__(self) -> None: ...

class Float8E4M3Type(max._core.Type):
    """
    An 8-bit floating point type with 1 sign bit, 4 bits exponent and 3 bits
    mantissa. This is not a standard type as defined by IEEE-754, but it
    follows similar conventions with the following characteristics:

      * bit encoding: S1E4M3
      * exponent bias: 7
      * infinities: supported with exponent set to all 1s and mantissa 0s
      * NaNs: supported with exponent bits set to all 1s and mantissa of
        (001, 010, 011, 100, 101, 110, 111)
      * denormals when exponent is 0
    """

    def __init__(self) -> None: ...

class Float8E5M2FNUZType(max._core.Type):
    """
    An 8-bit floating point type with 1 sign bit, 5 bits exponent and 2 bits
    mantissa. This is not a standard type as defined by IEEE-754, but it follows
    similar conventions, with the exception that there are no infinity values,
    no negative zero, and only one NaN representation. This type has the
    following characteristics:

      * bit encoding: S1E5M2
      * exponent bias: 16
      * infinities: Not supported
      * NaNs: Supported with sign bit set to 1, exponent bits and mantissa bits set to all 0s
      * denormals when exponent is 0

    Described in: https://arxiv.org/abs/2206.02915
    """

    def __init__(self) -> None: ...

class Float8E5M2Type(max._core.Type):
    """
    An 8-bit floating point type with 1 sign bit, 5 bits exponent and 2 bits
    mantissa. This is not a standard type as defined by IEEE-754, but it
    follows similar conventions with the following characteristics:

      * bit encoding: S1E5M2
      * exponent bias: 15
      * infinities: supported with exponent set to all 1s and mantissa 0s
      * NaNs: supported with exponent bits set to all 1s and mantissa of
        (01, 10, or 11)
      * denormals when exponent is 0

    Described in: https://arxiv.org/abs/2209.05433
    """

    def __init__(self) -> None: ...

class Float8E8M0FNUType(max._core.Type):
    """
    An 8-bit floating point type with no sign bit, 8 bits exponent and no
    mantissa. This is not a standard type as defined by IEEE-754; it is intended
    to be used for representing scaling factors, so it cannot represent zeros
    and negative numbers. The values it can represent are powers of two in the
    range [-127,127] and NaN.

      * bit encoding: S0E8M0
      * exponent bias: 127
      * infinities: Not supported
      * NaNs: Supported with all bits set to 1
      * denormals: Not supported

    Open Compute Project (OCP) microscaling formats (MX) specification:
    https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    """

    def __init__(self) -> None: ...

class FloatAttr(max._core.Attribute):
    """
    Syntax:

    ```
    float-attribute ::= (float-literal (`:` float-type)?)
                      | (hexadecimal-literal `:` float-type)
    ```

    A float attribute is a literal attribute that represents a floating point
    value of the specified [float type](#floating-point-types). It can be
    represented in the hexadecimal form where the hexadecimal value is
    interpreted as bits of the underlying binary representation. This form is
    useful for representing infinity and NaN floating point values. To avoid
    confusion with integer attributes, hexadecimal literals _must_ be followed
    by a float type to define a float attribute.

    Examples:

    ```
    42.0         // float attribute defaults to f64 type
    42.0 : f32   // float attribute of f32 type
    0x7C00 : f16 // positive infinity
    0x7CFF : f16 // NaN (one of possible values)
    42 : f32     // Error: expected integer type
    ```
    """

    @overload
    def __init__(self, type: max._core.Type, value: float) -> None: ...
    @overload
    def __init__(self, type: max._core.Type, value: float) -> None: ...
    @property
    def type(self) -> max._core.Type | None: ...
    @property
    def value(self) -> float: ...

class FloatTF32Type(max._core.Type):
    def __init__(self) -> None: ...

class FloatType(Protocol):
    """
    This type interface should be implemented by all floating-point types. It
    defines the LLVM APFloat semantics and provides a few helper functions.
    """

    def scale_element_bitwidth(self, arg: int, /) -> FloatType: ...

class FunctionType(max._core.Type):
    """
    Syntax:

    ```
    // Function types may have multiple results.
    function-result-type ::= type-list-parens | non-function-type
    function-type ::= type-list-parens `->` function-result-type
    ```

    The function type can be thought of as a function signature. It consists of
    a list of formal parameter types and a list of formal result types.

    #### Example:

    ```mlir
    func.func @add_one(%arg0 : i64) -> i64 {
      %c1 = arith.constant 1 : i64
      %0 = arith.addi %arg0, %c1 : i64
      return %0 : i64
    }
    ```
    """

    def __init__(
        self,
        inputs: Sequence[max._core.Type] = [],
        results: Sequence[max._core.Type] = [],
    ) -> None: ...
    @property
    def inputs(self) -> Sequence[max._core.Type]: ...
    @property
    def results(self) -> Sequence[max._core.Type]: ...

class FusedLoc(max._core.Attribute):
    """
    Syntax:

    ```
    fusion-metadata ::= `<` attribute-value `>`
    fused-location ::= `fused` fusion-metadata? `[` (location (`,` location)* )? `]`
    ```

    An instance of a `fused` location represents a grouping of several other
    source locations, with optional metadata that describes the context of the
    fusion. There are many places within a compiler in which several constructs
    may be fused together, e.g. pattern rewriting, that normally result partial
    or even total loss of location information. With `fused` locations, this is
    a non-issue.

    Example:

    ```mlir
    loc(fused["mysource.cc":10:8, "mysource.cc":22:8])
    loc(fused<"CSE">["mysource.cc":10:8, "mysource.cc":22:8])
    ```
    """

    def __init__(
        self, locations: Sequence[Location], metadata: max._core.Attribute
    ) -> None: ...
    @property
    def locations(self) -> Sequence[Location]: ...
    @property
    def metadata(self) -> max._core.Attribute | None: ...

class IndexType(max._core.Type):
    """
    Syntax:

    ```
    // Target word-sized integer.
    index-type ::= `index`
    ```

    The index type is a signless integer whose size is equal to the natural
    machine word of the target ( [rationale](../../Rationale/Rationale/#integer-signedness-semantics) )
    and is used by the affine constructs in MLIR.

    **Rationale:** integers of platform-specific bit widths are practical to
    express sizes, dimensionalities and subscripts.
    """

    def __init__(self) -> None: ...

class IntegerAttr(max._core.Attribute):
    """
    Syntax:

    ```
    integer-attribute ::= (integer-literal ( `:` (index-type | integer-type) )?)
                          | `true` | `false`
    ```

    An integer attribute is a literal attribute that represents an integral
    value of the specified integer or index type. `i1` integer attributes are
    treated as `boolean` attributes, and use a unique assembly format of either
    `true` or `false` depending on the value. The default type for non-boolean
    integer attributes, if a type is not specified, is signless 64-bit integer.

    Examples:

    ```mlir
    10 : i32
    10    // : i64 is implied here.
    true  // A bool, i.e. i1, value.
    false // A bool, i.e. i1, value.
    ```
    """

    @property
    def type(self) -> max._core.Type | None: ...
    @property
    def value(self) -> int: ...
    def __init__(self, type: IntegerType, value: int = 0) -> None: ...

class IntegerSet:
    pass

class IntegerSetAttr(max._core.Attribute):
    """
    Syntax:

    ```
    integer-set-attribute ::= `affine_set` `<` integer-set `>`
    ```

    Examples:

    ```mlir
    affine_set<(d0) : (d0 - 2 >= 0)>
    ```
    """

    def __init__(self, value: IntegerSet) -> None: ...
    @property
    def value(self) -> IntegerSet: ...

class IntegerType(max._core.Type):
    """
    Syntax:

    ```
    // Sized integers like i1, i4, i8, i16, i32.
    signed-integer-type ::= `si` [1-9][0-9]*
    unsigned-integer-type ::= `ui` [1-9][0-9]*
    signless-integer-type ::= `i` [1-9][0-9]*
    integer-type ::= signed-integer-type |
                     unsigned-integer-type |
                     signless-integer-type
    ```

    Integer types have a designated bit width and may optionally have signedness
    semantics.

    **Rationale:** low precision integers (like `i2`, `i4` etc) are useful for
    low-precision inference chips, and arbitrary precision integers are useful
    for hardware synthesis (where a 13 bit multiplier is a lot cheaper/smaller
    than a 16 bit one).
    """

    def __init__(
        self,
        width: int,
        signedness: SignednessSemantics = SignednessSemantics.signless,
    ) -> None: ...
    @property
    def width(self) -> int: ...
    @property
    def signedness(self) -> SignednessSemantics: ...

class MemRefElementTypeInterface(Protocol):
    """
    Indication that this type can be used as element in memref types.

    Implementing this interface establishes a contract between this type and the
    memref type indicating that this type can be used as element of ranked or
    unranked memrefs. The type is expected to:

      - model an entity stored in memory;
      - have non-zero size.

    For example, scalar values such as integers can implement this interface,
    but indicator types such as `void` or `unit` should not.

    The interface currently has no methods and is used by types to opt into
    being memref elements. This may change in the future, in particular to
    require types to provide their size or alignment given a data layout.
    """

class MemRefLayoutAttrInterface(Protocol):
    """
    This interface is used for attributes that can represent the MemRef type's
    layout semantics, such as dimension order in the memory, strides and offsets.
    Such a layout attribute should be representable as a
    [semi-affine map](Affine.md/#semi-affine-maps).

    Note: the MemRef type's layout is assumed to represent simple strided buffer
    layout. For more complicated case, like sparse storage buffers,
    it is preferable to use a separate type with a more specific layout, rather
    than introducing extra complexity to the builtin MemRef type.
    """

    @property
    def affine_map(self) -> AffineMap: ...
    @property
    def identity(self) -> bool: ...
    def verify_layout(
        self, arg0: Sequence[int], arg1: DiagnosticHandler, /
    ) -> bool: ...
    def get_strides_and_offset(
        self, arg0: Sequence[int], arg1: Sequence[int], arg2: int, /
    ) -> bool: ...

class MemRefType(max._core.Type):
    """
    Syntax:

    ```
    layout-specification ::= attribute-value
    memory-space ::= attribute-value
    memref-type ::= `memref` `<` dimension-list-ranked type
                    (`,` layout-specification)? (`,` memory-space)? `>`
    ```

    A `memref` type is a reference to a region of memory (similar to a buffer
    pointer, but more powerful). The buffer pointed to by a memref can be
    allocated, aliased and deallocated. A memref can be used to read and write
    data from/to the memory region which it references. Memref types use the
    same shape specifier as tensor types. Note that `memref<f32>`,
    `memref<0 x f32>`, `memref<1 x 0 x f32>`, and `memref<0 x 1 x f32>` are all
    different types.

    A `memref` is allowed to have an unknown rank (e.g. `memref<*xf32>`). The
    purpose of unranked memrefs is to allow external library functions to
    receive memref arguments of any rank without versioning the functions based
    on the rank. Other uses of this type are disallowed or will have undefined
    behavior.

    Are accepted as elements:

    - built-in integer types;
    - built-in index type;
    - built-in floating point types;
    - built-in vector types with elements of the above types;
    - another memref type;
    - any other type implementing `MemRefElementTypeInterface`.

    ##### Layout

    A memref may optionally have a layout that indicates how indices are
    transformed from the multi-dimensional form into a linear address. The
    layout must avoid internal aliasing, i.e., two distinct tuples of
    _in-bounds_ indices must be pointing to different elements in memory. The
    layout is an attribute that implements `MemRefLayoutAttrInterface`. The
    bulitin dialect offers two kinds of layouts: strided and affine map, each
    of which is available as an attribute. Other attributes may be used to
    represent the layout as long as they can be converted to a
    [semi-affine map](Affine.md/#semi-affine-maps) and implement the required
    interface. Users of memref are expected to fallback to the affine
    representation when handling unknown memref layouts. Multi-dimensional
    affine forms are interpreted in _row-major_ fashion.

    In absence of an explicit layout, a memref is considered to have a
    multi-dimensional identity affine map layout.  Identity layout maps do not
    contribute to the MemRef type identification and are discarded on
    construction. That is, a type with an explicit identity map is
    `memref<?x?xf32, (i,j)->(i,j)>` is strictly the same as the one without a
    layout, `memref<?x?xf32>`.

    ##### Affine Map Layout

    The layout may be represented directly as an affine map from the index space
    to the storage space. For example, the following figure shows an index map
    which maps a 2-dimensional index from a 2x2 index space to a 3x3 index
    space, using symbols `S0` and `S1` as offsets.

    ![Index Map Example](/includes/img/index-map.svg)

    Semi-affine maps are sufficiently flexible to represent a wide variety of
    dense storage layouts, including row- and column-major and tiled:

    ```mlir
    // MxN matrix stored in row major layout in memory:
    #layout_map_row_major = (i, j) -> (i, j)

    // MxN matrix stored in column major layout in memory:
    #layout_map_col_major = (i, j) -> (j, i)

    // MxN matrix stored in a 2-d blocked/tiled layout with 64x64 tiles.
    #layout_tiled = (i, j) -> (i floordiv 64, j floordiv 64, i mod 64, j mod 64)
    ```

    ##### Strided Layout

    Memref layout can be expressed using strides to encode the distance, in
    number of elements, in (linear) memory between successive entries along a
    particular dimension. For example, a row-major strided layout for
    `memref<2x3x4xf32>` is `strided<[12, 4, 1]>`, where the last dimension is
    contiguous as indicated by the unit stride and the remaining strides are
    products of the sizes of faster-variying dimensions. Strided layout can also
    express non-contiguity, e.g., `memref<2x3, strided<[6, 2]>>` only accesses
    even elements of the dense consecutive storage along the innermost
    dimension.

    The strided layout supports an optional _offset_ that indicates the
    distance, in the number of elements, between the beginning of the memref
    and the first accessed element. When omitted, the offset is considered to
    be zero. That is, `memref<2, strided<[2], offset: 0>>` and
    `memref<2, strided<[2]>>` are strictly the same type.

    Both offsets and strides may be _dynamic_, that is, unknown at compile time.
    This is represented by using a question mark (`?`) instead of the value in
    the textual form of the IR.

    The strided layout converts into the following canonical one-dimensional
    affine form through explicit linearization:

    ```mlir
    affine_map<(d0, ... dN)[offset, stride0, ... strideN] ->
                (offset + d0 * stride0 + ... dN * strideN)>
    ```

    Therefore, it is never subject to the implicit row-major layout
    interpretation.

    ##### Codegen of Unranked Memref

    Using unranked memref in codegen besides the case mentioned above is highly
    discouraged. Codegen is concerned with generating loop nests and specialized
    instructions for high-performance, unranked memref is concerned with hiding
    the rank and thus, the number of enclosing loops required to iterate over
    the data. However, if there is a need to code-gen unranked memref, one
    possible path is to cast into a static ranked type based on the dynamic
    rank. Another possible path is to emit a single while loop conditioned on a
    linear index and perform delinearization of the linear index to a dynamic
    array containing the (unranked) indices. While this is possible, it is
    expected to not be a good idea to perform this during codegen as the cost
    of the translations is expected to be prohibitive and optimizations at this
    level are not expected to be worthwhile. If expressiveness is the main
    concern, irrespective of performance, passing unranked memrefs to an
    external C++ library and implementing rank-agnostic logic there is expected
    to be significantly simpler.

    Unranked memrefs may provide expressiveness gains in the future and help
    bridge the gap with unranked tensors. Unranked memrefs will not be expected
    to be exposed to codegen but one may query the rank of an unranked memref
    (a special op will be needed for this purpose) and perform a switch and cast
    to a ranked memref as a prerequisite to codegen.

    Example:

    ```mlir
    // With static ranks, we need a function for each possible argument type
    %A = alloc() : memref<16x32xf32>
    %B = alloc() : memref<16x32x64xf32>
    call @helper_2D(%A) : (memref<16x32xf32>)->()
    call @helper_3D(%B) : (memref<16x32x64xf32>)->()

    // With unknown rank, the functions can be unified under one unranked type
    %A = alloc() : memref<16x32xf32>
    %B = alloc() : memref<16x32x64xf32>
    // Remove rank info
    %A_u = memref_cast %A : memref<16x32xf32> -> memref<*xf32>
    %B_u = memref_cast %B : memref<16x32x64xf32> -> memref<*xf32>
    // call same function with dynamic ranks
    call @helper(%A_u) : (memref<*xf32>)->()
    call @helper(%B_u) : (memref<*xf32>)->()
    ```

    The core syntax and representation of a layout specification is a
    [semi-affine map](Affine.md/#semi-affine-maps). Additionally,
    syntactic sugar is supported to make certain layout specifications more
    intuitive to read. For the moment, a `memref` supports parsing a strided
    form which is converted to a semi-affine map automatically.

    The memory space of a memref is specified by a target-specific attribute.
    It might be an integer value, string, dictionary or custom dialect attribute.
    The empty memory space (attribute is None) is target specific.

    The notionally dynamic value of a memref value includes the address of the
    buffer allocated, as well as the symbols referred to by the shape, layout
    map, and index maps.

    Examples of memref static type

    ```mlir
    // Identity index/layout map
    #identity = affine_map<(d0, d1) -> (d0, d1)>

    // Column major layout.
    #col_major = affine_map<(d0, d1, d2) -> (d2, d1, d0)>

    // A 2-d tiled layout with tiles of size 128 x 256.
    #tiled_2d_128x256 = affine_map<(d0, d1) -> (d0 div 128, d1 div 256, d0 mod 128, d1 mod 256)>

    // A tiled data layout with non-constant tile sizes.
    #tiled_dynamic = affine_map<(d0, d1)[s0, s1] -> (d0 floordiv s0, d1 floordiv s1,
                                 d0 mod s0, d1 mod s1)>

    // A layout that yields a padding on two at either end of the minor dimension.
    #padded = affine_map<(d0, d1) -> (d0, (d1 + 2) floordiv 2, (d1 + 2) mod 2)>


    // The dimension list "16x32" defines the following 2D index space:
    //
    //   { (i, j) : 0 <= i < 16, 0 <= j < 32 }
    //
    memref<16x32xf32, #identity>

    // The dimension list "16x4x?" defines the following 3D index space:
    //
    //   { (i, j, k) : 0 <= i < 16, 0 <= j < 4, 0 <= k < N }
    //
    // where N is a symbol which represents the runtime value of the size of
    // the third dimension.
    //
    // %N here binds to the size of the third dimension.
    %A = alloc(%N) : memref<16x4x?xf32, #col_major>

    // A 2-d dynamic shaped memref that also has a dynamically sized tiled
    // layout. The memref index space is of size %M x %N, while %B1 and %B2
    // bind to the symbols s0, s1 respectively of the layout map #tiled_dynamic.
    // Data tiles of size %B1 x %B2 in the logical space will be stored
    // contiguously in memory. The allocation size will be
    // (%M ceildiv %B1) * %B1 * (%N ceildiv %B2) * %B2 f32 elements.
    %T = alloc(%M, %N) [%B1, %B2] : memref<?x?xf32, #tiled_dynamic>

    // A memref that has a two-element padding at either end. The allocation
    // size will fit 16 * 64 float elements of data.
    %P = alloc() : memref<16x64xf32, #padded>

    // Affine map with symbol 's0' used as offset for the first dimension.
    #imapS = affine_map<(d0, d1) [s0] -> (d0 + s0, d1)>
    // Allocate memref and bind the following symbols:
    // '%n' is bound to the dynamic second dimension of the memref type.
    // '%o' is bound to the symbol 's0' in the affine map of the memref type.
    %n = ...
    %o = ...
    %A = alloc (%n)[%o] : <16x?xf32, #imapS>
    ```
    """

    @overload
    def __init__(
        self,
        shape: Sequence[int],
        element_type: max._core.Type,
        layout: MemRefLayoutAttrInterface = ...,
        memory_space: max._core.Attribute = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        shape: Sequence[int],
        element_type: max._core.Type,
        map: AffineMap = ...,
        memory_space: max._core.Attribute = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        shape: Sequence[int],
        element_type: max._core.Type,
        map: AffineMap,
        memory_space_ind: int,
    ) -> None: ...
    @property
    def shape(self) -> Sequence[int]: ...
    @property
    def element_type(self) -> max._core.Type | None: ...
    @property
    def layout(self) -> MemRefLayoutAttrInterface: ...
    @property
    def memory_space(self) -> max._core.Attribute | None: ...

class ModuleOp(max._core.Operation):
    """
    A `module` represents a top-level container operation. It contains a single
    [graph region](../LangRef.md#control-flow-and-ssacfg-regions) containing a single block
    which can contain any operations and does not have a terminator. Operations
    within this region cannot implicitly capture values defined outside the module,
    i.e. Modules are [IsolatedFromAbove](../Traits.md#isolatedfromabove). Modules have
    an optional [symbol name](../SymbolsAndSymbolTables.md) which can be used to refer
    to them in operations.

    Example:

    ```mlir
    module {
      func.func @foo()
    }
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        name: str | None = None,
    ) -> None: ...
    @overload
    def __init__(self, location: Location, name: str | None = None) -> None: ...
    @property
    def sym_name(self) -> str | None: ...
    @sym_name.setter
    def sym_name(self, arg: StringAttr, /) -> None: ...
    @property
    def sym_visibility(self) -> str | None: ...
    @sym_visibility.setter
    def sym_visibility(self, arg: StringAttr, /) -> None: ...
    @property
    def body(self) -> max._core.Block: ...

class NameLoc(max._core.Attribute):
    """
    Syntax:

    ```
    name-location ::= string-literal (`(` location `)`)?
    ```

    An instance of this location allows for attaching a name to a child location.
    This can be useful for representing the locations of variable, or node,
    definitions.

    #### Example:

    ```mlir
    loc("CSE"("mysource.cc":10:8))
    ```
    """

    @overload
    def __init__(self, name: StringAttr) -> None: ...
    @overload
    def __init__(self, name: StringAttr, child_loc: Location) -> None: ...
    @property
    def name(self) -> StringAttr: ...
    @property
    def child_loc(self) -> Location: ...

class NoneType(max._core.Type):
    """
    Syntax:

    ```
    none-type ::= `none`
    ```

    NoneType is a unit type, i.e. a type with exactly one possible value, where
    its value does not have a defined dynamic representation.

    #### Example:

    ```mlir
    func.func @none_type() {
      %none_val = "foo.unknown_op"() : () -> none
      return
    }
    ```
    """

    def __init__(self) -> None: ...

class OpaqueAttr(max._core.Attribute):
    """
    Syntax:

    ```
    opaque-attribute ::= dialect-namespace `<` attr-data `>`
    ```

    Opaque attributes represent attributes of non-registered dialects. These are
    attribute represented in their raw string form, and can only usefully be
    tested for attribute equality.

    Examples:

    ```mlir
    #dialect<"opaque attribute data">
    ```
    """

    def __init__(
        self, dialect: StringAttr, attr_data: str, type: max._core.Type
    ) -> None: ...
    @property
    def dialect_namespace(self) -> StringAttr: ...
    @property
    def attr_data(self) -> str: ...
    @property
    def type(self) -> max._core.Type | None: ...

class OpaqueType(max._core.Type):
    """
    Syntax:

    ```
    opaque-type ::= `opaque` `<` type `>`
    ```

    Opaque types represent types of non-registered dialects. These are types
    represented in their raw string form, and can only usefully be tested for
    type equality.

    #### Example:

    ```mlir
    opaque<"llvm", "struct<(i32, float)>">
    opaque<"pdl", "value">
    ```
    """

    def __init__(
        self, dialect_namespace: StringAttr, type_data: str = ""
    ) -> None: ...
    @property
    def dialect_namespace(self) -> StringAttr: ...
    @property
    def type_data(self) -> str: ...

class RankedTensorType(max._core.Type):
    """
    Syntax:

    ```
    tensor-type ::= `tensor` `<` dimension-list type (`,` encoding)? `>`
    dimension-list ::= (dimension `x`)*
    dimension ::= `?` | decimal-literal
    encoding ::= attribute-value
    ```

    Values with tensor type represents aggregate N-dimensional data values, and
    have a known element type and a fixed rank with a list of dimensions. Each
    dimension may be a static non-negative decimal constant or be dynamically
    determined (indicated by `?`).

    The runtime representation of the MLIR tensor type is intentionally
    abstracted - you cannot control layout or get a pointer to the data. For
    low level buffer access, MLIR has a [`memref` type](#memreftype). This
    abstracted runtime representation holds both the tensor data values as well
    as information about the (potentially dynamic) shape of the tensor. The
    [`dim` operation](MemRef.md/#memrefdim-mlirmemrefdimop) returns the size of a
    dimension from a value of tensor type.

    The `encoding` attribute provides additional information on the tensor.
    An empty attribute denotes a straightforward tensor without any specific
    structure. But particular properties, like sparsity or other specific
    characteristics of the data of the tensor can be encoded through this
    attribute. The semantics are defined by a type and attribute interface
    and must be respected by all passes that operate on tensor types.
    TODO: provide this interface, and document it further.

    Note: hexadecimal integer literals are not allowed in tensor type
    declarations to avoid confusion between `0xf32` and `0 x f32`. Zero sizes
    are allowed in tensors and treated as other sizes, e.g.,
    `tensor<0 x 1 x i32>` and `tensor<1 x 0 x i32>` are different types. Since
    zero sizes are not allowed in some other types, such tensors should be
    optimized away before lowering tensors to vectors.

    #### Example:

    ```mlir
    // Known rank but unknown dimensions.
    tensor<? x ? x ? x ? x f32>

    // Partially known dimensions.
    tensor<? x ? x 13 x ? x f32>

    // Full static shape.
    tensor<17 x 4 x 13 x 4 x f32>

    // Tensor with rank zero. Represents a scalar.
    tensor<f32>

    // Zero-element dimensions are allowed.
    tensor<0 x 42 x f32>

    // Zero-element tensor of f32 type (hexadecimal literals not allowed here).
    tensor<0xf32>

    // Tensor with an encoding attribute (where #ENCODING is a named alias).
    tensor<?x?xf64, #ENCODING>
    ```
    """

    def __init__(
        self,
        shape: Sequence[int],
        element_type: max._core.Type,
        encoding: max._core.Attribute = ...,
    ) -> None: ...
    @property
    def shape(self) -> Sequence[int]: ...
    @property
    def element_type(self) -> max._core.Type | None: ...
    @property
    def encoding(self) -> max._core.Attribute | None: ...

class ShapedType(Protocol):
    """
    This interface provides a common API for interacting with multi-dimensional
    container types. These types contain a shape and an element type.

    A shape is a list of sizes corresponding to the dimensions of the container.
    If the number of dimensions in the shape is unknown, the shape is "unranked".
    If the number of dimensions is known, the shape "ranked". The sizes of the
    dimensions of the shape must be positive, or kDynamic (in which case the
    size of the dimension is dynamic, or not statically known).
    """

    @property
    def element_type(self) -> max._core.Type | None: ...
    @property
    def shape(self) -> Sequence[int]: ...
    def clone_with(
        self, arg0: Sequence[int], arg1: max._core.Type, /
    ) -> ShapedType: ...
    def has_rank(self) -> bool: ...

class SignednessSemantics(enum.Enum):
    signless = 0

    signed = 1

    unsigned = 2

class StridedLayoutAttr(max._core.Attribute):
    """
    Syntax:

    ```
    strided-layout-attribute ::= `strided` `<` `[` stride-list `]`
                                 (`,` `offset` `:` dimension)? `>`
    stride-list ::= /*empty*/
                  | dimension (`,` dimension)*
    dimension ::= decimal-literal | `?`
    ```

    A strided layout attribute captures layout information of the memref type in
    the canonical form. Specifically, it contains a list of _strides_, one for
    each dimension. A stride is the number of elements in the linear storage
    one must step over to reflect an increment in the given dimension. For
    example, a `MxN` row-major contiguous shaped type would have the strides
    `[N, 1]`. The layout attribute also contains the _offset_ from the base
    pointer of the shaped type to the first effectively accessed element,
    expressed in terms of the number of contiguously stored elements.

    Strides must be positive and the offset must be non-negative. Both the
    strides and the offset may be _dynamic_, i.e. their value may not be known
    at compile time. This is expressed as a `?` in the assembly syntax and as
    `ShapedType::kDynamic` in the code. Stride and offset values
    must satisfy the constraints above at runtime, the behavior is undefined
    otherwise.

    See [Dialects/Builtin.md#memreftype](MemRef type) for more information.
    """

    def __init__(self, offset: int, strides: Sequence[int]) -> None: ...
    @property
    def offset(self) -> int: ...
    @property
    def strides(self) -> Sequence[int]: ...

class StringAttr(max._core.Attribute):
    """
    Syntax:

    ```
    string-attribute ::= string-literal (`:` type)?
    ```

    A string attribute is an attribute that represents a string literal value.

    Examples:

    ```mlir
    "An important string"
    "string with a type" : !dialect.string
    ```
    """

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, bytes: str, type: max._core.Type) -> None: ...
    @overload
    def __init__(self, bytes: str) -> None: ...
    @property
    def value(self) -> str: ...
    @property
    def type(self) -> max._core.Type | None: ...

class SymbolRefAttr(max._core.Attribute):
    """
    Syntax:

    ```
    symbol-ref-attribute ::= symbol-ref-id (`::` symbol-ref-id)*
    ```

    A symbol reference attribute is a literal attribute that represents a named
    reference to an operation that is nested within an operation with the
    `OpTrait::SymbolTable` trait. As such, this reference is given meaning by
    the nearest parent operation containing the `OpTrait::SymbolTable` trait. It
    may optionally contain a set of nested references that further resolve to a
    symbol nested within a different symbol table.

    **Rationale:** Identifying accesses to global data is critical to
    enabling efficient multi-threaded compilation. Restricting global
    data access to occur through symbols and limiting the places that can
    legally hold a symbol reference simplifies reasoning about these data
    accesses.

    See [`Symbols And SymbolTables`](../SymbolsAndSymbolTables.md) for more
    information.

    Examples:

    ```mlir
    @flat_reference
    @parent_reference::@nested_reference
    ```
    """

    def __init__(
        self,
        root_reference: StringAttr,
        nested_references: Sequence[FlatSymbolRefAttr],
    ) -> None: ...
    @property
    def root_reference(self) -> StringAttr: ...
    @property
    def nested_references(self) -> Sequence[FlatSymbolRefAttr]: ...

class TupleType(max._core.Type):
    """
    Syntax:

    ```
    tuple-type ::= `tuple` `<` (type ( `,` type)*)? `>`
    ```

    The value of `tuple` type represents a fixed-size collection of elements,
    where each element may be of a different type.

    **Rationale:** Though this type is first class in the type system, MLIR
    provides no standard operations for operating on `tuple` types
    ([rationale](../../Rationale/Rationale/#tuple-types)).

    #### Example:

    ```mlir
    // Empty tuple.
    tuple<>

    // Single element
    tuple<f32>

    // Many elements.
    tuple<i32, f32, tensor<i1>, i5>
    ```
    """

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, element_types: Sequence[max._core.Type]) -> None: ...
    @property
    def types(self) -> Sequence[max._core.Type]: ...

class TypeAttr(max._core.Attribute):
    """
    Syntax:

    ```
    type-attribute ::= type
    ```

    A type attribute is an attribute that represents a
    [type object](#type-system).

    Examples:

    ```mlir
    i32
    !dialect.type
    ```
    """

    def __init__(self, type: max._core.Type) -> None: ...
    @property
    def value(self) -> max._core.Type | None: ...

class TypedAttr(Protocol):
    """
    This interface is used for attributes that have a type. The type of an
    attribute is understood to represent the type of the data contained in the
    attribute and is often used as the type of a value with this data.
    """

    @property
    def type(self) -> max._core.Type | None: ...

class UnitAttr(max._core.Attribute):
    """
    Syntax:

    ```
    unit-attribute ::= `unit`
    ```

    A unit attribute is an attribute that represents a value of `unit` type. The
    `unit` type allows only one value forming a singleton set. This attribute
    value is used to represent attributes that only have meaning from their
    existence.

    One example of such an attribute could be the `swift.self` attribute. This
    attribute indicates that a function parameter is the self/context parameter.
    It could be represented as a [boolean attribute](#boolean-attribute)(true or
    false), but a value of false doesn't really bring any value. The parameter
    either is the self/context or it isn't.


    Examples:

    ```mlir
    // A unit attribute defined with the `unit` value specifier.
    func.func @verbose_form() attributes {dialectName.unitAttr = unit}

    // A unit attribute in an attribute dictionary can also be defined without
    // the value specifier.
    func.func @simple_form() attributes {dialectName.unitAttr}
    ```
    """

    def __init__(self) -> None: ...

class UnknownLoc(max._core.Attribute):
    """
    Syntax:

    ```
    unknown-location ::= `?`
    ```

    Source location information is an extremely integral part of the MLIR
    infrastructure. As such, location information is always present in the IR,
    and must explicitly be set to unknown. Thus, an instance of the `unknown`
    location represents an unspecified source location.

    Example:

    ```mlir
    loc(?)
    ```
    """

    def __init__(self) -> None: ...

class UnrankedMemRefType(max._core.Type):
    """
    Syntax:

    ```
    unranked-memref-type ::= `memref` `<*x` type (`,` memory-space)? `>`
    memory-space ::= attribute-value
    ```

    A `memref` type with an unknown rank (e.g. `memref<*xf32>`). The purpose of
    unranked memrefs is to allow external library functions to receive memref
    arguments of any rank without versioning the functions based on the rank.
    Other uses of this type are disallowed or will have undefined behavior.

    See [MemRefType](#memreftype) for more information on
    memref types.

    #### Examples:

    ```mlir
    memref<*f32>

    // An unranked memref with a memory space of 10.
    memref<*f32, 10>
    ```
    """

    @overload
    def __init__(
        self, element_type: max._core.Type, memory_space: max._core.Attribute
    ) -> None: ...
    @overload
    def __init__(
        self, element_type: max._core.Type, memory_space: int
    ) -> None: ...
    @property
    def element_type(self) -> max._core.Type | None: ...
    @property
    def memory_space(self) -> max._core.Attribute | None: ...

class UnrankedTensorType(max._core.Type):
    """
    Syntax:

    ```
    tensor-type ::= `tensor` `<` `*` `x` type `>`
    ```

    An unranked tensor is a type of tensor in which the set of dimensions have
    unknown rank. See [RankedTensorType](#rankedtensortype)
    for more information on tensor types.

    #### Examples:

    ```mlir
    tensor<*xf32>
    ```
    """

    def __init__(self, element_type: max._core.Type) -> None: ...
    @property
    def element_type(self) -> max._core.Type | None: ...

class UnrealizedConversionCastOp(max._core.Operation):
    """
    An `unrealized_conversion_cast` operation represents an unrealized
    conversion from one set of types to another, that is used to enable the
    inter-mixing of different type systems. This operation should not be
    attributed any special representational or execution semantics, and is
    generally only intended to be used to satisfy the temporary intermixing of
    type systems during the conversion of one type system to another.

    This operation may produce results of arity 1-N, and accept as input
    operands of arity 0-N.

    Example:

    ```mlir
    // An unrealized 0-1 conversion. These types of conversions are useful in
    // cases where a type is removed from the type system, but not all uses have
    // been converted. For example, imagine we have a tuple type that is
    // expanded to its element types. If only some uses of an empty tuple type
    // instance are converted we still need an instance of the tuple type, but
    // have no inputs to the unrealized conversion.
    %result = unrealized_conversion_cast to !bar.tuple_type<>

    // An unrealized 1-1 conversion.
    %result1 = unrealized_conversion_cast %operand : !foo.type to !bar.lowered_type

    // An unrealized 1-N conversion.
    %results2:2 = unrealized_conversion_cast %tuple_operand : !foo.tuple_type<!foo.type, !foo.type> to !foo.type, !foo.type

    // An unrealized N-1 conversion.
    %result3 = unrealized_conversion_cast %operand, %operand : !foo.type, !foo.type to !bar.tuple_type<!foo.type, !foo.type>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        outputs: Sequence[max._core.Value],
        inputs: Sequence[max._core.Value],
    ) -> None: ...
    @property
    def inputs(self) -> Sequence[max._core.Value]: ...

class VectorElementTypeInterface(Protocol):
    """
    Implementing this interface establishes a contract between this type and the
    vector type, indicating that this type can be used as element of vectors.

    Vector element types are treated as a bag of bits without any assumed
    structure. The size of an element type must be a compile-time constant.
    However, the bit-width may remain opaque or unavailable during
    transformations that do not depend on the element type.

    Note: This type interface is still evolving. It currently has no methods
    and is just used as marker to allow types to opt into being vector elements.
    This may change in the future, for example, to require types to provide
    their size or alignment given a data layout. Please post an RFC before
    adding this interface to additional types. Implementing this interface on
    downstream types is discourged, until we specified the exact properties of
    a vector element type in more detail.
    """

class VectorType(max._core.Type):
    """
    Syntax:

    ```
    vector-type ::= `vector` `<` vector-dim-list vector-element-type `>`
    vector-element-type ::= float-type | integer-type | index-type
    vector-dim-list := (static-dim-list `x`)?
    static-dim-list ::= static-dim (`x` static-dim)*
    static-dim ::= (decimal-literal | `[` decimal-literal `]`)
    ```

    The vector type represents a SIMD style vector used by target-specific
    operation sets like AVX or SVE. While the most common use is for 1D
    vectors (e.g. vector<16 x f32>) we also support multidimensional registers
    on targets that support them (like TPUs). The dimensions of a vector type
    can be fixed-length, scalable, or a combination of the two. The scalable
    dimensions in a vector are indicated between square brackets ([ ]).

    Vector shapes must be positive decimal integers. 0D vectors are allowed by
    omitting the dimension: `vector<f32>`.

    Note: hexadecimal integer literals are not allowed in vector type
    declarations, `vector<0x42xi32>` is invalid because it is interpreted as a
    2D vector with shape `(0, 42)` and zero shapes are not allowed.

    #### Examples:

    ```mlir
    // A 2D fixed-length vector of 3x42 i32 elements.
    vector<3x42xi32>

    // A 1D scalable-length vector that contains a multiple of 4 f32 elements.
    vector<[4]xf32>

    // A 2D scalable-length vector that contains a multiple of 2x8 f32 elements.
    vector<[2]x[8]xf32>

    // A 2D mixed fixed/scalable vector that contains 4 scalable vectors of 4 f32 elements.
    vector<4x[4]xf32>

    // A 3D mixed fixed/scalable vector in which only the inner dimension is
    // scalable.
    vector<2x[4]x8xf32>
    ```
    """

    def __init__(
        self,
        shape: Sequence[int],
        element_type: max._core.Type,
        scalable_dims: Sequence[bool] = [],
    ) -> None: ...
    @property
    def shape(self) -> Sequence[int]: ...
    @property
    def element_type(self) -> max._core.Type | None: ...
    @property
    def scalable_dims(self) -> Sequence[bool]: ...

class _ElementsAttrIndexer:
    pass
