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
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

import enum
from collections.abc import Callable, Sequence
from typing import Protocol, overload

import max._core
import max._core.dialects.builtin
import max._core.dialects.m
from max.mlir import Context, Location

# C++ overloads on different int types look the same in Python, ignore these
# mypy: disable-error-code="overload-cannot-match"

DiagnosticHandler = Callable

class FnMetadataAttrInterface(Protocol):
    """
    This interface describes attributes that are attached to a `!kgen.func`
    type. Function metadata attributes carry additional information about a
    callable on top of the information in the base `FuncType`. This
    interface defines the required methods for this metadata attribute,
    including verification and print hooks.
    """

    def verify_func_type(
        self,
        arg0: DiagnosticHandler,
        arg1: max._core.dialects.builtin.FunctionType,
        arg2: Sequence[ArgConvention],
        arg3: FnEffects,
        /,
    ) -> bool: ...
    def get_with_bound_pos_args(
        self, arg: int, /
    ) -> FnMetadataAttrInterface: ...
    def equals(self, arg: FnMetadataAttrInterface, /) -> bool: ...

class GeneratorMetadataAttrInterface(Protocol):
    """
    This interface describes attributes that are attached to a GeneratorType,
    and carries additional metadata about the list. This interface defines the
    required methods for this metadata attribute, including verification and
    print hooks.
    """

    def verify_generator(
        self,
        arg0: DiagnosticHandler,
        arg1: Sequence[max._core.Type],
        arg2: max._core.Type,
        /,
    ) -> bool: ...
    def get_specialized_metadata(
        self,
        arg0: ParameterEvaluator,
        arg1: max._core._BitVector,
        arg2: DiagnosticHandler,
        /,
    ) -> GeneratorMetadataAttrInterface: ...
    def prepend_contextual_params_from_ops(
        self,
        arg0: Sequence[max._core.dialects.builtin.StringAttr],
        arg1: Sequence[max._core.Operation],
        /,
    ) -> GeneratorMetadataAttrInterface: ...

class IndexRefAttrInterface(Protocol):
    """
    Index-based parameter references are a relative parameter referencing scheme
    that uses a pair of integers to reference parameters in a way that doesn't
    involve names. This is useful for later knowing if two types are equal, even
    if they have different parameter names.

    For example, these two aliases have equal types:

    ```mojo
    alias A: fn[T: AnyType](x: T)->None = ...
    alias B: fn[Y: AnyType](x: Y)->None = ...
    ```

    ...if those param-refs use indexes instead of names like:

    ```mojo
    alias A: fn[_: AnyType](x: *(0,0))->None = ...
    alias B: fn[_: AnyType](x: *(0,0))->None = ...
    ```

    All types in Mojo use `IndexRefAttrInterface` instead of parameter names.
    The above are `ParamIndexRefAttr` specifically.

    All `IndexRefAttrInterface` have two fields: a depth and an index.

    - depth: Which containing signature contains the parameter we're referring
      to. Non-negative integer. Zero means the nearest containing signature
      (like above), one means the signature containing that one, etc.
      Note they cannot refer to any op's parameter-decls, and you cannot always
      use a depth to refer to surrounding scopes, see DCRTODS.
    - index: index of the parameter decl in that signature (non-negative integer).

    See IRAIDAI for more details, context, and examples.

    These depths must be carefully handled and adjusted when dealing with
    multiple signatures or scopes, see STCHDDDOS.
    """

    @property
    def depth(self) -> int: ...
    @property
    def index(self) -> int: ...
    def replace(
        self,
        arg0: int,
        arg1: int,
        arg2: Sequence[max._core.Attribute],
        arg3: Sequence[max._core.Type],
        /,
    ) -> IndexRefAttrInterface: ...

class ParameterAttr(Protocol):
    """
    Any attribute that implements `TypedAttr` can be used as a parameter
    attribute in KGEN, but this interface allows parameter attributes to plug
    into specific parts of the KGEN parameter system.
    """

    @property
    def constant(self) -> bool: ...
    def is_less_than(self, arg: max._core.Attribute, /) -> bool: ...
    def validate_for_elaborator(self) -> None: ...

class ParameterScopeAttrInterface(Protocol):
    """
    The `ParameterScopeAttrInterface` describes an attribute that declares a
    nested parameter scope within a parameter expression. It enables
    `ParamIndexRefAttr` values inside the attribute to reference parameters
    declared in a scope.
    """

    @property
    def input_param_types(self) -> Sequence[max._core.Type]: ...

class AttrCtorDeferredAttr(max._core.Attribute):
    """
    The `#kgen.attr_ctor_deferred` attribute holds an array of StringAttr
    or `#kgen.to_string_deferred` attributes. In the elaborator, when
    attributes are concrete, the `#kgen.attr_ctor_deferred` concatenates them
    and builds and requested attribute.

    Example:

    ```mlir
    #kgen.attr_ctor_deferred<"#index<", "cmp_predicate", "sle>">>
    ```
    """

    @overload
    def __init__(
        self, strings: Sequence[max._core.dialects.builtin.TypedAttr]
    ) -> None: ...
    @overload
    def __init__(
        self, strings: Sequence[max._core.dialects.builtin.TypedAttr]
    ) -> None: ...
    @property
    def strings(self) -> Sequence[max._core.dialects.builtin.TypedAttr]: ...

class ClosureAttr(max._core.Attribute):
    """
    The `#kgen.closure` attribute represents an uncomputed
        parametric closure. A parametric closure is a set of parametric
        values that are captured from the enclosing function. This abstraction
        is useful in the case where transformations are applied between the
        the closure definition site and the closure lowering pass that may alter
        the set of parameters captured.

        Example:

        In the following example we define a closure that captures the parameter
        value `C`. We want to bind this closure to the "x" parameter of the
        function `consume`. We must also pass the parameters that the closure
        depends on, which in this case is just `C` but could include `D` if we
        apply a transformation between now and when we lift the closure. To
        postpone the calculation of the captures, we bind an abstract value
        to the capture struct parameter of `consume` called
        `#kgen.closure<@foo "fn">`. This placeholder is of type ClosureAttr and
        represents the parameter captures of the "fn" closure.

        ```mlir

        kgen.generator @foo<C,D>() {
         %0 = kgen.closure.init()() -> index {
                     %1 = kgen.param.constant = <mul(C, C)>
                     kgen.return %1 : index
          } : (), !kgen.pointer<!kgen.closure<@foo, "fn" registerpassable>>
          %2 = kgen.call @consume<:type #type_value,
           :!kgen.param<!kgen.param_closure<@foo "fn">> #kgen.closure<@foo "fn">
           >(%3) : (!kgen.pointer<!kgen.closure<@foo, "fn" nonescaping>>) -> index
          kgen.return
        }

        kgen.generator @consume<
          x: type,
          CAPTURE_INST: !kgen.param<get_witness(x, "closure_trait", "CAPTURE_TYPE")>
        >(%arg0: !kgen.param<x>) -> index {
            // BODY OMITTED FOR BREVITY
        }
        ```
    """

    def __init__(self, type: ParamClosureType) -> None: ...
    @property
    def type(self) -> ParamClosureType: ...

class ClosureMethodAttr(max._core.Attribute):
    """
    The `#kgen.closure_method` attribute represents the symbol of a closure method.

    Example:

    ```mlir
    #kgen.closure_method<call>
    ```
    """

    def __init__(self, value: ClosureMethod) -> None: ...
    @property
    def value(self) -> ClosureMethod: ...

class ClosureSymbolAttr(max._core.Attribute):
    """
    We want to model function calls to functions that have not been generated
    yet. These functions are with respect to a capture struct and a nested
    function. This attribute must contain enough information to pair it with
    the generated methods which includes the symbol of the enclosing method.

    Example:
    ```mlir
    #kgen.closure.symbol<@foo,
                         "fn",
                         #kgen.closure_method<call>,
                         <:!kgen.param_capture<@foo "fn"> ?>
                        >
    ```
    """

    def __init__(
        self,
        parent_symbol: max._core.dialects.builtin.SymbolRefAttr,
        nested_func_name: max._core.dialects.builtin.StringAttr,
        method: ClosureMethodAttr,
        param_values: Sequence[max._core.dialects.builtin.TypedAttr],
        type: FuncTypeGeneratorType,
    ) -> None: ...
    @property
    def parent_symbol(self) -> max._core.dialects.builtin.SymbolRefAttr: ...
    @property
    def nested_func_name(self) -> max._core.dialects.builtin.StringAttr: ...
    @property
    def method(self) -> ClosureMethodAttr: ...
    @property
    def param_values(
        self,
    ) -> Sequence[max._core.dialects.builtin.TypedAttr]: ...
    @property
    def type(self) -> FuncTypeGeneratorType: ...

class CompileAssemblyAttr(max._core.Attribute):
    """
    The `#kgen.compile_assembly` attribute is used to model compiling a function
    to assembly code for a given target and emission format.

    Example:

    ```mlir
    kgen.param.declare some_target: target = #kgen.target<
      triple="", arch="", features="", data_layout="", simd_bit_width=128
    > : !kgen.target

    #kgen.compile_assembly<
      some_target, =llvm, "", false, :() -> () @kernel>
    > : !kgen.string
    ```
    """

    def __init__(
        self,
        target: max._core.dialects.builtin.TypedAttr,
        emission_kind: max._core.dialects.builtin.TypedAttr,
        emission_options: max._core.dialects.builtin.TypedAttr,
        propagate_error: max._core.dialects.builtin.BoolAttr,
        func: max._core.dialects.builtin.TypedAttr,
        type: max._core.Type,
    ) -> None: ...
    @property
    def target(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def emission_kind(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def emission_options(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def propagate_error(self) -> max._core.dialects.builtin.BoolAttr: ...
    @property
    def func(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def type(self) -> max._core.Type | None: ...

class CompileOffloadClosureAttr(max._core.Attribute):
    """
    The `#kgen.compile_offload_closure` attribute is used to compile offload
    closures for a given target.

    Example:

    ```mlir
    #kgen.compile_offload_closure<
      #kgen.target<triple="", arch="", features="", data_layout="", simd_bit_width=128> : !kgen.target,
      #kgen.symbol.constant<@kernel> : !kgen.generator<() -> ()>
    > : !kgen.string
    ```
    """

    def __init__(
        self,
        target: max._core.dialects.builtin.TypedAttr,
        func: max._core.dialects.builtin.TypedAttr,
        type: max._core.Type,
    ) -> None: ...
    @property
    def target(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def func(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def type(self) -> max._core.Type | None: ...

class ConstraintAttr(max._core.Attribute):
    """
    The `#kgen.constraint` attribute represents a proposition that should hold
    and a location for where this constraint was declared, which is useful for
    error reporting.

    The proposition is an i1-typed parameter expression.

    Example:

    ```mlir
    #kgen.constraint<1, loc("file.mojo":10:5)>
    ```
    """

    @overload
    def __init__(
        self,
        proposition: max._core.dialects.builtin.TypedAttr,
        loc: max._core.LocationAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        proposition: max._core.dialects.builtin.TypedAttr,
        loc: max._core.LocationAttr,
    ) -> None: ...
    @property
    def proposition(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def loc(self) -> max._core.LocationAttr: ...

class DTypeConstantAttr(max._core.Attribute):
    """
    This is constant value for a dtype, whose elements correspond to DType.
    """

    def __init__(self, d_type: _KGENDType) -> None: ...
    @property
    def d_type(self) -> _KGENDType: ...

class DecoratorsAttr(max._core.Attribute):
    """
    The `#kgen.decorators` attribute represents a list of decorator invocations
    attached to an operation. Decorators are closures where the first argument
    of the function will be the operation the decorator is attached to. The
    expected signature of the closure is:

    ```mlir
    (!pdl.operation) capturing -> !pdl.operation
    ```

    Decorators in the list are invoked on the operation from first to last.
    Decorators may be invoked at different points in KGEN pass pipeline. Each
    decorator contains a tag indicating when it should be invoked. Successive
    decorators must have later invocation points than previous ones.
    (TODO: Not implemented yet)
    """

    def __init__(
        self, value: Sequence[max._core.dialects.builtin.TypedAttr]
    ) -> None: ...
    @property
    def value(self) -> Sequence[max._core.dialects.builtin.TypedAttr]: ...

class DeferredAttr(max._core.Attribute):
    """
    The `#kgen.deferred` attribute holds a non-typed attribute to allow it
    to be created later.

    Example:

    ```mlir
    #kgen.deferred #index<cmp_predicate sle>> : !kgen.deferred
    ```
    """

    @overload
    def __init__(self, attr: max._core.Attribute) -> None: ...
    @overload
    def __init__(self, attr: max._core.Attribute) -> None: ...
    @property
    def attr(self) -> max._core.Attribute | None: ...

class DowncastAttr(max._core.Attribute):
    """
    The `#kgen.upcast` attribute is used to convert from a (variadic of) typeValue
    to a (variadic of) typeValue of a more-derived trait.
    For example, this can represent a cast from AnyType to Movable.

    Note that parser does not (also can not) verify whether the downcast is
    legal and a illegal downcast can lead to elaboration time error.


    Example:

    ```mlir
    #kgen.downcast<:AnyType T> : !lit.trait<Movable>
    ```
    """

    @overload
    def __init__(
        self,
        type: max._core.Type,
        input_type_value: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        type: max._core.Type,
        input_type_value: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def type(self) -> max._core.Type | None: ...
    @property
    def input_type_value(self) -> max._core.dialects.builtin.TypedAttr: ...

class EnvAttr(max._core.Attribute):
    """
    The `#kgen.env` attribute defines a generic dictionary of environment
    parameters that can be accessed through parameter operators. The values
    contained can be:
    - integers, represented with the `index` type
    - strings, represented as `!kgen.string` types
    - unit attributes, the presence of which indicates something

    Note that EnvAttr does not support storing BoolAttr in it.
    Instead, a boolean true is represented as a UnitAttr, and boolean false
    is not represented at all (absence of a value evaluates to false).

    Example:

    ```mlir
    #kgen<env{intVal = 1 : index, unitAttr, strVal = "hello" : !kgen.string}>
    ```
    """

    @overload
    def __init__(
        self, values: max._core.dialects.builtin.DictionaryAttr
    ) -> None: ...
    @overload
    def __init__(
        self, values: max._core.dialects.builtin.DictionaryAttr
    ) -> None: ...
    @property
    def values(self) -> max._core.dialects.builtin.DictionaryAttr: ...

class ExportKindAttr(max._core.Attribute):
    """
    The `#kgen.export` attribute defines the export semantics of a symbol. A
    symbol can be:

    - Not exported: its linkage is internal and its visibility is hidden.
    - Exported: its linkage is external and its visibility is public.
    - C exported: like `exported`, but with a C-compatible name and ABI.
    - Package exported: like `exported`, but implicitly exported as part of a
                        package.

    Example:

    ```mlir
    #kgen.export_kind<not_exported>
    #kgen.export_kind<exported>
    #kgen.export_kind<c_exported>
    #kgen.export_kind<package_exported>
    ```
    """

    def __init__(self, value: ExportKind) -> None: ...
    @property
    def value(self) -> ExportKind: ...

class GeneratorAttr(max._core.Attribute):
    """
    This is a generator constant attribute that represents a generator whose
    body is a parameter expression. The GeneratorAttr natively encodes the input
    parameter types and metadata, and computes the overall type on demand. This
    encoding ensures that the type and the value of the body are always at the
    same level of nesting. If we instead stored a GeneratorType in this
    attribute, the body of the GeneratorType would be at a deeper level of
    nesting than the body of the GeneratorAttr, leading to inconsistencies.

    Example:

    ```mlir
    #kgen.gen<*(0,0) + 1> : !kgen.generator<<index> index>
    ```
    """

    @overload
    def __init__(
        self,
        input_param_types: Sequence[max._core.Type],
        body: max._core.dialects.builtin.TypedAttr,
        metadata: GeneratorMetadataAttrInterface = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        body: max._core.dialects.builtin.TypedAttr,
        input_param_types: Sequence[max._core.Type],
        metadata: GeneratorMetadataAttrInterface,
    ) -> None: ...
    @property
    def body(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def input_param_types(self) -> Sequence[max._core.Type]: ...
    @property
    def metadata(self) -> GeneratorMetadataAttrInterface: ...

class GetBaseTypeNameAttr(max._core.Attribute):
    """
    The `#kgen.get_base_type_name` attribute extracts the unqualified name of
    the base (unparameterized) type from a parameterized type. For example,
    given `List[Int]`, it returns the string `"List"`. For non-parameterized
    types, it returns the type's simple name.

    This is useful for reflection-based code that needs to identify the base
    type of parameterized types.

    Example:

    ```mlir
    #kgen.get_base_type_name<#List[Int]> : !kgen.string
    // Returns "List"
    ```
    """

    def __init__(
        self, type_value: max._core.dialects.builtin.TypedAttr
    ) -> None: ...
    @property
    def type_value(self) -> max._core.dialects.builtin.TypedAttr: ...

class GetLinkageNameAttr(max._core.Attribute):
    """
    The `#kgen.get_linkage_name` attribute is used to get the linkage name of
    a function symbol for a given target.

    Example:

    ```mlir
    #kgen.get_linkage_name<
      #kgen.target<triple="", arch="", features="", data_layout="", simd_bit_width=128> : !kgen.target,
      #kgen.symbol.constant<@return_one> : !kgen.generator<() -> index>
    > : !kgen.string
    ```
    """

    def __init__(
        self,
        target: max._core.dialects.builtin.TypedAttr,
        func: max._core.dialects.builtin.TypedAttr,
        type: max._core.Type,
    ) -> None: ...
    @property
    def target(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def func(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def type(self) -> max._core.Type | None: ...

class GetSourceNameAttr(max._core.Attribute):
    """
    The `#kgen.get_source_name` attribute is used to get the source name of a
    function symbol.

    Example:

    ```mlir
    #kgen.get_source_name<
      #kgen.symbol.constant<@return_two> : !kgen.generator<() -> index>
    > : !kgen.string
    ```
    """

    def __init__(
        self, func: max._core.dialects.builtin.TypedAttr, type: max._core.Type
    ) -> None: ...
    @property
    def func(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def type(self) -> max._core.Type | None: ...

class GetTypeNameAttr(max._core.Attribute):
    """
    The `#kgen.get_type_name` attribute is used to get the name of a struct
    symbol.

    Example:

    ```mlir
    #kgen.get_type_name<#Int>: !kgen.string
    ```
    """

    def __init__(
        self,
        type_value: max._core.dialects.builtin.TypedAttr,
        qualified_builtins: max._core.dialects.builtin.TypedAttr,
        type: max._core.Type,
    ) -> None: ...
    @property
    def type_value(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def qualified_builtins(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def type(self) -> max._core.Type | None: ...

class GetWitnessAttr(max._core.Attribute):
    """
    The `#kgen.get_witness` attribute is used to lookup a witness entry
    from a witness table given a type value and a trait conformance.

    Since type value definitions are symbols, this attribute can only be folded
    when a global symbol table is provided.

    Example:

    ```mlir
    #kgen.get_witness<#Int, "Boolable", "__bool__">
      : !kgen.generator<("self": !Int) -> i1>
    ```
    """

    @overload
    def __init__(
        self,
        type_value: max._core.dialects.builtin.TypedAttr,
        trait_name: max._core.dialects.builtin.StringAttr,
        witness_name: max._core.dialects.builtin.StringAttr,
        type: max._core.Type,
    ) -> None: ...
    @overload
    def __init__(
        self,
        type_value: max._core.dialects.builtin.TypedAttr,
        trait_name: max._core.dialects.builtin.StringAttr,
        witness_name: max._core.dialects.builtin.StringAttr,
        type: max._core.Type,
    ) -> None: ...
    @property
    def type_value(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def trait_name(self) -> max._core.dialects.builtin.StringAttr: ...
    @property
    def witness_name(self) -> max._core.dialects.builtin.StringAttr: ...
    @property
    def type(self) -> max._core.Type | None: ...

class IsStructTypeAttr(max._core.Attribute):
    """
    The `#kgen.is_struct_type` attribute returns true if the given type value
    refers to a Mojo struct type, false otherwise. This is useful for guarding
    reflection code that needs to avoid compiler errors on non-struct types.

    Example:

    ```mlir
    #kgen.is_struct_type<#MyStruct> : i1
    ```
    """

    def __init__(
        self,
        type_value: max._core.dialects.builtin.TypedAttr,
        type: max._core.dialects.builtin.IntegerType,
    ) -> None: ...
    @property
    def type_value(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def type(self) -> max._core.dialects.builtin.IntegerType: ...

class LLVMBitcodeLibArrayAttr(max._core.Attribute):
    """
    The `#kgen.llvm.bitcode.libs` attribute represents an array of LLVM
    bitcode libraries, each with their own usage tracking. This is typically
    attached to ModuleOp to store all bitcode libraries that should be
    linked during compilation.
    """

    def __init__(self, value: Sequence[LLVMBitcodeLibAttr]) -> None: ...
    @property
    def value(self) -> Sequence[LLVMBitcodeLibAttr]: ...

class LLVMBitcodeLibAttr(max._core.Attribute):
    """
    The `#kgen.llvm.bitcode.lib` attribute represents a single LLVM bitcode
    library with usage tracking. It contains:
    - `used`: A boolean flag indicating whether this library was used
    - `library`: The actual bitcode library, which can be either:
      - StringAttr: For bitcode libraries passed via command line
      - DenseResourceElementsAttr: For bitcode libraries from packages

    Example:
    ```mlir
    #kgen.llvm.bitcode.lib<used = false, library = "/path/to/lib.bc">
    #kgen.llvm.bitcode.lib<used = true, library = dense_resource<data> : ...>
    ```
    """

    @overload
    def __init__(self, used: bool, library: max._core.Attribute) -> None: ...
    @overload
    def __init__(
        self,
        used: max._core.dialects.builtin.BoolAttr,
        library: max._core.Attribute,
    ) -> None: ...
    @overload
    def __init__(
        self,
        used: max._core.dialects.builtin.BoolAttr,
        library: max._core.Attribute,
    ) -> None: ...
    @property
    def used(self) -> max._core.dialects.builtin.BoolAttr: ...
    @property
    def library(self) -> max._core.Attribute | None: ...

class LinkDependencyArrayAttr(max._core.Attribute):
    """
    The `#kgen.link.dependencies` attribute represents a list of link
    dependencies, which are flat symbol references.
    """

    def __init__(
        self, value: Sequence[max._core.dialects.builtin.FlatSymbolRefAttr]
    ) -> None: ...
    @property
    def value(
        self,
    ) -> Sequence[max._core.dialects.builtin.FlatSymbolRefAttr]: ...

class MLIROpAttr(max._core.Attribute):
    """
    The `#kgen.param.mlir_op` attribute represents an MLIR operation as a
    parameter expression. Its type is FuncTypeGeneratorType.

    Example:

    ```
    #kgen.param.mlir_op<"index.add", {}>
      : !kgen.generator<(index, index) -> index>
    ```

    Operation attributes can be specified using a dictionary attribute.

    Example:

    ```
    #kgen.param.mlir_op<"index.cmp", {pred = #index<cmp_predicate slt>}>
      : !kgen.generator<(index, index) -> i1>
    ```

    The operation can be parameterized on any of its attributes that are
    parametric -- that is, which are `TypedAttr`. These attributes are omitted
    from the attribute dictionary and are present in the signature.

    Example:

    ```
    #kgen.param.mlir_op<"pop.array.get", {}> : !kgen.generator<
      <*"index">(!pop.array<size, type>) -> !kgen.param<type>
    >
    ```

    Parametric operations can be bound using `bind_signature`.
    """

    @overload
    def __init__(
        self,
        name: max._core.dialects.builtin.StringAttr,
        attrs: max._core.dialects.builtin.DictionaryAttr,
        type: FuncTypeGeneratorType,
    ) -> None: ...
    @overload
    def __init__(
        self,
        name: max._core.dialects.builtin.StringAttr,
        attrs: max._core.dialects.builtin.DictionaryAttr,
        type: FuncTypeGeneratorType,
    ) -> None: ...
    @property
    def name(self) -> max._core.dialects.builtin.StringAttr: ...
    @property
    def attrs(self) -> max._core.dialects.builtin.DictionaryAttr: ...
    @property
    def type(self) -> FuncTypeGeneratorType: ...

class MemSymbolTripleAttr(max._core.Attribute):
    """
    The `#kgen.mem_symbol_triple` attribute holds the symbols of a memory value.
    The symbols it holds are copy, move, and del. The copy symbol is optional.
    This attribute is useful in the context of abstracted operations whose
    lowering depends on these core symbols.

    Example:

    ```mlir
      #kgen.mem_symbol_triple<@bar_move<:type index, :type index>>,
                              @bar_del<:type index, :type index>>
                               : !kgen.pointer<struct<(index, index)>>
    ```
    """

    def __init__(
        self,
        copy: SymbolConstantAttr,
        move: SymbolConstantAttr,
        del_: SymbolConstantAttr,
        is_move: max._core.dialects.builtin.UnitAttr,
    ) -> None: ...
    @property
    def copy(self) -> SymbolConstantAttr: ...
    @property
    def move(self) -> SymbolConstantAttr: ...
    @property
    def del_(self) -> SymbolConstantAttr: ...
    @property
    def is_move(self) -> max._core.dialects.builtin.UnitAttr: ...

class PackAttr(max._core.Attribute):
    """
    The `#kgen.pack` attribute contains a heterogenously typed list of constant
    elements. It can be used to represent constant pack values, and so is of
    pack type.

    Example:

    ```mlir
    // A pack of 3 elements.
    %0 = kgen.param.constant: !kgen.pack<[i8, ui4, i32]> = <<3, 1, 4>>
    // An empty pack.
    %1 = kgen.param.constant: !kgen.pack<[]> = <<>>
    ```
    """

    @overload
    def __init__(
        self,
        values: Sequence[max._core.dialects.builtin.TypedAttr],
        type: PackType,
    ) -> None: ...
    @overload
    def __init__(
        self,
        values: Sequence[max._core.dialects.builtin.TypedAttr],
        type: PackType,
    ) -> None: ...
    @property
    def values(self) -> Sequence[max._core.dialects.builtin.TypedAttr]: ...
    @property
    def type(self) -> PackType: ...

class ParamDeclArrayAttr(max._core.Attribute):
    @overload
    def __init__(self, param_decl: ParamDeclAttr) -> None: ...
    @overload
    def __init__(self, value: Sequence[ParamDeclAttr]) -> None: ...
    @property
    def value(self) -> Sequence[ParamDeclAttr]: ...

class ParamDeclAttr(max._core.Attribute):
    """
    This is a declaration of a parameter in the meta-programming domain for
    generators and related infrastructure.  These are typically owned by
    kgen.generator instances or other things that produce new attributes.
    """

    @overload
    def __init__(self, ref: ParamDeclRefAttr) -> None: ...
    @overload
    def __init__(
        self, name: max._core.dialects.builtin.StringAttr, type: max._core.Type
    ) -> None: ...
    @overload
    def __init__(self, name: str, type: max._core.Type) -> None: ...
    @overload
    def __init__(
        self, name: max._core.dialects.builtin.StringAttr, type: max._core.Type
    ) -> None: ...
    @property
    def name(self) -> max._core.dialects.builtin.StringAttr: ...
    @property
    def type(self) -> max._core.Type | None: ...

class ParamDeclRefAttr(max._core.Attribute):
    """
    The `#kgen.param.decl.ref` attribute is a typed attribute that represents
    a reference to a declared parameter. It contains the type of the referenced
    parameter and its name.

    Example:

    ```mlir
    // A reference to parameter "p" with type "i1".
    #kgen.param.decl.ref<"p"> : i1
    ```

    There are special rules governing when these can appear, or when
    ParamIndexRefAttr must be used instead, see DCRTODS.
    """

    @overload
    def __init__(self, decl: ParamDeclAttr) -> None: ...
    @overload
    def __init__(
        self, name: max._core.dialects.builtin.StringAttr, type: max._core.Type
    ) -> None: ...
    @overload
    def __init__(self, name: str, type: max._core.Type) -> None: ...
    @overload
    def __init__(
        self, name: max._core.dialects.builtin.StringAttr, type: max._core.Type
    ) -> None: ...
    @property
    def name(self) -> max._core.dialects.builtin.StringAttr: ...
    @property
    def type(self) -> max._core.Type | None: ...

class ParamIndexRefAttr(max._core.Attribute):
    """
    The `#kgen.param.index.ref` attribute is a reference to an input
    parameter of an enclosing signature. This attribute can only be used inside
    a `GeneratorType`. This attribute contains two fields:

    - depth: Which containing signature contains the parameter we're referring
      to. Non-negative integer. Zero means the nearest containing signature, one
      means the signature containing that one, etc.
      Note they cannot refer to any op's parameter-decls, and you cannot always
      use a depth to refer to surrounding scopes, see DCRTODS.
    - index: index of the parameter decl in that signature (non-negative
      integer).

    Example:

    ```mlir
    // Second input parameter of the nearest signature.
    #kgen.param.index.ref<0, 1> : index
    // First input parameter of next enclosing signature.
    #kgen.param.index.ref<1, 0> : !lit.struct<@Int>
    ```

    The latter would appear in something like this:

    ```
    alias bar: fn[
      D: DType,
      N: Int,
      f: fn[Y: AnyType](Y, SIMD[N, D])->None
    ](...) = ...
    ```

    The `SIMD[N, D]`'s `N` is a #kgen.param.index.ref<1, 0> : !lit.struct<@Int>.

    But, per DCRTODS, it can NOT be used inside a generator's body to refer to
    one of the generator's parameters, like this:

    ```
    fn foo[X: AnyType](x: X):
        alias zork: fn[...](
          # Cannot have: #kgen.param.index.ref<1, 0> : !lit.struct<@Int>
        )->None = ...
    ```

    These depths must be carefully handled and adjusted when dealing with
    multiple signatures or scopes, see STCHDDDOS.
    """

    @overload
    def __init__(self, index: int, type: max._core.Type) -> None: ...
    @overload
    def __init__(
        self, depth: int, index: int, type: max._core.Type
    ) -> None: ...
    @overload
    def __init__(
        self, depth: int, index: int, type: max._core.Type
    ) -> None: ...
    @property
    def depth(self) -> int: ...
    @property
    def index(self) -> int: ...
    @property
    def type(self) -> max._core.Type | None: ...

class ParamOperatorAttr(max._core.Attribute):
    @overload
    def __init__(
        self,
        opcode: POC,
        operands: Sequence[max._core.dialects.builtin.TypedAttr],
    ) -> None: ...
    @overload
    def __init__(
        self,
        opcode: POC,
        operands: Sequence[max._core.dialects.builtin.TypedAttr],
        type: max._core.Type,
    ) -> None: ...
    @overload
    def __init__(
        self,
        opcode: POC,
        lhs: max._core.dialects.builtin.TypedAttr,
        rhs: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        opcode: POC,
        operands: Sequence[max._core.dialects.builtin.TypedAttr],
        type: max._core.Type,
    ) -> None: ...
    @property
    def opcode(self) -> POC: ...
    @property
    def operands(self) -> Sequence[max._core.dialects.builtin.TypedAttr]: ...
    @property
    def type(self) -> max._core.Type | None: ...

class ParameterExprArrayAttr(max._core.Attribute):
    def __init__(
        self, value: Sequence[max._core.dialects.builtin.TypedAttr]
    ) -> None: ...
    @property
    def value(self) -> Sequence[max._core.dialects.builtin.TypedAttr]: ...

class PreservedAttr(max._core.Attribute):
    """
    The `#kgen.preserved` attribute contains an attribute and isolates it from
    all rewrites and lowerings. This is useful for keeping higher-level
    information, like source information, around in the IR in case users need to
    inspect them.

    Example:

    ```mlir
    #kgen.preserved<!lit.generator<() -> ()>>
    ```
    """

    @overload
    def __init__(self, value: max._core.Attribute) -> None: ...
    @overload
    def __init__(self, value: max._core.Attribute) -> None: ...
    @property
    def value(self) -> max._core.Attribute | None: ...

class StructAttr(max._core.Attribute):
    """
    The `#kgen.struct` attribute contains a heterogenous list of elements of
    struct type. It is used to represent constant struct values.

    Example:

    ```mlir
    #kgen.struct<3, 3.5> : !kgen.struct<(index, f32)>
    ```
    """

    @overload
    def __init__(
        self, values: Sequence[max._core.dialects.builtin.TypedAttr]
    ) -> None: ...
    @overload
    def __init__(
        self,
        values: Sequence[max._core.dialects.builtin.TypedAttr],
        type: StructType,
    ) -> None: ...
    @overload
    def __init__(
        self,
        values: Sequence[max._core.dialects.builtin.TypedAttr],
        type: StructType,
    ) -> None: ...
    @property
    def values(self) -> Sequence[max._core.dialects.builtin.TypedAttr]: ...
    @property
    def type(self) -> StructType: ...

class StructDefFieldAttr(max._core.Attribute):
    """
    The `#kgen.struct_def.field` attribute represents a field declared in a
    mojo struct. It keeps track of the name and type of the field.

    Example:

    ```mlir
    #kgen.struct_def.field<"num" : Int>
    ```
    """

    @overload
    def __init__(
        self,
        name: max._core.dialects.builtin.StringAttr,
        type_value: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        name: max._core.dialects.builtin.StringAttr,
        type_value: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def name(self) -> max._core.dialects.builtin.StringAttr: ...
    @property
    def type_value(self) -> max._core.dialects.builtin.TypedAttr: ...

class StructExtractAttr(max._core.Attribute):
    """
    The `#kgen.struct.extract` attribute represents a field reference from a
    constant struct, which may be parametric.

    Example:

    ```mlir
    #kgen.struct.extract<p, 4> : index
    ```
    """

    @overload
    def __init__(
        self, struct_value: max._core.dialects.builtin.TypedAttr, field_no: int
    ) -> None: ...
    @overload
    def __init__(
        self,
        struct_value: max._core.dialects.builtin.TypedAttr,
        field_no: max._core.dialects.builtin.TypedAttr,
        result_type: max._core.Type,
    ) -> None: ...
    @overload
    def __init__(
        self,
        struct_value: max._core.dialects.builtin.TypedAttr,
        field_no: max._core.dialects.builtin.TypedAttr,
        result_type: max._core.Type,
    ) -> None: ...
    @property
    def struct_value(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def field_no(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def type(self) -> max._core.Type | None: ...

class StructFieldIndexByNameAttr(max._core.Attribute):
    """
    The `#kgen.struct_field_index_by_name` attribute returns the index of a field
    in a struct type given the field name. Produces a compile error if the
    field name does not exist in the struct.

    The fieldName parameter should resolve to a StringAttr (kgen.string) after
    parameter evaluation.

    Example:

    ```mlir
    #kgen.struct_field_index_by_name<#MyStruct, "x"> : index
    ```
    """

    def __init__(
        self,
        type_value: max._core.dialects.builtin.TypedAttr,
        field_name: max._core.dialects.builtin.TypedAttr,
        type: max._core.dialects.builtin.IndexType,
    ) -> None: ...
    @property
    def type_value(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def field_name(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def type(self) -> max._core.dialects.builtin.IndexType: ...

class StructFieldNamesAttr(max._core.Attribute):
    """
    The `#kgen.struct_field_names` attribute returns the names of all fields
    in a struct type as a variadic sequence of strings.

    Example:

    ```mlir
    #kgen.struct_field_names<#MyStruct> : !kgen.variadic<!kgen.string>
    ```
    """

    def __init__(
        self,
        type_value: max._core.dialects.builtin.TypedAttr,
        type: VariadicType,
    ) -> None: ...
    @property
    def type_value(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def type(self) -> VariadicType: ...

class StructFieldOffsetByIndexAttr(max._core.Attribute):
    """
    The `#kgen.struct_field_offset_by_index` attribute returns the byte offset
    of a field in a struct type given the field index. Produces a compile error
    if the field index is out of bounds.

    The offset is computed using the target's data layout to determine field
    sizes and alignment requirements.

    Example:

    ```mlir
    #kgen.struct_field_offset_by_index<#MyStruct, 0 : index, #target> : index
    ```
    """

    def __init__(
        self,
        type_value: max._core.dialects.builtin.TypedAttr,
        field_index: max._core.dialects.builtin.TypedAttr,
        target: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def type_value(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def field_index(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def target(self) -> max._core.dialects.builtin.TypedAttr: ...

class StructFieldOffsetByNameAttr(max._core.Attribute):
    """
    The `#kgen.struct_field_offset_by_name` attribute returns the byte offset
    of a field in a struct type given the field name. Produces a compile error
    if the field name does not exist in the struct.

    The fieldName parameter should resolve to a StringAttr (kgen.string) after
    parameter evaluation.

    The offset is computed using the target's data layout to determine field
    sizes and alignment requirements.

    Example:

    ```mlir
    #kgen.struct_field_offset_by_name<#MyStruct, "x", #target> : index
    ```
    """

    def __init__(
        self,
        type_value: max._core.dialects.builtin.TypedAttr,
        field_name: max._core.dialects.builtin.TypedAttr,
        target: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def type_value(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def field_name(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def target(self) -> max._core.dialects.builtin.TypedAttr: ...

class StructFieldTypeByNameAttr(max._core.Attribute):
    """
    The `#kgen.struct_field_type_by_name` attribute returns the type of a field
    in a struct given the field name. Produces a compile error if the field
    name does not exist in the struct.

    The fieldName parameter should resolve to a StringAttr (kgen.string) after
    parameter evaluation.

    Example:

    ```mlir
    #kgen.struct_field_type_by_name<#MyStruct, "x"> : !kgen.type
    ```
    """

    def __init__(
        self,
        type_value: max._core.dialects.builtin.TypedAttr,
        field_name: max._core.dialects.builtin.TypedAttr,
        type: TypeType,
    ) -> None: ...
    @property
    def type_value(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def field_name(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def type(self) -> TypeType: ...

class StructFieldTypesAttr(max._core.Attribute):
    """
    The `#kgen.struct_field_types` attribute returns the types of all fields
    in a struct type as a variadic sequence.

    Example:

    ```mlir
    #kgen.struct_field_types<#MyStruct> : !kgen.variadic<!kgen.type>
    ```
    """

    def __init__(
        self,
        type_value: max._core.dialects.builtin.TypedAttr,
        type: VariadicType,
    ) -> None: ...
    @property
    def type_value(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def type(self) -> VariadicType: ...

class SugarAttr(max._core.Attribute):
    """
    The `#kgen.sugar` attribute represents a syntax sugar overlaid on some other
    value e.g. an alias or expanded builtin function call. It maintains the
    original sugared form as well as the "one level expanded" form, and
    fully expanded "canonical" version of the attribute.
    """

    def __init__(
        self,
        kind: SugarKind,
        member_name: max._core.dialects.builtin.StringAttr,
        sugared: max._core.dialects.builtin.TypedAttr,
        expanded: max._core.dialects.builtin.TypedAttr,
        canonical: max._core.dialects.builtin.TypedAttr = ...,
    ) -> None: ...
    @property
    def kind(self) -> SugarKind: ...
    @property
    def member_name(self) -> max._core.dialects.builtin.StringAttr: ...
    @property
    def sugared(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def expanded(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def canonical(self) -> max._core.dialects.builtin.TypedAttr: ...

class SymbolConstantAttr(max._core.Attribute):
    """
    This is a value of FuncTypeGenerator type, which refers to a func or
    generator.  This may optionally bind the input parameter values at time of
    formation - when this happens, the result type is non-parametric.
    """

    @overload
    def __init__(self, func: GeneratorOp) -> None: ...
    @overload
    def __init__(self, func: FuncOp) -> None: ...
    @overload
    def __init__(
        self,
        symbol: max._core.dialects.builtin.SymbolRefAttr,
        type: FuncTypeGeneratorType,
        param_values: Sequence[max._core.dialects.builtin.TypedAttr] = [],
    ) -> None: ...
    @overload
    def __init__(
        self,
        name: max._core.dialects.builtin.StringAttr,
        type: FuncTypeGeneratorType,
        param_values: Sequence[max._core.dialects.builtin.TypedAttr] = [],
    ) -> None: ...
    @overload
    def __init__(
        self,
        func: GeneratorOp,
        type: FuncTypeGeneratorType,
        param_values: Sequence[max._core.dialects.builtin.TypedAttr] = [],
    ) -> None: ...
    @overload
    def __init__(
        self,
        symbol: max._core.dialects.builtin.SymbolRefAttr,
        param_values: Sequence[max._core.dialects.builtin.TypedAttr],
        type: FuncTypeGeneratorType,
    ) -> None: ...
    @property
    def symbol(self) -> max._core.dialects.builtin.SymbolRefAttr: ...
    @property
    def param_values(
        self,
    ) -> Sequence[max._core.dialects.builtin.TypedAttr]: ...
    @property
    def type(self) -> FuncTypeGeneratorType: ...

class TailKindAttr(max._core.Attribute):
    """
    The `#kgen.tailkind` attribute defines the export semantics of a symbol. A
    symbol can be:

    - None: Unspecified.
    - Musttail: Compilation will fail if this call cannot be tail call
                optimized.
    - NoTail: Do not tail call optimize this call.

    Example:

    ```mlir
    #kgen.tailkind<none>
    #kgen.tailkind<musttail>
    #kgen.tailkind<notail>
    ```
    """

    def __init__(self, value: TailKind) -> None: ...
    @property
    def value(self) -> TailKind: ...

class TargetParamAttr(max._core.Attribute):
    """
    The `#kgen.target` is an attribute representing a target of type
    `!kgen.target`. It contains the target configuration information.

    Example:

    ```mlir
    #kgen.target<triple="triple", cpu="cpu", features="features",
                 data_layout="p:32:32", simd_bit_width=128>
    ```

    Target types can be manipulated using target operators.

    Example:

    ```mlir
    kgen.generator @target_host<t0: target>()
        constraints <[eq(:target
            #kgen.target<triple="triple", cpu="cpu", features="features",
                         data_layout="", simd_bit_width=128>,
            t0),
          "Must support the target!"
        ]> {
      kgen.return
    }
    ```
    """

    @overload
    def __init__(self, target: max._core.dialects.m.TargetInfoAttr) -> None: ...
    @overload
    def __init__(self, target: max._core.dialects.m.TargetInfoAttr) -> None: ...
    @property
    def target(self) -> max._core.dialects.m.TargetInfoAttr: ...

class ToStringDeferredAttr(max._core.Attribute):
    """
    The `#kgen.to_string_deferred` attribute holds an array of StringAttr
    and concatenates them into a single StringAttr in Elaborator

    Example:

    ```mlir
    #kgen.to_string_deferred<"#index<", "cmp_predicate", "sle>">>
    ```
    """

    @overload
    def __init__(
        self, attr: max._core.Attribute, need_elide_type: bool
    ) -> None: ...
    @overload
    def __init__(
        self,
        attr: max._core.Attribute,
        need_elide_type: max._core.dialects.builtin.UnitAttr,
    ) -> None: ...
    @property
    def attr(self) -> max._core.Attribute | None: ...
    @property
    def need_elide_type(self) -> max._core.dialects.builtin.UnitAttr: ...

class TypeConformsToTraitAttr(max._core.Attribute):
    """
    This represents a flag to indicate the type, specified by `typeValue`,
    conforms to specific traits, specified by a list of trait names.

    FIXME: The only reason that we uses a list of string to represent trait in
    this attr is because trait types are not preserved after lower-lit, meaning
    that the only way to refer to a specific trait during elaboration time is
    through symbol names.
    """

    @overload
    def __init__(
        self,
        type_value: max._core.dialects.builtin.TypedAttr,
        trait_names: VariadicAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        type_value: max._core.dialects.builtin.TypedAttr,
        trait_names: VariadicAttr,
    ) -> None: ...
    @property
    def type_value(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def trait_names(self) -> VariadicAttr: ...

class TypeGeneratorRefAttr(max._core.Attribute):
    """
    This is a symbolic reference to a type-value generator. Its type is the
    metatype of the type-value. If the type-value is parametric, additional
    parameter values may be bound.

    TODO: Merge SymbolConstantAttr into this.
    """

    @overload
    def __init__(
        self,
        symbol: max._core.dialects.builtin.SymbolRefAttr,
        type: max._core.Type,
    ) -> None: ...
    @overload
    def __init__(
        self,
        symbol: max._core.dialects.builtin.SymbolRefAttr,
        param_values: Sequence[max._core.dialects.builtin.TypedAttr],
        type: max._core.Type,
    ) -> None: ...
    @overload
    def __init__(
        self,
        symbol: max._core.dialects.builtin.SymbolRefAttr,
        param_values: Sequence[max._core.dialects.builtin.TypedAttr],
        type: max._core.Type,
    ) -> None: ...
    @property
    def symbol(self) -> max._core.dialects.builtin.SymbolRefAttr: ...
    @property
    def param_values(
        self,
    ) -> Sequence[max._core.dialects.builtin.TypedAttr]: ...
    @property
    def type(self) -> max._core.Type | None: ...

class TypeInstanceRefAttr(max._core.Attribute):
    """
    This is a symbolic reference to a concrete type-value instance. Its type
    is the metatype of the type-value.
    """

    @overload
    def __init__(
        self,
        symbol: max._core.dialects.builtin.SymbolRefAttr,
        type: max._core.Type,
    ) -> None: ...
    @overload
    def __init__(
        self,
        symbol: max._core.dialects.builtin.SymbolRefAttr,
        type: max._core.Type,
    ) -> None: ...
    @property
    def symbol(self) -> max._core.dialects.builtin.SymbolRefAttr: ...
    @property
    def type(self) -> max._core.Type | None: ...

class TypeParamAttr(max._core.Attribute):
    """
    This represents a parameter whose value is an MLIR type.  It is similar to
    `TypeAttr` in that it is an attribute that refers to a type.  The difference
    is that it is a `TypedAttr` so it can be a parameter expression, and has a
    metatype.

    The `typeValue` field encodes the value-representation of a type, while the
    `mlirType` field encodes the type-representation of the type.

    Example:

    ```mlir
    // Default asm format.
    #kgen.type<!myTypeValue, !myMlirType> : !kgen.type

    // MlirType is omitted if same as typeValue.
    #kgen.type<!myTypeValue> : !kgen.type
    ```
    """

    @overload
    def __init__(
        self, mlir_type: max._core.Type, type: max._core.Type
    ) -> None: ...
    @overload
    def __init__(
        self,
        type_value: max._core.Type,
        mlir_type: max._core.Type,
        type: max._core.Type,
    ) -> None: ...
    @overload
    def __init__(
        self,
        ctx: Context,
        type_value: max._core.Type,
        mlir_type: max._core.Type,
        type: max._core.Type,
    ) -> None: ...
    @property
    def type_value(self) -> max._core.Type | None: ...
    @property
    def mlir_type(self) -> max._core.Type | None: ...
    @property
    def type(self) -> max._core.Type | None: ...

class UnboundAttr(max._core.Attribute):
    """
    The `#kgen.unbound` attribute represents an unbound parameter value. It is
    a special placeholder that appears in collections of parameters that have
    been partially bound.
    """

    @overload
    def __init__(self, type: max._core.Type) -> None: ...
    @overload
    def __init__(self, type: max._core.Type) -> None: ...
    @property
    def type(self) -> max._core.Type | None: ...

class UnknownAttr(max._core.Attribute):
    """
    The `#kgen.unknown` attribute represents an unknown parameter value. It is
    a special placeholder that represents a parameter that may only be known
    dynamically.
    """

    @overload
    def __init__(self, type: max._core.Type) -> None: ...
    @overload
    def __init__(self, type: max._core.Type) -> None: ...
    @property
    def type(self) -> max._core.Type | None: ...

class UpcastAttr(max._core.Attribute):
    """
    The `#kgen.upcast` attribute is used to convert from a (variadic of) typeValue
    to a (variadic of) typeValue of a less-derived trait.
    For example, this can represent a cast from Movable to AnyType, handling the
    rebind of the `__del__` member.

    This aggressively canonicalizes, e.g. when the operand is a simple type
    value like a struct, it will return a TypeParamAttr.

    Example:

    ```mlir
    #kgen.upcast<#kgen.param.decl.ref<"T"> : !lit.trait<Movable>> : !lit.trait<AnyType>
    ```
    """

    @overload
    def __init__(
        self,
        type: max._core.Type,
        input_type_value: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        type: max._core.Type,
        input_type_value: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def type(self) -> max._core.Type | None: ...
    @property
    def input_type_value(self) -> max._core.dialects.builtin.TypedAttr: ...

class VariadicAttr(max._core.Attribute):
    """
    The `#kgen.variadic` attribute contains a homogeneous list of elements of an
    variadic type. It is used to represent constant variadic sequence values.

    Example:

    ```mlir
    #kgen.variadic<1, 2> : !kgen.variadic<index>
    ```
    """

    @overload
    def __init__(
        self,
        values: Sequence[max._core.dialects.builtin.TypedAttr],
        type: VariadicType,
    ) -> None: ...
    @overload
    def __init__(
        self,
        values: Sequence[max._core.dialects.builtin.TypedAttr],
        type: VariadicType,
    ) -> None: ...
    @property
    def values(self) -> Sequence[max._core.dialects.builtin.TypedAttr]: ...
    @property
    def type(self) -> VariadicType: ...

class VariadicConcatAttr(max._core.Attribute):
    """
    The `#kgen.variadic.concat` attribute is used to concatenate a variadic of
    variadic (type) value.

    Example:
    ```mlir
    #kgen.variadic.concat<[[Int, Int], [Float, Float]]> : !variadic<!AnyType>
    // ->
    #kgen.variadic<[Int, Int, Float, Float]> : !variadic<!AnyType>
    ```
    """

    @overload
    def __init__(
        self,
        type: VariadicType,
        variadics: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        type: VariadicType,
        variadics: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def type(self) -> VariadicType: ...
    @property
    def variadics(self) -> max._core.dialects.builtin.TypedAttr: ...

class VariadicReduceAttr(max._core.Attribute):
    """
    The `#kgen.variadic.reduce` attribute is used to reduce a variadic of
    (type) value to a (type) value by repeatedly applying the provided reducer
    on each element of the variadic.
    """

    def __init__(
        self,
        type: max._core.Type,
        base: max._core.dialects.builtin.TypedAttr,
        variadic: max._core.dialects.builtin.TypedAttr,
        generator: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def type(self) -> max._core.Type | None: ...
    @property
    def base(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def variadic(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def generator(self) -> max._core.dialects.builtin.TypedAttr: ...

class VariadicSizeAttr(max._core.Attribute):
    @overload
    def __init__(
        self,
        type: max._core.dialects.builtin.IndexType,
        variadic: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        type: max._core.dialects.builtin.IndexType,
        variadic: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def type(self) -> max._core.dialects.builtin.IndexType: ...
    @property
    def variadic(self) -> max._core.dialects.builtin.TypedAttr: ...

class VariadicSplatAttr(max._core.Attribute):
    """
    The `#kgen.variadic.splat` creates a variadic by splatting the same value
    to the given times.

    Example:
    ```mlir
    #kgen.variadic.splat<Int, 5> : !variadic<!AnyType>
    // ->
    #kgen.variadic<[Int, Int, Int, Int, Int]> : !variadic<!AnyType>
    ```
    """

    @overload
    def __init__(
        self,
        type: VariadicType,
        element: max._core.dialects.builtin.TypedAttr,
        count: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        type: VariadicType,
        element: max._core.dialects.builtin.TypedAttr,
        count: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def type(self) -> VariadicType: ...
    @property
    def element(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def count(self) -> max._core.dialects.builtin.TypedAttr: ...

class VariadicZipAttr(max._core.Attribute):
    """
    The `#kgen.variadic.zip` attribute is used to zip a variadic of
    variadic (type) value.

    Example:
    ```mlir
    #kgen.variadic.zip<[[Int, Int], [Float, Float]]> : !variadic<!variadic<!AnyType>>
    // ->
    #kgen.variadic<[[Int, Float], [Int, Float]]> : !variadic<!variadic<!AnyType>>
    ```

    At the moment, when the provided variadics are of different lengths, we zip
    till the shortest variadic are consumed. In the future, we might want to
    extend the attribute to accept an "default" value for "zip_longest".
    """

    @overload
    def __init__(
        self,
        type: VariadicType,
        input_type_value: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        type: VariadicType,
        input_type_value: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def type(self) -> VariadicType: ...
    @property
    def variadics(self) -> max._core.dialects.builtin.TypedAttr: ...

class VariantAttr(max._core.Attribute):
    """
    The `#kgen.variant` attribute represents a constant `!kgen.variant` value.
    It contains a single typed attribute where the typed of the value is
    expected to equal one of the corresponding variant types.

    Example:

    ```mlir
    #kgen.variant<3 : index> : !kgen.variant<index, i32, f32>
    ```
    """

    @overload
    def __init__(
        self,
        value: max._core.dialects.builtin.TypedAttr,
        index: int,
        type: VariantType,
    ) -> None: ...
    @overload
    def __init__(
        self,
        value: max._core.dialects.builtin.TypedAttr,
        index: int,
        type: VariantType,
    ) -> None: ...
    @property
    def value(self) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def index(self) -> int: ...
    @property
    def type(self) -> VariantType: ...

class ComputeKind(enum.Enum):
    addition = 0

    comparison = 1

    division = 2

    multiplication = 3

    multiply_add = 4

    other = 5

class EmitAs(enum.Enum):
    asm = 0

    llvm = 1

    llvm_opt = 2

    object = 3

    llvm_bitcode = 4

    llvm_opt_bitcode = 5

class ArgConvention(enum.Enum):
    read = 0

    read_mem = 1

    owned = 2

    owned_in_mem = 3

    deinit_mem = 4

    mut = 5

    ref = 6

    mutref = 7

    byref_result = 8

    byref_error = 9

class ArgConventionAttr(max._core.Attribute):
    def __init__(self, arg0: Context, arg1: ArgConvention, /) -> None: ...
    @property
    def value(self) -> ArgConvention: ...

class ClosureMemoryKind(enum.Enum):
    escaping = 0

    nonescaping = 1

    trivial = 2

    register_passable = 3

class ClosureMethod(enum.Enum):
    call = 0

    del_ = 1

    move = 2

    copy = 3

    none = 4

class ExportKind(enum.Enum):
    not_exported = 0

    exported = 1

    c_exported = 2

    package_exported = 3

class FnEffects(enum.Enum):
    none = 0

    throws = 1

    async_ = 2

    capturing = 4

    escaping = 16

    refresult = 32

    unified = 64

    register_passable = 128

    extern = 256

class InlineLevel(enum.Enum):
    automatic = 0

    always = 1

    always_nodebug = 2

    always_builtin = 3

    never = 4

class InlineLevelAttr(max._core.Attribute):
    def __init__(self, arg0: Context, arg1: InlineLevel, /) -> None: ...
    @property
    def value(self) -> InlineLevel: ...

class POC(enum.Enum):
    add = 0

    mul = 1

    mul_no_wrap = 2

    and_ = 3

    or_ = 4

    xor = 5

    max = 6

    min = 7

    shl = 8

    shr = 9

    div = 10

    mod = 11

    eq = 12

    lt = 13

    le = 14

    in_ = 15

    cond = 16

    current_target = 17

    target_has_feature = 18

    target_get_field = 19

    accelerator_arch = 20

    cross_compilation = 21

    get_env = 22

    get_sizeof = 23

    get_alignof = 24

    apply = 25

    apply_result_slot = 26

    rebind = 27

    variadic_get = 28

    ptr_bitcast = 34

    load_from_mem = 35

    variadic_ptr_map = 36

    variadic_ptrremove_map = 37

    attr_to_str = 39

    data_to_str = 40

    string_address = 41

    str_concat = 42

    function_get_arg_types = 43

    div_s = 44

    div_u = 45

    ceil_div_s = 46

    ceil_div_u = 47

    floor_div_s = 48

    rem_s = 49

    rem_u = 50

class POCAttr(max._core.Attribute):
    def __init__(self, arg0: Context, arg1: POC, /) -> None: ...
    @property
    def value(self) -> POC: ...

class SugarKind(enum.Enum):
    aibuiltin = 0

    preserved = 1

    member_alias = 2

    alias = 3

class TailKind(enum.Enum):
    none = 0

    musttail = 1

    notail = 2

    tail = 3

class CallIndirectOp(max._core.Operation):
    """
    The `kgen.call_indirect` operation takes an SSA value of `!kgen.generator`
    type (that wraps a `!kgen.func` type) and invokes it with the provided
    operands as arguments.

    Example:

    ```mlir
    %0 = kgen.call_indirect %closure(%arg0, %arg1)
      : (index, index) capturing -> index
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        results: Sequence[max._core.Type],
        callee: max._core.Value[FuncTypeGeneratorType],
        arguments: Sequence[max._core.Value[max._core.Type]],
        tail_kind: TailKindAttr,
    ) -> None: ...
    @property
    def callee(self) -> max._core.Value[FuncTypeGeneratorType]: ...
    @property
    def arguments(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def tail_kind(self) -> TailKind: ...
    @tail_kind.setter
    def tail_kind(self, arg: TailKindAttr, /) -> None: ...

class CallParamOp(max._core.Operation):
    """
    The `kgen.call_param` operation invokes a parametric callee. The callee is
    a parameter expression with a signature type, which in practice is either a
    symbol constant (a KGEN function or generator), or a region
    body passed from higher up the call stack.

    Example:

    ```mlir
    // Symbol constant callee.
    %0 = kgen.call_param[(index) -> index: @someFn](%arg0)

    // Parameter reference callee.
    %1 = kgen.call_param[<N>() -> index: foo]<N = 4>()
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        results: Sequence[max._core.Type],
        callee: max._core.dialects.builtin.TypedAttr,
        operands: Sequence[max._core.Value[max._core.Type]],
        tail_kind: TailKindAttr,
    ) -> None: ...
    @property
    def callee(self) -> max._core.dialects.builtin.TypedAttr: ...
    @callee.setter
    def callee(self, arg: max._core.dialects.builtin.TypedAttr, /) -> None: ...
    @property
    def operands(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def tail_kind(self) -> TailKind: ...
    @tail_kind.setter
    def tail_kind(self, arg: TailKindAttr, /) -> None: ...

class CaptureListCopyOp(max._core.Operation):
    """
    The `kgen.capture_list.copy` operation allocates heap memory and
    copies the capture list for a closure.

    Example:

    ```mlir
    %cl = kgen.capture_list.copy %orig :() capturing -> index @cap_closure
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: PointerType,
        orig: max._core.Value[PointerType],
        callee: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def orig(self) -> max._core.Value[PointerType]: ...
    @property
    def callee(self) -> max._core.dialects.builtin.TypedAttr: ...
    @callee.setter
    def callee(self, arg: max._core.dialects.builtin.TypedAttr, /) -> None: ...

class CaptureListCreateOp(max._core.Operation):
    """
    The `kgen.capture_list.create` operation allocates heap memory and
    initializes the capture list for a closure.

    Example:

    ```mlir
    %cl = kgen.capture_list.create :() capturing -> index @cap_closure
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: PointerType,
        callee: max._core.dialects.builtin.TypedAttr,
        args: Sequence[max._core.Value[max._core.Type]],
    ) -> None: ...
    @property
    def callee(self) -> max._core.dialects.builtin.TypedAttr: ...
    @callee.setter
    def callee(self, arg: max._core.dialects.builtin.TypedAttr, /) -> None: ...
    @property
    def args(self) -> Sequence[max._core.Value[max._core.Type]]: ...

class CaptureListExpandOp(max._core.Operation):
    """
    The `kgen.capture_list.expand` operation unpacks the capture list of the
    enclosing function.

    Example:

    ```mlir
    kgen.capture_list.expand %cl
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        capture_list: max._core.Value[PointerType],
    ) -> None: ...
    @property
    def capture_list(self) -> max._core.Value[PointerType]: ...

class ClosureInitOp(max._core.Operation):
    """
    The `kgen.closure.init` operation represents the instantiation of a closure
    struct. The closure struct holds the captured values. It maps the nested
    function to the capture struct.

    The closure init op contains a list of captured values. For each value,
    there is an optional symbol list. The first symbol is the copy or move
    symbol used to copy or move the captured value into the closure.

    The second and third symbols are the move and the del methods for that
    capture. If the first symbol is the move symbol then the second symbol
    is the del method. The assumption is that copyable values are movable.
    Movable values may be copyable but since closures are not copyable we
    do not need the copy symbol if the captured value is not captured by
    copy.

    Example:
    ```mlir
    kgen.generator @closure_types(%arg0 : index,
       %foo: !kgen.pointer<struct<(index,index)>>) {
      %3 = kgen.closure.init(%foo[@copy, @move, @del])(%arg1: index) -> index {
      %0 = kgen.struct.gep %foo[0] : !kgen.pointer<struct<(index,index)>>
      %1 = pop.load %0 : !kgen.pointer<index>
      kgen.return %1 : index
      } : (!kgen.pointer<struct<(index,index)>>),
          <!kgen.closure<@closure_types, "name" escaping>>

    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.Type,
        func_type_generator: max._core.dialects.builtin.TypeAttr,
        function_type: max._core.dialects.builtin.TypeAttr,
        captures: Sequence[max._core.Value[max._core.Type]],
        move_or_copy_capture_symbols: max._core.dialects.builtin.ArrayAttr,
        input_params: ParamDeclArrayAttr,
        inline_level: InlineLevelAttr,
        nested_fn_scope: max._core.Attribute,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.Type,
        func_type_generator: FuncTypeGeneratorType,
        function_type: max._core.dialects.builtin.FunctionType,
        captures: Sequence[max._core.Value[max._core.Type]],
        move_or_copy_capture_symbols: max._core.dialects.builtin.ArrayAttr,
        input_params: Sequence[ParamDeclAttr],
        inline_level: InlineLevel,
    ) -> None: ...
    @property
    def func_type_generator(self) -> FuncTypeGeneratorType: ...
    @func_type_generator.setter
    def func_type_generator(
        self, arg: max._core.dialects.builtin.TypeAttr, /
    ) -> None: ...
    @property
    def function_type(self) -> max._core.dialects.builtin.FunctionType: ...
    @function_type.setter
    def function_type(
        self, arg: max._core.dialects.builtin.TypeAttr, /
    ) -> None: ...
    @property
    def captures(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def move_or_copy_capture_symbols(
        self,
    ) -> max._core.dialects.builtin.ArrayAttr: ...
    @move_or_copy_capture_symbols.setter
    def move_or_copy_capture_symbols(
        self, arg: max._core.dialects.builtin.ArrayAttr, /
    ) -> None: ...
    @property
    def input_params(self) -> Sequence[ParamDeclAttr]: ...
    @input_params.setter
    def input_params(self, arg: ParamDeclArrayAttr, /) -> None: ...
    @property
    def inline_level(self) -> InlineLevel: ...
    @inline_level.setter
    def inline_level(self, arg: InlineLevelAttr, /) -> None: ...
    @property
    def nested_fn_scope(self) -> max._core.Attribute | None: ...
    @nested_fn_scope.setter
    def nested_fn_scope(self, arg: max._core.Attribute, /) -> None: ...

class CodegenReachableOp(max._core.Operation):
    """
    The `kgen.codegen.reachable` operation checks if
    codegen into runtimne code is allowed or not.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        cond: max._core.dialects.builtin.TypedAttr,
        message: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def cond(self) -> max._core.dialects.builtin.TypedAttr: ...
    @cond.setter
    def cond(self, arg: max._core.dialects.builtin.TypedAttr, /) -> None: ...
    @property
    def message(self) -> max._core.dialects.builtin.TypedAttr: ...
    @message.setter
    def message(self, arg: max._core.dialects.builtin.TypedAttr, /) -> None: ...

class CompileOffloadOp(max._core.Operation):
    """
    The `kgen.compile_offload` operation indicates compilation to a
    heterogenous target.

    `target_type` is target the offload function is compiled to.

    `emission_kind` is the output type for compile this offload function,
    i.e. asm, shared object, etc.

    `emission_option` is for extra compilation options for compiling
    this offload function.

    `func` is the offload function.

    `kernelID` is an integer number to identify this op from compiled results
    where multiple functions of the same target are bundled together for
    compilation.

    Example:

    ```mlir
    %0 = kgen.compile_offload<nvptx, 0, "", : ()->() @kernel> : !kgen.none
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.Type,
        target_type: max._core.dialects.builtin.TypedAttr,
        emission_kind: max._core.dialects.builtin.TypedAttr,
        emission_option: max._core.dialects.builtin.TypedAttr,
        func: max._core.dialects.builtin.TypedAttr,
        kernel_id: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def target_type(self) -> max._core.dialects.builtin.TypedAttr: ...
    @target_type.setter
    def target_type(
        self, arg: max._core.dialects.builtin.TypedAttr, /
    ) -> None: ...
    @property
    def emission_kind(self) -> max._core.dialects.builtin.TypedAttr: ...
    @emission_kind.setter
    def emission_kind(
        self, arg: max._core.dialects.builtin.TypedAttr, /
    ) -> None: ...
    @property
    def emission_option(self) -> max._core.dialects.builtin.TypedAttr: ...
    @emission_option.setter
    def emission_option(
        self, arg: max._core.dialects.builtin.TypedAttr, /
    ) -> None: ...
    @property
    def func(self) -> max._core.dialects.builtin.TypedAttr: ...
    @func.setter
    def func(self, arg: max._core.dialects.builtin.TypedAttr, /) -> None: ...
    @property
    def kernel_id(self) -> max._core.dialects.builtin.TypedAttr | None: ...
    @kernel_id.setter
    def kernel_id(
        self, arg: max._core.dialects.builtin.TypedAttr, /
    ) -> None: ...

class ConformanceOp(max._core.Operation):
    """
    The `kgen.conformance` operation defines the conformance table of a struct
    type for a trait.

    Its body contains the conformance table entries that map trait requirements
    to the struct type's definitions.

    - The optional `traitRef` parameter is a reference to the trait being
      conformed to.
    - The `immediateParents` parameter contains the conformance tables that this
      conformance table directly inherits from. It only includes the first level
      of parents, not any further ancestors.
    - The `constraint` parameter specifies the condition under which this
      conformance applies. This is used for conditional trait conformance,
      where a struct only conforms to a trait when certain conditions are met.
      Unconditional conformances use a trivially true constraint (proposition =
      constant 1) and are not printed with a `where` clause. Note: the builder
      canonicalizes null constraints to the trivially true constraint.

    Logically, a ConformanceOp represents a witness table whose contents is a
    concatenation of each parent ConformanceOp's conformance table followed by
    the entries of the ConformanceOp itself. The parent conformance tables are
    ordered by the name of the ConformanceOps (also the order in
    `immediateParents`).

    Example:

    ```mlir
    kgen.struct.generator @SIMD<type: dtype, size> = ... {
      kgen.conformance @Boolable {
        kgen.witness @"__bool__" : (!pop.simd<size, type>) -> i1
          = @"SIMD::__bool__(::SIMD[$0, $1])"<:dtype type, size>
      }
      ...
    }
    ```

    Example with conditional conformance:

    ```mlir
    kgen.struct.generator @List<T: type> = ... {
      kgen.conformance @Copyable where #kgen.constraint<conforms_to(T, Copyable), loc> {
        ...
      }
      ...
    }
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        sym_name: max._core.dialects.builtin.StringAttr,
        trait_ref: max._core.dialects.builtin.SymbolRefAttr,
        immediate_parents: max._core.dialects.m.SymbolRefArrayAttr,
        constraint: ConstraintAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        sym_name: max._core.dialects.builtin.StringAttr,
        trait_ref: max._core.dialects.builtin.SymbolRefAttr,
        immediate_parents: max._core.dialects.m.SymbolRefArrayAttr,
    ) -> None: ...
    @property
    def sym_name(self) -> str: ...
    @sym_name.setter
    def sym_name(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...
    @property
    def trait_ref(self) -> max._core.dialects.builtin.SymbolRefAttr | None: ...
    @trait_ref.setter
    def trait_ref(
        self, arg: max._core.dialects.builtin.SymbolRefAttr, /
    ) -> None: ...
    @property
    def immediate_parents(
        self,
    ) -> Sequence[max._core.dialects.builtin.SymbolRefAttr]: ...
    @immediate_parents.setter
    def immediate_parents(
        self, arg: max._core.dialects.m.SymbolRefArrayAttr, /
    ) -> None: ...
    @property
    def constraint(self) -> ConstraintAttr: ...
    @constraint.setter
    def constraint(self, arg: ConstraintAttr, /) -> None: ...

class CostOfOp(max._core.Operation):
    """
    The `kgen.cost_of` operation takes a parametric callee and returns
    characteristics of the function that can be used to develop heuristics for
    the cost of the function. This operation must be resolved at compile time.

    Currently, `kgen.cost_of` returns the number of loads, stores, additions,
    comparisons, divisions, multiplications, multiply-adds, and other
    operations, that is, operations that do not fall into any of the above
    categories. The cost is evaluated on the function at the output of
    elaboration, without running any post-elaboration passes.

    Example:

    ```mlir
    %loads, %stores, %additions, %comparisions, %divisions, %multiplications,
    %multiply_adds, %other = kgen.cost_of[(si8) -> si8: @foo]
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        loads: max._core.dialects.builtin.IndexType,
        stores: max._core.dialects.builtin.IndexType,
        additions: max._core.dialects.builtin.IndexType,
        comparisons: max._core.dialects.builtin.IndexType,
        divisions: max._core.dialects.builtin.IndexType,
        multiplications: max._core.dialects.builtin.IndexType,
        multiply_adds: max._core.dialects.builtin.IndexType,
        other: max._core.dialects.builtin.IndexType,
        callee: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def callee(self) -> max._core.dialects.builtin.TypedAttr: ...
    @callee.setter
    def callee(self, arg: max._core.dialects.builtin.TypedAttr, /) -> None: ...

class CreateClosureOp(max._core.Operation):
    """
    The `kgen.create_closure` operation represents the instantiation of
    a closure. This operation models closure creation in terms of a function
    and local captured variables.

    ```mlir
    %idx0 = index.constant 0
    %0 = kgen.create_closure[(index) -> index: @h](%idx0)
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: FuncTypeGeneratorType,
        callee: max._core.dialects.builtin.TypedAttr,
        captures: Sequence[max._core.Value[max._core.Type]],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        callee: max._core.dialects.builtin.TypedAttr,
        captures: Sequence[max._core.Value[max._core.Type]],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        callee: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def callee(self) -> max._core.dialects.builtin.TypedAttr: ...
    @callee.setter
    def callee(self, arg: max._core.dialects.builtin.TypedAttr, /) -> None: ...
    @property
    def captures(self) -> Sequence[max._core.Value[max._core.Type]]: ...

class CreateRegStubOp(max._core.Operation):
    """
    This op is similar to CreateClosureOp (without captures).
    But during LowerArgConvention, some arguments are promoted from by-memory
    to by-value.
    This would also affect the signature type of CreateClosureOp.

    With CreateRegStubOp, during LowerArgConvention, the signature of `callee`
    might change, but the result signature always remains the same.
    The compiler also adds the correct load / stores around `callee`'s call.

    This is needed for interoperability. We can hold mojo register_passable
    types by pointer in C++, and create a function pointer with CreateRegStub.
    Even if the compiled mojo functions excepts value arguments, you can have
    a function pointer to a wrapper that takes arguments by pointer.

    For example:
    ```mlir
    kgen.create_reg_stub [
      (!kgen.pointer<index> owned_in_mem, !kgen.pointer<index> byref_result)
      -> !kgen.none: @regtype__moveinit__] :
      <(!kgen.pointer<struct<(index) memoryOnly>> owned_in_mem,
        !kgen.pointer<struct<(index) memoryOnly>> byref_result) -> !kgen.none>
    ```

    The type is wrapped around a memory struct to preserve the signature,
    and also to indicate to LLVM that pointers don't alias.

    After LowerCallConvention, only `callee`'s signature change:
    ```mlir
    kgen.create_reg_stub [(index) -> index: @regtype__moveinit__] :
      <(!kgen.pointer<struct<(index) memoryOnly>> owned_in_mem,
        !kgen.pointer<struct<(index) memoryOnly>> byref_result) -> !kgen.none>
    ```

    Callee can also take any other kind of arguments (eg memory mojo objects,
    scalars, raw pointers) that don't get promoted:
    ```mlir
    kgen.create_reg_stub [
      (!kgen.pointer<struct<(index) memoryOnly>> owned_in_mem,
      !pop.scalar<si16> borrow) -> !kgen.none: @foo] :
      <(!kgen.pointer<struct<(index) memoryOnly>> owned_in_mem,
      !pop.scalar<si16> borrow) -> !kgen.none>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: FuncTypeGeneratorType,
        callee: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        callee: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def callee(self) -> max._core.dialects.builtin.TypedAttr: ...
    @callee.setter
    def callee(self, arg: max._core.dialects.builtin.TypedAttr, /) -> None: ...

class DeferredOp(max._core.Operation):
    """
    The `kgen.deferred` is used to encapsulate operation that does have at least
    one !kgen.deferred typed attribute. That is, when operation cannot be
    constructed by the parser as it has non-typed attributes that require
    elaboration.
    It's expected that elaborator will replace `kgen.deferred` with the
    operation and attributes it holds.

    Example:

    ```mlir
    kgen.deferred "index.cmp"(%a, %b : !Int, !Int) {
      %pred = #kgen<deferred #index<cmp_predicate sle>> : !kgen.deferred } : i1

    kgen.deferred "index.cmp"(%a, %b : !Int, !Int) {
      pred = #kgen.param.expr<apply,
       #kgen.bind_params<:!lit.generator<<"cmp": !Bool>()
         -> !kgen.deferred> *"select_pred[::Bool]()", cmp> :
          !kgen.generator<!lit.generator<()
            -> !kgen.deferred>>> : !kgen.deferred} : i1
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        results: Sequence[max._core.Type],
        operands: Sequence[max._core.Value[max._core.Type]],
        op_name: max._core.dialects.builtin.StringAttr,
        op_attrs: max._core.dialects.builtin.DictionaryAttr,
    ) -> None: ...
    @property
    def operands(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def op_name(self) -> str: ...
    @op_name.setter
    def op_name(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...
    @property
    def op_attrs(self) -> max._core.dialects.builtin.DictionaryAttr: ...
    @op_attrs.setter
    def op_attrs(
        self, arg: max._core.dialects.builtin.DictionaryAttr, /
    ) -> None: ...

class ExternGeneratorOp(max._core.Operation):
    """
    The `kgen.extern.generator` operation declares a KGEN generator with a
    external definition. The definition and its dependencies must be made
    available prior to elaboration, but this operations allows manipulating
    sections of IR without requiring all transitive dependencies be present in
    the module.

    Example:

    ```mlir
    kgen.extern.generator @kernel<simd_width>(!pop.simd<simd_width, f32>)
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        sym_name: max._core.dialects.builtin.StringAttr,
        func_type_generator: max._core.dialects.builtin.TypeAttr,
        function_type: max._core.dialects.builtin.TypeAttr,
        input_params: ParamDeclArrayAttr,
        export_kind: ExportKindAttr,
    ) -> None: ...
    @property
    def sym_name(self) -> str: ...
    @sym_name.setter
    def sym_name(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...
    @property
    def func_type_generator(self) -> FuncTypeGeneratorType: ...
    @func_type_generator.setter
    def func_type_generator(
        self, arg: max._core.dialects.builtin.TypeAttr, /
    ) -> None: ...
    @property
    def function_type(self) -> max._core.dialects.builtin.FunctionType: ...
    @function_type.setter
    def function_type(
        self, arg: max._core.dialects.builtin.TypeAttr, /
    ) -> None: ...
    @property
    def input_params(self) -> Sequence[ParamDeclAttr]: ...
    @input_params.setter
    def input_params(self, arg: ParamDeclArrayAttr, /) -> None: ...
    @property
    def export_kind(self) -> ExportKind: ...
    @export_kind.setter
    def export_kind(self, arg: ExportKindAttr, /) -> None: ...

class FuncOp(max._core.Operation):
    """
    The `kgen.func` operation represents a concrete KGEN function. It has no
    input parameters, no results parameters, and cannot contain any parametric
    operations. A `kgen.func` represents an elaborated `kgen.generator` that can
    be lowered and compiled to an executable.

    The body of a `kgen.func` is a single block region terminated by a
    `kgen.return` whose operands represent the return values of the function.

    Example:

    ```mlir
    kgen.func @kernel(%arg0: index, %arg1: index) -> index {
      %0 = index.add %arg0, %arg1
      kgen.return %0 : index
    }
    ```

    There are cases where we might have a `kgen.func` with a
    `precompiledBodyRef` attribute, like so:

    ```mlir
    kgen.func @someFn(%arg0: index) -> index attributes {
      precompiledBodyRef = @importedLib
    } {
      ...
    }
    ```

    The presence of this attribute means that the function was already compiled
    all the way to an object file once, so we should avoid doing it again.

    The function can contain opaque metadata to pass onto LLVM in its
    `llvmMetadata` dictionary attribute. Note that this is not the same as LLVM
    function attributes, which is contained in the `llvmAttrs` dictionary
    attribute.
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        sym_name: max._core.dialects.builtin.StringAttr,
        func_type_generator: max._core.dialects.builtin.TypeAttr,
        decorators: DecoratorsAttr,
        inline_level: InlineLevelAttr,
        export_kind: ExportKindAttr,
        external: max._core.dialects.builtin.UnitAttr,
        convergent: max._core.dialects.builtin.UnitAttr,
        _llvm_metadata: max._core.dialects.builtin.DictionaryAttr,
        _llvm_arg_metadata: max._core.dialects.builtin.ArrayAttr,
        cross_device_captures: max._core.dialects.m.StringArrayAttr,
        coroutine_type: max._core.dialects.builtin.TypeAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        name: max._core.dialects.builtin.StringAttr,
        type: FuncType,
        inline_level: InlineLevel = InlineLevel.automatic,
        export_kind: ExportKind = ExportKind.not_exported,
        external: bool = False,
        convergent: bool = False,
        decorators: Sequence[max._core.dialects.builtin.TypedAttr] = [],
        llvm_metadata: max._core.dialects.builtin.DictionaryAttr = ...,
        llvm_arg_metadata: max._core.dialects.builtin.ArrayAttr = ...,
    ) -> None: ...
    @property
    def sym_name(self) -> str: ...
    @sym_name.setter
    def sym_name(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...
    @property
    def func_type_generator(self) -> FuncTypeGeneratorType: ...
    @func_type_generator.setter
    def func_type_generator(
        self, arg: max._core.dialects.builtin.TypeAttr, /
    ) -> None: ...
    @property
    def decorators(self) -> Sequence[max._core.dialects.builtin.TypedAttr]: ...
    @decorators.setter
    def decorators(self, arg: DecoratorsAttr, /) -> None: ...
    @property
    def inline_level(self) -> InlineLevel: ...
    @inline_level.setter
    def inline_level(self, arg: InlineLevelAttr, /) -> None: ...
    @property
    def export_kind(self) -> ExportKind: ...
    @export_kind.setter
    def export_kind(self, arg: ExportKindAttr, /) -> None: ...
    @property
    def external(self) -> bool: ...
    @external.setter
    def external(self, arg: max._core.dialects.builtin.UnitAttr, /) -> None: ...
    @property
    def convergent(self) -> bool: ...
    @convergent.setter
    def convergent(
        self, arg: max._core.dialects.builtin.UnitAttr, /
    ) -> None: ...
    @property
    def _llvm_metadata(self) -> max._core.dialects.builtin.DictionaryAttr: ...
    @_llvm_metadata.setter
    def _llvm_metadata(
        self, arg: max._core.dialects.builtin.DictionaryAttr, /
    ) -> None: ...
    @property
    def _llvm_arg_metadata(self) -> max._core.dialects.builtin.ArrayAttr: ...
    @_llvm_arg_metadata.setter
    def _llvm_arg_metadata(
        self, arg: max._core.dialects.builtin.ArrayAttr, /
    ) -> None: ...
    @property
    def cross_device_captures(
        self,
    ) -> Sequence[max._core.dialects.builtin.StringAttr]: ...
    @cross_device_captures.setter
    def cross_device_captures(
        self, arg: max._core.dialects.m.StringArrayAttr, /
    ) -> None: ...
    @property
    def coroutine_type(self) -> max._core.Type | None: ...
    @coroutine_type.setter
    def coroutine_type(
        self, arg: max._core.dialects.builtin.TypeAttr, /
    ) -> None: ...

class GeneratorOp(max._core.Operation):
    """
    The `kgen.generator` operation defines a function generator. A generator is
    a parametric template for a function. It can have input parameters and
    result parameters. Uses of parameters inside the function must obey the
    parameter use-def graph; there can be no cycles. Each input parameter as it
    is declared in the signature of the generator can be used in the declaration
    of subsequent parameters only and in the declaration of all function
    arguments and results.

    Example:

    ```mlir
    kgen.generator @add<rhs>(%lhs: index) -> index {
      %0 = kgen.param.constant = <rhs>
      %1 = index.add %lhs, %0
      kgen.return %1 : index
    }
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        sym_name: max._core.dialects.builtin.StringAttr,
        source_name: max._core.dialects.builtin.StringAttr,
        func_type_generator: max._core.dialects.builtin.TypeAttr,
        function_type: max._core.dialects.builtin.TypeAttr,
        input_params: ParamDeclArrayAttr,
        decorators: DecoratorsAttr,
        inline_level: InlineLevelAttr,
        export_kind: ExportKindAttr,
        external: max._core.dialects.builtin.UnitAttr,
        inlined_form: max._core.dialects.builtin.TypedAttr,
        _llvm_metadata_array: max._core.dialects.builtin.ArrayAttr,
        _llvm_arg_metadata_array: max._core.dialects.builtin.ArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        sym_name: max._core.dialects.builtin.StringAttr,
        source_name: max._core.dialects.builtin.StringAttr,
        type: FuncTypeGeneratorType,
        function_type: max._core.dialects.builtin.FunctionType,
        input_params: Sequence[ParamDeclAttr],
        inline_level: InlineLevel = InlineLevel.automatic,
        inlined_form: max._core.dialects.builtin.TypedAttr = ...,
        llvm_metadata_array: max._core.dialects.builtin.ArrayAttr = ...,
        llvm_arg_metadata_array: max._core.dialects.builtin.ArrayAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        sym_name: max._core.dialects.builtin.StringAttr,
        type: FuncTypeGeneratorType,
    ) -> None: ...
    @property
    def sym_name(self) -> str: ...
    @sym_name.setter
    def sym_name(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...
    @property
    def source_name(self) -> str | None: ...
    @source_name.setter
    def source_name(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...
    @property
    def func_type_generator(self) -> FuncTypeGeneratorType: ...
    @func_type_generator.setter
    def func_type_generator(
        self, arg: max._core.dialects.builtin.TypeAttr, /
    ) -> None: ...
    @property
    def function_type(self) -> max._core.dialects.builtin.FunctionType: ...
    @function_type.setter
    def function_type(
        self, arg: max._core.dialects.builtin.TypeAttr, /
    ) -> None: ...
    @property
    def input_params(self) -> Sequence[ParamDeclAttr]: ...
    @input_params.setter
    def input_params(self, arg: ParamDeclArrayAttr, /) -> None: ...
    @property
    def decorators(self) -> Sequence[max._core.dialects.builtin.TypedAttr]: ...
    @decorators.setter
    def decorators(self, arg: DecoratorsAttr, /) -> None: ...
    @property
    def inline_level(self) -> InlineLevel: ...
    @inline_level.setter
    def inline_level(self, arg: InlineLevelAttr, /) -> None: ...
    @property
    def export_kind(self) -> ExportKind: ...
    @export_kind.setter
    def export_kind(self, arg: ExportKindAttr, /) -> None: ...
    @property
    def external(self) -> bool: ...
    @external.setter
    def external(self, arg: max._core.dialects.builtin.UnitAttr, /) -> None: ...
    @property
    def inlined_form(self) -> max._core.dialects.builtin.TypedAttr | None: ...
    @inlined_form.setter
    def inlined_form(
        self, arg: max._core.dialects.builtin.TypedAttr, /
    ) -> None: ...
    @property
    def _llvm_metadata_array(self) -> max._core.dialects.builtin.ArrayAttr: ...
    @_llvm_metadata_array.setter
    def _llvm_metadata_array(
        self, arg: max._core.dialects.builtin.ArrayAttr, /
    ) -> None: ...
    @property
    def _llvm_arg_metadata_array(
        self,
    ) -> max._core.dialects.builtin.ArrayAttr: ...
    @_llvm_arg_metadata_array.setter
    def _llvm_arg_metadata_array(
        self, arg: max._core.dialects.builtin.ArrayAttr, /
    ) -> None: ...

class IsCompileTimeOp(max._core.Operation):
    """
    The `kgen.is_compile_time` represents a boolean value which is `true`
    during compile time and `false` otherwise.
    When used as condition for control flow, for example,
    only the `true` branch will be evaluated during compile
    time, while the other branch will be compiled to generated code.
    This helps to have efficient compile time (interpreter) for generic
    program without loss of runtime generated code efficiency.

    Example:

    ```mlir
      kgen.is_compile_time : i1
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.builtin.IntegerType,
    ) -> None: ...

class PackCreateOp(max._core.Operation):
    """
    The `kgen.pack.create` operation creates a value of `!kgen.pack` type,
    populated with the given SSA values.

    Example:

    ```mlir
    kgen.generator @pack<Ts: variadic<!kgen.type>>(
      %arg0: f32, %arg1: si8
    ) {
      // Create a pack of two elements.
      %0 = kgen.pack.create(%arg0, %arg1) : !kgen.pack<[f32, si8]>

      // Create that same pack of two elements, but with a parameterized result.
      %1 = kgen.pack.create(%arg0 : f32, %arg1 : si8) : !kgen.pack<Ts>

      // Create an empty pack.
      %2 = kgen.pack.create() : !kgen.pack<[]>

      kgen.return
    }
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: PackType,
        elements: Sequence[max._core.Value[max._core.Type]],
    ) -> None: ...
    @property
    def elements(self) -> Sequence[max._core.Value[max._core.Type]]: ...

class PackExtractOp(max._core.Operation):
    """
    The `kgen.pack.extract` operation returns the element of a pack at the
    provided index.  The index must be a parameter of index type.  This
    operation is resolved post-elaboration when the pack details become known.

    Example:

    ```mlir
    kgen.generator @pack<Ts: variadic<!kgen.type>, T: type, I: index>(
      %arg0: !kgen.pack<i32, T>
      %arg1: Ts,
    ) {
      // Get the first element, of type `i32`.
      %0 = kgen.pack.extract %arg0[0] : <[i32, T]>
      // Get the second element, of type `!kgen.param<T>`.
      %1 = kgen.pack.extract %arg0[1] : <[i32, T]>

      // Get the element at an offset `I + 1`.
      %2 = kgen.pack.extract %arg0[add(I, 1)] : <[i32, T]>

      // Get the element at index 3.
      %3 = kgen.pack.extract %arg1[3] : <Ts>

      kgen.return
    }
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.Type,
        pack: max._core.Value[PackType],
        index: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        pack: max._core.Value[PackType],
        index: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def pack(self) -> max._core.Value[PackType]: ...
    @property
    def index(self) -> max._core.dialects.builtin.TypedAttr: ...
    @index.setter
    def index(self, arg: max._core.dialects.builtin.TypedAttr, /) -> None: ...

class PackGepOp(max._core.Operation):
    """
    The `kgen.pack.gep` operation returns a pointer to the element of a pack at
    the provided index.  The index must be a parameter of index type.  This
    operation is resolved post-elaboration when the pack details become known.
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: PointerType,
        pack: max._core.Value[PointerType],
        index: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        pack: max._core.Value[PointerType],
        index: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def pack(self) -> max._core.Value[PointerType]: ...
    @property
    def index(self) -> max._core.dialects.builtin.TypedAttr: ...
    @index.setter
    def index(self, arg: max._core.dialects.builtin.TypedAttr, /) -> None: ...

class PackLoadOp(max._core.Operation):
    """
    The `kgen.pack.load` operation takes a pack of !kgen.pointer values and
    loads each one into a pack without the pointer type.  This requires
    elements with trivially loadable types supported by pop.load.
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: PackType,
        pack: max._core.Value[PackType],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        pack: max._core.Value[PackType],
    ) -> None: ...
    @property
    def pack(self) -> max._core.Value[PackType]: ...

class PackSizeOp(max._core.Operation):
    """
    The `kgen.pack.size` operation takes an operand with a `!kgen.pack` type and
    returns the number of elements in the pack.

    Example:

    ```mlir
    // Get the size of a pack.
    kgen.pack.size %0 : <Ts>
    // Get the size of a concrete pack with 2 elements.
    kgen.pack.size %1 : <[i32, f32]>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.builtin.IndexType,
        operand: max._core.Value[PackType],
    ) -> None: ...
    @property
    def operand(self) -> max._core.Value[PackType]: ...

class ParamApplyOp(max._core.Operation):
    """
    The `kgen.param.apply` operation is a call operation entirely in the
    parameter domain. The operands to the call are parameter expressions and the
    results of the call are bound to parameter declarations. This is the 'apply'
    operator as an operation.

    The callee is limited to a single result value with no result parameters.

    Example:

    ```mlir
    kgen.param.declare sw: i1 = <1>
    kgen.param.apply A = [(i1) -> !pop.simd<8, f32>: callee](sw)
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        param_decl: ParamDeclAttr,
        callee: max._core.dialects.builtin.TypedAttr,
        operands: ParameterExprArrayAttr,
    ) -> None: ...
    @property
    def param_decl(self) -> ParamDeclAttr: ...
    @param_decl.setter
    def param_decl(self, arg: ParamDeclAttr, /) -> None: ...
    @property
    def callee(self) -> max._core.dialects.builtin.TypedAttr: ...
    @callee.setter
    def callee(self, arg: max._core.dialects.builtin.TypedAttr, /) -> None: ...
    @property
    def operands(self) -> Sequence[max._core.dialects.builtin.TypedAttr]: ...
    @operands.setter
    def operands(self, arg: ParameterExprArrayAttr, /) -> None: ...

class ParamAssertOp(max._core.Operation):
    """
    The `kgen.param.assert` operation ensures that a parameter expression is
    true at elaboration time.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        cond: max._core.dialects.builtin.TypedAttr,
        message: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def cond(self) -> max._core.dialects.builtin.TypedAttr: ...
    @cond.setter
    def cond(self, arg: max._core.dialects.builtin.TypedAttr, /) -> None: ...
    @property
    def message(self) -> max._core.dialects.builtin.TypedAttr: ...
    @message.setter
    def message(self, arg: max._core.dialects.builtin.TypedAttr, /) -> None: ...

class ParamDeclareOp(max._core.Operation):
    """
    The `kgen.param.declare` operation declares a single parameter and binds
    its value to a parameter expression. The parameter is visible within and
    below the scope of the enclosing `DeclInterface` operation.

    Example:

    ```mlir
    kgen.param.declare A = <5>
    kgen.param.declare A_plus_one = <add(A, 1)>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        param_decl: ParamDeclAttr,
        value: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def param_decl(self) -> ParamDeclAttr: ...
    @param_decl.setter
    def param_decl(self, arg: ParamDeclAttr, /) -> None: ...
    @property
    def value(self) -> max._core.dialects.builtin.TypedAttr: ...
    @value.setter
    def value(self, arg: max._core.dialects.builtin.TypedAttr, /) -> None: ...

class ParamDeclareRegionOp(max._core.Operation):
    """
    The `kgen.param.declare.region` operation declares a signature-type
    parameter whose value is an MLIR region (a closure). The region can be
    isolated from above, in which case it is treated by the elaborator as a
    stateless closure and inlined at wherever it ends up being called. The
    region can also capture values from above, in which case the elaborator
    treats it as a stateful closure and processes it by inlining every call that
    forwards this parameter down to all callsites.

    Parameters defined by a `kgen.param.declare.region` that are not isolated
    from above can only be used by function calls. They cannot be used, for
    example, by `kgen.addressof`.

    Example:

    ```mlir
    kgen.param.declare.region AddIt[my_add] = <N>(%arg0: index) -> index {
      %0 = kgen.param.constant = <N>
      %1 = index.add %0, %arg0
      kgen.return %1 : index
    }

    kgen.param.declare.region SubtractIt = <N>(%arg0: index) -> index {
      %0 = kgen.param.constant = <N>
      %1 = index.sub %0, %arg0
      kgen.return %1 : index
    }
    ```

    The operation carries an extra bit `isolated` to indicate that they are
    parametrically isolated from above.
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        param_decl: ParamDeclAttr,
        source_name: max._core.dialects.builtin.StringAttr,
        func_type_generator: max._core.dialects.builtin.TypeAttr,
        function_type: max._core.dialects.builtin.TypeAttr,
        input_params: ParamDeclArrayAttr,
        inline_level: InlineLevelAttr,
        _llvm_metadata_array: max._core.dialects.builtin.ArrayAttr,
        _llvm_arg_metadata_array: max._core.dialects.builtin.ArrayAttr,
        isolated: max._core.dialects.builtin.UnitAttr,
    ) -> None: ...
    @property
    def param_decl(self) -> ParamDeclAttr: ...
    @param_decl.setter
    def param_decl(self, arg: ParamDeclAttr, /) -> None: ...
    @property
    def source_name(self) -> str: ...
    @source_name.setter
    def source_name(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...
    @property
    def func_type_generator(self) -> FuncTypeGeneratorType: ...
    @func_type_generator.setter
    def func_type_generator(
        self, arg: max._core.dialects.builtin.TypeAttr, /
    ) -> None: ...
    @property
    def function_type(self) -> max._core.dialects.builtin.FunctionType: ...
    @function_type.setter
    def function_type(
        self, arg: max._core.dialects.builtin.TypeAttr, /
    ) -> None: ...
    @property
    def input_params(self) -> Sequence[ParamDeclAttr]: ...
    @input_params.setter
    def input_params(self, arg: ParamDeclArrayAttr, /) -> None: ...
    @property
    def inline_level(self) -> InlineLevel: ...
    @inline_level.setter
    def inline_level(self, arg: InlineLevelAttr, /) -> None: ...
    @property
    def _llvm_metadata_array(self) -> max._core.dialects.builtin.ArrayAttr: ...
    @_llvm_metadata_array.setter
    def _llvm_metadata_array(
        self, arg: max._core.dialects.builtin.ArrayAttr, /
    ) -> None: ...
    @property
    def _llvm_arg_metadata_array(
        self,
    ) -> max._core.dialects.builtin.ArrayAttr: ...
    @_llvm_arg_metadata_array.setter
    def _llvm_arg_metadata_array(
        self, arg: max._core.dialects.builtin.ArrayAttr, /
    ) -> None: ...
    @property
    def isolated(self) -> bool: ...
    @isolated.setter
    def isolated(self, arg: max._core.dialects.builtin.UnitAttr, /) -> None: ...

class ParamForBreakOp(max._core.Operation):
    """
    The `kgen.param.for.break` operation represents an exit from a
    `kgen.param.for`, branching to the end of all generated iterations.
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
    ) -> None: ...
    @overload
    def __init__(
        self, builder: max._core.OpBuilder, location: Location
    ) -> None: ...
    @property
    def operands(self) -> Sequence[max._core.Value[max._core.Type]]: ...

class ParamForContinueOp(max._core.Operation):
    """
    The `kgen.param.for.continue` operation branches to the next generated
    iteration of the loop.
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
    ) -> None: ...
    @overload
    def __init__(
        self, builder: max._core.OpBuilder, location: Location
    ) -> None: ...
    @property
    def operands(self) -> Sequence[max._core.Value[max._core.Type]]: ...

class ParamForGotoElseOp(max._core.Operation):
    """
    The `kgen.param.for.goto.else` operation jumps to the 'else' block of a
    `kgen.param.for`.  It only exists to help the interface between the Mojo
    parser and the LowerSemanticCF pass work more easily.
    """

    def __init__(
        self, builder: max._core.OpBuilder, location: Location
    ) -> None: ...

class ParamForOp(max._core.Operation):
    """
    The `kgen.param.for` operation instantiates its body with values according
    to its iterator. It takes an initial iterator value, a 'hasNext' function
    that take an iterator and indicates whether more elements exist, and a
    'getNextIter' function that takes an iterator instance and returns the next
    iterator value.

    This operation can have loop-carried values - the "operands" inputs and
    results, which are values promoted within the loop by mem2reg.
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        results: Sequence[max._core.Type],
        initial: max._core.dialects.builtin.TypedAttr,
        has_next: max._core.dialects.builtin.TypedAttr,
        get_next_iter: max._core.dialects.builtin.TypedAttr,
        param_decl: ParamDeclAttr,
        operands: Sequence[max._core.Value[max._core.Type]],
        body_isolated: max._core.dialects.builtin.UnitAttr,
        else_isolated: max._core.dialects.builtin.UnitAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        initial: max._core.dialects.builtin.TypedAttr,
        has_next: max._core.dialects.builtin.TypedAttr,
        get_next_iter: max._core.dialects.builtin.TypedAttr,
        param_decl: ParamDeclAttr,
    ) -> None: ...
    @property
    def initial(self) -> max._core.dialects.builtin.TypedAttr: ...
    @initial.setter
    def initial(self, arg: max._core.dialects.builtin.TypedAttr, /) -> None: ...
    @property
    def has_next(self) -> max._core.dialects.builtin.TypedAttr: ...
    @has_next.setter
    def has_next(
        self, arg: max._core.dialects.builtin.TypedAttr, /
    ) -> None: ...
    @property
    def get_next_iter(self) -> max._core.dialects.builtin.TypedAttr: ...
    @get_next_iter.setter
    def get_next_iter(
        self, arg: max._core.dialects.builtin.TypedAttr, /
    ) -> None: ...
    @property
    def param_decl(self) -> ParamDeclAttr: ...
    @param_decl.setter
    def param_decl(self, arg: ParamDeclAttr, /) -> None: ...
    @property
    def operands(self) -> Sequence[max._core.Value[max._core.Type]]: ...
    @property
    def body_isolated(self) -> bool: ...
    @body_isolated.setter
    def body_isolated(
        self, arg: max._core.dialects.builtin.UnitAttr, /
    ) -> None: ...
    @property
    def else_isolated(self) -> bool: ...
    @else_isolated.setter
    def else_isolated(
        self, arg: max._core.dialects.builtin.UnitAttr, /
    ) -> None: ...

class ParamIfOp(max._core.Operation):
    """
    The `kgen.param.if` op provides a compile-time guarantee that if `cond`
    is false, only the `else` block will be elaborated, and if `cond` is true,
    only the `then` block will be elaborated. Whichever block is elaborated
    will be directly inlined into the scope. This op is fully removed
    during elaboration.

    The result parameter will be resolved by the parameter on the
    `param.yield` in the branch that is actually elaborated.

    ```mlir
    %0 = kgen.param.if <condition> -> index {
      %i0 = index.constant 0
      kgen.param.yield %i0
    } else {
      %i1 = index.constant 1
      kgen.param.yield %i1
    }

    kgen.param.if <condition -> result> {
      kgen.param.yield<1>
    } else {
      kgen.param.yield<0>
    }
    %0 = kgen.param.constant = <result> // == <condition ? 1 : 0>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        results: Sequence[max._core.Type],
        cond: max._core.dialects.builtin.TypedAttr,
        then_isolated: max._core.dialects.builtin.UnitAttr,
        else_isolated: max._core.dialects.builtin.UnitAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        cond: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def cond(self) -> max._core.dialects.builtin.TypedAttr: ...
    @cond.setter
    def cond(self, arg: max._core.dialects.builtin.TypedAttr, /) -> None: ...
    @property
    def then_isolated(self) -> bool: ...
    @then_isolated.setter
    def then_isolated(
        self, arg: max._core.dialects.builtin.UnitAttr, /
    ) -> None: ...
    @property
    def else_isolated(self) -> bool: ...
    @else_isolated.setter
    def else_isolated(
        self, arg: max._core.dialects.builtin.UnitAttr, /
    ) -> None: ...

class ParamYieldOp(max._core.Operation):
    """
    The `kgen.param.yield` operation is used to denote the terminator of a
    block in the `kgen.param.if` op. Conceptually, it branches to the next
    op after a `kgen.param.if`, but in reality it's to make sure the IR
    stays in a reasonable state and is possible to inline directly.

    This op defines the parameter that is used as the result parameter on the
    enclosing `kgen.param.if`.

    Example:

    ```mlir
    %0 = kgen.param.if <condition> -> index {
      kgen.param.yield %arg1 : index
    } else {
      kgen.param.yield %arg2 : index
    }
    ```

    This operation can always be marked pure because the control-flow edge from
    this operation leads only to after the parent operation.
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
    ) -> None: ...
    @overload
    def __init__(
        self, builder: max._core.OpBuilder, location: Location
    ) -> None: ...
    @property
    def operands(self) -> Sequence[max._core.Value[max._core.Type]]: ...

class RebindOp(max._core.Operation):
    """
    The `kgen.rebind` operation rebinds values with parametric types into other
    parameter domains. The operation can be used to convert between parameter
    domains or between concrete types. During elaboration, rebind operations
    must resolve to the same input and output types. Otherwise, elaboration will
    fail.

    Example:

    ```mlir
    // Rebind a parameterized type to a concrete type.
    %0 = kgen.rebind %arg0 : !kgen.param<type> to !pop.scalar<f32>

    // Rebind between parameter domains.
    %1 = kgen.rebind %arg1 : !kgen.param<type> to !pop.simd<size, dtype>

    // Unbind a concrete type to one with a parameter.
    %2 = kgen.rebind %arg2 : !pop.scalar<f32> to !pop.scalar<dtype>

    // ERROR: Cannot rebind between different concrete types.
    %3 = kgen.rebind %arg2 : !pop.scalar<f32> to !pop.scalar<si32>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        output: max._core.Type,
        input: max._core.Value,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value: ...

class ReturnOp(max._core.Operation):
    """
    The `kgen.return` operation specifies the returned values for a `kgen.func`
    or `kgen.generator`.

    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the
    `kgen.generator` (or enclosing region) that contains the operation. For
    example:

    ```mlir
      kgen.generator @foo() : i32, f8 {
        ...
        kgen.return %0, %1 : i32, f8
      }
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
    ) -> None: ...
    @overload
    def __init__(
        self, builder: max._core.OpBuilder, location: Location
    ) -> None: ...
    @property
    def operands(self) -> Sequence[max._core.Value[max._core.Type]]: ...

class SourceLocOp(max._core.Operation):
    """
    The `kgen.source_loc` operation enables implementing location capture of
    call stacks. When defined in an `@always_inline` or
    `@always_inline("nodebug")` function, it will return the capturing the call
    location of the enclosing function. By specifying an `inlineCount` larger
    than 0, it can generalize this behavior, e.g. for `inlineCount = 1`, it will
    return the call location of two steps up the call stack (as long as both
    calls are to `@always_inline` or `@always_inline("nodebug")` functions). In
    parameter contexts, no location context can be recovered, so the op will be
    interpreted as a constant of dummy location info.

    Example:
    ```mlir
     %line, %col, %fileName = kgen.source_loc[0]
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        line: max._core.dialects.builtin.IndexType,
        col: max._core.dialects.builtin.IndexType,
        file_name: StringType,
        inline_count: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def inline_count(self) -> max._core.dialects.builtin.TypedAttr: ...
    @inline_count.setter
    def inline_count(
        self, arg: max._core.dialects.builtin.TypedAttr, /
    ) -> None: ...

class StageClosureOp(max._core.Operation):
    """
    The `kgen.stage_closure` operation declares a named concrete region. The
    region can capture values from its parent region. This operation models
    nested callable functions within their original scope before lifting them
    into functions so that we may perform transformations that depend on the
    original structure of the scopes.

    ```mlir
    %0 = kgen.stage_closure = () capturing -> index {
      kgen.return %arg0 : index
    }
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: FuncTypeGeneratorType,
    ) -> None: ...

class StructCreateOp(max._core.Operation):
    """
    The `kgen.struct.create` operation creates a struct of the given type
    from values corresponding to its element types.

    Example:

    ```mlir
    %0 = kgen.struct.create(%f32, %f64)
      : !kgen.struct<(scalar<f32>, scalar<f64>)>

    %1 = kgen.struct.create(%arr, %ptr, %v)
      : !kgen.struct<(array<size, type>, pointer<type>, type)>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: StructType,
        elements: Sequence[max._core.Value[max._core.Type]],
    ) -> None: ...
    @property
    def elements(self) -> Sequence[max._core.Value[max._core.Type]]: ...

class StructExtractOp(max._core.Operation):
    """
    The `kgen.struct.extract` operation gets the struct element at the given
    index from the provided struct.

    Example:

    ```mlir
    // Extract the !pop.scalar<f32> at index 0.
    %0 = kgen.struct.extract %struct[0]
      : !kgen.struct<(scalar<f32>, scalar<f64>)>

    // Extract the !pop.scalar<f64> at index 1.
    %1 = kgen.struct.extract %struct[1]
      : !kgen.struct<(scalar<f32>, scalar<f64>)>

    // Extract an element at a parametric index.
    kgen.generator @example<I: index>(%struct: !kgen.struct<(i32, f32)>) {
      %0 = kgen.struct.extract %struct[I] : !kgen.struct<(i32, f32)>
      kgen.return
    }
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.Type,
        container: max._core.Value[StructType],
        index: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        container: max._core.Value[StructType],
        index: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        container: max._core.Value,
        index: int,
    ) -> None: ...
    @property
    def container(self) -> max._core.Value[StructType]: ...
    @property
    def index(self) -> max._core.dialects.builtin.TypedAttr: ...
    @index.setter
    def index(self, arg: max._core.dialects.builtin.TypedAttr, /) -> None: ...

class StructGepOp(max._core.Operation):
    """
    The `kgen.struct.gep` operation takes a pointer to a `!kgen.struct` and an
    index and returns a pointer to the struct element at that index. The index
    can be either a constant integer or a parametric expression that is resolved
    post-elaboration.

    Example:

    ```mlir
    %struct = pop.stack_allocation 1 : !kgen.struct<(i32, i64)>
    // Constant index - result type is inferred
    %i64Ptr = kgen.struct.gep %struct[1] : <struct<(i32, i64)>>
    ```

    With a parametric index:

    ```mlir
    kgen.generator @example<I: index>(%struct: !kgen.pointer<struct<(i32, i64)>>) {
      // Parametric index - result type must be specified
      %ptr = kgen.struct.gep %struct[I] : <struct<(i32, i64)>> -> <i64>
    }
    ```

    To get a pointer to a struct from a value-semantic struct, the struct value
    must be stored into a block of allocated memory first.

    ```mlir
    ^bb0(%struct: !kgen.struct<(i32, i64)>):
      %mem = pop.stack_allocation 1 : !kgen.struct<(i32, i64)>
      pop.store %struct, %mem : !kgen.pointer<struct<(i32, i64)>>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        container: max._core.Value,
        index: int,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result_type: max._core.Type,
        container: max._core.Value,
        index: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def container(self) -> max._core.Value[PointerType]: ...
    @property
    def index(self) -> max._core.dialects.builtin.TypedAttr: ...
    @index.setter
    def index(self, arg: max._core.dialects.builtin.TypedAttr, /) -> None: ...

class StructGeneratorOp(max._core.Operation):
    """
    The `kgen.struct.generator` operation defines a type-value generator. It is
    a parametric template for a type-value. It can have input parameters.

    Its body contains the definition for a potentially parametric struct type
    as a type-value.

    Example:

    ```mlir
    kgen.struct.generator @struct_SIMD<dt: dtype, size> : type
      = struct_inst<"struct_SIMD"[dt, size]<:dtype dt, size>(data: struct<()>)>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        sym_name: max._core.dialects.builtin.StringAttr,
        input_params: ParamDeclArrayAttr,
        value_domain_type: max._core.dialects.builtin.TypeAttr,
        meta_type: max._core.dialects.builtin.TypeAttr,
    ) -> None: ...
    @property
    def sym_name(self) -> str: ...
    @sym_name.setter
    def sym_name(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...
    @property
    def input_params(self) -> Sequence[ParamDeclAttr]: ...
    @input_params.setter
    def input_params(self, arg: ParamDeclArrayAttr, /) -> None: ...
    @property
    def value_domain_type(self) -> max._core.Type | None: ...
    @value_domain_type.setter
    def value_domain_type(
        self, arg: max._core.dialects.builtin.TypeAttr, /
    ) -> None: ...
    @property
    def meta_type(self) -> max._core.Type | None: ...
    @meta_type.setter
    def meta_type(
        self, arg: max._core.dialects.builtin.TypeAttr, /
    ) -> None: ...

class StructInstanceOp(max._core.Operation):
    """
    The `kgen.struct.instance` operation defines a concretized struct type
    instance.

    Its body contains the definition for a non-parametric struct type as a
    type-value.

    Example:

    ```mlir
    kgen.struct.instance @"struct_SIMD,dt=f32,size=4" : type
      = struct_inst<"struct_SIMD"[dt, size]<:dtype f32, 4>(data: struct<()>)>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        sym_name: max._core.dialects.builtin.StringAttr,
        value_domain_type: max._core.dialects.builtin.TypeAttr,
        meta_type: max._core.dialects.builtin.TypeAttr,
    ) -> None: ...
    @property
    def sym_name(self) -> str: ...
    @sym_name.setter
    def sym_name(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...
    @property
    def value_domain_type(self) -> max._core.Type | None: ...
    @value_domain_type.setter
    def value_domain_type(
        self, arg: max._core.dialects.builtin.TypeAttr, /
    ) -> None: ...
    @property
    def meta_type(self) -> max._core.Type | None: ...
    @meta_type.setter
    def meta_type(
        self, arg: max._core.dialects.builtin.TypeAttr, /
    ) -> None: ...

class StructLoadIndirectOp(max._core.Operation):
    """
    The `kgen.struct.load_indirect` operation takes a struct of !kgen.pointer
    values and loads each one into a struct without the pointer type.  This
    requires elements with trivially loadable types supported by pop.load.
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: StructType,
        struct_value: max._core.Value[StructType],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        struct_value: max._core.Value[StructType],
    ) -> None: ...
    @property
    def struct_value(self) -> max._core.Value[StructType]: ...

class StructReplaceOp(max._core.Operation):
    """
    The `kgen.struct.replace` operation inserts the given value into the struct
    at the given index. It returns a new struct with the inserted value.

    Example:

    ```mlir
    // Insert the !pop.scalar<f32> at index 0.
    %0 = kgen.struct.replace %f32, %struct[0]
      : !kgen.struct<(scalar<f32>, scalar<f64>)>

    // Insert the !pop.scalar<f64> at index 1.
    %1 = kgen.struct.replace %f64, %struct[1]
      : !kgen.struct<(scalar<f32>, scalar<f64>)>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: StructType,
        value: max._core.Value,
        container: max._core.Value[StructType],
        index: max._core.dialects.builtin.IntegerAttr,
    ) -> None: ...
    @property
    def value(self) -> max._core.Value: ...
    @property
    def container(self) -> max._core.Value[StructType]: ...
    @property
    def index(self) -> int: ...
    @index.setter
    def index(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...

class UnreachableOp(max._core.Operation):
    """
    The `kgen.unreachable` operation is a block terminator that indicates that
    the previous operations cannot continue control flow.  This is used after
    infinite loops and no-return functions to simplify dataflow analysis.

    Example:

    ```mlir
    hlcf.loop () {
      kgen.call @printHello()
      hlcf.continue
    }
    kgen.unreachable
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        is_after_unreachable_call: max._core.dialects.builtin.BoolAttr,
    ) -> None: ...
    @property
    def is_after_unreachable_call(self) -> bool: ...
    @is_after_unreachable_call.setter
    def is_after_unreachable_call(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...

class VariantCreateOp(max._core.Operation):
    """
    The `kgen.variant.create` operation creates a variant of the referred type
    with a value provided for one of its possible types.

    Example:

    ```mlir
    // Create an `std::optional<T>` variant.
    %none = kgen.struct.create() : !kgen.struct<()>
    %0 = kgen.variant.create %none, 0 : !kgen.struct<()>
        -> !kgen.variant<struct<()>, T>

    // Create a variant of either a scalar float or integer.
    %1 = kgen.param.constant: scalar<f64> = <<"0.0">>
    %2 = kgen.variant.create %1, 1 : !pop.scalar<f64>
        -> !kgen.variant<scalar<i64>, scalar<f64>>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: VariantType,
        operand: max._core.Value,
        index: max._core.dialects.builtin.IntegerAttr,
    ) -> None: ...
    @property
    def operand(self) -> max._core.Value: ...
    @property
    def index(self) -> int: ...
    @index.setter
    def index(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...

class VariantGetOp(max._core.Operation):
    """
    The `kgen.variant.get` operation interprets the given variant as one of its
    possible types and gets it as that type. This operation does NOT check
    whether the variant is that type. If the result type does not match the
    actual type of the variant, the operation's result is a poison value.  From
    an ownership perspective, this consumes the input variant and returns an
    owned register value.

    Example:

    ```mlir
    %optional = ...
    %intVal = kgen.variant.get %optional, 1 : !kgen.variant<struct<()>, i32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.Type,
        variant: max._core.Value[VariantType],
        index: max._core.dialects.builtin.IntegerAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        variant: max._core.Value[VariantType],
        index: max._core.dialects.builtin.IntegerAttr,
    ) -> None: ...
    @property
    def variant(self) -> max._core.Value[VariantType]: ...
    @property
    def index(self) -> int: ...
    @index.setter
    def index(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...

class VariantIsOp(max._core.Operation):
    """
    The `kgen.variant.is` operation checks whether the given variant contains
    a particular type. Returns an `i1` that indicates whether the variant is the
    particular type.

    Example:

    ```mlir
    %optional = ...
    %isNone = kgen.variant.is %optional, 0 : !kgen.variant<none, T>
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.builtin.IntegerType,
        variant: max._core.Value[VariantType],
        index: max._core.dialects.builtin.IntegerAttr,
    ) -> None: ...
    @property
    def variant(self) -> max._core.Value[VariantType]: ...
    @property
    def index(self) -> int: ...
    @index.setter
    def index(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...

class WitnessOp(max._core.Operation):
    """
    The `kgen.witness` operation defines a witness table entry in a
    conformance table. It represents a single requirement satisfied by a struct
    type for the trait being conformed to.

    Example:

    ```mlir
    kgen.struct.generator @SIMD<type: dtype, size> = ... {
      kgen.conformance @Boolable {
        kgen.witness @"__bool__" : (!pop.simd<size, type>) -> i1
          = @"SIMD::__bool__(::SIMD[$0, $1])"<:dtype type, size>
      }
      ...
    }
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        sym_name: max._core.dialects.builtin.StringAttr,
        value: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def sym_name(self) -> str: ...
    @sym_name.setter
    def sym_name(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...
    @property
    def value(self) -> max._core.dialects.builtin.TypedAttr: ...
    @value.setter
    def value(self, arg: max._core.dialects.builtin.TypedAttr, /) -> None: ...

class PackageLinkOp(max._core.Operation):
    """
    A `kgen.package.link` defines a link to the compiled artifacts of a package.
    It contains a reference to all precompiled packages, providing an anchor for
    functions and other operations defined within the package during the
    lowering pipeline. It may also contain the post-parse bodies which can be
    used to compile the package.

    Example:

    ```mlir
    kgen.package.link @foo
      dependencies([@std])
      post_parse(dense_resource<...> : tensor<...xi8>)
    ```
    """

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        sym_name: max._core.dialects.builtin.StringAttr,
        post_parse_module: max._core.dialects.builtin.DenseResourceElementsAttr,
        dependencies: LinkDependencyArrayAttr,
    ) -> None: ...
    @property
    def sym_name(self) -> str: ...
    @sym_name.setter
    def sym_name(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...
    @property
    def post_parse_module(
        self,
    ) -> max._core.dialects.builtin.DenseResourceElementsAttr | None: ...
    @post_parse_module.setter
    def post_parse_module(
        self, arg: max._core.dialects.builtin.DenseResourceElementsAttr, /
    ) -> None: ...
    @property
    def dependencies(
        self,
    ) -> Sequence[max._core.dialects.builtin.FlatSymbolRefAttr] | None: ...
    @dependencies.setter
    def dependencies(self, arg: LinkDependencyArrayAttr, /) -> None: ...

class ParameterScopeTypeInterface(Protocol):
    """
    The `ParameterScopeTypeInterface` describes a type that declares a nested
    parameter scope within a type expression. It enables `ParamIndexRefAttr`
    values inside the type to reference parameters declared in a scope.

    For example, if we have this Mojo code:

    ```mojo
    fn foo[T: AnyType]():
      alias bork: fn[
        T: AnyType,
        inner_f: fn[Y: AnyType](t: T, y: Y) -> None
      ] -> None = ...
    ```

    The `fn` after `bork:` is a `kgen.generator` which is a
    `ParameterScopeTypeInterface`.

    `ParameterScopeTypeInterface` also causes the `depth` fields of
    `ParamIndexRefAttr`/`ImplicitOriginRefAttr`/etc that are contained (even
    indirectly) within this object to be higher, see PSTIAIRAID.
    """

    @property
    def input_param_types(self) -> Sequence[max._core.Type]: ...

class ParameterTypeInterface(Protocol):
    """
    The `ParameterTypeInterface` can be used by types to plug into various parts
    of the KGEN parameter system, including pretty parsing of attribute values.
    """

    @property
    def meta_type(self) -> bool: ...

class SugaredTypeInterface(Protocol):
    """This interface can be used to customize SugarAttr behavior."""

    def can_elide_sugar_for(
        self, arg: max._core.dialects.builtin.TypedAttr, /
    ) -> SugarKind | None: ...
    def get_cached_canonical_type(
        self, arg: max._core.Type, /
    ) -> max._core.Type | None: ...

class BuildInfoType(max._core.Type):
    """
    A `!kgen.build_info` is the type of a build info. It is used for
    parameterizing kernels by how they are built.

    Example:
    ```mlir
    kgen.generator @target_params<bi: !kgen.build_info>() {
      ...
    }
    ```
    """

    def __init__(self) -> None: ...

class ClosureType(max._core.Type):
    """
    A `!kgen.closure` type represents a struct of captures.
    Example:
    ```mlir
    kgen.generator @parent(%x: index) {
       %0 = kgen.closure.init(%x)(){
          kgen.call @foo(%x)
       } : (index), !kgen.closure<@parent, "closure", escaping>
       %1 = kgen.closure.init(%x)(){
          kgen.call @foo(%x)
       } : (index), !kgen.closure<@parent, "closure", nonescaping>
       %2 = kgen.closure.init(%x)(){
          kgen.call @foo(%x)
       } : (index), !kgen.closure<@parent, "closure", registerpassable>
    }
    }
    ```
    """

    def __init__(
        self,
        parent_symbol: max._core.dialects.builtin.SymbolRefAttr,
        name: max._core.dialects.builtin.StringAttr,
        closure_memory_kind: ClosureMemoryKind,
    ) -> None: ...
    @property
    def parent_symbol(self) -> max._core.dialects.builtin.SymbolRefAttr: ...
    @property
    def name(self) -> max._core.dialects.builtin.StringAttr: ...
    @property
    def closure_memory_kind(self) -> ClosureMemoryKind: ...

class DTypeType(max._core.Type):
    """
    This type corresponds to the DType runtime class, representing an
    element type (aka "dtype") specifier for a data storage types.
    """

    def __init__(self) -> None: ...

class DeferredType(max._core.Type):
    """
    A `!kgen.deferred` type represents a deferred attribute that will be
    concretized at elaborator

    Example:

    ```mlir
    #kgen<deferred #index<cmp_predicate sle>> : !kgen.deferred
    #kgen.param.decl.ref<"(lifted)apply_0"> : !kgen.deferred
    ```
    """

    def __init__(self) -> None: ...

class FuncType(max._core.Type):
    """
    This type describes the type of a function KGEN, which can have input/output
    value arguments and results.

    The metadata attribute is used to store additional information about a
    function, such as its argument calling conventions and the effects of the
    function itself. When the metadata only contains default values, it isn't
    printed in the textual MLIR format.
    """

    @overload
    def __init__(
        self,
        inputs: Sequence[max._core.Type],
        results: Sequence[max._core.Type],
    ) -> None: ...
    @overload
    def __init__(
        self,
        values: max._core.dialects.builtin.FunctionType,
        arg_convs: Sequence[ArgConvention] = [],
        effects: FnEffects = FnEffects.none,
        metadata: max._core.Attribute = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        values: max._core.dialects.builtin.FunctionType,
        arg_conventions: Sequence[ArgConvention],
        fn_effects: FnEffects,
        metadata: FnMetadataAttrInterface,
    ) -> None: ...
    @property
    def values(self) -> max._core.dialects.builtin.FunctionType: ...
    @property
    def arg_conventions(self) -> Sequence[ArgConvention]: ...
    @property
    def fn_effects(self) -> FnEffects: ...
    @property
    def metadata(self) -> FnMetadataAttrInterface: ...

class GeneratorType(max._core.Type):
    """
    This type describes a generator (i.e. a parameterized value). It describes
    the generator's parameter signature (the types of input parameters), and the
    generator's output type (potentially in terms of the input parameters).
    """

    @overload
    def __init__(
        self,
        input_param_types: Sequence[max._core.Type],
        body: max._core.Type,
        metadata: max._core.Attribute = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        input_param_types: Sequence[max._core.Type],
        body: max._core.Type,
        metadata: GeneratorMetadataAttrInterface,
    ) -> None: ...
    @property
    def input_param_types(self) -> Sequence[max._core.Type]: ...
    @property
    def body(self) -> max._core.Type | None: ...
    @property
    def metadata(self) -> GeneratorMetadataAttrInterface: ...

class NeverType(max._core.Type):
    """
    A `!kgen.never` type represents a type that cannot be
    constructed. This is used to represent functions that never return or
    generic thrown types that resolve to a concrete type of "not actually
    thrown".
    ```
    """

    def __init__(self) -> None: ...

class NoneType(max._core.Type):
    """
    The `!kgen.none` type represents a value with no contents. When used as a
    function result, it is equivalent to a void result or no results.
    """

    def __init__(self) -> None: ...

class PackType(max._core.Type):
    """
    A `!kgen.pack` type represents a sequence of heterogeneously typed elements.
    This type can be used to represent a tuple of 0 or more elements.

    Example:

    ```mlir
    // A concrete pack type with no element types.
    !kgen.pack<[]>

    // A concrete pack type with two element types.
    !kgen.pack<[i32, i64]>

    kgen.generator @pack<Ts: variadic<!kgen.type>, T0: type, T1: type>(
      // A pack type parameterized on a variadic sequence of elements.
      %0: !kgen.pack<Ts>,
      // A pack type parameterized on two element types.
      %1: !kgen.pack<[T0, T1]>,
    ) { kgen.return }
    ```
    """

    @overload
    def __init__(
        self, variadic: max._core.dialects.builtin.TypedAttr
    ) -> None: ...
    @overload
    def __init__(
        self, variadic: max._core.dialects.builtin.TypedAttr
    ) -> None: ...
    @property
    def variadic(self) -> max._core.dialects.builtin.TypedAttr: ...

class ParamClosureType(max._core.Type):
    """
    A `!kgen.param_closure` type represents a struct of  parameter
    captures. It is used as a placeholder before the captured parameters
    are calculated. Once calculated, the capture type is replaced with
    a !kgen.struct type with the capture types as the struct field types.

    Example:
    ```mlir
        #type_value = #kgen.type<!kgen.closure<@foo, "fn" registerpassable>,
                {"__call__" :
                   <!kgen.param_closure<@foo “fn”>>
             (!kgen.closure<@foo, "fn" registerpassable>) -> index =
             @foo_fn<:!kgen.param<!kgen.param_closure<@foo “fn”>> ?>
           }> : !kgen.type

       kgen.generator @foo<C>(%arg0 : index) {
         %1 = kgen.closure.init()(%arg1: index) -> index {
         %0 = kgen.param.constant = <add(C, C)>
         kgen.return %0 : index
         } : (), !kgen.pointer<!kgen.closure<@foo, "fn" nonescaping>>
         %2 = kgen.call @consume<
           :type #type_value,
           :type !kgen.param_closure<@foo “fn”>,
           :!kgen.param<!kgen.param_closure<@foo “fn”>> #kgen.capture<@foo, “fn”>>
           (%3) :
           (!kgen.pointer<!kgen.closure<@foo, "fn" nonescaping>>) -> index
         kgen.return
       }

       kgen.generator @consume<x: type,
                       CAPTURE_TYPE: type,
                       CAPTURE_INST: !kgen.param<CAPTURE_TYPE>
                       >(%arg0: !kgen.param<x>) -> index {
         %0 = kgen.call_param[(!kgen.param<x>) -> index:
                      bind_params(:<!kgen.param<CAPTURE_TYPE>>
                        (!kgen.none, index) -> index
                        get_witness(x, "closure_trait", "__call__"),
                      CAPTURE_INST)](%arg0, %arg1)
         kgen.return %0 : index
       }
    ```
    """

    def __init__(
        self,
        parent_symbol: max._core.dialects.builtin.SymbolRefAttr,
        name: max._core.dialects.builtin.StringAttr,
    ) -> None: ...
    @property
    def parent_symbol(self) -> max._core.dialects.builtin.SymbolRefAttr: ...
    @property
    def name(self) -> max._core.dialects.builtin.StringAttr: ...

class ParamType(max._core.Type):
    """
    This is a symbolic type represented with a parameter expression that is
    resolved by the KGEN elaborator.  Once this parameter is substituted with a
    type constant, this ParamType is folded away and the MlirType inside
    the type constant is returned.
    """

    @overload
    def __init__(self, param: max._core.dialects.builtin.TypedAttr) -> None: ...
    @overload
    def __init__(
        self, context: Context, param: max._core.dialects.builtin.TypedAttr
    ) -> None: ...
    @property
    def param(self) -> max._core.dialects.builtin.TypedAttr: ...

class StringType(max._core.Type):
    """
    This kgen type represents an string value, used for parameterizing
    generators and working with string-returning expressions.
    """

    def __init__(self) -> None: ...

class StructInstanceType(max._core.Type):
    """
    This type represents a concretized mojo source struct type. It is the result
    of flattening an AppliedStructType.

    Example:

    ```mlir
    // A trivially-concretized struct with no parameters and no fields.
    !kgen.struct_inst<"Foo">

    // A parameterized memory-only struct with primitive element types.
    !kgen.struct_inst<
      "Bar"
      [elemT]
      <:dtype f32>
      (first: !kgen.scalar<f32>, second: !kgen.scalar<f32>)
      memoryOnly
    >
    ```
    """

    @overload
    def __init__(
        self,
        name: max._core.dialects.builtin.StringAttr,
        param_names: Sequence[max._core.dialects.builtin.StringAttr],
        param_values: Sequence[max._core.dialects.builtin.TypedAttr],
        fields: Sequence[StructDefFieldAttr],
        is_memory_only: bool,
    ) -> None: ...
    @overload
    def __init__(
        self,
        name: max._core.dialects.builtin.StringAttr,
        param_names: Sequence[max._core.dialects.builtin.StringAttr],
        param_values: Sequence[max._core.dialects.builtin.TypedAttr],
        fields: Sequence[StructDefFieldAttr],
        is_memory_only: bool,
    ) -> None: ...
    @property
    def name(self) -> max._core.dialects.builtin.StringAttr: ...
    @property
    def param_names(
        self,
    ) -> Sequence[max._core.dialects.builtin.StringAttr]: ...
    @property
    def param_values(
        self,
    ) -> Sequence[max._core.dialects.builtin.TypedAttr]: ...
    @property
    def fields(self) -> Sequence[StructDefFieldAttr]: ...
    @property
    def is_memory_only(self) -> bool: ...

class StructType(max._core.Type):
    """
    This type represents a struct. A struct contains element types arranged in
    their order of declaration and a flag indicating register-passability.

    This is the "anonymous" version of `!kgen.concrete_source_struct` that
    strips away all information except for those necessary for understanding the
    memory layout of the data it describes.

    The element types are stored as a `TypedAttr` which can be either:
    - A concrete `VariadicAttr` with resolved types
    - A parametric expression (e.g., a variadic parameter reference) before elaboration

    Example:

    ```mlir
    // A struct with primitive element types.
    !kgen.struct<(scalar<f32>, simd<4, ui64>)>

    // A memory-only struct with primitive element types.
    !kgen.struct<(scalar<f32>, simd<4, ui64>) memoryOnly>

    // A struct with nested types.
    !kgen.struct<(
      !kgen.pointer<simd<4, si8>>,
      !kgen.array<24, scalar<si64>>,
      !kgen.struct<(
        !kgen.scalar<f32>,
        !kgen.scalar<f64>
      )>
    )>

    // A struct with parameterized element types.
    !kgen.struct<(type, array<size, scalar<dtype>>)>

    // A struct parameterized on a variadic sequence of element types.
    kgen.generator @example<Ts: variadic<!kgen.type>>(
      %0: !kgen.struct<(Ts)>,
    ) { kgen.return }
    ```
    """

    @overload
    def __init__(
        self,
        variadic: max._core.dialects.builtin.TypedAttr,
        is_memory_only: bool = False,
    ) -> None: ...
    @overload
    def __init__(
        self, types: Sequence[max._core.Type], is_memory_only: bool = False
    ) -> None: ...
    @overload
    def __init__(
        self,
        variadic: max._core.dialects.builtin.TypedAttr,
        is_memory_only: bool = False,
        min_alignment: max._core.dialects.builtin.TypedAttr = ...,
    ) -> None: ...
    @property
    def element_types_variadic(
        self,
    ) -> max._core.dialects.builtin.TypedAttr: ...
    @property
    def is_memory_only(self) -> bool: ...
    @property
    def min_alignment(self) -> max._core.dialects.builtin.TypedAttr: ...

class TargetType(max._core.Type):
    """
    A `!kgen.target` is the type of a target. It is used for parameterizing
    kernels by target.

    Example:
    ```mlir
    kgen.generator @target_params<t: !kgen.target>() {
      ...
    }
    ```
    """

    def __init__(self) -> None: ...

class TypeType(max._core.Type):
    """
    This kgen type represents an arbitrary type, used for parameterizing
    type-generic functions and structs.

    It cannot be materialized into an SSA value, because it has no runtime
    representation.
    """

    def __init__(self) -> None: ...

class TypeValueType(max._core.Type):
    """
    This is the value domain counterpart to KGEN ParamType. It allows
    losslessly using a parameter expression as a MLIR Type: Even when its
    type value parameter is substituted with a type constant, it does not fold
    into the MlirType itself (unless the type constant is a trivial mlir type).
    """

    @overload
    def __init__(
        self, type_value: max._core.dialects.builtin.TypedAttr
    ) -> None: ...
    @overload
    def __init__(
        self, context: Context, type_value: max._core.dialects.builtin.TypedAttr
    ) -> None: ...
    @property
    def type_value(self) -> max._core.dialects.builtin.TypedAttr: ...

class VariadicSplatType(max._core.Type):
    """
    The `!kgen.variadic_splat` type represents deferred type that splats
    element type specified number of times. The type cannot be used standalone
    and has to be used either within `!kgen.struct` or `!llvm.struct` types.

    ```mlir
    !kgen.struct<(!kgen.variadic_splat<index, 3>)>
    !llvm.struct<(!kgen.variadic_splat<index, 5>)>
    ```

    will be concretized to

    ```mlir
    !kgen.struct<(index, index, index)>
    !llvm.struct<(index, index, index, index, index)>
    ```
    """

    def __init__(
        self,
        element_type: max._core.Type,
        count: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def element_type(self) -> max._core.Type | None: ...
    @property
    def count(self) -> max._core.dialects.builtin.TypedAttr: ...

class VariadicType(max._core.Type):
    """
    The `!kgen.variadic` type represents a homogeneously typed variadic sequence
    of zero or more elements.  It also includes the original argument convention
    so clients know if the input argument is supposed to be owned, read,
    mut, etc.

    Example:

    ```mlir
    // A variadic sequence of scalar floats.
    !kgen.variadic<scalar<f32>>

    // A parameterized variadic sequence.
    !kgen.variadic<type>
    ```
    """

    @overload
    def __init__(self, element_type: max._core.Type) -> None: ...
    @overload
    def __init__(self, element_type: max._core.Type) -> None: ...
    @property
    def element_type(self) -> max._core.Type | None: ...

class VariantType(max._core.Type):
    """
    A `!kgen.variant` type is a structural variant type that represents a value
    that is one of a list of types (discriminated union).

    Variants of zero types or one type are allowed, e.g. `!kgen.variant<i32>`,
    and all variant operations are defined on the even though they aren't
    particularly useful types.

    Example:

    ```mlir
    !kgen.variant<i32, scalar<dtype>, T>
    ```
    """

    @overload
    def __init__(self, types: Sequence[max._core.Type]) -> None: ...
    @overload
    def __init__(
        self, variadic: max._core.dialects.builtin.TypedAttr
    ) -> None: ...
    @property
    def variadic(self) -> max._core.dialects.builtin.TypedAttr: ...

class PointerType(max._core.Type):
    """
    This type represents a pointer. The pointee type is parameterized with a
    `!kgen.type` type. An optional `addressSpace` argument can be
    specified (default to 0).

    Example:

    ```mlir
    // A pointer to a scalar.
    !kgen.pointer<scalar<f32>>

    // A pointer to a SIMD vector.
    !kgen.pointer<simd<4, f32>>

    // A parameterized scalar pointer.
    !kgen.pointer<scalar<type>>

    // A completely parameterized pointer.
    !kgen.pointer<elementType>

    // The address space parameter is optional, but can be specified.
    !kgen.pointer<scalar<si32>, 2>

    // The address space also works on parametrized pointers.
    !kgen.pointer<elementType, 5>
    ```
    """

    @overload
    def __init__(
        self, element_type: max._core.Type, address_space: int = 0
    ) -> None: ...
    @overload
    def __init__(
        self,
        element_type: max._core.Type,
        address_space: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        element_type: max._core.Type,
        address_space: max._core.dialects.builtin.TypedAttr,
    ) -> None: ...
    @property
    def element_type(self) -> max._core.Type | None: ...
    @property
    def address_space(self) -> max._core.dialects.builtin.TypedAttr: ...

class FuncTypeGeneratorType(GeneratorType):
    def __init__(
        self,
        input_param_types: Sequence[max._core.Type],
        fn_type: max._core.dialects.builtin.FunctionType,
        arg_convs: Sequence[ArgConvention] = [],
        effects: FnEffects = FnEffects.none,
        fn_metadata: max._core.Attribute = ...,
        gen_metadata: max._core.Attribute = ...,
    ) -> None: ...

class _KGENDType:
    pass

class ParamDefValue:
    pass

class ParameterEvaluator:
    pass
