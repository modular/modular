//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the KGEN dialect.
//
//===----------------------------------------------------------------------===//

include "KGEN/KGENDialect/KGENAttrs.td"
include "KGEN/KGENDialect/KGENDialect.td"
include "KGEN/KGENDialect/KGENEnums.td"
include "KGEN/KGENDialect/KGENInterfaces.td"
include "KGEN/KGENDialect/KGENTypes.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// FIXME: KGENDialect should not depend on POPDialect.
include "KGEN/POPDialect/POPTypes.td"

// Base class for the operation in this dialect.
class KGENOp<string mnemonic, list<Trait> traits = []> :
    Op<KGENDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def KGEN_FuncOp
  : KGENOp<"func", [HasParent<"mlir::ModuleOp">, FunctionOpInterface,
                    Symbol, IsolatedFromAbove,
                    SingleBlockImplicitTerminator<"ReturnOp">,
                    DeclareOpInterfaceMethods<CallableOpInterface>,
                    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                    DeclInterface, FuncInterface]> {
  let summary = "concrete function definition";

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<KGEN_SignatureType>:$signature);
  let regions = (region AnyRegion:$bodyRegion);

  let extraClassDeclaration = [{
    using mlir::OpTrait::SingleBlockImplicitTerminator<ReturnOp>::Impl<FuncOp>::getBody;

    ReturnOp getReturnOp();
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GeneratorOp
//===----------------------------------------------------------------------===//

def KGEN_GeneratorOp
  : KGENOp<"generator", [HasParent<"mlir::ModuleOp">, FunctionOpInterface,
                         Symbol, IsolatedFromAbove,
                         SingleBlockImplicitTerminator<"ReturnOp">,
                         DeclareOpInterfaceMethods<CallableOpInterface>,
                         DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                         DeclInterface, FuncInterface]> {
  let summary = "Function generator definition";

  /// If the `implements` attribute is present then this generator is an
  /// implementation of the specified generator.interface.
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<KGEN_SignatureType>:$signature,
                       ConstraintArrayAttr:$constraints,
                       OptionalAttr<FlatSymbolRefAttr>:$implements);
  let regions = (region AnyRegion:$bodyRegion);

  let extraClassDeclaration = [{
    using mlir::OpTrait::SingleBlockImplicitTerminator<ReturnOp>::Impl<GeneratorOp>::getBody;

    ReturnOp getReturnOp();
  }];

  let hasCustomAssemblyFormat = 1;
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GeneratorInterfaceOp
//===----------------------------------------------------------------------===//

def KGEN_GeneratorInterfaceOp : KGENOp<"generator.interface", [
    HasParent<"mlir::ModuleOp">,
    DeclareOpInterfaceMethods<CallableOpInterface>,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    KGEN_CallOpInterface, FunctionOpInterface, Symbol, IsolatedFromAbove,
    DeclInterface, FuncInterface]> {
  let summary = "Generator interface definition";

  /// input and result parameters are stored in a single flat array named
  /// `paramDecls` and the `numInputParameters` field indicates which are input
  /// and which are result parameters.
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<KGEN_SignatureType>:$signature,
                       ConstraintArrayAttr:$constraints,
                       OptionalAttr<KGEN_SymbolConstantAttr>:$evaluator,
                       OptionalAttr<KGEN_SymbolConstantAttr>:$defaultImpl);

  // We want to work with the FunctionOpInterface, which always requires a body
  // region.  We are "declaration like" though, so ours never has a block.
  let regions = (region SizedRegion<0>:$body);

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// AddressOf
//===----------------------------------------------------------------------===//

def KGEN_AddressOfOp : KGENOp<"addressof", [KGEN_CallOpInterface]> {
  let summary = "Get the address of a function.";
  let description = [{
    The `kgen.addressof` operation references a parametric function and returns
    an address to a concrete instance of that function.

    Example:

    ```mlir
    kgen.generator.interface @foo<size>(index) -> index

    // Take the address of the interface. This will be resolved to the address
    // of the selected implemented of the interface post-elaboration.
    %0 = kgen.addressof @foo<size = 5> : (index) -> index
    ```

    The operation accepts region parameters like `kgen.call`.

    Example:

    ```mlir
    kgen.generator.interface @foo<fn: () -> ()>()

    %0 = kgen.addressof @foo<fn: () -> () = region> : () -> ()
    fn() {
      kgen.return
    }
    ```
  }];

  let arguments = (ins KGEN_SymbolConstantAttr:$callee,
                       KGEN_ParamDeclArrayAttr:$paramDecls);
  let regions = (region VariadicRegion<SizedRegion<1>>:$regions);
  let results = (outs FunctionType:$result);
  let assemblyFormat = [{
    custom<AddressOfOp>($callee, $paramDecls, type($result))
    attr-dict
    custom<CallRegions>($regions, ref($callee))
  }];

  let builders = [
    OpBuilder<(ins "Type":$type, "SymbolConstantAttr":$callee,
                   CArg<"ArrayRef<ParamDeclAttr>", "{}">:$resultParams)>
  ];

  let extraClassDeclaration = [{
    /// Return the symbol reference to the callee.
    SymbolRefAttr getCalleeSymbol() { return getCallee().getSymbol(); };
    /// Return the bound input parameter values.
    ArrayRef<ParamBindAttr> getParamValues() {
      return getCallee().getParamValues().getValue();
    }
  }];
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def KGEN_CallOp : KGENOp<"call", [KGEN_CallOpInterface,
    DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "Call a func, generator, or generator interface";
  let description = [{
    The `kgen.call` operation invokes a parametric function. The callee must be
    a flat symbol reference to a KGEN function, generator, or interface. The
    operation must bind parameter values as required by the callee. The
    operation's operands are used as function arguments and the operation's
    results match the return values of the function.

    Example:

    ```mlir
    kgen.func @foo(%arg0: index) {
      kgen.return %arg0 : index
    }

    %0 = kgen.call @foo(%arg0) : (index) -> index
    ```

    Specify values for the input parameters to the callee using a parameter
    binding list and declarations for result parameters with a parameter
    declaration list.

    Example:

    ```mlir
    kgen.generator @add2<N -> index>() {
      kgen.return<add(N, 2)>
    }

    kgen.call @add2<N = 2 -> NPlus2>() : () -> ()
    ```

    Region parameters can be specified using a special region binding attribute
    to indicate that a parameter's value is one of the operation's regions. The
    regions must be isolated from above.

    Example:

    ```mlir
    kgen.generator.interface @passRegion<fn: <N>(index) -> index>()

    kgen.call @passRegion<fn: <N>(index) = region>() : () -> ()
    fn<N>(%arg0: index) {
      kgen.return %arg0 : index
    }
    ```
  }];

  let arguments = (ins
    KGEN_SymbolConstantAttr:$callee,
    KGEN_ParamDeclArrayAttr:$paramDecls,
    Variadic<AnyType>:$operands
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region VariadicRegion<SizedRegion<1>>:$regions);

  let assemblyFormat = [{
    custom<CallOp>($callee, $paramDecls, $operands,
                   type($operands), type($results)) attr-dict
    custom<CallRegions>($regions, ref($callee))
  }];

  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes, "SymbolConstantAttr":$callee,
                   "ValueRange":$operands,
                   CArg<"ArrayRef<ParamDeclAttr>", "{}">:$resultParams)>
  ];

  let extraClassDeclaration = [{
    /// Return the symbol reference to the callee.
    SymbolRefAttr getCalleeSymbol() { return getCallee().getSymbol(); };
    /// Return the bound input parameter values.
    ArrayRef<ParamBindAttr> getParamValues() {
      return getCallee().getParamValues().getValue();
    }
  }];
}

//===----------------------------------------------------------------------===//
// CallParamOp
//===----------------------------------------------------------------------===//

// NOTE: This cannot be a CallOpInterface, because the callee is not a
// SymbolRefAttr, it is a different kind of attribute.
def KGEN_CallParamOp : KGENOp<"call_param", [KGEN_CallOpInterface]> {
  let summary = "Call declaration specified by a parameter expression";
  let description = [{
    The `kgen.call_param` operation invokes a parametric callee. The callee is
    a parameter expression with a signature type, which in practice is either a
    symbol constant, so KGEN function, generator, or interface, or a region
    body passed from higher up the call stack.

    Example:

    ```mlir
    // Symbol constant callee.
    %0 = kgen.call_param[(index) -> index: @someFn](%arg0)

    // Parameter reference callee.
    %1 = kgen.call_param[<N>() -> index: foo]<N = 4>()
    ```

    The operation can bind regions as parameters as well, but they must be
    isolated from above.

    ```mlir
    kgen.call_param[<fn: <N>(index) -> index>(index) -> (): callee]<
      fn: <N>(index) -> index = region>(%arg0)
    fn<N>(%arg0: index) {
      %0 = kgen.param.constant = <N>
      %1 = index.add %0, %arg0
      kgen.return %1 : index
    }
    ```
  }];

  let arguments = (ins
    AttrOfType<KGEN_SignatureType>:$callee,
    ParamBindArrayAttr:$paramValues,
    KGEN_ParamDeclArrayAttr:$paramDecls,
    Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region VariadicRegion<SizedRegion<1>>:$regions);

  let assemblyFormat = [{
    `` custom<CallParamCallee>($callee, $paramValues, $paramDecls,
                            type($operands), type($results))
    `(` $operands `)` attr-dict
    custom<CallRegions>($regions, ref($paramValues))
  }];

  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes, "TypedAttr":$callee,
                   "ArrayRef<ParamBindAttr>":$inputParams,
                   "ArrayRef<ParamDeclAttr>":$resultParams,
                   "ValueRange":$operands)>
  ];

  let extraClassDeclaration = [{
    bool isAllowedInFunc() { return false; }
  }];

  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// RegionBodyOp
//===----------------------------------------------------------------------===//

def KGEN_RegionBodyOp : KGENOp<"region.body", [Terminator, DeclInterface,
    FuncInterface, SingleBlockImplicitTerminator<"ReturnOp">,
    ParentOneOf<["AddressOfOp", "CallOp", "CallParamOp", "ParamDeclareRegionOp"]>]> {
  let summary = "Define a region parameter body.";
  let description = [{
    The `kgen.region.body` operation defines the contents of a region parameter.
    It is used in conjunction with `kgen.param.declare.region` to define a
    region parameter. See that operation for more details.
    ```
  }];

  let arguments = (ins TypeAttrOf<KGEN_SignatureType>:$signature,
                       ConstraintArrayAttr:$constraints,
                       UnitAttr:$isolated);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = [{
    custom<RegionBody>($signature, $constraints, $bodyRegion) attr-dict
  }];

  let extraClassDeclaration = [{
    /// Get the terminator.
    ReturnOp getReturnOp();

    bool isIsolatedFromAbove() { return getIsolated(); }
    void notifyKnownIsolatedFromAbove() { setIsolated(true); }
  }];

  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CallIndirectOp
//===----------------------------------------------------------------------===//

def KGEN_CallIndirectOp : KGENOp<"call_indirect"> {
  let summary = "call a signature-typed value";
  let description = [{
    The `kgen.call_indirect` operation takes an SSA value of `!kgen.signature`
    type but which does not have input or output parameters and invokes it with
    the provided operands as arguments.

    Example:

    ```mlir
    %0:2 = kgen.call_indirect %fn(%arg0, %arg1)
      : (index, !kgen.declref<@Foo>) -> (!pop.simd<2, si32>, !kgen.declref<@Bar>)
    ```
  }];

  let arguments = (ins KGEN_SignatureType:$callee, Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, $results)
    `` custom<CallIndirectCalleeType>(type($callee), ref(type($inputs)),
                                      ref(type($results)))
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// PartialApplyOp
//===----------------------------------------------------------------------===//

def KGEN_PartialApplyOp : KGENOp<"partial_apply",
      [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "partially apply arguments to a callable";
  let description = [{
    The `kgen.partial_apply` operation takes a callable of `!kgen.signature`
    type but which does not have input or output parameters and binds values
    to its arguments, partially specializing the callable and returning a
    closure. The operation can bind any number of the callee's arguments.

    Example:

    ```mlir
    // The result has the signature type `(index, f32) -> !pop.scalar<address>`.
    %0 = kgen.partial_apply %fn(?, %arg0, ?, %arg1)
      : (index, i32, f32, !pop.scalar<index>) -> !pop.scalar<address>
    ```

    The syntax of the operation uses `?` to denote arguments that are left
    unbound. These arguments form the signature of the resulting callable.
    In the above example, the second and fourth arguments to the callable `%fn`
    are bound to `%arg0` and `%arg1` respectively, returning a callable that
    has two fewer arguments.
  }];

  let arguments = (ins KGEN_SignatureType:$callee, Variadic<AnyType>:$inputs,
                       DenseI64ArrayAttr:$boundInputs);
  let results = (outs KGEN_SignatureType:$result);
  let assemblyFormat = [{
    $callee `` custom<BoundInputs>($inputs, $boundInputs, type($inputs),
                                   type($result), type($callee)) attr-dict
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// ParamConstantOp
//===----------------------------------------------------------------------===//

def KGEN_ParamConstantOp : KGENOp<"param.constant",
                             [FirstAttrDerivedResultType, Pure,
                              ConstantLike]> {
  let summary = [{
    Return the value of a parameter expression as an SSA value that may be used
    by other ops.
  }];

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat =
     "`` custom<ParamConstantOpValue>($value, qualified(type($result))) attr-dict";

  let builders = [OpBuilder<(ins "TypedAttr":$value)>];

  let hasFolder = true;
}

//===----------------------------------------------------------------------===//
// ParamDeclareOp
//===----------------------------------------------------------------------===//

def KGEN_ParamDeclareOp : KGENOp<"param.declare"> {
  let summary = [{
    Declare a new parameter with a value.
  }];

  let arguments = (ins ParamDeclSize1ArrayAttr:$paramDecls,
                       TypedAttrInterface:$value);
  let assemblyFormat =
     "custom<ParamDeclareOpValue>($paramDecls, $value) attr-dict";

  let builders = [
    OpBuilder<(ins "ParamDeclAttr":$decl, "Attribute":$value)>,
  ];

  let extraClassDeclaration = [{
    /// Return the single parameter declaration as a well typed attribute.
    ParamDeclAttr getParamDecl();
    StringAttr getName() { return getParamDecl().getName(); }
    Type getType() { return getParamDecl().getType(); }
    void setParamDecl(ParamDeclAttr decl);
  }];
}

//===----------------------------------------------------------------------===//
// ParamDeclareRegion
//===----------------------------------------------------------------------===//

def KGEN_ParamDeclareRegionOp : KGENOp<"param.declare.region",
    [SingleBlockImplicitTerminator<"RegionBodyOp">]> {
  let summary = "declare a region parameter";
  let description = [{
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
    kgen.param.declare.region AddIt = <N>(%arg0: index) -> index {
      %0 = kgen.param.constant = <N>
      %1 = index.add %0, %arg0
      kgen.return %1 : index
    }
    ```

    The declared region can also define constraints and result parameters.

    Example:

    ```mlir
    kgen.param.declare.region LessThan2 = <N -> index>()
        constraints <[lt(N, 2), "must be less than 2"]> {
      kgen.return<add(N, 2)>
    }
    ```

    The operations carry an extra bit `isolated` to indicate that they are
    parametrically isolated from above.

    Note: this operation carries a single `RegionBodyOp` since operations cannot
    both define parameters in the current scope while declaring a new scope at
    the same time: it is not possible to check redeclarations at the same scope.
  }];

  let arguments = (ins ParamDeclSize1ArrayAttr:$paramDecls);
  let results = (outs KGEN_SignatureType:$result);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    custom<RegionDeclaration>($paramDecls, type($result), $body) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// ParamSearchOp
//===----------------------------------------------------------------------===//

def KGEN_ParamSearchOp : KGENOp<"param.search"> {
  let summary = [{
    Declare a new parameter that gets one out of a set of values at elaboration
    time.
  }];

  let arguments = (ins ParamDeclSize1ArrayAttr:$paramDecls,
                       KGEN_ParameterExprArrayAttr:$values);
  let assemblyFormat =
     "custom<ParamSearchOpValue>($paramDecls, $values) attr-dict";

  let extraClassDeclaration = [{
    /// Return the single parameter declaration as a well typed attribute.
    ParamDeclAttr getParamDecl();
  }];
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def KGEN_ReturnOp : KGENOp<"return", [Terminator, ReturnLike]> {
  let summary = "return operation for kgen declarations";
  let description = [{
    The `kgen.return` operation specifies the returned values and output
    parameters for a `kgen.func` or generator.

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
  }];

  let arguments = (ins KGEN_ParameterExprArrayAttr:$parameters,
                       Variadic<AnyType>:$operands);
  let assemblyFormat = [{
    `` custom<ParameterValues>($parameters) attr-dict
    ($operands^ `:` type($operands))?
  }];

  let extraClassDeclaration = [{
    /// Verify that the operands of this ReturnOp match the specified set of
    /// types. Only check the result types if they were provided.
    LogicalResult checkArgumentTypes(ArrayRef<Type> resultParamTypes,
                                     Optional<TypeRange> types);
  }];
}

//===----------------------------------------------------------------------===//
// ParamAssertOp
//===----------------------------------------------------------------------===//

def KGEN_ParamAssertOp : KGENOp<"param.assert", []> {
  let summary = "check a parameter expression at elaboration time";
  let description = [{
    The `kgen.param.assert` operation ensures that a parameter expression is
    true at elaboration time.  It is a more flexible form of the constraint list
    on a generator.
  }];

  let arguments = (ins AttrOfType<I1>:$cond, StrAttr:$message);
  let assemblyFormat =
    "` ` `<` custom<ParamAssertOpValue>($cond) `>` `,` $message attr-dict";

  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// IncludeOp
//===----------------------------------------------------------------------===//

def KGEN_IncludeOp : KGENOp<"include", []> {
  let summary = "include an external file";
  let description = [{
    The `kgen.include` operation includes an external file and makes all the
    symbols contained within available for use.

    This op only exists in the IR before the include is resolved, once it's
    resolved the IncludeOp is removed. Includes are resolved by the elaborator
    by providing a list of search paths.
  }];

  let arguments = (ins StrAttr:$fileName);
  let assemblyFormat = "$fileName attr-dict-with-keyword";
}

//===----------------------------------------------------------------------===//
// RebindOp
//===----------------------------------------------------------------------===//

def KGEN_RebindOp : KGENOp<"rebind"> {
  let summary = "rebind parametric types";
  let description = [{
    The `kgen.rebind` operation rebinds values with parametric types into other
    parameter domains. The operation can be used to convert between parameter
    domains or between concrete types. During elaboration, rebind operations
    must resolve to the same input and output types. Otherwise, elaboration will
    fail.

    Example:

    ```mlir
    // Rebind a parameterized type to a concrete type.
    %0 = kgen.rebind %arg0 : !kgen.paramref<type> to !pop.scalar<f32>

    // Rebind between parameter domains.
    %1 = kgen.rebind %arg1 : !kgen.paramref<type> to !pop.simd<size, dtype>

    // Unbind a concrete type to one with a parameter.
    %2 = kgen.rebind %arg2 : !pop.scalar<f32> to !pop.scalar<dtype>

    // ERROR: Cannot rebind between different concrete types.
    %3 = kgen.rebind %arg2 : !pop.scalar<f32> to !pop.scalar<si32>
    ```
  }];

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// StructDeclOp
//===----------------------------------------------------------------------===//

def KGEN_StructDeclOp : KGENOp<"struct.decl", [Symbol, SymbolTable,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>, DeclInterface,
    SingleBlock, NoTerminator]> {
  let summary = "Struct declaration.";
  let description = [{
    The `kgen.struct.decl` operation is a top-level operation used to declare
    parametric KGEN struct types. Struct declarations are named with a symbol
    and have a parameter declaration list. The body contains
    `kgen.struct.field` operations that declare the contents of the struct.

    Example:

    ```mlir
    kgen.struct.decl @Buffer<size, dtype: dtype> {
      kgen.struct.field size : index
      kgen.struct.field ptr : !pop.pointer<scalar<dtype>>
      kgen.struct.field dtype : !kgen.dtype
    }
    ```

    The parameters of the declaration can be constraints with optional
    constraints.

    Example:

    ```mlir
    kgen.struct.decl @SmallString<size>
        constraints <[le(size, 8), "string must be small"]> {
      data : !pop.array<size, scalar<si8>>
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       KGEN_ParamDeclArrayAttr:$inputParamDecls);
  let regions = (region SizedRegion<1>:$fields);
  let assemblyFormat = [{
    $sym_name `` custom<OptionalParameterSpec>($inputParamDecls)
    attr-dict-with-keyword
    $fields
  }];

  let extraClassDeclaration = [{
    /// Get the field declarations.
    auto getFieldDecls() { return getFields().getOps<StructFieldOp>(); }
    auto field_begin() { return getFieldDecls().begin(); }
    auto field_end() { return getFieldDecls().end(); }
  }];

  let builders = [
    // Create a struct with a region but no body block. The caller must create
    // it and fill it in.
    OpBuilder<(ins "StringAttr":$name)>,
  ];

  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// StructFieldOp
//===----------------------------------------------------------------------===//

def KGEN_StructFieldOp : KGENOp<"struct.field", [HasParent<"StructDeclOp">]> {
  let summary = "Struct field declaration.";
  let description = [{
    The `kgen.struct.field` operations declares a named struct field and type.

    Example:

    ```mlir
    // Declare a field named "foo".
    kgen.struct.field foo : !pop.scalar<f32>
    ```
  }];

  let arguments = (ins StrAttr:$name, TypeAttr:$type);
  let assemblyFormat = "custom<KeywordAsString>($name) attr-dict `:` $type";
}

//===----------------------------------------------------------------------===//
// StructCreateOp
//===----------------------------------------------------------------------===//

def KGEN_StructCreateOp : KGENOp<"struct.create", [Pure,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Create a parametric struct.";
  let description = [{
    The `kgen.struct.create` operation creates a struct of the referred type
    with the provided values as fields. The operands must match the fields in
    the order they are declared.

    Example:

    ```mlir
    kgen.struct.decl @MyStruct<T: dtype> {
      a : !pop.scalar<T>
      b : !pop.scalar<f32>
      c : !pop.simd<2, T>
    }

    %0 = kgen.struct.create(%a, %b, %c) :
      (!pop.scalar<type>, !pop.scalar<f32>, !pop.simd<2, type>)
      -> !kgen.declref<@MyStruct<T: dtype = type>>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs KGEN_DeclRefType:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// StructInsertOp
//===----------------------------------------------------------------------===//

def KGEN_StructInsertOp : KGENOp<"struct.insert", [Pure,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    AllTypesMatch<["container", "result"]>]> {
  let summary = "Insert a value into a struct.";
  let description = [{
    The `kgen.struct.insert` operation inserts a value into a parametric struct.
    The original struct is immutable, so a new copy with the inserted value is
    returned.

    Example:

    ```mlir
    kgen.struct.decl @MyStruct<T: dtype> {
      a : !pop.scalar<T>
    }

    %0 = kgen.struct.insert %fieldVal, %struct[a]
      : !pop.scalar<f64> into !kgen.declref<@MyStruct<T: dtype = f64>>
    ```
  }];

  let arguments = (ins StrAttr:$field, AnyType:$value, KGEN_DeclRefType:$container);
  let results = (outs KGEN_DeclRefType:$result);
  let assemblyFormat = [{
    $value `,` $container `[` custom<KeywordAsString>($field) `]` attr-dict `:`
    type($value) `into` qualified(type($container))
  }];
}

//===----------------------------------------------------------------------===//
// StructExtractOp
//===----------------------------------------------------------------------===//

def KGEN_StructExtractOp : KGENOp<"struct.extract", [Pure,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Extract a value from a struct.";
  let description = [{
    The `kgen.struct.extract` operation extracts the value of a field by name
    from a parametric struct.

    Example:

    ```mlir
    kgen.struct.decl @MyStruct<T: dtype> {
      a : !pop.scalar<T>
    }

    %0 = kgen.struct.extract %struct[a]
      : !pop.scalar<f64> from !kgen.declref<@MyStruct<T: dtype = f64>>
    ```
  }];

  let arguments = (ins StrAttr:$field, KGEN_DeclRefType:$container);
  let results = (outs AnyType:$value);
  let assemblyFormat = [{
    $container `[` custom<KeywordAsString>($field) `]` attr-dict `:`
    type($value) `from` qualified(type($container))
  }];

  let builders = [
    // Infer the result type based on the result of lookup.
    OpBuilder<(ins "Value":$structBase, "StructFieldOp":$field)>
  ];
}

//===----------------------------------------------------------------------===//
// StructGEPOp
//===----------------------------------------------------------------------===//

def KGEN_StructGEPOp : KGENOp<"struct.gep", [Pure,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Get a pointer to a struct field.";
  let description = [{
    The `kgen.struct.gep` operation computes a pointer to a struct field by
    name from a parametric struct.

    Example:

    ```mlir
    kgen.struct.decl @MyStruct<T: dtype> {
      a : !pop.scalar<T>
    }

    %0 = kgen.struct.gep %struct[a]
      : <scalar<f64>> from <@MyStruct<T: dtype = f64>>
    ```
  }];

  let arguments = (ins StrAttr:$field, PointerOf<KGEN_DeclRefType>:$container);
  let results = (outs POP_PointerType:$result);
  let assemblyFormat = [{
    $container `[` custom<KeywordAsString>($field) `]` attr-dict `:`
    type($result) `from` type($container)
  }];

  let builders = [
    // Infer the result type based on the result of lookup.
    OpBuilder<(ins "Value":$structBasePtr, "StructFieldOp":$field)>
  ];

}

//===----------------------------------------------------------------------===//
// ListGetOp
//===----------------------------------------------------------------------===//

def KGEN_ListGetOp : KGENOp<"list.get", [Pure, TypesMatchWith<"list element type",
    "list", "result",
    "ParamRefType::get($_self.cast<ListType>().getElementType())">]> {
  let summary = "Get a list element.";
  let description = [{
    The `kgen.list.get` operation returns the element of a list at the provided
    index. The index can be an expression, but if it resolves to an out-of-range
    index, an error is emitted.

    Example:

    ```mlir
    %0 = kgen.list.get %list[add(A, N)] : <index[N]>
    ```
  }];

  let arguments = (ins KGEN_ListType:$list, AttrOfType<Index>:$index);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $list `[` custom<IndexParamValue>($index) `]` attr-dict `:` type($list)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ListCreateOp
//===----------------------------------------------------------------------===//

def KGEN_ListCreateOp : KGENOp<"list.create", [Pure,
    TypesMatchWith<"list element type", "result", "operands",
      "ParamRefType::get($_self.cast<ListType>().getElementType())",
      "typeRangeMatches">]> {
  let summary = "Form a list.";
  let description = [{
    The `kgen.list.create` operation forms a list using the provided operands.

    Example:

    ```mlir
    %0 = kgen.list.create(%arg0, %arg1) : <index[2]>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs KGEN_ListType:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type($result)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ExportOp
//===----------------------------------------------------------------------===//

def KGEN_ExportOp : KGENOp<"export", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Declare a list of exported symbols.";
  let description = [{
    The `kgen.export` operation specifies a list of exported symbols from a
    given module. The list must not be empty.

    Example:

    ```mlir
    kgen.export [@foo, @bar]
    ```
  }];

  let arguments = (ins FlatSymbolRefArrayAttr:$exports);
  let results = (outs);
  let assemblyFormat = "$exports attr-dict";
}

//===----------------------------------------------------------------------===//
// ExternFuncOp
//===----------------------------------------------------------------------===//

def KGEN_ExternFuncOp : KGENOp<"extern.func", [DeclareOpInterfaceMethods<Symbol>]> {
  let summary = "Declare an external function.";
  let description = [{
    The `kgen.extern.func` operation provides a signature for an external function.

    Example:

    ```mlir
    kgen.extern.func @foo(si32) -> (f32, f32)
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type);
  let results = (outs);
  let assemblyFormat = "$sym_name `` $function_type attr-dict";
}

//===----------------------------------------------------------------------===//
// ExternVariableOp
//===----------------------------------------------------------------------===//

def KGEN_ExternVariableOp : KGENOp<"extern.variable",
    [DeclareOpInterfaceMethods<Symbol>]> {
  let summary = "Declare an external variable.";
  let description = [{
    The `kgen.extern.variable` operation provides a signature for an
    external variable.

    Example:

    ```mlir
    kgen.extern.variable @foo : !foo.someType
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$type);
  let results = (outs);
  let assemblyFormat = "$sym_name `:` $type attr-dict";
}
