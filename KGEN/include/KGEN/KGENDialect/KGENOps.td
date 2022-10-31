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
include "KGEN/KGENDialect/KGENDeclInterface.td"
include "KGEN/KGENDialect/KGENDialect.td"
include "KGEN/KGENDialect/KGENEnums.td"
include "KGEN/KGENDialect/KGENTypes.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// FIXME: KGENDialect should not depend on POPDialect.
include "KGEN/POPDialect/POPTypes.td"

// Base class for the operation in this dialect.
class KGENOp<string mnemonic, list<Trait> traits = []> :
    Op<KGENDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def FuncOp
  : KGENOp<"func", [HasParent<"mlir::ModuleOp">, FunctionOpInterface,
                    Symbol, IsolatedFromAbove,
                    SingleBlockImplicitTerminator<"ReturnOp">,
                    DeclareOpInterfaceMethods<CallableOpInterface>,
                    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                    DeclareOpInterfaceMethods<KGENDeclInterface>]> {
  let summary = "concrete function definition";

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       LinkageAttr:$linkage,
                       ParamDeclArrayAttr:$paramDecls,
                       KTypeArrayAttr:$resultParamTypes);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let extraClassDeclaration = [{
    using mlir::OpTrait::SingleBlockImplicitTerminator<ReturnOp>::Impl<FuncOp>::getBody;
    ReturnOp getReturnOp();

    /// Returns the argument types of this invokable.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this invokable.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let hasCustomAssemblyFormat = 1;

  let builders = [
    // Create a func with a region but no body block.  The caller must create
    // it and fill it in.
    OpBuilder<(ins "StringAttr":$name,
                   "FunctionType":$signature, "LinkageAttr":$linkage,
                   "ArrayRef<Type>":$outputParamTypes)>,
    // Create a func with an empty body, `argLocs` specifies the locations for
    // all the block arguments.
    OpBuilder<(ins "StringAttr":$name,
                   "FunctionType":$signature, "LinkageAttr":$linkage,
                   "ArrayRef<Type>":$outputParamTypes,
                   "ArrayRef<Location>":$argLocs)>,
  ];
}

//===----------------------------------------------------------------------===//
// GeneratorOp
//===----------------------------------------------------------------------===//

def GeneratorOp
  : KGENOp<"generator", [HasParent<"mlir::ModuleOp">, FunctionOpInterface,
                         Symbol, IsolatedFromAbove,
                         SingleBlockImplicitTerminator<"ReturnOp">,
                         DeclareOpInterfaceMethods<CallableOpInterface>,
                         DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                         DeclareOpInterfaceMethods<KGENDeclInterface>]> {
  let summary = "Function generator definition";

  /// If the `implements` attribute is present then this generator is an
  /// implementation of the specified generator.interface.
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       LinkageAttr:$linkage,
                       ParamDeclArrayAttr:$paramDecls,
                       KTypeArrayAttr:$resultParamTypes,
                       ConstraintArrayAttr:$constraints,
                       OptionalAttr<FlatSymbolRefAttr>:$implements);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let extraClassDeclaration = [{
    using mlir::OpTrait::SingleBlockImplicitTerminator<ReturnOp>::Impl<GeneratorOp>::getBody;
    ReturnOp getReturnOp();

    /// Returns the argument types of this invokable.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this invokable.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// GeneratorInterfaceOp
//===----------------------------------------------------------------------===//

def GeneratorInterfaceOp : KGENOp<"generator.interface", [
    HasParent<"mlir::ModuleOp">,
    FunctionOpInterface,
    Symbol,
    IsolatedFromAbove,
    DeclareOpInterfaceMethods<CallableOpInterface>,
    DeclareOpInterfaceMethods<KGENDeclInterface>,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Generator interface definition";

  /// input and result parameters are stored in a single flat array named
  /// `paramDecls` and the `numInputParameters` field indicates which are input
  /// and which are result parameters.
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       LinkageAttr:$linkage,
                       ParamDeclArrayAttr:$paramDecls,
                       KTypeArrayAttr:$resultParamTypes,
                       ConstraintArrayAttr:$constraints,
                       OptionalAttr<SymbolConstantAttr>:$evaluator);

  // We want to work with the FunctionOpInterface, which always requires a body
  // region.  We are "declaration like" though, so ours never has a block.
  let regions = (region SizedRegion<0>:$body);

  let extraClassDeclaration = [{
    /// Returns the argument types of this invokable.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this invokable.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// AddressOf
//===----------------------------------------------------------------------===//

def AddressOfOp : KGENOp<"addressof",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
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

  let arguments = (ins FlatSymbolRefAttr:$callee,
                       ParamBindArrayAttr:$paramValues,
                       ParamDeclArrayAttr:$paramDecls);
  let regions = (region VariadicRegion<SizedRegion<1>>:$regions);
  let results = (outs FunctionType:$result);
  let assemblyFormat = [{
    $callee `` custom<CallOpParams>($paramValues, $paramDecls) attr-dict `:`
    type($result) custom<CallRegions>($regions, ref($paramValues))
  }];

  let hasRegionVerifier = 1;

  let builders = [
    OpBuilder<(ins "Type":$type, "StringAttr":$callee,
                   "ArrayRef<ParamBindAttr>":$inputParams,
                   "ArrayRef<ParamDeclAttr>":$resultParams)>
  ];

  let extraClassDeclaration = [{
    /// Return a signature for this function reference.
    SignatureType getSignature();
  }];
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def CallOp
  : KGENOp<"call", [CallOpInterface,
                    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {

  let summary = "Call a func, generator, or generator interface";

  let arguments =
   (ins FlatSymbolRefAttr:$callee, ParamBindArrayAttr:$paramValues,
        ParamDeclArrayAttr:$paramDecls, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);
  let regions = (region VariadicRegion<SizedRegion<1>>:$regions);

  let extraClassDeclaration = [{
    FunctionType getFunctionType() {
      return FunctionType::get(getContext(), getOperandTypes(),
                               getResultTypes());
    }

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    /// Return a signature for this call.
    SignatureType getSignature();
  }];

  let assemblyFormat = [{
    $callee `` custom<CallOpParams>($paramValues, $paramDecls) `(` $operands `)`
    attr-dict `:` functional-type($operands, results)
    custom<CallRegions>($regions, ref($paramValues))
  }];

  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes, "StringAttr":$callee,
                   "ArrayRef<ParamBindAttr>":$inputParams,
                   "ArrayRef<ParamDeclAttr>":$resultParams,
                   "ValueRange":$operands)>,
  ];

  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CallParamOp
//===----------------------------------------------------------------------===//

// NOTE: This cannot be a CallOpInterface, because the callee is not a
// SymbolRefAttr, it is a different kind of attribute.
def CallParamOp : KGENOp<"call_param"> {
  let summary = "Call declaration specified by a parameter expression";

  let arguments =
   (ins TypedParameterExprAttr<SignatureType>:$callee,
        ParamBindArrayAttr:$paramValues,
        ParamDeclArrayAttr:$paramDecls, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region VariadicRegion<SizedRegion<1>>:$regions);

  let extraClassDeclaration = [{
    FunctionType getFunctionType() {
      return FunctionType::get(getContext(), getOperandTypes(),
                               getResultTypes());
    }

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return a signature for this call.
    SignatureType getSignature();
  }];

  let assemblyFormat = [{
    `` custom<CallParamCallee>($callee, $paramValues, $paramDecls,
                            type($operands), type($results))
    `(` $operands `)`
    attr-dict
    custom<CallRegions>($regions, ref($paramValues))
  }];

  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes, "TypedAttr":$callee,
                   "ArrayRef<ParamBindAttr>":$inputParams,
                   "ArrayRef<ParamDeclAttr>":$resultParams,
                   "ValueRange":$operands)>,
  ];

  let hasCanonicalizeMethod = 1;
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// InlinedCallOp
//===----------------------------------------------------------------------===//

def InlinedCallOp : KGENOp<"inlined_call"> {
  let summary = "Parametric call that allows implicit capture.";
  let description = [{
    The `kgen.inlined_call` operation is a parametric call similar to
    `kgen.call_param` in that the callee is a parameter. However, any region
    parameters specified by the operation do not have to be isolated from above.
    `kgen.inlined_call` can capture values within scope of the call.

    Example:

    ```mlir
    %c0 = index.constant 0
    %0 = kgen.inlined_call[<fn: () -> index>() -> index: foo]
                          <fn: () -> index = region>()
    fn() {
      kgen.return %c0 : index
    }
    ```

    The elaborator will inline the callee and any nested calls that use these
    region parameters.
  }];

  let arguments = (ins
    TypedParameterExprAttr<SignatureType>:$callee,
    ParamBindArrayAttr:$paramValues,
    ParamDeclArrayAttr:$paramDecls,
    Variadic<AnyType>:$operands
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region VariadicRegion<SizedRegion<1>>:$regions);
  let assemblyFormat = [{
    `` custom<CallParamCallee>($callee, $paramValues, $paramDecls,
                               type($operands), type($results))
    `(` $operands `)`
    attr-dict
    custom<InPlaceCallRegions>($regions, ref($paramValues))
  }];

  let hasRegionVerifier = 1;

  let extraClassDeclaration = [{
    /// Return the function type of this call.
    FunctionType getFunctionType();
    /// Return the signature type of this call.
    SignatureType getSignature();
  }];
}

//===----------------------------------------------------------------------===//
// RegionBodyOp
//===----------------------------------------------------------------------===//

def RegionBodyOp : KGENOp<"region.body",
    [IsolatedFromAbove, Terminator,
     ParentOneOf<["AddressOfOp", "CallOp", "CallParamOp"]>,
     SingleBlockImplicitTerminator<"ReturnOp">,
     DeclareOpInterfaceMethods<KGENDeclInterface>]> {
  let summary = "Define an isolated from above region parameter body.";
  let description = [{
    The `kgen.region.body` operation defines a region parameter that is isolated
    from above. It is used by `kgen.addressof`, `kgen.call`, and
    `kgen.call_param` to specify region parameters. The operation defines a
    parameter signature with input and output parameters for the callable body
    and constraints over them.

    Example:

    ```mlir
    kgen.call @foreach<fn: (index) -> index = region>() : () -> ()
    fn(%arg1: index) {
      kgen.return %arg1 : index
    }
    ```
  }];

  let arguments = (ins ParamDeclArrayAttr:$paramDecls,
                       KTypeArrayAttr:$resultParamTypes,
                       ConstraintArrayAttr:$constraints);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = [{
    `` custom<OptionalParameterSpec>($paramDecls, $resultParamTypes)
    `` custom<OptionalConstraints>($constraints)
    `` custom<RegionBody>($bodyRegion) attr-dict
  }];

  let extraClassDeclaration = [{
    /// Get the terminator.
    ReturnOp getReturnOp();
  }];

  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// RegionOpenBodyOp
//===----------------------------------------------------------------------===//

def RegionOpenBodyOp : KGENOp<"region.open_body",
    [Terminator, HasParent<"InlinedCallOp">,
     SingleBlockImplicitTerminator<"ReturnOp">,
     DeclareOpInterfaceMethods<KGENDeclInterface>]> {
  let summary = "Define a region parameter body.";
  let description = [{
    The `kgen.region.open_body` operation defines a region parameter that is not
    isolated from above. It is used by `kgen.inlined_call` to specify region
    parameters to the callee that will be inlined.

    Example:

    ```mlir
    %0 = index.constant 0
    %1 = kgen.inlined_call[<fn: () -> index>() -> index: @foo]
                          <fn: () -> index = region>()
    fn() {
      // Return a captured value.
      kgen.return %0 : index
    }
    ```
  }];

  let arguments = (ins ParamDeclArrayAttr:$paramDecls,
                       KTypeArrayAttr:$resultParamTypes,
                       ConstraintArrayAttr:$constraints);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = [{
    `` custom<OptionalParameterSpec>($paramDecls, $resultParamTypes)
    `` custom<OptionalConstraints>($constraints)
    `` custom<RegionBody>($bodyRegion) attr-dict
  }];

  let extraClassDeclaration = [{
    /// Get the terminator.
    ReturnOp getReturnOp();
  }];

  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ParamConstantOp
//===----------------------------------------------------------------------===//

def ParamConstantOp : KGENOp<"param.constant",
                             [FirstAttrDerivedResultType, Pure,
                              ConstantLike]> {
  let summary = [{
    Return the value of a parameter expression as an SSA value that may be used
    by other ops.
  }];

  let arguments = (ins ParameterExprAttr:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat =
     "`` custom<ParamConstantOpValue>($value, qualified(type($result))) attr-dict";

  let builders = [OpBuilder<(ins "TypedAttr":$value)>];

  let hasFolder = true;
}

//===----------------------------------------------------------------------===//
// ParamDeclareOp
//===----------------------------------------------------------------------===//

def ParamDeclareOp : KGENOp<"param.declare", []> {
  let summary = [{
    Declare a new parameter with a value.
  }];

  let arguments = (ins ParamDeclSize1ArrayAttr:$paramDecls,
                       ParameterExprAttr:$value);
  let assemblyFormat =
     "custom<ParamDeclareOpValue>($paramDecls, $value) attr-dict";

  let builders = [
    OpBuilder<(ins "ParamDeclAttr":$decl, "Attribute":$value)>,
  ];

  let extraClassDeclaration = [{
    /// Return the single parameter declaration as a well typed attribute.
    ParamDeclAttr getParamDecl();
  }];
}

//===----------------------------------------------------------------------===//
// ParamSearchOp
//===----------------------------------------------------------------------===//

def ParamSearchOp : KGENOp<"param.search", []> {
  let summary = [{
    Declare a new parameter that gets one out of a set of values at elaboration
    time.
  }];

  let arguments = (ins ParamDeclSize1ArrayAttr:$paramDecls,
                       ParameterExprArrayAttr:$values);
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

def ReturnOp : KGENOp<"return", [Terminator, ReturnLike]> {
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

  let arguments = (ins ParameterExprArrayAttr:$parameters,
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

def ParamAssertOp : KGENOp<"param.assert", []> {
  let summary = "check a parameter expression at elaboration time";
  let description = [{
    The `kgen.param.assert` operation ensures that a parameter expression is
    true at elaboration time.  It is a more flexible form of the constraint list
    on a generator.
  }];

  let arguments = (ins TypedParameterExprAttr<I1>:$cond, StrAttr:$message);
  let assemblyFormat =
    "` ` `<` custom<ParamAssertOpValue>($cond) `>` `,` $message attr-dict";

  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// IncludeOp
//===----------------------------------------------------------------------===//

def IncludeOp : KGENOp<"include", []> {
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

def RebindOp : KGENOp<"rebind"> {
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

def StructDeclOp : KGENOp<"struct.decl", [HasParent<"mlir::ModuleOp">, Symbol,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    DeclareOpInterfaceMethods<KGENDeclInterface>, NoTerminator]> {
  let summary = "Struct declaration.";
  let description = [{
    The `kgen.struct.decl` operation is a top-level operation used to declare
    parametric KGEN struct types. Struct declarations are named with a symbol
    and have a parameter declaration list. The body contains only
    `kgen.struct.field` operations that declare the contents of the struct.

    Example:

    ```mlir
    kgen.struct.decl @Buffer<size, dtype: dtype> {
      size : index
      ptr : !pop.pointer<scalar<dtype>>
      dtype : !kgen.dtype
    }
    ```

    The parameters of the declaration can be constraints with optional
    constraints.

    Example:

    ```mlir
    kgen.struct.decl @SmallString<size>
        constraints [<le(size, 8), "string must be small">] {
      data : !pop.array<size, scalar<si8>>
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, ParamDeclArrayAttr:$paramDecls,
                       // Needed for KGENDeclInterface. Always empty.
                       KTypeArrayAttr:$resultParamTypes);
  let regions = (region SizedRegion<1>:$fields);
  let assemblyFormat = [{
    $sym_name `` custom<OptionalParameterSpec>($paramDecls, $resultParamTypes)
    custom<StructFields>($fields)
    attr-dict
  }];

  let extraClassDeclaration = [{
    /// Get the field declarations.
    auto getFieldDecls() { return getFields().getOps<StructFieldOp>(); }
    auto field_begin() { return getFieldDecls().begin(); }
    auto field_end() { return getFieldDecls().end(); }
  }];

  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// StructFieldOp
//===----------------------------------------------------------------------===//

def StructFieldOp : KGENOp<"struct.field", [HasParent<"StructDeclOp">]> {
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
  let assemblyFormat = "`` custom<KeywordAsString>($name) attr-dict `:` $type";
}

//===----------------------------------------------------------------------===//
// StructCreateOp
//===----------------------------------------------------------------------===//

def StructCreateOp : KGENOp<"struct.create", [Pure,
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
      -> !kgen.ref<@MyStruct<T: dtype = type>>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs RefType:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// StructInsertOp
//===----------------------------------------------------------------------===//

def StructInsertOp : KGENOp<"struct.insert", [Pure,
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
      : !pop.scalar<f64> into !kgen.ref<@MyStruct<T: dtype = f64>>
    ```
  }];

  let arguments = (ins StrAttr:$field, AnyType:$value, RefType:$container);
  let results = (outs RefType:$result);
  let assemblyFormat = [{
    $value `,` $container `[` custom<KeywordAsString>($field) `]` attr-dict `:`
    type($value) `into` qualified(type($container))
  }];
}

//===----------------------------------------------------------------------===//
// StructExtractOp
//===----------------------------------------------------------------------===//

def StructExtractOp : KGENOp<"struct.extract", [Pure,
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
      : !pop.scalar<f64> from !kgen.ref<@MyStruct<T: dtype = f64>>
    ```
  }];

  let arguments = (ins StrAttr:$field, RefType:$container);
  let results = (outs AnyType:$value);
  let assemblyFormat = [{
    $container `[` custom<KeywordAsString>($field) `]` attr-dict `:`
    type($value) `from` qualified(type($container))
  }];
}

//===----------------------------------------------------------------------===//
// StructGEPOp
//===----------------------------------------------------------------------===//

def StructGEPOp : KGENOp<"struct.gep", [Pure,
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

  let arguments = (ins StrAttr:$field, PointerOf<RefType>:$container);
  let results = (outs POP_PointerType:$result);
  let assemblyFormat = [{
    $container `[` custom<KeywordAsString>($field) `]` attr-dict `:`
    type($result) `from` type($container)
  }];
}

//===----------------------------------------------------------------------===//
// PrecompiledOpBase
//===----------------------------------------------------------------------===//

class PrecompiledOpBase<string mnemonic, dag additionalArgs>
  : KGENOp<"precompiled." # mnemonic,
    [HasParent<"mlir::ModuleOp">,
     FunctionOpInterface,
     Symbol,
     DeclareOpInterfaceMethods<CallableOpInterface>,
     DeclareOpInterfaceMethods<KGENDeclInterface>]> {

  let arguments = !con((ins SymbolNameAttr:$sym_name,
                            TypeAttrOf<FunctionType>:$function_type,
                            LinkageAttr:$linkage,
                            ParamDeclArrayAttr:$paramDecls,
                            KTypeArrayAttr:$resultParamTypes), additionalArgs);

  // We want to work with the FunctionOpInterface, which always requires a body
  // region.  We are "declaration like" though, so ours never has a block.
  let regions = (region SizedRegion<0>:$body);

  let extraClassDeclaration = [{
   /// Returns the argument types of this precompiled op.
   ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

   /// Returns the result types of this precompiled op.
   ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let extraClassDefinition = [{
    /// Parse an instance of this precompiled op.
    ParseResult $cppClass::parse(::mlir::OpAsmParser &parser,
                                 ::mlir::OperationState &result) {
      return parsePrecompiledOp(parser, result);
    }

    /// Print an instance of this precompiled op.
    void $cppClass::print(::mlir::OpAsmPrinter &p) {
      printPrecompiledOp(p, *this);
    }

    /// Return null to indicate an external callable.
    Region *$cppClass::getCallableRegion() { return nullptr; }

    /// Return the results of calling this operation.
    ArrayRef<Type> $cppClass::getCallableResults() { return getResultTypes(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// PrecompiledLLVMOp
//===----------------------------------------------------------------------===//

def PrecompiledLLVMOp : PrecompiledOpBase<"llvm",
    (ins TargetInfoAttr:$compiledFor, StrAttr:$llvm)> {
  let summary = "Declare a precompiled `kgen.func`'s LLVM lowering";
  let description = [{
    The `kgen.precompiled.llvm` operation provides an IR handle for a
    `kgen.func` that is already compiled to LLVM. It stores a cache key to
    the LLVM representation of this function (the `llvm` attr). We can back
    track up the compilation stack by preserving a mapping from a
    `kgen.precompiled.llvm` back to a serialized `kgen.func`.

    `kgen.precompiled.llvm` is target-specific, as the LLVM lowering requires
    a data layout.
  }];

  let builders = [
    // Build this from a FuncOp.
    OpBuilder<(ins "M::KGEN::FuncOp":$func,
                   "M::KGEN::TargetInfoAttr":$compiledFor,
                   "llvm::StringRef":$llvm)>
  ];
}

//===----------------------------------------------------------------------===//
// PrecompiledObjectOp
//===----------------------------------------------------------------------===//

def PrecompiledObjectOp : PrecompiledOpBase<"object",
    (ins TargetInfoAttr:$compiledFor, StrAttr:$object)> {
  let summary = "Declare a precompiled `kgen.func`'s cached object";
  let description = [{
    The `kgen.precompiled.object` operation provides an IR handle for a
    `kgen.func` that is already compiled to an object file. It stores a cache
    key to the object file (the `object` attr). We can back track up the
    compilation stack by preserving a mapping from a `kgen.precompiled.object`
    back to a serialized `kgen.precompiled.llvm`.

    `kgen.precompiled.object` is target-specific, it contains the raw bytes
    of a machine-specific object file.
  }];

  let builders = [
    // Build this from a PrecompiledLLVMOp.
    OpBuilder<(ins "M::KGEN::PrecompiledLLVMOp":$func,
                   "llvm::StringRef":$object)>
  ];
}
