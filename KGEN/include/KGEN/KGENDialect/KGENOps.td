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
include "Support/HLCFDialect/HLCFInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for the operation in this dialect.
class KGENOp<string mnemonic, list<Trait> traits = []> :
    Op<KGENDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def KGEN_FuncOp : KGENOp<"func", [
    HasParent<"mlir::ModuleOp">, FunctionOpInterface, Symbol, IsolatedFromAbove,
    SingleBlockImplicitTerminator<"ReturnOp">,
    DeclareOpInterfaceMethods<CallableOpInterface>,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>, DeclInterface,
    FuncInterface]> {
  let summary = "Concrete function definition.";
  let description = [{
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
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<KGEN_SignatureType>:$signature,
                       KGEN_AlwaysInlineLevelAttr:$alwaysInlineLevel);
  let regions = (region AnyRegion:$bodyRegion);

  let extraClassDeclaration = [{
    using OpTrait::SingleBlockImplicitTerminator<ReturnOp>::Impl<
        FuncOp>::getBody;

    ReturnOp getReturnOp();
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GeneratorOp
//===----------------------------------------------------------------------===//

def KGEN_GeneratorOp : KGENOp<"generator", [
    HasParent<"mlir::ModuleOp">, FunctionOpInterface, Symbol, IsolatedFromAbove,
    SingleBlockImplicitTerminator<"ReturnOp">,
    DeclareOpInterfaceMethods<CallableOpInterface>,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>, DeclInterface,
    FuncInterface]> {
  let summary = "Function generator definition.";
  let description = [{
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
  }];

  /// If the `implements` attribute is present then this generator is an
  /// implementation of the specified generator.interface.
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<KGEN_SignatureType>:$signature,
                       ConstraintArrayAttr:$constraints,
                       OptionalAttr<FlatSymbolRefAttr>:$implements,
                       KGEN_AlwaysInlineLevelAttr:$alwaysInlineLevel);
  let regions = (region AnyRegion:$bodyRegion);

  let extraClassDeclaration = [{
    using OpTrait::SingleBlockImplicitTerminator<ReturnOp>::Impl<
        GeneratorOp>::getBody;

    ReturnOp getReturnOp();
  }];

  let builders = [
    OpBuilder<(ins "StringAttr":$name, "SignatureType":$signature), [{
      build($_builder, $_state, name, signature, {}, {}, {});
    }]>
  ];

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

  let extraClassDeclaration = [{
    /// Interfaces cannot be `always_inline` because they have no body.
    AlwaysInlineLevel getAlwaysInlineLevel() {
      return AlwaysInlineLevel::Disabled;
    }
  }];
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
  }];

  let arguments = (ins KGEN_SymbolConstantAttr:$callee,
                       KGEN_ParamDeclArrayAttr:$paramDecls);
  let results = (outs FunctionType:$result);
  let assemblyFormat = [{
    custom<AddressOfOp>($callee, $paramDecls, type($result)) attr-dict
  }];

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
  }];

  let arguments = (ins
    KGEN_SymbolConstantAttr:$callee,
    KGEN_ParamDeclArrayAttr:$paramDecls,
    Variadic<AnyType>:$operands
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    custom<CallOp>($callee, $paramDecls, $operands, type($operands),
                   type($results)) attr-dict
  }];

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

def KGEN_CallParamOp : KGENOp<"call_param", [KGEN_CallOpInterface,
    CalleeArgumentTypes, CalleeResultTypes]> {
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
  }];

  let arguments = (ins
    AttrOfType<KGEN_SignatureType>:$callee,
    KGEN_ParamDeclArrayAttr:$paramDecls,
    Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    `` custom<ParametricCallee>($callee, $paramDecls) `(` $operands `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    bool isAllowedInFunc() { return false; }
  }];

  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// ParamConstantOp
//===----------------------------------------------------------------------===//

def KGEN_ParamConstantOp : KGENOp<"param.constant",
    [AllTypesMatch<["value", "result"]>, Pure, ConstantLike]> {
  let summary = [{
    The `kgen.param.constant` operation materializes the value of a parameter
    expression as an SSA value that may be used by other operations.
    Conceptually, it bridges the parameter/compile-time value domain to the
    dynamic/runtime value domain.

    Example:

    ```mlir
    %cst0 = kgen.param.constant: scalar<f32> = <"1.5">
    %cst1 = kgen.param.constant = <apply(:() -> index @makeIndex)>
    ```
  }];

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "`` custom<ParamConstantOpValue>($value) attr-dict";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// ParamDeclareOp
//===----------------------------------------------------------------------===//

def KGEN_ParamDeclareOp : KGENOp<"param.declare"> {
  let summary = "Declare a single parameter value.";
  let description = [{
    The `kgen.param.declare` operation declares a single parameter and binds
    its value to a parameter expression. The parameter is visible within and
    below the scope of the enclosing `DeclInterface` operation.

    Example:

    ```mlir
    kgen.param.declare A = <5>
    kgen.param.declare A_plus_one = <add(A, 1)>
    ```
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
// ParamDeclareRegionOp
//===----------------------------------------------------------------------===//

def KGEN_ParamDeclareRegionOp : KGENOp<"param.declare.region", [
    DeclInterface, FuncInterface, FunctionOpInterface,
    SingleBlockImplicitTerminator<"ReturnOp">]> {
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

    The operation carries an extra bit `isolated` to indicate that they are
    parametrically isolated from above.
  }];

  let arguments = (ins
    ParamDeclSize1ArrayAttr:$paramDecls,
    TypeAttrOf<KGEN_SignatureType>:$signature,
    ConstraintArrayAttr:$constraints,
    UnitAttr:$isolated,
    KGEN_AlwaysInlineLevelAttr:$alwaysInlineLevel
  );
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = [{
    custom<RegionDeclaration>($paramDecls, $signature, $constraints,
                              $alwaysInlineLevel, $bodyRegion) attr-dict
  }];

  let extraClassDeclaration = [{
    bool isIsolatedFromAbove(unsigned regionNum);
    void notifyKnownIsolatedFromAbove(unsigned regionNum);

    /// Region body op doesn't form a symbol.
    bool isOptionalSymbol() { return true; }

    /// Return the single parameter declaration.
    ParamDeclAttr getParamDecl();
  }];

  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ParamSearchOp
//===----------------------------------------------------------------------===//

def KGEN_ParamSearchOp : KGENOp<"param.search"> {
  let summary = [{
    The `kgen.param.search` operation declares a single parameter and binds its
    value to one of a set values at elaboration time. This operation represents
    a fork of the current generator by multi-versioning the generator based on
    the possible values of the declared parameter.

    Example:

    ```mlir
    // Fork the current generator into 3 children, each a different one of the
    // possible values of `A`.
    kgen.param.search A = <1, 2, 3>
    ```
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
}

//===----------------------------------------------------------------------===//
// ParamAssertOp
//===----------------------------------------------------------------------===//

def KGEN_ParamAssertOp : KGENOp<"param.assert", []> {
  let summary = "check a parameter expression at elaboration time";
  let description = [{
    The `kgen.param.assert` operation ensures that a parameter expression is
    true at elaboration time. It is a more flexible form of the constraint list
    on a generator.
  }];

  let arguments = (ins AttrOfType<I1>:$cond,
                       AttrOfType<KGEN_StringType>:$message);
  let assemblyFormat = [{
    ` ` `<` custom<ParamValue>($cond, "$_builder.getI1Type()") `>` `,`
    custom<StringParam>($message) attr-dict
  }];

  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// ParamIfOp
//===----------------------------------------------------------------------===//

def KGEN_ParamIfOp : KGENOp<"param.if", [DeclInterface,
    DeclareOpInterfaceMethods<ControlFlowNode>]> {
  let summary = "compile-time constant if";
  let description = [{
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
  }];

  let arguments = (ins AttrOfType<I1>:$cond,
                       KGEN_ParamDeclArrayAttr:$paramDecls);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$thenRegion, SizedRegion<1>:$elseRegion);

  let extraClassDeclaration = [{
    Block::OpListType &getThenOps() {
      return getThenRegion().front().getOperations();
    }
    Block::OpListType &getElseOps() {
      return getElseRegion().front().getOperations();
    }
    /// For DeclInterface - this op never defines input parameters.
    ArrayRef<ParamDeclAttr> getInputParamDecls() { return {}; }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let builders = [OpBuilder<(ins "TypedAttr":$cond), [{
    build($_builder, $_state, {}, cond,
          ParamDeclArrayAttr::get(cond.getContext(), {}));
  }]>];
}

//===----------------------------------------------------------------------===//
// ParamYieldOp
//===----------------------------------------------------------------------===//

def KGEN_ParamYieldOp : KGENOp<"param.yield", [Pure, HasParent<"ParamIfOp">,
    Terminator, DeclareOpInterfaceMethods<ControlFlowTerminator>]> {
  let summary = "conditional yield operation";
  let description = [{
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

    kgen.param.if <condition -> result> {
      kgen.param.yield<1> // `then` definition for `result`
    } else {
      kgen.param.yield<0> // `else` definition for `result`
    }
    %0 = kgen.param.constant = <result>
    ```

    This operation can always be marked pure because the control-flow edge from
    this operation leads only to after the parent operation.
  }];

  let arguments = (ins Variadic<AnyType>:$operands,
                       KGEN_ParameterExprArrayAttr:$parameters);

  let builders = [
    OpBuilder<(ins), [{
      return build($_builder, $_state, {},
                   $_builder.getAttr<ParameterExprArrayAttr>(
                     ArrayRef<TypedAttr>{}));
    }]>
  ];

  let assemblyFormat = [{
    `` custom<ParameterValues>($parameters) attr-dict
    ($operands^ `:` type($operands))?
  }];
}

//===----------------------------------------------------------------------===//
// RebindOp
//===----------------------------------------------------------------------===//

def KGEN_RebindOp : KGENOp<"rebind", [Pure]> {
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
// ExportOp
//===----------------------------------------------------------------------===//

def KGEN_ExportOp : KGENOp<"export", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Declare an exported symbol.";
  let description = [{
    The `kgen.export` operation specifies an exported symbol from a
    given module. The provided alias name will be used as the actual
    symbol name in the C header and object file generation.

    Example:

    ```mlir
    kgen.export @foo as @my_foo

    // The two examples below are equivalent.
    kgen.export @foo as @foo
    kgen.export @foo
    ```
  }];

  let arguments = (ins SymbolRefAttr:$exported, SymbolNameAttr:$alias);
  let results = (outs);
  let assemblyFormat = [{
      $exported custom<ExportOp>(ref($exported), $alias) attr-dict
  }];
  let hasVerifier = 1;
}
