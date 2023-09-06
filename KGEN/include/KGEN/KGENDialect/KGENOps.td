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
include "KGEN/HLCFDialect/HLCFInterfaces.td"
include "KGEN/Interpreter/InterpreterInterface.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
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
    SingleBlock, DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    DeclInterface, FuncInterface, FunctionLike, ExportInterface]> {
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

    There are cases where we might have a `kgen.func` with a
    `precompiledBodyRef` attribute, like so:
    ```mlir
    kgen.func @someFn(%arg0: index) -> index attributes {
      precompiledBodyRef = @importedLib
    } {
      ...
    }

    The presence of this attribute means that the function was already compiled
    all the way to an object file once, so we should avoid doing it again.
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<KGEN_SignatureType>:$signature,
                       KGEN_DecoratorsAttr:$decorators,
                       KGEN_InlineLevelAttr:$inlineLevel,
                       KGEN_ExportKindAttr:$exportKind,
                       // If the body was precompiled, this will hold a
                       // reference to the link directive.
                       OptionalAttr<SymbolRefAttr>:$precompiledBodyRef);
  let regions = (region MaxSizedRegion<1>:$bodyRegion);

  let extraClassDeclaration = [{
    using OpTrait::SingleBlock<FuncOp>::getBody;

    ArrayRef<ParamDeclAttr> getInputParams() { return {}; }
    ArrayRef<ParamDeclAttr> getResultParams() { return {}; }
    FunctionType getFunctionType() { return getSignature().getValues(); }
    Region *getCallableRegion() { return &getBodyRegion(); }
  }];

  let builders = [
    // Builder to be used when there is no precompiled body.
    OpBuilder<(ins "StringAttr":$name, "SignatureType":$signature,
                   CArg<"InlineLevel",
                        "InlineLevel::Automatic">:$inlineLevel,
                   CArg<"ExportKind", "ExportKind::NotExported">:$exportKind,
                   CArg<"ArrayRef<TypedAttr>", "{}">:$decorators), [{
      build($_builder, $_state, name, signature, decorators,
            inlineLevel, exportKind, /*precompiledBodyRef=*/{});
    }]>,
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ExternFuncOp
//===----------------------------------------------------------------------===//

def KGEN_ExternFuncOp : KGENOp<"extern.func", [
    HasParent<"mlir::ModuleOp">, Symbol, DeclInterface, FuncInterface,
    ExportInterface, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "External function declaration.";
  let description = [{
    The `kgen.extern.func` operation represents a concrete KGEN function that
    is defined in an external library. It has no body. It cannot represent any
    parametric functions.

    Example:

    ```mlir
    kgen.link "libsomelib.a" as @importedLib

    kgen.extern.func @someFn(%arg0: index) -> index from @importedLib
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<KGEN_SignatureType>:$signature,
                       KGEN_ExportKindAttr:$exportKind,
                       OptionalAttr<SymbolRefAttr>:$importedFrom);

  let extraClassDeclaration = [{
    InlineLevel getInlineLevel() { return InlineLevel::Never; }
    ArrayRef<ParamDeclAttr> getInputParams() { return {}; }
    ArrayRef<ParamDeclAttr> getResultParams() { return {}; }
    FunctionType getFunctionType() { return getSignature().getValues(); }
  }];

  let assemblyFormat = [{
    custom<SymbolExport>($exportKind) $sym_name `` custom<Signature>($signature)
    (`from` $importedFrom^)? attr-dict-with-keyword
  }];
}

//===----------------------------------------------------------------------===//
// GeneratorOp
//===----------------------------------------------------------------------===//

def KGEN_GeneratorOp : KGENOp<"generator", [
    HasParent<"mlir::ModuleOp">, FunctionOpInterface, Symbol, IsolatedFromAbove,
    SingleBlock, DeclInterface, FuncInterface, FunctionLike, ExportInterface]> {
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
                       TypeAttrOf<FunctionType>:$functionType,
                       KGEN_ParamDeclArrayAttr:$inputParams,
                       KGEN_ParamDeclArrayAttr:$resultParams,
                       ConstraintArrayAttr:$constraints,
                       KGEN_DecoratorsAttr:$decorators,
                       KGEN_InlineLevelAttr:$inlineLevel,
                       KGEN_ExportKindAttr:$exportKind);
  let regions = (region MaxSizedRegion<1>:$bodyRegion);

  let extraClassDeclaration = [{
    using OpTrait::SingleBlock<GeneratorOp>::getBody;
    Region *getCallableRegion() { return &getBodyRegion(); }
  }];

  let builders = [
    OpBuilder<(ins "StringAttr":$name, "SignatureType":$signature,
                   "FunctionType":$functionType,
                   "ArrayRef<ParamDeclAttr>":$inputParams,
                   "ArrayRef<ParamDeclAttr>":$resultParams), [{
      build($_builder, $_state, name, signature, functionType, inputParams,
            resultParams, /*constraints=*/{}, /*decorators=*/{},
            InlineLevel::Automatic, ExportKind::NotExported);
    }]>,
    OpBuilder<(ins "StringAttr":$name, "SignatureType":$signature,
                   "FunctionType":$functionType), [{
      build($_builder, $_state, name, signature, functionType, {}, {});
    }]>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ExternGeneratorOp
//===----------------------------------------------------------------------===//

def KGEN_ExternGeneratorOp : KGENOp<"extern.generator", [
    HasParent<"mlir::ModuleOp">, Symbol, DeclInterface, FuncInterface,
    ExportInterface]> {
  let summary = "External generator declaration.";
  let description = [{
    The `kgen.extern.generator` operation declares a KGEN generator with a
    external definition. The definition and its dependencies must be made
    available prior to elaboration, but this operations allows manipulating
    sections of IR without requiring all transitive dependencies be present in
    the module.

    Example:

    ```mlir
    kgen.extern.generator @kernel<simd_width>(!pop.simd<simd_width, f32>)
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<KGEN_SignatureType>:$signature,
                       TypeAttrOf<FunctionType>:$functionType,
                       KGEN_ParamDeclArrayAttr:$inputParams,
                       KGEN_ParamDeclArrayAttr:$resultParams,
                       KGEN_ExportKindAttr:$exportKind);

  let extraClassDeclaration = [{
    InlineLevel getInlineLevel() { return InlineLevel::Never; }
  }];

  let assemblyFormat = [{
    custom<SymbolExport>($exportKind) $sym_name ``
    custom<ExternGenerator>($signature, $functionType, $inputParams,
                            $resultParams) attr-dict-with-keyword
  }];
}

//===----------------------------------------------------------------------===//
// CallOpBase
//===----------------------------------------------------------------------===//

class KGEN_CallOpBase<string mnemonic, list<Trait> traits = []>
    : KGENOp<mnemonic, traits # [
        DeclareOpInterfaceMethods<KGEN_CallOpInterface>,
        DeclareOpInterfaceMethods<KGEN_GeneratorUserOpInterface>,
        DeclareOpInterfaceMethods<ParamOpInterface,
                                  ["isImplicitlyParametric"]>]>,
      CallLikeOperation;

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def KGEN_CallOp : KGEN_CallOpBase<"call", [CalleeArgumentTypes,
    CalleeResultTypes, DeclareOpInterfaceMethods<CallOpInterface>]> {
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
      kgem.param.result_bind<add(N, 2)>
      kgen.return
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
    ArrayRef<TypedAttr> getParamValues() {
      return getCallee().getParamValues();
    }
  }];
}

//===----------------------------------------------------------------------===//
// CallParamOp
//===----------------------------------------------------------------------===//

def KGEN_CallParamOp : KGEN_CallOpBase<"call_param",
    [CalleeArgumentTypes, CalleeResultTypes]> {
  let summary = "Call declaration specified by a parameter expression";
  let description = [{
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

def KGEN_ParamConstantOp : KGENOp<"param.constant", [Pure, ConstantLike,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "create a constant value from a parameter value";
  let description = [{
    The `kgen.param.constant` operation materializes the value of a parameter
    expression as a constant SSA value that may be used by other operations.
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
  let assemblyFormat = [{
    `` custom<ParamConstantOpValue>($value, type($result)) attr-dict
  }];

  let builders = [
    OpBuilder<(ins "TypedAttr":$value),
              [{ build($_builder, $_state, value.getType(), value); }]>
  ];

  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ParamMaterializeOp
//===----------------------------------------------------------------------===//

def KGEN_ParamMaterializeOp : KGENOp<"param.materialize",
    [DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = [{
    The `kgen.param.materialize` operation materializes the value of a parameter
    expression as an SSA value that may be used by other operations. It bridges
    the parameter domain to the runtime domain. This operation assumes the
    materialization of the parameter expression may have memory allocation
    side-effects.

    Example:

    ```mlir
    %value0 = kgen.param.materialize = <1>
    %value1 = kgen.param.materialize: pointer<f32> = <ptr_val>
    ```
  }];

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs Res<AnyType, "", [MemAlloc]>:$result);
  let assemblyFormat = [{
    `` custom<ParamConstantOpValue>($value, type($result)) attr-dict
  }];

  let builders = [
    OpBuilder<(ins "TypedAttr":$value),
              [{ build($_builder, $_state, value.getType(), value); }]>
  ];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// ParamDeclareOp
//===----------------------------------------------------------------------===//

def KGEN_ParamDeclareOp : KGENOp<"param.declare",
    [DeclareOpInterfaceMethods<ParamOpInterface>]>, OneParamDecl {
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

  let arguments = (ins KGEN_ParamDeclAttr:$paramDecl,
                       TypedAttrInterface:$value);
  let assemblyFormat = [{
    custom<ParamDeclareOpValue>($paramDecl, $value) attr-dict
  }];

  let extraClassDeclaration = [{
    StringAttr getName() { return getParamDecl().getName(); }
    Type getType() { return getParamDecl().getType(); }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ParamDeclareRegionOp
//===----------------------------------------------------------------------===//

def KGEN_ParamDeclareRegionOp : KGENOp<"param.declare.region", [
    DeclInterface, FuncInterface, FunctionOpInterface, FunctionLike,
    DeclareOpInterfaceMethods<ParamOpInterface, ["collectParameterUses"]>,
    SingleBlock
  ]>, OneParamDecl {
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
      kgem.param.result_bind<add(N, 2)>
      kgen.return
    }
    ```

    The operation carries an extra bit `isolated` to indicate that they are
    parametrically isolated from above.
  }];

  let arguments = (ins
    KGEN_ParamDeclAttr:$paramDecl,
    TypeAttrOf<KGEN_SignatureType>:$signature,
    TypeAttrOf<FunctionType>:$functionType,
    KGEN_ParamDeclArrayAttr:$inputParams,
    KGEN_ParamDeclArrayAttr:$resultParams,
    ConstraintArrayAttr:$constraints,
    UnitAttr:$isolated,
    KGEN_InlineLevelAttr:$inlineLevel
  );
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = [{
    custom<RegionDeclaration>($paramDecl, $inputParams, $resultParams,
                              $functionType, $signature, $constraints,
                              $inlineLevel, $bodyRegion)
    attr-dict
  }];

  let extraClassDeclaration = [{
    bool isIsolatedFromAbove(unsigned regionNum);
    void notifyKnownIsolatedFromAbove(unsigned regionNum);
    Region *getCallableRegion() { return &getBodyRegion(); }

    /// Region body op doesn't form a symbol.
    bool isOptionalSymbol() { return true; }
  }];
}

//===----------------------------------------------------------------------===//
// ParamForkOp
//===----------------------------------------------------------------------===//

def KGEN_ParamForkOp : KGENOp<"param.fork",
    [DeclareOpInterfaceMethods<ParamOpInterface>]>, OneParamDecl {
  let summary = [{
    The `kgen.param.fork` operation declares a tuple of parameters and binds its
    value to one of a set values at elaboration time. This operation represents
    a fork of the current generator by multi-versioning the generator based on
    the possible values of the declared parameter.

    Example:

    ```mlir
    kgen.param.declare PARAMS: !kgen.variadic<!kgen.list<index[2]>> =
                              <#kgen.variadic<[1, 11], [2, 22], [3, 33]>>
    kgen.param.fork PARAM_TUPLE : list<index[2]> = <PARAMS>
    ```
  }];

  let arguments = (ins KGEN_ParamDeclAttr:$paramDecl,
                       AttrOfType<KGEN_VariadicType>:$values);
  let assemblyFormat = [{
    custom<ParamForkOpValue>($paramDecl, $values) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// ParamApplyOp
//===----------------------------------------------------------------------===//

def KGEN_ParamApplyOp : KGENOp<"param.apply",
    [DeclareOpInterfaceMethods<KGEN_GeneratorUserOpInterface>,
     DeclareOpInterfaceMethods<ParamOpInterface>]> {
  let summary = "Function call in parameter domain.";
  let description = [{
    The `kgen.param.apply` operation is a call operation entirely in the
    parameter domain. The operands to the call are parameter expressions and the
    results of the call are bound to parameter declarations. This is the 'apply'
    operator as an operation.

    This form of the 'apply' operator is beneficial for the elaborator, because
    it gives the elaborator a "save point" for multiversioning functions, if
    the 'apply' callee has multiple implementations.

    The callee is limited to a single result value with no result parameters.

    Example:

    ```mlir
    kgen.param.declare sw: i1 = <1>
    kgen.param.apply A = [(i1) -> !pop.simd<8, f32>: callee](sw)
    ```
  }];

  let arguments = (ins KGEN_ParamDeclAttr:$paramDecl,
                       AttrOfType<KGEN_SignatureType>:$callee,
                       KGEN_ParameterExprArrayAttr:$operands);
  let assemblyFormat = [{
    custom<ParamApplyOp>($paramDecl, $callee, $operands) attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CostOfOp
//===----------------------------------------------------------------------===//

def KGEN_CostOfOp : KGENOp<"cost_of", [
    DeclareOpInterfaceMethods<InterpreterOpInterface>,
    DeclareOpInterfaceMethods<KGEN_GeneratorUserOpInterface>,
    DeclareOpInterfaceMethods<ParamOpInterface, ["isImplicitlyParametric"]>
  ]>, CallLikeOperation {
  let summary = "Return the 'cost' of the function.";
  let description = [{
    The `kgen.cost_of` operation takes a parametric callee and returns
    characteristics of the function that can be used to develop heuristics for
    the cost of the function. This operation must be resolved at compile time.

    Currently, `kgen.cost_of` returns the number of operations in the callee.
    Specifically, the cost is evaluated on the function at the output of
    elaboration, without running any post-elaboration passes.

    Example:

    ```mlir
    %numOps = kgen.cost_of[(index) -> index: @evaluateMe]
    ```
  }];

  let arguments = (ins TypedAttrInterface:$callee,
                       KGEN_ParamDeclArrayAttr:$paramDecls);
  let results = (outs Index:$result);

  let assemblyFormat = [{
    `` custom<ParametricCallee>($callee, $paramDecls) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// ParamEvaluateOp
//===----------------------------------------------------------------------===//

def KGEN_ParamEvaluateOp : KGENOp<"param.evaluate",
    [DeclareOpInterfaceMethods<ParamOpInterface>]> {
  let summary = "Evaluate candidates of functions with an evaluator.";
  let description = [{
    The `kgen.param.evaluate` operation takes an evaluator function reference
    and variadic function references and declares a parameter that represents
    the selected function candidate using the evaluator.

    This form of the 'evaluate' operator is beneficial for the elaborator,
    because it gives the elaborator a "save point" for multiversioning
    functions.

    Example:

    ```mlir
    kgen.param.evaluate best: (index) -> i1 = [@fooImpl, @barImpl]
      with [(!kgen.pointer<(index) -> i1>, index) -> index: @evaluator]
    ```
  }];

  let arguments = (ins KGEN_ParamDeclAttr:$paramDecl,
                       AttrOfType<KGEN_SignatureType>:$evaluator,
                       AttrOfType<KGEN_VariadicType>:$candidates);
  let assemblyFormat = [{
    custom<ParamEvaluateOp>($paramDecl, $evaluator, $candidates) attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ParamResultBindOp
//===----------------------------------------------------------------------===//

def KGEN_ParamResultBindOp : KGENOp<"param.result_bind",
    [DeclareOpInterfaceMethods<ParamOpInterface, ["isImplicitlyParametric"]>]>,
    NoParamDecls {
  let summary = "bind values for the result parameters of the nearest scope";
  let description = [{
    The `kgen.param.result_bind` operation binds values for the result
    parameters of the nearest enclosing parameter scope. Values for the result
    parameters must be bound at least once.

    Example:

    ```mlir
    kgen.generator @result_params<in: i1 -> out>() {
      kgen.param.if <in -> value> {
        kgen.param.result_bind<1>
        kgen.param.yield
      } else {
        kgen.param.result_bind<2>
        kgen.param.yield
      }
      kgen.param.result_bind<value>
      kgen.return
    }
    ```
  }];

  let arguments = (ins KGEN_ParameterExprArrayAttr:$parameters);
  let assemblyFormat = "`` custom<ParameterValues>($parameters) attr-dict";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def KGEN_ReturnOp : KGENOp<"return", [Terminator, ReturnLike,
    DeclareOpInterfaceMethods<ControlFlowTerminator>]> {
  let summary = "return operation for kgen declarations";
  let description = [{
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
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>
  ];
}

//===----------------------------------------------------------------------===//
// UnreachableOp
//===----------------------------------------------------------------------===//

def KGEN_UnreachableOp : KGENOp<"unreachable", [Terminator,
    DeclareOpInterfaceMethods<ControlFlowTerminator>]> {
  let summary = "Unreachable operation";
  let description = [{
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
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// ParamAssertOp
//===----------------------------------------------------------------------===//

def KGEN_ParamAssertOp : KGENOp<"param.assert",
    [DeclareOpInterfaceMethods<ParamOpInterface, ["isImplicitlyParametric"]>]>,
    NoParamDecls {
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
    DeclareOpInterfaceMethods<ControlFlowNode>,
    DeclareOpInterfaceMethods<ParamOpInterface, ["isImplicitlyParametric",
      "collectParameterUsesBelow"]>]> {
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

  let arguments = (ins
    AttrOfType<I1>:$cond,
    KGEN_ParamDeclArrayAttr:$resultParams,
    UnitAttr:$thenIsolated,
    UnitAttr:$elseIsolated
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$thenRegion, SizedRegion<1>:$elseRegion);

  let extraClassDeclaration = [{
    bool isIsolatedFromAbove(unsigned regionNum);
    void notifyKnownIsolatedFromAbove(unsigned regionNum);

    Block::OpListType &getThenOps() {
      return getThenRegion().front().getOperations();
    }
    Block::OpListType &getElseOps() {
      return getElseRegion().front().getOperations();
    }

    /// Parametric ifs have no input parameters.
    ArrayRef<ParamDeclAttr> getInputParams() { return {}; }
  }];

  let assemblyFormat = [{
    ` ` `<` custom<ParamValue>($cond, "$_builder.getI1Type()") ``
    custom<OptionalParamDecls>($resultParams) `>` (`->` type($results)^)?
    $thenRegion `else` $elseRegion attr-dict
  }];

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
    ```

    This operation can always be marked pure because the control-flow edge from
    this operation leads only to after the parent operation.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ return build($_builder, $_state, {}); }]>
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
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

  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// UndefOp
//===----------------------------------------------------------------------===//

def KGEN_UndefOp : KGENOp<"undef", [Pure]> {
  let summary = "Return an undefined value.";
  let description = [{
    The `kgen.undef` operation returns an undefined value of a given type.
    Accessing an undefined value is undefined behaviour (duh).

    Example:

    ```mlir
    %0 = kgen.undef : !kgen.declref<@Int>
    %1 = kgen.undef : !pop.struct<array<2, index>>
    ```
  }];

  let results = (outs AnyType:$value);
  let assemblyFormat = "attr-dict `:` type($value)";
}

//===----------------------------------------------------------------------===//
// CallSignatureOp
//===----------------------------------------------------------------------===//

def KGEN_CallSignatureOp : KGENOp<"call_signature", [
    TypesMatchWith<"operands match callee argument types","callee",
    "arguments", "cast<SignatureType>($_self).getValueInputs()">,
    CalleeResultTypes]> {
  let summary = "Call a signature typed instance";
  let description = [{
    The `kgen.call_signature` operation takes an SSA value of `!kgen.signature`
    type and invokes it with the provided operands as arguments.

    Example:

    ```mlir
    %0 = kgen.call_signature %closure(%arg0, %arg1)
      : (index, index) capturing -> index
    ```
  }];

  let arguments = (ins KGEN_SignatureType:$callee,
                       Variadic<AnyType>:$arguments);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $callee `(` $arguments `)` `:`
    custom<CallSignature>(type($callee), type($arguments), type($results))
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// StageClosureOp
//===----------------------------------------------------------------------===//

// TODO(LLVM Issue #59877): Why isn't this in BuiltinAttributes.td?
def Builtin_LocationAttr : Builtin_Attr<"Location", []> {}

def KGEN_StageClosureOp : KGENOp<"stage_closure",
  [Pure, FunctionLike, DebugInfo_InlinedSubprogramScoped]
>{
  let summary = "Stage a closure declaring a concrete region";
  let description = [{
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
  }];

  let arguments = (ins OptionalAttr<Builtin_LocationAttr>:$callLoc);
  let results = (outs KGEN_SignatureType:$result);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = [{
    custom<StageClosureOp>(type($result), $bodyRegion) attr-dict
    (`callLoc` `(` $callLoc^ `)`)?
  }];

  let extraClassDeclaration = [{
    ArrayRef<Type> getResultTypes() {
       return getResult().getType().getValues().getResults();
    }
  }];
}

//===----------------------------------------------------------------------===//
// CreateClosureOp
//===----------------------------------------------------------------------===//

def KGEN_CreateClosureOp : KGEN_CallOpBase<"create_closure",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Create a closure from variables and a function.";
  let description = [{
    The `kgen.create_closure` operation represents the instantiation of
    a closure. This operation models closure creation in terms of a function
    and local captured variables.

    ```mlir
    %idx0 = index.constant 0
    %0 = kgen.create_closure [(index) -> index: @h](%idx0)
    ```
  }];

  let arguments = (ins AttrOfType<KGEN_SignatureType>:$callee,
                       Variadic<AnyType>:$captures,
                       KGEN_ParamDeclArrayAttr:$paramDecls);
  let results = (outs KGEN_SignatureType:$result);
  let assemblyFormat = [{
    custom<ParametricCallee>($callee, $paramDecls) `(` $captures `)`
    custom<ClosureCaptureTypes>(ref($callee), ref($captures), type($captures))
    attr-dict
  }];

  let builders = [
    OpBuilder<(ins "TypedAttr":$callee, "ValueRange":$captures)>,
    OpBuilder<(ins "Type":$type, "TypedAttr":$callee, "ValueRange":$captures)>
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

def KGEN_GlobalOp : KGENOp<"global", [Symbol, NoTerminator,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>, ExportInterface]> {
  let summary = "Declare a global variable with an constructor.";
  let description = [{
    The `kgen.global` operation declares a global variable bound to a symbol
    name. It contains the type of the global variable, and optionally a
    constructor function, a destructor function, and an initialization priority.
    The initialization priority determines the order in which the constructors
    of all the globals in the program will be invoked (lower is earlier) and
    the reverse order in which the destructors are invoked.

    The constructors and destructors are optional, and so a global variable
    that does not require constructors and destructors can be defined by:

    ```mlir
    kgen.global @my_global : f32
    ```

    The optional constructor and destructor functions are expected to have
    zero arguments and zero results.

    Example:

    ```mlir
    kgen.global @foo : i32 [@foo_ctor, @foo_dtor](2)
    ```

    The global `@foo` with initialization priority `2` will be initialized
    after a global with priority lower than `2` and destroyed before.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name, TypeAttr:$type, OptionalAttr<I32Attr>:$priority,
    OptionalAttr<SymbolRefAttr>:$ctor, OptionalAttr<SymbolRefAttr>:$dtor,
    KGEN_ExportKindAttr:$exportKind);

  let assemblyFormat = [{
    `` custom<SymbolExport>($exportKind) $sym_name `:` $type
    ` ` (`[` $ctor `,` $dtor `]` `` `(` $priority^ `)`)? attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GlobalAddressOp
//===----------------------------------------------------------------------===//

def KGEN_GlobalAddressOp : KGENOp<"global.address", [Pure,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Get the address of a global variable.";
  let description = [{
    The `kgen.global.address` operation returns the address of a referenced
    global variable. The type of the result is a `!kgen.pointer` to the global
    type.

    Example:

    ```mlir
    %ptr = kgen.global.address @foo : <i32>
    ```
  }];

  let arguments = (ins SymbolRefAttr:$global);
  let results = (outs PointerType:$result);

  let assemblyFormat = "$global attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// LinkOp
//===----------------------------------------------------------------------===//

def KGEN_LinkOp : KGENOp<"link", [Symbol]> {
  let summary = "Link directive.";
  let description = [{
    The `kgen.link` operation defines a symbol that can be used to import
    C functions from other libraries. The `kgen.link` directives are uniqued
    by their symbol name - it is illegal to attempt to link more than one
    library as a given symbol name.

    For example:
    ```mlir
    kgen.link "/path/to/musl.libc.a" as @libc
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$linkPath,
    OptionalAttr<Builtin_DenseResourceElementsAttr>:$linkBytes);

  let assemblyFormat = "($linkPath^)? ($linkBytes^)? `as` $sym_name attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// IntLiteralCmp
//===----------------------------------------------------------------------===//

def KGEN_IntLiteralCmp : KGENOp<"int_literal.cmp", [Pure]> {
  let summary = "Comparison operator for int_literals.";
  let description = [{
    The `kgen.int_literal.cmp` operator does equality/inequality testing
    for `!kgen.int_literal` values and returns a bool.
  }];

  let arguments = (ins
    KGEN_IntLiteralCmpPredAttr:$pred,
    KGEN_IntLiteralType:$lhs,
    KGEN_IntLiteralType:$rhs
    );
  let results = (outs I1:$result);

  let assemblyFormat = "$pred `(` $lhs `,` $rhs `)` attr-dict";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// IntLiteralBinop
//===----------------------------------------------------------------------===//

def KGEN_IntLiteralBinop : KGENOp<"int_literal.binop", [Pure]> {
  let summary = "Binary operations for int_literals";
  let description = [{
    The `kgen.int_literal.binop` operator does standard arithmetic binary
    operations on `!kgen.int_literal` values that return a `!kgen.int_literal`.
  }];

  let arguments = (ins
    KGEN_IntLiteralBinopKindAttr:$oper,
    KGEN_IntLiteralType:$lhs,
    KGEN_IntLiteralType:$rhs);
  let results = (outs KGEN_IntLiteralType:$result);

  let assemblyFormat = "$oper `(` $lhs `,` $rhs `)` attr-dict";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// IntLiteralCastOp
//===----------------------------------------------------------------------===//

def KGEN_IntLiteralCastOp : KGENOp<"int_literal.cast", [Pure]> {
  let summary = "Cast integer literal to int type usable at runtime.";
  let description = [{
    The `kgen.int_literal.cast` operator casts an infinite precision integer
    literal to a fixed width integer type.
  }];

  let arguments = (ins KGEN_IntLiteralType:$input);
  let results = (outs Index:$output);

  let assemblyFormat = "`(` $input `)` attr-dict";
  let hasFolder = 1;
}
