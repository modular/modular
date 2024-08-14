//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the KGEN dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENOPS_TD
#define KGEN_KGENDIALECT_KGENOPS_TD

include "KGEN/KGENDialect/KGENAttrs.td"
include "KGEN/KGENDialect/KGENDialect.td"
include "KGEN/KGENDialect/KGENEnums.td"
include "KGEN/KGENDialect/KGENInterfaces.td"
include "KGEN/KGENDialect/KGENTypes.td"
include "KGEN/HLCFDialect/HLCFInterfaces.td"
include "KGEN/Interpreter/InterpreterInterface.td"
include "Support/MDialect/MAttrs.td"
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
    SingleBlock, DeclInterface, FuncInterface, FunctionLike, ExportInterface]> {
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
    ```

    The presence of this attribute means that the function was already compiled
    all the way to an object file once, so we should avoid doing it again.

    The function can contain opaque metadata to pass onto LLVM in its
    `llvmMetadata` dictionary attribute. Note that this is not the same as LLVM
    function attributes, which is contained in the `llvmAttrs` dictionary
    attribute.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<KGEN_SignatureType>:$signature,
    KGEN_DecoratorsAttr:$decorators,
    KGEN_InlineLevelAttr:$inlineLevel,
    KGEN_ExportKindAttr:$exportKind,
    DefaultValuedAttr<DictionaryAttr, "std::nullopt">:$LLVMMetadata,
    DefaultValuedAttr<StringArrayAttr, "std::nullopt">:$crossDeviceCaptures,
    OptionalAttr<TypeAttr>:$coroutineType
  );
  let regions = (region MaxSizedRegion<1>:$bodyRegion);

  let assemblyFormat = [{
    `` custom<FuncOp>($exportKind, $sym_name, $signature, $inlineLevel,
                     $decorators, attr-dict, $bodyRegion)
  }];

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
                   CArg<"ArrayRef<TypedAttr>", "{}">:$decorators,
                   CArg<"DictionaryAttr", "{}">:$llvmMetadata), [{
      if (!llvmMetadata)
        llvmMetadata = DictionaryAttr::get(name.getContext());
      build($_builder, $_state, name, signature, decorators, inlineLevel,
            exportKind, llvmMetadata, /*crossDeviceCaptures=*/{},
            /*coroutineType=*/{});
    }]>,
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GeneratorOp
//===----------------------------------------------------------------------===//

def KGEN_GeneratorOp : KGENOp<"generator", [
    HasParent<"mlir::ModuleOp">, FunctionOpInterface, Symbol, IsolatedFromAbove,
    SingleBlock, DeclInterface, FuncInterface, FunctionLike, ExportInterface,
    KGEN_GeneratorOpInterface]> {
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

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<KGEN_SignatureType>:$signature,
    TypeAttrOf<FunctionType>:$functionType,
    KGEN_ParamDeclArrayAttr:$inputParams,
    KGEN_ParamDeclArrayAttr:$resultParams,
    KGEN_DecoratorsAttr:$decorators,
    KGEN_InlineLevelAttr:$inlineLevel,
    KGEN_ExportKindAttr:$exportKind,
    DefaultValuedAttr<DictionaryAttr, "std::nullopt">:$LLVMMetadata
  );
  let regions = (region MaxSizedRegion<1>:$bodyRegion);

  let assemblyFormat = [{
    `` custom<GeneratorOp>($exportKind, $sym_name, $signature, $functionType,
                           $inputParams, $resultParams, $inlineLevel,
                           $decorators, attr-dict, $bodyRegion)
  }];

  let extraClassDeclaration = [{
    using OpTrait::SingleBlock<GeneratorOp>::getBody;
    Region *getCallableRegion() { return &getBodyRegion(); }
  }];

  let builders = [
    OpBuilder<(ins "StringAttr":$name, "SignatureType":$signature,
                   "FunctionType":$functionType,
                   "ArrayRef<ParamDeclAttr>":$inputParams,
                   "ArrayRef<ParamDeclAttr>":$resultParams,
                   CArg<"InlineLevel", "InlineLevel::Automatic">:$inlineLevel), [{
      build($_builder, $_state, name, signature, functionType, inputParams,
            resultParams, /*decorators=*/{}, inlineLevel,
            ExportKind::NotExported, DictionaryAttr::get(name.getContext()));
    }]>,
    OpBuilder<(ins "StringAttr":$name, "SignatureType":$signature), [{
      build($_builder, $_state, name, signature, signature.getValues(), {}, {});
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// ExternGeneratorOp
//===----------------------------------------------------------------------===//

def KGEN_ExternGeneratorOp : KGENOp<"extern.generator", [
    HasParent<"mlir::ModuleOp">, Symbol, DeclInterface, FuncInterface,
    ExportInterface, DeclareOpInterfaceMethods<FunctionOpInterface>]> {
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
                       KGEN_ExportKindAttr:$exportKind,
                       OptionalAttr<FlatSymbolRefAttr>:$preCompiledModuleRef);
  let regions = (region SizedRegion<0>:$body);

  let extraClassDeclaration = [{
    InlineLevel getInlineLevel() { return InlineLevel::Never; }
    Region *getCallableRegion() { return &getBody(); }
  }];

  let assemblyFormat = [{
    `` custom<SymbolExport>($exportKind) $sym_name ``
    custom<ExternGenerator>($signature, $functionType, $inputParams,
                            $resultParams) attr-dict-with-keyword
    ($body^)?
  }];
}

//===----------------------------------------------------------------------===//
// StructGeneratorOp
//===----------------------------------------------------------------------===//

def KGEN_StructGeneratorOp : KGENOp<"struct.generator", [
    HasParent<"mlir::ModuleOp">, Symbol, DeclInterface, SingleBlock,
    KGEN_GeneratorOpInterface]> {
  let summary = "Struct generator definition.";
  let description = [{
    The `kgen.struct.generator` operation defines a type-value generator. It is
    a parametric template for a type-value. It can have input parameters.

    Its body contains the definition for a potentially parametric struct type
    as a type-value.

    Example:

    ```mlir
    kgen.struct.generator @struct_SIMD<dt: dtype, size> {
      kgen.struct.info = #type_value : !kgen.type
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       KGEN_ParamDeclArrayAttr:$inputParams,
                       TypeAttr:$type);
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    ArrayRef<ParamDeclAttr> getResultParams() { return {}; }
  }];

  let assemblyFormat = [{
    $sym_name `` custom<StructGeneratorSpec>($inputParams, $type)
    attr-dict-with-keyword
    ($body^)?
  }];
}

//===----------------------------------------------------------------------===//
// StructInfoOp
//===----------------------------------------------------------------------===//

def KGEN_StructInfoOp : KGENOp<"struct.info", [Terminator]> {
  let summary = "Struct type info.";
  let description = [{
    The `kgen.struct.info` operation contains the definition of a parameterized
    type-value.
  }];

  let arguments = (ins TypedAttrInterface:$typeConstant);

  let assemblyFormat = "custom<ColonTypeParamValue>($typeConstant) attr-dict";
}

//===----------------------------------------------------------------------===//
// CallOpBase
//===----------------------------------------------------------------------===//

class KGEN_CallOpBase<string mnemonic, list<Trait> traits = []>
    : KGENOp<mnemonic, traits # [
        DeclareOpInterfaceMethods<KGEN_CallOpInterface>,
        DeclareOpInterfaceMethods<KGEN_GeneratorUserOpInterface>]>;

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def KGEN_CallOp : KGEN_CallOpBase<"call",
    [DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
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
    Variadic<AnyType>:$operands
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    custom<CallOp>($callee, $operands, type($operands), type($results))
    attr-dict
  }];

  let extraClassDeclaration = [{
    void concretizeCallee(mlir::IRRewriter &b, SymbolConstantAttr callee);

    /// Return the symbol reference to the callee.
    FlatSymbolRefAttr getCalleeSymbol() {
      return cast<FlatSymbolRefAttr>(getCallee().getSymbol());
    }

    /// Return the bound input parameter values.
    ArrayRef<TypedAttr> getParamValues() {
      return getCallee().getParamValues();
    }
  }];

  let hasVerifier = 1;
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
    Variadic<AnyType>:$operands
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    `` custom<ParametricCallee>($callee) `(` $operands `)` attr-dict
  }];

  let extraClassDeclaration = [{
    void concretizeCallee(mlir::IRRewriter &b, SymbolConstantAttr callee);
    bool isAllowedInFunc() { return false; }
  }];

  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ParamConstantOp
//===----------------------------------------------------------------------===//

def KGEN_ParamConstantOp : KGENOp<"param.constant", [Pure, ConstantLike,
    DeclareOpInterfaceMethods<ParamOpInterface, ["isImplicitlyParametric"]>,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]>,
    NoParamDecls {
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

def KGEN_ParamMaterializeOp : KGENOp<"param.materialize", [
    DeclareOpInterfaceMethods<ParamOpInterface, ["isImplicitlyParametric"]>,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]>,
    NoParamDecls {
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

    The operation carries an extra bit `isolated` to indicate that they are
    parametrically isolated from above.
  }];

  let arguments = (ins
    KGEN_ParamDeclAttr:$paramDecl,
    TypeAttrOf<KGEN_SignatureType>:$signature,
    TypeAttrOf<FunctionType>:$functionType,
    KGEN_ParamDeclArrayAttr:$inputParams,
    KGEN_ParamDeclArrayAttr:$resultParams,
    UnitAttr:$isolated,
    KGEN_InlineLevelAttr:$inlineLevel
  );
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = [{
    custom<RegionDeclaration>($paramDecl, $inputParams, $resultParams,
                              $functionType, $signature, $inlineLevel,
                              $bodyRegion)
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
    DeclareOpInterfaceMethods<KGEN_GeneratorUserOpInterface>]> {
  let summary = "Return the 'cost' of the function.";
  let description = [{
    The `kgen.cost_of` operation takes a parametric callee and returns
    characteristics of the function that can be used to develop heuristics for
    the cost of the function. This operation must be resolved at compile time.

    Currently, `kgen.cost_of` returns the number of loads, stores, additions,
    comparisions, divisions, multiplications, multiply-adds, and other
    operations, that is, operations that do not fall into any of the above
    categories. The cost is evaluated on the function at the output of
    elaboration, without running any post-elaboration passes.

    Example:

    ```mlir
    %loads, %stores, %additions, %comparisions, %divisions, %multiplications,
    %multiply_adds, %other = kgen.cost_of[(si8) -> si8: @foo]
    ```
  }];

  let arguments = (ins TypedAttrInterface:$callee);
  let results = (outs Index:$loads,
                      Index:$stores,
                      Index:$additions,
                      Index:$comparisons,
                      Index:$divisions,
                      Index:$multiplications,
                      Index:$multiplyAdds,
                      Index:$other);

  let assemblyFormat = [{
    `` custom<ParametricCallee>($callee) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def KGEN_ReturnOp : KGENOp<"return", [Terminator, ReturnLike,
    DeclareOpInterfaceMethods<InterpreterOpInterface>,
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
  let hasVerifier = 1;
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
// ParamForOp
//===----------------------------------------------------------------------===//

def KGEN_ParamForOp : KGENOp<"param.for", [DeclInterface,
    DeclareOpInterfaceMethods<ControlFlowNode>,
    DeclareOpInterfaceMethods<ParamOpInterface, ["isImplicitlyParametric",
      "collectParameterUsesBelow"]>]>, NoParamDecls {
  let summary = "compile-time for loop";
  let description = [{
    The `kgen.param.for` operation instantiates its body with values according
    to its iterator. It takes an initial value, an iterate method that takes an
    iterator instance and returns the next value plus the next iterator.
  }];

  let arguments = (ins
    TypedAttrInterface:$initial,
    AttrOfType<KGEN_SignatureType>:$iterate,
    KGEN_ParamDeclAttr:$paramDecl,
    Variadic<AnyType>:$operands,
    UnitAttr:$bodyIsolated,
    UnitAttr:$elseIsolated
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body, SizedRegion<1>:$elseRegion);

  let builders = [
    OpBuilder<(ins "TypedAttr":$initial, "TypedAttr":$iterate,
                   "ParamDeclAttr":$paramDecl), [{
      build($_builder, $_state, {}, initial, iterate, paramDecl, {}, {}, {});
    }]>
  ];

  let assemblyFormat = [{
    custom<ParamDecl>($paramDecl) `in` custom<ColonTypeParamValue>($initial)
    `iter` custom<ColonTypeParamValue>($iterate) `\n`
    custom<Loop>($operands, type($operands), type($results), $body)
    `else` custom<RegionWithArgs>($elseRegion) attr-dict
  }];

  let extraClassDeclaration = [{
    bool isIsolatedFromAbove(unsigned regionNum);
    void notifyKnownIsolatedFromAbove(unsigned regionNum);
    ArrayRef<ParamDeclAttr> getInputParams();
    ArrayRef<ParamDeclAttr> getResultParams() { return {}; }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ParamForBreakOp
//===----------------------------------------------------------------------===//

def KGEN_ParamForBreakOp : KGENOp<"param.for.break", [Terminator,
    DeclareOpInterfaceMethods<ControlFlowTerminator>]> {
  let summary = "param loop exit operation";
  let description = [{
    The `kgen.param.for.break` operation represents an exit from a
    `kgen.param.for`, branching to the end of all generated iterations.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, ValueRange()); }]>,
  ];

  let assemblyFormat = "attr-dict ($operands `:` type($operands)^)?";
}

//===----------------------------------------------------------------------===//
// ParamForContinueOp
//===----------------------------------------------------------------------===//

def KGEN_ParamForContinueOp : KGENOp<"param.for.continue", [Terminator,
    DeclareOpInterfaceMethods<ControlFlowTerminator>]> {
  let summary = "param loop continue operation";
  let description = [{
    The `kgen.param.for.continue` operation branches to the next generated
    iteration of the loop.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, ValueRange()); }]>,
  ];

  let assemblyFormat = "attr-dict ($operands `:` type($operands)^)?";
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

  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// ParamYieldOp
//===----------------------------------------------------------------------===//

def KGEN_ParamYieldOp : KGENOp<"param.yield", [Pure, Terminator,
    ParentOneOf<["ParamIfOp", "ParamForOp"]>,
    DeclareOpInterfaceMethods<ControlFlowTerminator>]> {
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

def KGEN_UndefOp : KGENOp<"undef", [ConstantLike, Pure]> {
  let summary = "Return an undefined value.";
  let description = [{
    The `kgen.undef` operation returns an undefined value of a given type.
    Accessing an undefined value is undefined behaviour (duh).

    Example:

    ```mlir
    %0 = kgen.undef : !lit.struct<@Int>
    %1 = kgen.undef : !kgen.struct<(array<2, index>)>
    ```
  }];

  let results = (outs AnyType:$value);
  let assemblyFormat = "attr-dict `:` type($value)";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// CallIndirectOp
//===----------------------------------------------------------------------===//

def KGEN_CallIndirectOp : KGENOp<"call_indirect", [
    TypesMatchWith<"operands match callee argument types", "callee",
    "arguments", "cast<SignatureType>($_self).getArguments()">,
    CalleeResultTypes, DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "Call a signature typed instance";
  let description = [{
    The `kgen.call_indirect` operation takes an SSA value of `!kgen.signature`
    type and invokes it with the provided operands as arguments.

    Example:

    ```mlir
    %0 = kgen.call_indirect %closure(%arg0, %arg1)
      : (index, index) capturing -> index
    ```
  }];

  let arguments = (ins
    KGEN_SignatureType:$callee,
    Variadic<AnyType>:$arguments,
    DefaultValuedAttr<KGEN_TailKindAttr, "{}">:$tailKind
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    custom<OptionalEnum>($tailKind, "symbolizeTailKind") $callee
    `(` $arguments `)` `:` custom<Signature>(type($callee)) attr-dict
  }];

  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// StageClosureOp
//===----------------------------------------------------------------------===//

// TODO(LLVM Issue #59877): Why isn't this in BuiltinAttributes.td?
def Builtin_LocationAttr : Builtin_Attr<"Location", "location", []> {}

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

  let results = (outs KGEN_SignatureType:$result);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = [{
    custom<StageClosureOp>(type($result), $bodyRegion) attr-dict
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

def KGEN_CreateClosureOp : KGEN_CallOpBase<"create_closure", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "Create a closure from variables and a function.";
  let description = [{
    The `kgen.create_closure` operation represents the instantiation of
    a closure. This operation models closure creation in terms of a function
    and local captured variables.

    ```mlir
    %idx0 = index.constant 0
    %0 = kgen.create_closure[(index) -> index: @h](%idx0)
    ```
  }];

  let arguments = (ins AttrOfType<KGEN_SignatureType>:$callee,
                       Variadic<AnyType>:$captures);
  let results = (outs KGEN_SignatureType:$result);
  let assemblyFormat = [{
    `` custom<ParametricCallee>($callee) `(` $captures `)`
    custom<ClosureCaptureTypes>(ref($callee), ref($captures), type($captures))
    attr-dict
  }];

  let builders = [
    OpBuilder<(ins "TypedAttr":$callee), [{
      build($_builder, $_state, callee, {});
    }]>,
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CreateRegStubOp
//===----------------------------------------------------------------------===//

def KGEN_CreateRegStubOp : KGENOp<"create_reg_stub", [Pure,
  DeclareOpInterfaceMethods<KGEN_GeneratorUserOpInterface>]> {
  let summary = "Create a stub to enforce by-memory arguments for `callee`.";
  let description = [{

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
      (!kgen.pointer<index> init_self, !kgen.pointer<index> owned_in_mem)
      -> !kgen.none: @regtype__moveinit__] :
      <(!kgen.pointer<struct<(index) memoryOnly>> init_self,
      !kgen.pointer<struct<(index) memoryOnly>> owned_in_mem) -> !kgen.none>
    ```

    The type is wrapped aound a memory struct to preverse the signature,
    and also to indicate to LLVM that pointers don't alias.

    After LowerCallConvention, only `callee`'s signature change:
    ```mlir
    kgen.create_reg_stub [(index) -> index: @regtype__moveinit__] :
      <(!kgen.pointer<struct<(index) memoryOnly>> init_self,
      !kgen.pointer<struct<(index) memoryOnly>> owned_in_mem) -> !kgen.none>
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
  }];

  let arguments = (ins AttrOfType<KGEN_SignatureType>:$callee);
  let results = (outs KGEN_SignatureType:$result);

  let assemblyFormat = [{
    custom<ParametricCallee>($callee) attr-dict `:` type($result)
  }];

  let builders = [
    OpBuilder<(ins "TypedAttr":$callee)>,
  ];

  let extraClassDeclaration = [{
    /// Get the signature type of the output stub.
    /// `calleeSign` is the original callee with by-memory operands.
    static SignatureType getStubSignatureType(SignatureType calleeSign);

    /// The stub signature wraps some arguments to preserve the semantics.
    /// Returns the original (unwrapped) argument type.
    Type getOriginalArgType(unsigned index);

    /// Returns the type of the arg `index` in callee's signature.
    /// It takes into account that after callee is lowered some arguments are
    /// now outputs.
    Type getCalleeArgType(unsigned index);
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SourceLocOp
//===----------------------------------------------------------------------===//

def KGEN_SourceLocOp : KGENOp<"source_loc", [Pure,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "Track source location and update conditionally during inlining";
  let description = [{
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
  }];

  let arguments = (ins IntProperty<"int64_t">:$inlineCount);
  let results = (outs Index:$line, Index:$col, KGEN_StringType:$fileName);
  let assemblyFormat = [{
    `[` custom<IntProperty>($inlineCount) `]` attr-dict
  }];
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
// PackageLinkOp
//===----------------------------------------------------------------------===//

def PackageLinkOp : KGENOp<"package.link", [Symbol]> {
  let summary = "link to a precompiled package's artifacts";
  let description = [{
    A `kgen.package.link` defines a link to the compiled artifacts of a package.
    It contains a reference to all precompiled packages, providing an anchor for
    functions and other operations defined within the package during the
    lowering pipeline. It may also contain the post-parse bodies which can be
    used to compile the package.

    Example:

    ```mlir
    kgen.package.link @foo
      dependencies([@stdlib])
      post_parse(dense_resource<...> : tensor<...xi8>)
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<Builtin_DenseResourceElementsAttr>:$postParseModule,
    OptionalAttr<KGEN_LinkDependencyArrayAttr>:$dependencies
  );

  let assemblyFormat = [{
    $sym_name
    (`dependencies` `(` $dependencies^ `)`)?
    (`post_parse` `(` $postParseModule^ `)`)?
    attr-dict-with-keyword
  }];
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
// IntLiteralConvertOp
//===----------------------------------------------------------------------===//

def KGEN_IntLiteralConvertOp : KGENOp<"int_literal.convert",
    [DeclareOpInterfaceMethods<InterpreterOpInterface>, Pure]> {
  let summary = "Convert integer literal to int type usable at runtime.";
  let description = [{
    The `kgen.int_literal.convert` operator casts an infinite precision integer
    literal to a fixed width integer type.
  }];

  let arguments = (ins KGEN_IntLiteralType:$input,
                       UnitAttr:$treatIndexAsUnsigned);
  let results = (outs AnyTypeOf<[AnyInteger, Index]>:$output);

  let assemblyFormat = [{
    $input attr-dict `:` `to` qualified(type($output))
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// IntLiteralBitWidthOp
//===----------------------------------------------------------------------===//

def KGEN_IntLiteralBitWidthOp : KGENOp<"int_literal.bit_width", [Pure]> {
  let summary = "Return the number of bits needed to represent int.";
  let description = [{
    The `kgen.int_literal.bit_width` returns the number of bits needed to
    represent the given int_literal.  It is returned as another int_literal.

    Example:

    ```mlir
    %0 = kgen.int_literal.bit_width %arg0
    ```
  }];

  let arguments = (ins KGEN_IntLiteralType:$input);
  let results = (outs KGEN_IntLiteralType:$output);

  let assemblyFormat = "$input attr-dict";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// IntLiteralToFloatLiteralOp
//===----------------------------------------------------------------------===//

def KGEN_IntLiteralToFloatLiteralOp : KGENOp<"int_literal.to_float_literal",
    [DeclareOpInterfaceMethods<InterpreterOpInterface>, Pure]> {
  let summary = "Convert int_literal to float_literal type.";
  let description = [{
    The `kgen.int_literal.to_float_literal` operator converts from int_literal
    to float_literal, of course.

    Example:

    ```mlir
    %0 = kgen.int_literal.to_float_literal %arg0
    ```
  }];

  let arguments = (ins KGEN_IntLiteralType:$input);
  let results = (outs KGEN_FloatLiteralType:$output);

  let assemblyFormat = "$input attr-dict";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// FloatLiteralIsa
//===----------------------------------------------------------------------===//

def KGEN_FloatLiteralIsa : KGENOp<"float_literal.isa", [Pure]> {
  let summary = "Check if the float_literal is of the given special value.";
  let description = [{
    The `kgen.float_literal.isa` operator checks if the input value is of the
    given special value, including 'normal'.

    Example:

    ```mlir
    %0 = kgen.float_literal.isa normal %arg0
    %1 = kgen.float_literal.isa nan %arg1
    %2 = kgen.float_literal.isa nan %arg2
    ```
  }];

  let arguments = (ins
    KGEN_FloatLiteralSpecialValuesAttr:$special,
    KGEN_FloatLiteralType:$input
  );
  let results = (outs I1:$result);

  let assemblyFormat = "$special $input attr-dict";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// FloatLiteralCmp
//===----------------------------------------------------------------------===//

def KGEN_FloatLiteralCmp : KGENOp<"float_literal.cmp", [Pure]> {
  let summary = "Comparison operator for float_literals.";
  let description = [{
    The `kgen.float_literal.cmp` operator does equality/inequality testing
    for `!kgen.float_literal` values and returns a bool.

    Example:

    ```mlir
    %0 = kgen.float_literal.cmp lt(%arg0, %arg1)
    ```
  }];

  let arguments = (ins
    KGEN_FloatLiteralCmpPredAttr:$pred,
    KGEN_FloatLiteralType:$lhs,
    KGEN_FloatLiteralType:$rhs
    );
  let results = (outs I1:$result);

  let assemblyFormat = "$pred `(` $lhs `,` $rhs `)` attr-dict";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// FloatLiteralBinop
//===----------------------------------------------------------------------===//

def KGEN_FloatLiteralBinop : KGENOp<"float_literal.binop", [Pure]> {
  let summary = "Binary operations for float_literals.";
  let description = [{
    The `kgen.float_literal.binop` operator does standard arithmetic binary
    operations on `!kgen.float_literal` values that return a
    `!kgen.float_literal`.

    Example:

    ```mlir
    %0 = kgen.float_literal.binop add(%arg0, %arg1)
    ```
  }];

  let arguments = (ins
    KGEN_FloatLiteralBinopKindAttr:$oper,
    KGEN_FloatLiteralType:$lhs,
    KGEN_FloatLiteralType:$rhs);
  let results = (outs KGEN_FloatLiteralType:$result);

  let assemblyFormat = "$oper `(` $lhs `,` $rhs `)` attr-dict";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// FloatLiteralConvertOp
//===----------------------------------------------------------------------===//

def KGEN_FloatLiteralConvertOp : KGENOp<"float_literal.convert",
    [DeclareOpInterfaceMethods<InterpreterOpInterface>, Pure]> {
  let summary = "Convert float literal to float type usable at runtime.";
  let description = [{
    The `kgen.float_literal.convert` operator casts an infinite precision float
    literal to a fixed width float type.

    Example:

    ```mlir
    %0 = kgen.float_literal.convert %arg0 : to f64
    %1 = kgen.float_literal.convert %arg0 : to f16
    ```
  }];

  let arguments = (ins KGEN_FloatLiteralType:$input);
  let results = (outs AnyFloat:$output);

  let assemblyFormat = [{
    $input attr-dict `:` `to` qualified(type($output))
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// FloatLiteralToIntLiteralOp
//===----------------------------------------------------------------------===//

def KGEN_FloatLiteralToIntLiteralOp : KGENOp<"float_literal.to_int_literal",
    [DeclareOpInterfaceMethods<InterpreterOpInterface>, Pure]> {
  let summary = "Convert float_literal to int_literal type.";
  let description = [{
    The `kgen.float_literal.to_int_literal` operator converts from float_literal
    to int_literal by truncating at the decimal.  IE it rounds toward zero.

    Example:

    ```mlir
    %0 = kgen.float_literal.to_int_literal %arg0
    ```
  }];

  let arguments = (ins KGEN_FloatLiteralType:$input);
  let results = (outs KGEN_IntLiteralType:$output);

  let assemblyFormat = [{$input attr-dict}];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// PackCreateOp
//===----------------------------------------------------------------------===//

def KGEN_PackCreateOp : KGENOp<"pack.create", [Pure]> {
  let summary = "create a pack";
  let description = [{
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
  }];

  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs KGEN_PackType:$result);
  let assemblyFormat = [{
    `(` $elements `)` attr-dict `:`
    custom<PackCreateType>(type($result), type($elements))
  }];

  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// PackExtractOp
//===----------------------------------------------------------------------===//

def KGEN_PackExtractOp : KGENOp<"pack.extract",
                     [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "pack get element";
  let description = [{
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
      // Get the second element, of type `!kgen.paramref<T>`.
      %1 = kgen.pack.extract %arg0[1] : <[i32, T]>

      // Get the element at an offset `I + 1`.
      %2 = kgen.pack.extract %arg0[add(I, 1)] : <[i32, T]>

      // Get the element at index 3.
      %3 = kgen.pack.extract %arg1[3] : <Ts>

      kgen.return
    }
    ```
  }];

  let arguments = (ins KGEN_PackType:$pack, AttrOfType<Index>:$index);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $pack `[` custom<IndexParamValue>($index) `]` attr-dict `:` type($pack)
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// PackGEPOp
//===----------------------------------------------------------------------===//

def KGEN_PackGEPOp : KGENOp<"pack.gep",
                     [Pure,
                      DeclareOpInterfaceMethods<InferTypeOpInterface>,
                      DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "pack get element pointer";
  let description = [{
    The `kgen.pack.gep` operation returns a pointer to the element of a pack at
    the provided index.  The index must be a parameter of index type.  This
    operation is resolved post-elaboration when the pack details become known.
  }];

  let arguments = (ins PointerOf<KGEN_PackType>:$pack, AttrOfType<Index>:$index);
  let results = (outs PointerType:$result);
  let assemblyFormat = [{
    $pack `[` custom<IndexParamValue>($index) `]` attr-dict `:` type($pack)
  }];
}

//===----------------------------------------------------------------------===//
// PackSizeOp
//===----------------------------------------------------------------------===//

def KGEN_PackSizeOp : KGENOp<"pack.size", [Pure]> {
  let summary = "given a pack, return its size";
  let description = [{
    The `kgen.pack.size` operation takes an operand with a `!kgen.pack` type and
    returns the number of elements in the pack.

    Example:

    ```mlir
    // Get the size of a pack.
    kgen.pack.size %0 : <Ts>
    // Get the size of a concrete pack with 2 elements.
    kgen.pack.size %1 : <[i32, f32]>
    ```
  }];

  let arguments = (ins KGEN_PackType:$operand);
  let results = (outs Index:$result);

  let assemblyFormat = "$operand attr-dict `:` type($operand)";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// PackLoadOp
//===----------------------------------------------------------------------===//

def KGEN_PackLoadOp : KGENOp<"pack.load",
                      [Pure,
                       DeclareOpInterfaceMethods<InferTypeOpInterface>,
                       DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "load all pack elements";
  let description = [{
    The `kgen.pack.load` operation takes a pack of !kgen.pointer values and
    loads each one into a pack without the pointer type.  This requires
    elements with trivially loadable types supported by pop.load.
  }];

  let arguments = (ins KGEN_PackType:$pack);
  let results = (outs KGEN_PackType:$result);
  let assemblyFormat = "$pack attr-dict `:` type($pack)";
}

//===----------------------------------------------------------------------===//
// StructCreateOp
//===----------------------------------------------------------------------===//

def KGEN_StructCreateOp : KGENOp<"struct.create", [Pure,
    TypesMatchWith<"operand types match struct element types",
                   "result", "elements",
                   "llvm::cast<StructType>($_self).getElementTypes()">
]> {
  let summary = "create an instance of a struct";
  let description = [{
    The `kgen.struct.create` operation creates a struct of the given type
    from values corresponding to its element types.

    Example:

    ```mlir
    %0 = kgen.struct.create(%f32, %f64)
      : !kgen.struct<(scalar<f32>, scalar<f64>)>

    %1 = kgen.struct.create(%arr, %ptr, %v)
      : !kgen.struct<(array<size, type>, pointer<type>, type)>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs KGEN_StructType:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` qualified(type($result))
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// StructExtractOp
//===----------------------------------------------------------------------===//

def KGEN_StructExtractOp : KGENOp<"struct.extract", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "struct extract value";
  let description = [{
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
    ```
  }];

  let arguments = (ins KGEN_StructType:$container, UnsignedIndexAttr:$index);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $container `[` $index `]` attr-dict `:` qualified(type($container))
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// StructReplaceOp
//===----------------------------------------------------------------------===//

def KGEN_StructReplaceOp : KGENOp<"struct.replace", [Pure,
    AllTypesMatch<["container", "result"]>]> {
  let summary = "struct insert value";
  let description = [{
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
  }];

  let arguments = (ins AnyType:$value, KGEN_StructType:$container,
                       UnsignedIndexAttr:$index);
  let results = (outs KGEN_StructType:$result);
  let assemblyFormat = [{
    $value `,` $container `[` $index `]` attr-dict `:`
    qualified(type($container)) ``
    custom<StructValueType>(type($value), ref(type($container)), ref($index))
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// StructGEPOp
//===----------------------------------------------------------------------===//

def KGEN_StructGEPOp : KGENOp<"struct.gep", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "struct get element pointer";
  let description = [{
    The `kgen.struct.gep` operation takes a pointer to a `!kgen.struct` and a
    constant index and returns a pointer to the struct element at that index.

    Example:

    ```mlir
    %struct = pop.stack_allocation 1 : !kgen.struct<(i32, i64)>
    %i64Ptr = kgen.struct.gep %struct[1] : <struct<(i32, i64)>>
    ```

    To get a pointer to a struct from a value-semantic struct, the struct value
    must be stored into a block of allocated memory first.

    ```mlir
    ^bb0(%struct: !kgen.struct<(i32, i64)>):
      %mem = pop.stack_allocation 1 : !kgen.struct<(i32, i64)>
      pop.store %struct, %mem : !kgen.pointer<struct<(i32, i64)>>
    ```
  }];

  let arguments = (ins PointerOf<KGEN_StructType>:$container,
                       UnsignedIndexAttr:$index);
  let results = (outs PointerType:$result);
  let assemblyFormat = [{
    $container `[` $index `]` attr-dict `:` type($container)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// VariantCreateOp
//===----------------------------------------------------------------------===//

def KGEN_VariantCreateOp : KGENOp<"variant.create", [Pure]> {
  let summary = "Create a parametric variant.";
  let description = [{
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
  }];

  let arguments = (ins AnyType:$operand, UnsignedIndexAttr:$index);
  let results = (outs KGEN_VariantType:$result);
  let assemblyFormat = [{
    $operand `,` $index attr-dict `:` type($result)
    custom<VariantElementType>(type($operand), ref(type($result)), ref($index))
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// VariantIsOp
//===----------------------------------------------------------------------===//

def KGEN_VariantIsOp : KGENOp<"variant.is", [Pure]> {
  let summary = "Check the field contained by a variant.";
  let description = [{
    The `kgen.variant.is` operation checks whether the given variant contains
    a particular type. Returns an `i1` that indicates whether the variant is the
    particular type.

    Example:

    ```mlir
    %optional = ...
    %isNone = kgen.variant.is %optional, 0 : !kgen.variant<none, T>
    ```
  }];

  let arguments = (ins KGEN_VariantType:$variant, UnsignedIndexAttr:$index);
  let results = (outs I1:$result);
  let assemblyFormat = "$variant `,` $index attr-dict `:` type($variant)";

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// VariantGetOp
//===----------------------------------------------------------------------===//

def KGEN_VariantGetOp : KGENOp<"variant.get", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Take a register value out of a variant.";
  let description = [{
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
  }];

  let arguments = (ins KGEN_VariantType:$variant, UnsignedIndexAttr:$index);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$variant `,` $index attr-dict `:` type($variant)";

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// CaptureListCreateOp
//===----------------------------------------------------------------------===//

def KGEN_CaptureListCreateOp : KGENOp<"capture_list.create",
    [DeclareOpInterfaceMethods<KGEN_GeneratorUserOpInterface>]> {
  let summary = "Create a capture list for a capturing closure.";
  let description = [{
    The `kgen.capture_list.create` operation allocates heap memory and
    initializes the capture list for a closure.

    Example:

    ```mlir
    %cl = kgen.capture_list.create :() capturing -> index @cap_closure
    ```
  }];

  let arguments = (ins TypedAttrInterface:$callee, Variadic<AnyType>:$args);
  let results = (outs PointerOf<KGEN_NoneType>:$result);

  let assemblyFormat = [{
    attr-dict custom<ColonTypeParamValue>($callee)
    (`(` $args^ `:` type($args) `)`)?
  }];
}

//===----------------------------------------------------------------------===//
// CaptureListCopyOp
//===----------------------------------------------------------------------===//

def KGEN_CaptureListCopyOp : KGENOp<"capture_list.copy",
    [DeclareOpInterfaceMethods<KGEN_GeneratorUserOpInterface>]> {
  let summary = "Copy a capture list for a capturing closure.";
  let description = [{
    The `kgen.capture_list.copy` operation allocates heap memory and
    copies the capture list for a closure.

    Example:

    ```mlir
    %cl = kgen.capture_list.copy %orig :() capturing -> index @cap_closure
    ```
  }];

  let arguments = (ins PointerOf<KGEN_NoneType>:$orig,
                       TypedAttrInterface:$callee);
  let results = (outs PointerOf<KGEN_NoneType>:$result);

  let assemblyFormat = "$orig attr-dict custom<ColonTypeParamValue>($callee)";
}

//===----------------------------------------------------------------------===//
// CaptureListExpandOp
//===----------------------------------------------------------------------===//

def KGEN_CaptureListExpandOp : KGENOp<"capture_list.expand"> {
  let summary = "Expand the capture list from a capturing closure.";
  let description = [{
    The `kgen.capture_list.expand` operation unpacks the capture list of the
    enclosing function.

    Example:

    ```mlir
    kgen.capture_list.expand %cl
    ```
  }];

  let arguments = (ins PointerOf<KGEN_NoneType>:$captureList);
  let assemblyFormat = "$captureList attr-dict";
}

//===----------------------------------------------------------------------===//
// CustomOpImplsOp
//===----------------------------------------------------------------------===//

def KGEN_CustomOpImplsOp : KGENOp<"custom.op_impls",
    [Symbol, ExportInterface, DeclareOpInterfaceMethods<ExportInterface>]> {
  let summary = "Link `custom` ops names to their implementation";
  let description = [{
    The `kgen.custom.op_impl` operation defines the implementation of
    `custom` ops.

    The operation always have the symbol name @CustomOpImplSymbol, and thus
    this operation should only appear once in the IR.

    Example:
    ```mlir
    kgen.custom.op_impls @CustomOpImplSymbol [<"custom.a", <@impl_a()">>]
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       KGEN_CustomOpImplArrayAttr:$impls);
  let assemblyFormat = "$sym_name $impls attr-dict";

  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "ArrayRef<CustomOpImplAttr>":$impls), [{
      auto symbolAttr = StringAttr::get($_builder.getContext(), kSymbolName);
      build($_builder, $_state, symbolAttr, impls);
    }]>,
  ];

  let extraClassDeclaration = [{
    /// The expected operation symbol
    static constexpr StringLiteral kSymbolName = "__CustomOpImplSymbol";

    /// Lookup the unique CustomOpImplsOp from the given operation.
    /// It will walk to the toplevel ModuleOp, and then use the operation
    /// symbol to find the CustomOpImplsOp.
    static CustomOpImplsOp lookupOp(Operation *op);
  }];
}

#endif // KGEN_KGENDIALECT_KGENOPS_TD
