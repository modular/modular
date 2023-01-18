//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the LIT dialect.
//
//===----------------------------------------------------------------------===//

include "KGEN/LITDialect/LITDialect.td"
include "KGEN/LITDialect/LITTypes.td"
include "KGEN/KGENDialect/KGENAttrs.td"
include "KGEN/KGENDialect/KGENEnums.td"
include "KGEN/KGENDialect/KGENInterfaces.td"
include "KGEN/POPDialect/POPTypes.td"
include "Support/HLCFDialect/HLCFInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for the operation in this dialect.
class LITOp<string mnemonic, list<Trait> traits = []> :
    Op<LITDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// LITFuncOp
//===----------------------------------------------------------------------===//

def LIT_FuncOp : LITOp<"func", [
    FunctionOpInterface, Symbol, SymbolTable, NoTerminator, SingleBlock,
    DeclareOpInterfaceMethods<CallableOpInterface>,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
    DeclInterface, FuncInterface]> {
  let summary = "High level generator definition";

  let description = [{
    The `lit.func` operation models a def/fn declaration at the Lit source
    level.
    @interface and @staticmethod decorators in Lit correspond to the
    attributes `isInterface` and `isStatic` respectively.
    When `isInterface` is set, `lit.func` has an empty body block.
    The optional `implements` attribute refers to the interface name this
    `lit.func` implement and cannot be set alongside `isInterface`.
  }];

  /// If the `implements` attribute is present then this generator is an
  /// implementation of the specified generator.interface.
  let arguments = (ins SymbolNameAttr:$sym_name,
                       KGEN_StringArrayAtrr:$valueParamNames,
                       TypeAttrOf<KGEN_SignatureType>:$signature,
                       ConstraintArrayAttr:$constraints,
                       UnitAttr:$isStatic,
                       UnitAttr:$isInterface,
                       UnitAttr:$isDef,  // Present if 'def' absent if 'fn'.
                       UnitAttr:$noDebugInline,
                       OptionalAttr<DefaultArgumentArrayAttr>:$defaults,
                       OptionalAttr<SymbolRefAttr>:$implements,
                       OptionalAttr<KGEN_SymbolConstantAttr>:$evaluator);
  let regions = (region AnyRegion:$bodyRegion);

  let extraClassDeclaration = [{
    Block *getBody() { return &getFunctionBody().back(); }

    /// Returns the result type, which is guaranteed to be present.
    Type getResultType() { return getFunctionType().getResults()[0]; }

    /// Return a SymbolConstantAttr for this function, optionally bound to a set
    /// of parameter bindings.
    SymbolConstantAttr getBoundReference(ParamBindArrayAttr bindings = {});
  }];

  let builders = [
    // Create a func with a region and a body block.
    OpBuilder<(ins)>,
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// FileModuleOp
//===----------------------------------------------------------------------===//

def FileModuleOp : LITOp<"file_module", [
    IsolatedFromAbove, NoRegionArguments, SymbolTable, Symbol
  ] # GraphRegionNoTerminator.traits> {
  let summary = "source file module";
  let description = [{
    A `lit.file_module` models a file module at the lit source level. It
    contains any functions, modules, variables, etc. defined within the module.

    Example:

    ```mlir
    lit.file_module @bar {
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let builders = [
    OpBuilder<(ins "StringAttr":$name)>,
  ];
  let assemblyFormat = "$sym_name attr-dict-with-keyword $bodyRegion";
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// TryOp
//===----------------------------------------------------------------------===//

def LIT_TryOp : LITOp<"try", [RecursivelySpeculatable, RecursiveMemoryEffects,
    DeclareOpInterfaceMethods<ControlFlowNode>]> {
  let summary = "A try-except-else block.";
  let description = [{
    The `lit.try` operation represents a try-except-else statement. When inside
    the 'try' region, any implicitly propagated errors interrupt execution and
    branch to the 'except' region. If the 'try' region completes execution
    without any exceptions, control-flow branches to the 'else' region.

    Example:

    ```mlir
    lit.try {
      %0 = kgen.call @maybeRaises() : () throws -> !kgen.declref<@Int>
      lit.try.yield
    } except (%err: !kgen.declref<@Error>) {
      lit.try.yield
    } else {
      lit.try.yield
    }
    ```
  }];

  let regions = (region MinSizedRegion<1>:$tryRegion,
                        MinSizedRegion<1>:$exceptRegion,
                        MinSizedRegion<1>:$elseRegion);
  let assemblyFormat = [{
    $tryRegion `except` custom<ExceptRegion>($exceptRegion) `else` $elseRegion
    attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TryYieldOp
//===----------------------------------------------------------------------===//

def LIT_TryYieldOp : LITOp<"try.yield", [Pure, Terminator,
    HasParent<"TryOp">, DeclareOpInterfaceMethods<ControlFlowTerminator>]> {
  let summary = "Try block terminator.";
  let description = [{
    The `lit.try.yield` operation is used as a terminator to `lit.try` regions.
  }];
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// TryRaiseOp
//===----------------------------------------------------------------------===//

def LIT_TryRaiseOp : LITOp<"try.raise", [Terminator,
    DeclareOpInterfaceMethods<ControlFlowTerminator>]> {
  let summary = "Try block raise an error.";
  let description = [{
    The `lit.try.raise` operation is used within a `lit.try` try region to
    propagate an error to the except region.
  }];

  let arguments = (ins AnyTypeOf<[KGEN_DeclRefType, POP_StructType]>:$error);
  let assemblyFormat = "$error attr-dict `:` type($error)";
}

//===----------------------------------------------------------------------===//
// LocalAliasDeclOp
//===----------------------------------------------------------------------===//

// TODO(LLVM Issue #59877): Why isn't this in BuiltinAttributes.td?
def Builtin_LocationAttr : Builtin_Attr<"Location", []> {
}

def LIT_AliasForwardDeclOp : LITOp<"alias.fwd.decl"> {
  let summary = "`alias x: T` declaration for a callee return parameter";

  let description = [{
    This op represents an alias decl after its signature has been resolved and
    it was found to have no initializer.  It is therefore a forward declaration
    of a return parameter from a function call.

    `resultParamLoc` is filled in with the location of the result parameter that
    ultimately actually defines this.
  }];

  let arguments = (ins StrAttr:$name, TypeAttr:$type,
                   OptionalAttr<Builtin_LocationAttr>:$resultParamLoc);
  let results = (outs);
  let assemblyFormat = "$name `:` $type attr-dict";
}

//===----------------------------------------------------------------------===//
// LetDeclOp / VarDeclOp
//===----------------------------------------------------------------------===//

def LIT_LetDeclOp : LITOp<"let.decl",
         [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
          Pure, AllTypesMatch<["value", "result"]>,]> {
  let summary = "Local constant declaration";

  let arguments = (ins StrAttr:$name, AnyType:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$name `=` $value `:` type($result) attr-dict";

  let builders = [
    // Create a 'let' without a value, this is malformed until specified.
    OpBuilder<(ins "Type":$type, "StringAttr":$name)>,
  ];
}

def LIT_VarDeclOp : LITOp<"var.decl",
         [DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Local variable declaration";

  let arguments = (ins StrAttr:$name);
  let results = (outs POP_PointerType:$result);
  let assemblyFormat = "$name `:` type($result) attr-dict";
}

//===----------------------------------------------------------------------===//
// AsyncCallOp
//===----------------------------------------------------------------------===//

def LIT_AsyncCallOp : LITOp<"async_call", [CalleeArgumentTypes,
    TypesMatchWith<"coroutine of callee result types", "callee", "coroutine",
                    "getCoroutineOfResultTypes($_self)">]> {
  let summary = "Call to an async function.";
  let description = [{
    The `lit.async_call` operation calls an `async` parametric callable. It
    returns a `!pop.coroutine` of the callable's result types.

    Example:

    ```mlir
    // `%coro` has type `!pop.coroutine<index>`.
    %coro = lit.async_call<() -> bar>[:<() -> index>() -> index: @foo]()
    ```
  }];

  let arguments = (ins
    AttrOfType<KGEN_SignatureType>:$callee,
    KGEN_ParamDeclArrayAttr:$paramDecls,
    Variadic<AnyType>:$operands);
  let results = (outs POP_CoroutineType:$coroutine);

  let assemblyFormat = [{
    `` custom<ParametricCallee>($callee, $paramDecls) `(` $operands `)`
    attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AsyncAwaitOp
//===----------------------------------------------------------------------===//

def LIT_AsyncAwaitOp : LITOp<"async_await", [TypesMatchWith<
    "results match coroutine result types", "coroutine", "results",
    "$_self.cast<POP::CoroutineType>().getSignature().getValueResults()">]> {
  let summary = "Await a coroutine.";
  let description = [{
    The `lit.async_await` operation awaits a coroutine.

    Example:

    ```mlir
    %res0, %res1 = lit.async_await %hdl : <() -> (i32, index)>
    %errOr = lit.async_await %hdl : <() throws -> i64>
    ```
  }];

  let arguments = (ins POP_CoroutineType:$coroutine);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$coroutine attr-dict `:` type($coroutine)";
}

//===----------------------------------------------------------------------===//
// EndFuncOp
//===----------------------------------------------------------------------===//

def LIT_EndFuncOp : LITOp<"end_func", [Terminator, HasParent<"LIT::FuncOp">]> {
  let summary = "Function terminator.";
  let description = [{
    The `lit.end_func` operation denotes the end of a `lit.func` body. This
    terminator corresponds to the end of the function body suite in Lit. It is
    generated by the parser as the default terminator of a `lit.func`.

    Example:

    ```mlir
    lit.end_func
    ```
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def LIT_ReturnOp : LITOp<"return"> {
  let summary = "Lexical return statement.";
  let description = [{
    The `lit.return` operation is not a terminator, but it corresponds to a
    return statement in Lit. The operands are the desired return values at the
    return statement. This operation exists to model the fact that Lit functions
    can have multiple returns and that those returns do not have to be the last
    statement in a suite.

    Example:

    ```mlir
    lit.func @foobar(%foo: i32, %bar: i32) -> i32 {
      lit.return %foo : i32
      lit.return %bar : i32
      lit.end_func
    }
    ```

    The operation also specifies the return parameters of the function.

    Example:

    ```mlir
    lit.func @foobar<() -> index>() {
      lit.return<5>
      lit.end_func
    }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands,
                       KGEN_ParameterExprArrayAttr:$parameters);
  let assemblyFormat = [{
    `` custom<ParameterValues>($parameters) $operands attr-dict
    (`:` type($operands)^)?
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// RaiseOp
//===----------------------------------------------------------------------===//

def LIT_RaiseOp : LITOp<"raise"> {
  let summary = "Lexical raise statement.";
  let description = [{
    The `lit.raise` operation is not a terminator. It corresponds to a raise
    statement in Lit. It takes a single opearand: the error to raise. This
    operation is only allowed inside the 'try' region of a `lit.try` operation
    or within a `lit.func` marked as `throws`.

    Example:

    ```mlir
    lit.raise %err : !kgen.declref<@Error>
    ```
  }];

  let arguments = (ins KGEN_DeclRefType:$error);
  let assemblyFormat = "$error attr-dict `:` type($error)";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// BreakOp
//===----------------------------------------------------------------------===//

def LIT_BreakOp : LITOp<"break"> {
  let summary = "Lexical break statement.";
  let description = [{
    The `lit.break` operation is not a terminator, but it corresponds to a break
    statement in Lit. The operation indicates that control-flow should branch
    to after the enclosing loop.

    Example:

    ```mlir
    lit.break
    ```
  }];

  let assemblyFormat = "attr-dict";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ContinueOp
//===----------------------------------------------------------------------===//

def LIT_ContinueOp : LITOp<"continue"> {
  let summary = "Lexical continue statement.";
  let description = [{
    The `lit.continue` operation is not a terminator, but it corresponds to a
    continue statement in Lit. The operation indicates that control-flow should
    branch to after the enclosing loop.

    Example:

    ```mlir
    lit.continue
    ```
  }];

  let assemblyFormat = "attr-dict";

  let hasVerifier = 1;
}
