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

// FIXME: Interfaces don't have bodies.  When they go away, we should remove
// NoTerminator and simplify FuncOp::verifyRegions to remove its check.
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
                       KGEN_StringArrayAttr:$valueParamNames,
                       TypeAttrOf<KGEN_SignatureType>:$signature,
                       ConstraintArrayAttr:$constraints,
                       UnitAttr:$isStatic,
                       UnitAttr:$isInterface,
                       UnitAttr:$isDef,  // Present if 'def' absent if 'fn'.
                       OptionalAttr<SymbolRefAttr>:$implements,
                       OptionalAttr<KGEN_SymbolConstantAttr>:$evaluator,
                       KGEN_AlwaysInlineLevelAttr:$alwaysInlineLevel);
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
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// FileModuleOp
//===----------------------------------------------------------------------===//

def FileModuleOp : LITOp<"file_module", [
    IsolatedFromAbove, NoRegionArguments, SymbolTable, Symbol,
    DeclareOpInterfaceMethods<DeclInterface>,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
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
// StructDeclOp
//===----------------------------------------------------------------------===//

def LIT_StructDeclOp : LITOp<"struct.decl", [Symbol, SymbolTable,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>, DeclInterface,
    SingleBlock, NoTerminator]> {
  let summary = "Struct declaration.";
  let description = [{
    The `lit.struct.decl` operation is a top-level operation used to declare
    parametric LIT struct types. Struct declarations are named with a symbol
    and have a parameter declaration list. The body contains
    `lit.struct.field` operations that declare the contents of the struct.

    Example:

    ```mlir
    lit.struct.decl @Buffer<size, dtype: dtype> {
      lit.struct.field size : index
      lit.struct.field ptr : !pop.pointer<scalar<dtype>>
      lit.struct.field dtype : !kgen.dtype
    }
    ```

    The parameters of the declaration can be constraints with optional
    constraints.

    Example:

    ```mlir
    lit.struct.decl @SmallString<size>
        constraints <[le(size, 8), "string must be small"]> {
      lit.struct.field data : !pop.array<size, scalar<si8>>
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

def LIT_StructFieldOp : LITOp<"struct.field", [HasParent<"StructDeclOp">]> {
  let summary = "Struct field declaration.";
  let description = [{
    The `lit.struct.field` operations declares a named struct field and type.

    Example:

    ```mlir
    // Declare a field named "foo".
    lit.struct.field foo : !pop.scalar<f32>
    ```
  }];

  let arguments = (ins StrAttr:$name, TypeAttr:$type);
  let assemblyFormat = "custom<KeywordAsString>($name) attr-dict `:` $type";
}

//===----------------------------------------------------------------------===//
// StructCreateOp
//===----------------------------------------------------------------------===//

def LIT_StructCreateOp : LITOp<"struct.create", [Pure,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Create a parametric struct.";
  let description = [{
    The `lit.struct.create` operation creates a struct of the referred type
    with the provided values as fields. The operands must match the fields in
    the order they are declared.

    Example:

    ```mlir
    lit.struct.decl @MyStruct<T: dtype> {
      lit.struct.field a : !pop.scalar<T>
      lit.struct.field b : !pop.scalar<f32>
      lit.struct.field c : !pop.simd<2, T>
    }

    %0 = lit.struct.create(a=%a, b=%b, c=%c) :
      (!pop.scalar<type>, !pop.scalar<f32>, !pop.simd<2, type>)
      -> !kgen.declref<@MyStruct<T: dtype = type>>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands,
                       KGEN_StringArrayAttr:$fields);
  let results = (outs KGEN_DeclRefType:$result);
  let hasFolder = 1;
  let assemblyFormat = [{
    `` custom<OperandsAndFields>($operands,$fields) attr-dict `:`
    functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// StructInsertOp
//===----------------------------------------------------------------------===//

def LIT_StructInsertOp : LITOp<"struct.insert", [Pure,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    AllTypesMatch<["container", "result"]>]> {
  let summary = "Insert a value into a struct.";
  let description = [{
    The `lit.struct.insert` operation inserts a value into a parametric struct.
    The original struct is immutable, so a new copy with the inserted value is
    returned.

    Example:

    ```mlir
    lit.struct.decl @MyStruct<T: dtype> {
      lit.struct.field a : !pop.scalar<T>
    }

    %0 = lit.struct.insert %fieldVal, %struct[a]
      : !pop.scalar<f64> into !kgen.declref<@MyStruct<T: dtype = f64>>
    ```
  }];

  let arguments = (ins StrAttr:$field, AnyType:$value,
                       KGEN_DeclRefType:$container);
  let results = (outs KGEN_DeclRefType:$result);
  let assemblyFormat = [{
    $value `,` $container `[` custom<KeywordAsString>($field) `]` attr-dict `:`
    type($value) `into` qualified(type($container))
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// StructExtractOp
//===----------------------------------------------------------------------===//

def LIT_StructExtractOp : LITOp<"struct.extract", [Pure,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Extract a value from a struct.";
  let description = [{
    The `lit.struct.extract` operation extracts the value of a field by name
    from a parametric struct.

    Example:

    ```mlir
    lit.struct.decl @MyStruct<T: dtype> {
      lit.struct.field a : !pop.scalar<T>
    }

    %0 = lit.struct.extract %struct[a]
      : !pop.scalar<f64> from !kgen.declref<@MyStruct<T: dtype = f64>>
    ```
  }];

  let arguments = (ins KGEN_DeclRefType:$container, StrAttr:$field);
  let results = (outs AnyType:$value);
  let assemblyFormat = [{
    $container `[` custom<KeywordAsString>($field) `]` attr-dict `:`
    type($value) `from` qualified(type($container))
  }];
  let hasFolder = 1;

  let builders = [
    // Infer the result type based on the result of lookup.
    OpBuilder<(ins "Value":$structBase, "StructFieldOp":$field)>
  ];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// StructGEPOp
//===----------------------------------------------------------------------===//

def LIT_StructGEPOp : LITOp<"struct.gep", [Pure,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Get a pointer to a struct field.";
  let description = [{
    The `lit.struct.gep` operation computes a pointer to a struct field by
    name from a parametric struct.

    Example:

    ```mlir
    lit.struct.decl @MyStruct<T: dtype> {
      lit.struct.field a : !pop.scalar<T>
    }

    %0 = lit.struct.gep %struct[a]
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

//===----------------------------------------------------------------------===//
// UnresolvedImportOp
//===----------------------------------------------------------------------===//

def LIT_UnresolvedImportOp : LITOp<"unresolved_import"> {
  let summary = "Lexical unresolved import.";
  let description = [{
    The `lit.unresolved_import` operation represents a module import that is yet
    to be resolved. If a decl name is provided, this operation represents an
    import of that decl, not the module itself.

    Example:

    ```mlir
    // Import a module as the given name.
    lit.unresolved_import @module as @newModule
    // Import a function within the module as the given name.
    lit.unresolved_import @module -> @func as @newFunc
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$moduleName,
    SymbolNameAttr:$importName,
    OptionalAttr<SymbolNameAttr>:$declName
  );
  let assemblyFormat = [{
    $moduleName (`->` $declName^)? `as` $importName attr-dict
  }];
}
