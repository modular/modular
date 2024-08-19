//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the LIT dialect.
//
//===----------------------------------------------------------------------===//

include "KGEN/LITDialect/LITAttrs.td"
include "KGEN/LITDialect/LITEnums.td"
include "KGEN/LITDialect/LITDialect.td"
include "KGEN/LITDialect/LITInterfaces.td"
include "KGEN/LITDialect/LITTypes.td"
include "KGEN/CODialect/COTypes.td"
include "KGEN/KGENDialect/KGENAttrs.td"
include "KGEN/KGENDialect/KGENEnums.td"
include "KGEN/KGENDialect/KGENInterfaces.td"
include "KGEN/POPDialect/POPTypes.td"
include "KGEN/HLCFDialect/HLCFAttrs.td"
include "KGEN/HLCFDialect/HLCFInterfaces.td"
include "Support/DebugInfoDialect/IR/DebugInfoAttrs.td"
include "Support/DebugInfoDialect/IR/DebugInfoInterfaces.td"
include "KGEN/Interpreter/InterpreterInterface.td"
include "Support/MDialect/MAttrs.td"
include "Support/Compiler/Properties.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for the operation in this dialect.
class LITOp<string mnemonic, list<Trait> traits = []> :
    Op<LITDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def LIT_FuncOp : LITOp<"func", [
    FunctionLike, FunctionOpInterface, SymbolTable, SingleBlock,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
    DeclareOpInterfaceMethods<ParamOpInterface, ["collectParameterUses"]>,
    ASTDeclInterface, DeclInterface, FuncInterface, ExportInterface]> {
  let summary = "High level generator definition";
  let description = [{
    The `lit.func` operation models a def/fn declaration at the Lit source
    level.

    @staticmethod decorators correspond to the attribute `isStatic`.
  }];

  let arguments = (ins
    OptionalAttr<SymbolNameAttr>:$sym_name,
    OptionalAttr<KGEN_ParamDeclAttr>:$paramDecl,
    TypeAttrOf<LIT_SignatureType>:$signature,
    TypeAttrOf<FunctionType>:$functionType,
    KGEN_ParamDeclArrayAttr:$params,
    KGEN_DecoratorsAttr:$decorators,
    UnitAttr:$isStatic,
    UnitAttr:$isDef,  // Present if 'def' absent if 'fn'.
    UnitAttr:$isInherited,
    UnitAttr:$isSynthetic,
    KGEN_ExportKindAttr:$exportKind,
    KGEN_InlineLevelAttr:$inlineLevel,
    DefaultValuedOptionalAttr<I8Attr, "0">:$specialFnKind,
    // Used for binary packages, a reference to the symbol containing the
    // precompiled module that contains this function.
    OptionalAttr<FlatSymbolRefAttr>:$preCompiledModuleRef,
    // The linkage name of this function within the compiled artifact. If null,
    // the flattened full symbol reference of the function will be used.
    OptionalAttr<StrAttr>:$linkageName,
    // The name of this function in the pre-elaborated IR stage.
    OptionalAttr<StrAttr>:$preElaborationName,
    OptionalAttr<StrAttr>:$sourceName,
    OptionalAttr<StrAttr>:$namedResult,
    OptionalAttr<LIT_DocStringAttr>:$docString,
    OptionalAttr<StrAttr>:$deprecationWarning,
    DefaultValuedAttr<DictionaryAttr, "std::nullopt">:$LLVMMetadata
  );
  let regions = (region SizedRegion<1>:$bodyRegion);

  let extraClassDeclaration = [{
    ArrayRef<ParamDeclAttr> getInputParams() { return getParams(); }
    ArrayRef<ParamDeclAttr> getResultParams() { return {}; }
    Block *getBody() { return &getFunctionBody().back(); }
    Region *getCallableRegion() { return &getBodyRegion(); }
    bool isOptionalSymbol() { return getParamDecl().has_value(); }

    /// Returns the result type of the MLIR function (not the user defined
    /// result type).  This is guaranteed to be present.
    Type getMLIRResultType() { return getResultTypes().front(); }

    /// Returns the user-defined result type, looking through implicit memory
    /// results and stripping off the variant from error throwing results if
    /// needed.
    Type getUserResultType();

    /// Return a reference for this function, optionally bound to a set of
    /// parameter bindings.
    TypedAttr getBoundReference(ParameterExprArrayAttr bindings = {});
    SymbolConstantAttr getBoundSymbolRef(ParameterExprArrayAttr bindings = {});

    /// Return the SpecialFunctionKind ID that indicates if this is a special
    /// function like __init__ or __radd__.
    SpecialFunctionKind getSpecialFunctionKind();
    const SpecialFunctionInfo &getSpecialFunctionInfo();

    /// Check if a function is coming from an external package. This is
    /// signified by having the pre-compiled metadata on the func itself.
    bool isExternal() { return getPreCompiledModuleRef().has_value(); }

    /// Return the decl name of the function.
    StringAttr getDeclName() { return getSourceNameAttr(); }

    /// Return the full signature of this declaration, including parameters from
    /// enclosing struct declarations.
    LITSignatureType getFullSignature();

    /// Return a collection of the ParamDeclAttr's that are declared on this
    /// function and any enclosing unit, ordered as will be required by a call.
    SmallVector<ParamDeclAttr> collectAllParams(bool includeImplLifetimes);

    /// Return whether the function is compiler generated or not.
    bool isSynthetic();

    /// Return true if this was defined as a 'def', not a 'fn'.
    bool isDef() { return getIsDef(); }
  }];

  let builders = [
    // Create a func with a region and a body block.
    OpBuilder<(ins)>,
    OpBuilder<(ins "StringAttr":$declName, "StringRef":$sourceName,
                   "FunctionType":$funcType,
                   "ArrayRef<ParamDeclAttr>":$paramDecls,
                   CArg<"FnEffects", "{}">:$fnEffects,
                   CArg<"InlineLevel", "InlineLevel::Automatic">:$inlineLevel)>,
    OpBuilder<(ins "StringAttr":$name, "StringAttr":$sourceName,
                   "SignatureType":$signature,
                   CArg<"SpecialFunctionKind", "{}">:$specialFnID)>,
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def LIT_CallOp : LITOp<"call", [
    DeclareOpInterfaceMethods<KGEN_CallOpInterface>,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "call a function";
  let description = [{
    The `lit.call` operation is a static call to a `lit.func`. It references its
    callee via symbol constant. It contains optional parameter declarations for
    any callee result parameters and bindings for implicit lifetime parameters.

    Example:

    ```mlir
    lit.call @callee[*"a`"]<1, :dtype f32 -> out: type>(%arg0) : (!Int) -> ()
    ```
  }];

  let arguments = (ins
    AttrOfType<LIT_SignatureType>:$callee,
    KGEN_ParameterExprArrayAttr:$implicitLifetimes,
    Variadic<AnyType>:$operands
  );
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `` custom<CallOp>($callee, $implicitLifetimes, $operands,
                      type($operands), type($results)) attr-dict
  }];

  let extraClassDeclaration = [{
    /// Get the direct callee symbol if this is a direct call.
    SymbolRefAttr getDirectCallee();
    /// Get the callee signature type.
    LITSignatureType getCalleeType() {
      return cast<LITSignatureType>(getCallee().getType());
    }
  }];
}

//===----------------------------------------------------------------------===//
// CallIndirectOp
//===----------------------------------------------------------------------===//

def LIT_CallIndirectOp : LITOp<"call_indirect"> {
  let summary = "indirect call";
  let description = [{
    The `lit.call_indirect` operation is an indirect call to a signature-typed
    dynamic callee. It references its callee via an SSA operand. It contains
    optional bindings for implicit lifetime parameters.

    Example:

    ```mlir
    lit.call_indirect %callee[*"x`"](%x) : (!lit.ref<mut index, *"x`">) -> ()
    ```
  }];

  let arguments = (ins
    LIT_SignatureType:$callee,
    KGEN_ParameterExprArrayAttr:$implicitLifetimes,
    Variadic<AnyType>:$arguments
  );
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $callee `` custom<LifetimeParams>($implicitLifetimes) `(` $arguments `)` `:`
    custom<Signature>(type($callee)) attr-dict ``
    custom<CallOpTypes>(type($arguments), type($results), ref(type($callee)),
                        ref($implicitLifetimes))
  }];

  let extraClassDeclaration = [{
    LITSignatureType getCalleeType() { return getCallee().getType(); }
  }];
}

//===----------------------------------------------------------------------===//
// FileModuleOp
//===----------------------------------------------------------------------===//

def LIT_FileModuleOp : LITOp<"file_module", [
    ASTDeclInterface, IsolatedFromAbove, NoRegionArguments, SymbolTable, Symbol,
    DeclareOpInterfaceMethods<DeclInterface>,
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

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<LIT_DocStringAttr>:$docString
  );
  let regions = (region SizedRegion<1>:$bodyRegion);

  let extraClassDeclaration = [{
    /// Return the decl name of the file.
    StringAttr getDeclName() { return getSymNameAttr(); }
  }];

  let builders = [
    OpBuilder<(ins "StringAttr":$name)>
  ];
  let assemblyFormat = "$sym_name attr-dict-with-keyword $bodyRegion";
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// PackageOp
//===----------------------------------------------------------------------===//

def LIT_PackageOp : LITOp<"package", [
    ASTDeclInterface, IsolatedFromAbove, NoRegionArguments, SymbolTable, Symbol,
    DeclareOpInterfaceMethods<DeclInterface>,
  ] # GraphRegionNoTerminator.traits> {
  let summary = "mojo source package";
  let description = [{
    A `lit.package` models a mojo package. It contains a set of additional
    `lit.file_module` and `lit.package` operations.

    If the package was precompiled it may also contain:
    - The full post-parser IR module, `postParseModule`.
    - The set of link dependencies for this package, `dependencies`.

    Example:

    ```mlir
    lit.package @foo {
      lit.file_module @bar {
      }
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<LIT_DocStringAttr>:$docString,
    // The module containing the post-parser IR of this package when it was
    // compiled.
    OptionalAttr<Builtin_DenseResourceElementsAttr>:$postParseModule,
    // References to the dependencies of this package.
    OptionalAttr<KGEN_LinkDependencyArrayAttr>:$dependencies
  );
  let regions = (region SizedRegion<1>:$bodyRegion);

  let extraClassDeclaration = [{
    /// Return the decl name of the package.
    StringAttr getDeclName() { return getSymNameAttr(); }
  }];

  let builders = [
    OpBuilder<(ins "StringAttr":$name)>
  ];
  let assemblyFormat = "$sym_name attr-dict-with-keyword $bodyRegion";
  let skipDefaultBuilders = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// StructDeclOp
//===----------------------------------------------------------------------===//

def LIT_StructDeclOp : LITOp<"struct.decl", [
    IsolatedFromAbove, Symbol, SymbolTable, DeclInterface, SingleBlock,
    ASTDeclInterface, NoTerminator,
    DeclareOpInterfaceMethods<DebugInfo_ScopedLocation>
  ]> {
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
      lit.struct.field ptr : !kgen.pointer<scalar<dtype>>
      lit.struct.field dtype : !kgen.dtype
    }
    ```

    The `registerPassable` is used to indicate structs that are preferrably
    passed around as SSA values instead of being processed in memory.  This is
    true for simple values like F32 and Bool but not for complex aggregates like
    a SmallVector that can't be bitwise copied around or a Dictionary that
    shouldn't be.

    This includes a 'destructor' field if the type has a non-trivial destructor,
    allowing SSA passes to perform lifetime optimizations.  It includes a
    'moveInit' field if the type implements the `__moveinit__` special
    functions if present.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<LIT_TypeSignatureType>:$signature,
    KGEN_ParamDeclArrayAttr:$params,
    KGEN_DecoratorsAttr:$decorators,
    TypeLineageArrayAttr:$parentTypes,
    UnitAttr:$isSynthetic,
    OptionalAttr<TypeAttr>:$nonmaterializableTarget,
    OptionalAttr<KGEN_SymbolConstantAttr>:$destructor,
    OptionalAttr<KGEN_SymbolConstantAttr>:$moveInit,
    OptionalAttr<KGEN_SymbolConstantAttr>:$copyInit,
    OptionalAttr<TypeAttrOf<KGEN_SignatureType>>:$closureSignature,
    OptionalAttr<LIT_DocStringAttr>:$docString,
    OptionalAttr<StrAttr>:$deprecationWarning,
    OptionalAttr<SourceNameAttr>:$sourceName,
    EnumProperty<"TypeConvention">:$convention
  );
  let regions = (region SizedRegion<1>:$fields);
  let assemblyFormat = [{
    $sym_name ``
    custom<StructParameterSpec>($params, $signature, $parentTypes) ``
    custom<TypeConvention>($convention) attr-dict-with-keyword
    (`\n` `destructor` custom<Symbol>($destructor)^ ``)?
    (`\n` `move` custom<Symbol>($moveInit)^ ``)?
    (`\n` `copy` custom<Symbol>($copyInit)^ ``)?
    custom<OptionalDecorators>($decorators) $fields
  }];

  let extraClassDeclaration = [{
    /// Get the field declarations.
    auto getFieldDecls() { return getFields().getOps<StructFieldOp>(); }
    auto field_begin() { return getFieldDecls().begin(); }
    auto field_end() { return getFieldDecls().end(); }

    /// Struct declarations just have input parameters.
    ArrayRef<ParamDeclAttr> getResultParams() { return {}; }
    ArrayRef<ParamDeclAttr> getInputParams() { return getParams(); }

    bool isRegisterPassable() {
      return getConvention() != TypeConvention::MemoryOnly;
    }
    bool isRegisterPassableTrivial() {
      return getConvention() == TypeConvention::RegisterPassableTrivial;
    }

    /// Return the decl name of the struct.
    StringAttr getDeclName() { return getSymNameAttr(); }

    /// Get a bound reference to the struct type.
    StructType bindReference(ArrayRef<TypedAttr> paramValues = {});

    /// Get whether this struct is compiler generated.
    bool isSynthetic();
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

def LIT_StructFieldOp : LITOp<"struct.field", [
    ASTDeclInterface, HasParent<"StructDeclOp">
  ]> {
  let summary = "Struct field declaration.";
  let description = [{
    The `lit.struct.field` operations declares a named struct field and type.

    Example:

    ```mlir
    // Declare a field named "foo".
    lit.struct.field foo : !pop.scalar<f32>
    ```
  }];

  let arguments = (ins
    StrAttr:$name,
    TypeAttr:$type,
    OptionalAttr<LIT_DocStringAttr>:$docString
  );
  let assemblyFormat = "custom<KeywordAsString>($name) attr-dict `:` $type";

  let builders = [
    OpBuilder<(ins "StringAttr":$name, "Type":$type)>,
    OpBuilder<(ins "const Twine &":$name, "Type":$type)>,
  ];

  let extraClassDeclaration = [{
    /// Given a struct base value type that may have bound parameters,
    /// substitute those bound parameters into our type and return the result.
    Type getReboundType(StructType structSelfType);
  }];
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
      -> !lit.struct<@MyStruct<T: dtype = type>>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands,
                       StringArrayAttr:$fields);
  let results = (outs LIT_StructType:$result);
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
      : !pop.scalar<f64> into !lit.struct<@MyStruct<T: dtype = f64>>
    ```
  }];

  let arguments = (ins StrAttr:$field, AnyType:$value,
                       LIT_StructType:$container);
  let results = (outs LIT_StructType:$result);
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
      : !pop.scalar<f64> from !lit.struct<@MyStruct<T: dtype = f64>>
    ```
  }];

  let arguments = (ins LIT_StructType:$container, StrAttr:$field);
  let results = (outs AnyType:$value);
  let assemblyFormat = [{
    $container `[` custom<KeywordAsString>($field) `]` attr-dict `:`
    type($value) `from` qualified(type($container))
  }];

  let builders = [
    // Infer the result type based on the result of lookup.
    OpBuilder<(ins "Value":$structBase, "StructFieldOp":$field)>
  ];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// TraitDeclOp
//===----------------------------------------------------------------------===//

def LIT_TraitDeclOp : LITOp<"trait.decl", [
    IsolatedFromAbove, Symbol, SymbolTable, DeclInterface, SingleBlock,
    ASTDeclInterface, NoTerminator,
    DeclareOpInterfaceMethods<DebugInfo_ScopedLocation>
  ]> {
  let summary = "Trait declaration.";
  let description = [{
    The `lit.trait.decl` operation is a top-level operation used to declare
    LIT trait types. Trait declarations are named with a symbol.
    The body contains `lit.trait.field` operations that declare the contents
    of the struct.

    Example:

    ```mlir
    lit.trait.decl @Trait<T: !Trait> {
      lit.func @f() -> !kgen.none {
        lit.trait_func
      }
    }
    ```

    TODO: Add decorators support.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<LIT_TypeSignatureType>:$signature,
    KGEN_ParamDeclArrayAttr:$params,
    TypeLineageArrayAttr:$parentTypes,
    OptionalAttr<TypeAttrOf<KGEN_SignatureType>>:$dtorSig,
    OptionalAttr<LIT_DocStringAttr>:$docString,
    OptionalAttr<StrAttr>:$deprecationWarning
  );

  let regions = (region SizedRegion<1>:$fields);

  let assemblyFormat = [{
    $sym_name ``
    custom<StructParameterSpec>($params, $signature, $parentTypes)
    attr-dict-with-keyword $fields
  }];

  let extraClassDeclaration = [{
    /// Trait declarations only have input parameters.
    ArrayRef<ParamDeclAttr> getInputParams() { return getParams(); }
    ArrayRef<ParamDeclAttr> getResultParams() { return {}; }

    /// Get a bound reference to the Trait type.
    TraitType bindReference();
  }];

  let builders = [
    // Create a trait with a region but no body block. The caller must create
    // it and fill it in.
    OpBuilder<(ins "StringAttr":$name)>,
  ];
}

//===----------------------------------------------------------------------===//
// TryOp
//===----------------------------------------------------------------------===//

def LIT_TryOp : LITOp<"try", [DeclareOpInterfaceMethods<ControlFlowNode>]> {
  let summary = "A try-except-else block.";
  let description = [{
    The `lit.try` operation represents a try-except-else statement. When inside
    the 'try' region, any implicitly propagated errors interrupt execution and
    branch to the 'except' region. If the 'try' region completes execution
    without any exceptions, control-flow branches to the 'else' region.

    Example:

    ```mlir
    lit.try {
      %0 = kgen.call @maybeRaises() : () throws -> !lit.struct<@Int>
      lit.try.yield
    } except (%err: !lit.struct<@Error>) {
      lit.try.yield
    } else {
      lit.try.yield
    } finally {
      lit.try.yield
    }
    ```

    The 'finally' region is only present during semantic control-flow
    representation mode and is not modeled through the HLCF abstraction.
  }];

  let arguments = (ins
    Optional<LIT_RefType>:$err,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$suppressWarnings);
  let regions = (region SizedRegion<1>:$tryRegion, SizedRegion<1>:$exceptRegion,
                        SizedRegion<1>:$elseRegion,
                        VariadicRegion<SizedRegion<1>>:$finallyRegions);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    ($err^ `:` type($err))? (`->` type($results)^)? $tryRegion
    `except` custom<RegionWithArgs>($exceptRegion)
    `else` custom<RegionWithArgs>($elseRegion)
    (`finally` $finallyRegions^)? attr-dict
  }];

  let builders = [
    OpBuilder<(ins "Value":$err, CArg<"bool", "false">:$suppressWarnings),
      [{ build($_builder, $_state, {}, err, suppressWarnings, 1); }]>,
  ];

  let extraClassDeclaration = [{
    /// Get the finally region, assuming it is present.
    Region &getFinallyRegion() { return getFinallyRegions().front(); }

    /// Return true if the finally block is trivial.
    bool hasTrivialFinally();
  }];
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
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";

  let builders = [OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>];
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

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>
  ];
}

//===----------------------------------------------------------------------===//
// AliasDeclOp
//===----------------------------------------------------------------------===//

def LIT_AliasDeclOp : LITOp<"alias.decl", [
    ASTDeclInterface, DeclareOpInterfaceMethods<ParamOpInterface>
  ]>, OneParamDecl {
  let summary = "`alias x: T = value` declaration";
  let description = [{
    This op represents an alias decl bound with the given initializer.

    Example:

    ```mlir
    lit.alias.decl A = <5>
    lit.alias.decl A_plus_one = <add(A, 1)>
    ```
  }];

  let arguments = (ins
    KGEN_ParamDeclAttr:$paramDecl,
    TypedAttrInterface:$value,
    OptionalAttr<LIT_DocStringAttr>:$docString,
    OptionalAttr<StrAttr>:$deprecationWarning
  );
  let assemblyFormat = [{
    custom<AliasDeclOpValue>($paramDecl, $value) attr-dict
  }];

  let extraClassDeclaration = [{
    StringAttr getName() { return getParamDecl().getName(); }
    StringAttr getDeclName() { return getName(); }
    Type getType() { return getParamDecl().getType(); }
  }];

  let builders = [
    OpBuilder<(ins "ParamDeclAttr":$decl, "TypedAttr":$value), [{
      build($_builder, $_state, decl, value, /*deprecationWarning=*/{},
            /*docString=*/{});
    }]>
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// VarDeclOp
//===----------------------------------------------------------------------===//

def LIT_VarDeclOp : LITOp<"var.decl", [
    ASTDeclInterface,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    DeclareOpInterfaceMethods<ParamOpInterface>,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]>, OneParamDecl {
  let summary = "Addressible local var declaration";
  let description = [{
    This declaration is used for var declarations with an exposed
    address.  `kind` encodes how this declaration was created, either via
    explicit `var` expressions, implicitly declared, or internally
    synthesized (which suppresses some warnings).  This type defines a lifetime
    and returns a mutable reference to its result in that lifetime.

    For implicitly declared VarDecls, an optional `argShadowIndex` specifies
    which argument number this VarDecl was created to shadow.

    Example:

    ```mlir
    %foo = lit.var.decl "foo" var : !lit.ref<index, foo>
    ```
  }];

  let arguments = (ins
    StrAttr:$name,
    LIT_VarDeclKindAttr:$kind,
    KGEN_ParamDeclAttr:$paramDecl,
    OptionalAttr<I32Attr>:$argShadowIndex
  );
  let results = (outs LIT_RefType:$result);

  let assemblyFormat = [{
    $name `` $kind (`(` $argShadowIndex^ `)`)? `:`
    custom<VarDeclType>(type($result), $paramDecl) attr-dict
  }];

  let builders = [
    OpBuilder<(ins "Type":$elementType, "StringRef":$name,
                   "StringRef":$lifetimeName,
                   "VarDeclKind":$kind)>
  ];
  let extraClassDeclaration = [{
    bool isSynthetic();
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// VarLifetimeStartOp
//===----------------------------------------------------------------------===//

def LIT_VarLifetimeStartOp : LITOp<"var.lifetime.start"> {
  let arguments = (ins LIT_RefType:$arg);
  let assemblyFormat = "$arg attr-dict `:` type($arg)";
}

//===----------------------------------------------------------------------===//
// VarLifetimeEndOp
//===----------------------------------------------------------------------===//

def LIT_VarLifetimeEndOp : LITOp<"var.lifetime.end"> {
  let arguments = (ins LIT_RefType:$arg);
  let assemblyFormat = "$arg attr-dict `:` type($arg)";
}

//===----------------------------------------------------------------------===//
// GlobalVarDeclOp
//===----------------------------------------------------------------------===//

def LIT_GlobalVarDeclOp : LITOp<"globalvar.decl", [
    ASTDeclInterface, Symbol, NoTerminator, ExportInterface,
    DeclareOpInterfaceMethods<DebugInfo_ScopedLocation>
  ]> {
  let summary = "Global variable declaration.";
  let description = [{
    The `lit.globalvar.decl` operation corresponds to a global variable
    declaration. It is referenced with a symbol name. It contains a constructor
    and destructor region where the initializer expression and the destructor
    call (if there is one) for the variable go.

    Example:

    ```mlir
    lit.globalvar.decl @global_var : !lit.struct<@String> isVar {
      %0 = lit.globalvar.ref @global_var : @String
      kgen.call @String::@__init__(%0)
    }, {
      %0 = lit.globalvar.ref @global_var : @String
      kgen.call @String::@__del__(%0)
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttr:$type,
    KGEN_ExportKindAttr:$exportKind,
    KGEN_DecoratorsAttr:$decorators,
    OptionalAttr<StrAttr>:$linkageName,
    OptionalAttr<LIT_DocStringAttr>:$docString,
    OptionalAttr<StrAttr>:$deprecationWarning
  );
  let regions = (region SizedRegion<1>:$ctor, SizedRegion<1>:$dtor);

  let assemblyFormat = [{
    `` custom<SymbolExport>($exportKind) $sym_name `:` $type
    attr-dict-with-keyword custom<OptionalDecorators>($decorators)
    custom<NoArgRegion>($ctor) `,`
    custom<NoArgRegion>($dtor)
  }];

  let extraClassDeclaration = [{
    /// Return the decl name of the variable.
    StringAttr getDeclName() { return getSymNameAttr(); }
  }];

  let builders = [
    OpBuilder<(ins "StringRef":$sym_name, "Type":$type), [{
      build($_builder, $_state, sym_name, type, ExportKind::NotExported,
            DecoratorsAttr::get(type.getContext(), {}), /*linkageName=*/{},
            /*docString=*/{}, /*deprecationWarning=*/{});
    }]>
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GlobalVarRefOp
//===----------------------------------------------------------------------===//

def LIT_GlobalVarRefOp : LITOp<"globalvar.ref",
    [Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Reference to a global variable.";
  let description = [{
    The `lit.globalvar.ref` operation forms a reference to a global variable.
    The result reference is always mutable and immortal.

    Example:

    ```mlir
    %0 = lit.globalvar.ref @global_var : <mut @Int, #lit.lifetime>
    ```
  }];

  let arguments = (ins SymbolRefAttr:$global);
  let results = (outs LIT_RefType:$result);

  let builders = [
    OpBuilder<(ins "GlobalVarDeclOp":$op)>
  ];

  let assemblyFormat = "$global attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// AsyncCallOp
//===----------------------------------------------------------------------===//

def LIT_AsyncCallOp : LITOp<"async.call", [
    DeclareOpInterfaceMethods<KGEN_CallOpInterface>]> {
  let summary = "Call to an async function.";
  let description = [{
    The `lit.async.call` operation calls an `async` parametric callable. It
    returns a `!co.routine`.

    Example:

    ```mlir
    %coro = lit.async.call[:() -> index: @foo]()
    ```
  }];

  let arguments = (ins
    AttrOfType<KGEN_SignatureType>:$callee,
    KGEN_ParameterExprArrayAttr:$implicitLifetimes,
    Variadic<AnyType>:$operands
  );
  let results = (outs CoroutineType:$coroutine);

  let assemblyFormat = [{
    `` custom<ParametricCallee>($callee) ``
    custom<LifetimeParams>($implicitLifetimes) ``
    custom<AsyncCallOpTypes>(type($operands), ref($callee),
                             ref($implicitLifetimes))
    `(` $operands `)` attr-dict
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
// ExternFuncOp
//===----------------------------------------------------------------------===//

def LIT_ExternFuncOp : LITOp<"extern_func", [Terminator,
    HasParent<"LIT::FuncOp">]> {
  let summary = "Denotes an externalized function.";
  let description = [{
    The `lit.extern_func` operation denotes the end of a `lit.func` body,
    indicating that the function's body has been externalized or elided.

    Example:

    ```mlir
    lit.func @extern() {
      lit.extern_func
    }
    ```
  }];

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TraitFuncOp
//===----------------------------------------------------------------------===//

def LIT_TraitFuncOp : LITOp<"trait_func", [Terminator,
    HasParent<"LIT::FuncOp">]> {
  let summary = "Denotes a trait function.";
  let description = [{
    The `lit.trait_func` operation denotes the end of a `lit.func` body,
    indicating that the function is a trait function whose body is empty.

    Example:

    ```mlir
    lit.trait.decl @Trait {
      lit.func @f() -> !kgen.none {
        lit.trait_func
      }
    }
    ```
  }];

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def LIT_ReturnOp : LITOp<"return",
    [DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
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

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict (`:` type($operands)^)?";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ErrorReturnOp
//===----------------------------------------------------------------------===//

def LIT_ErrorReturnOp : LITOp<"error_return", [Terminator, ReturnLike,
    DeclareOpInterfaceMethods<ControlFlowTerminator>]> {
  let summary = "Represents an early return due to failure.";
  let description = [{
    The `lit.error_return` operation is a terminator version
    of `lit.raise` that represents exit from function call due
    to failure.

    Example:

    ```mlir
    lit.func @throwing_func() throws -> !kgen.variant<@Error, none> {
      %1 = lit.struct.create(a=%0) : (index) -> !lit.struct<@Error>
      %2 = kgen.variant.create %1 :
        !lit.struct<@Error> -> !kgen.variant<@Error, none>
      lit.error_return %2 : !kgen.variant<@Error, none>
    }
    ```
  }];

  let arguments = (ins AnyType:$result);
  let assemblyFormat = "$result attr-dict `:` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// RaiseOp
//===----------------------------------------------------------------------===//

def LIT_RaiseOp : LITOp<"raise"> {
  let summary = "Lexical raise statement.";
  let description = [{
    The `lit.raise` operation is not a terminator. It corresponds to a raise
    statement in Lit. This operation is only allowed inside the 'try' region of
    a `lit.try` operation or within a `lit.func` marked as `throws`.

    Example:

    ```mlir
    lit.raise
    ```
  }];

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// LoopConditionOp
//===----------------------------------------------------------------------===//

def LIT_LoopConditionOp : LITOp<"loop.condition", [
    HasParent<"LoopOp">, Terminator]> {
  let summary = "Condition for the loop.";
  let description = [{
    The `lit.loop.condition` operation is a terminator. It yields the condition
    of a loop.

    Example:

    ```mlir
    lit.loop.condition %cond
    ```
  }];

  let arguments = (ins I1:$operand);

  let assemblyFormat = "attr-dict $operand `:` type($operand)";
}

//===----------------------------------------------------------------------===//
// LoopContinueOp
//===----------------------------------------------------------------------===//

def LIT_LoopContinueOp : LITOp<"loop.continue", [
    HasParent<"LoopOp">, Terminator]> {

  let summary = "Loop continue.";
  let description = [{
    The `lit.loop.continue` operation is a terminator. The operation indicates
    that control-flow should branch to the beginning of the enclosing loop. It
    is only used as the terminator for the body of a `lit.loop`.

    Example:

    ```mlir
    lit.loop.continue
    ```
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// LoopYieldOp
//===----------------------------------------------------------------------===//

def LIT_LoopYieldOp : LITOp<"loop.yield", [
    HasParent<"LoopOp">, Terminator]> {
  let summary = "Loop yield.";
  let description = [{
    The `lit.loop.yield` operation is a terminator. The operation indicates
    that control-flow should branch to the end of the enclosing loop (after the
    else region finishes execution).

    Example:

    ```mlir
    lit.loop.yield
    ```
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// BreakOp
//===----------------------------------------------------------------------===//

def LIT_BreakOp : LITOp<"break"> {
  let summary = "Lexical break statement.";
  let description = [{
    The `lit.break` operation is not a terminator, but it corresponds to a break
    statement in Lit. The operation indicates that control-flow should branch
    to after the enclosing loop.  This is lowered by LowerSemanticCF.

    Example:

    ```mlir
    lit.break
    ```
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// ContinueOp
//===----------------------------------------------------------------------===//

def LIT_ContinueOp : LITOp<"continue"> {
  let summary = "Lexical continue statement.";
  let description = [{
    The `lit.continue` operation is not a terminator, but it corresponds to a
    continue statement in Lit. The operation indicates that control-flow should
    branch to after the enclosing loop.  This is lowered by LowerSemanticCF.

    Example:

    ```mlir
    lit.continue
    ```
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// UnresolvedImportOp
//===----------------------------------------------------------------------===//

// TODO(LLVM Issue #59877): Why isn't this in BuiltinAttributes.td?
def Builtin_LocationAttr : Builtin_Attr<"Location", "location", []> {}

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
    // Import a relative module as the given name.
    lit.unresolved_import @package.module as @newModule
    // Import a function within the module as the given name.
    lit.unresolved_import @module -> @func as @newFunc
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$moduleName,
    SymbolNameAttr:$importName,
    OptionalAttr<SymbolNameAttr>:$declName,
    OptionalAttr<Builtin_LocationAttr>:$importNameLoc,
    OptionalAttr<Builtin_LocationAttr>:$declNameLoc
  );
  let assemblyFormat = [{
    $moduleName (`->` $declName^)? `as` $importName
    (`declNameLoc` `(` $declNameLoc^ `)`)?
    (`importNameLoc` `(` $importNameLoc^ `)`)?
    attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// UnresolvedWildcardImportOp
//===----------------------------------------------------------------------===//

def LIT_UnresolvedWildcardImportOp : LITOp<"unresolved_wildcard_import"> {
  let summary = "Lexical unresolved wildcard import.";
  let description = [{
    The `lit.unresolved_wildcard_import` operation represents a wildcard module
    import that is yet to be resolved. If `fullImport` is set, all symbols are
    considered from import, not just public symbols.

    Example:

    ```mlir
    // Import from the given module.
    lit.unresolved_wildcard_import (`all` $fullImport^)? from @module
    ```
  }];

  let arguments = (ins SymbolNameAttr:$moduleName, UnitAttr:$fullImport);
  let assemblyFormat = "`from` $moduleName attr-dict";

  let builders = [OpBuilder<(ins
    "StringAttr":$moduleName, CArg<"bool", "false">:$isFullImport), [{
    build($_builder, $_state, moduleName,
          isFullImport ? $_builder.getUnitAttr() : mlir::UnitAttr());
  }]>];
}

//===----------------------------------------------------------------------===//
// UnboundRegionOp
//===----------------------------------------------------------------------===//

def LIT_UnboundRegionOp : LITOp<"unbound_region"> {
  let summary = "A transient container for a region.";
  let description = [{
    The `lit.unbound_region` operation is a container for an MLIR region that
    has yet to be bound to an MLIR operation. This is used by the parser to
    attach regions to MLIR operations. The operation itself is never valid. It
    must be moved by the parser into an MLIR operation.
  }];

  let regions = (region SizedRegion<1>:$region);

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// LoadConsumeOp
//===----------------------------------------------------------------------===//

def LIT_LoadConsumeOp : LITOp<"load.consume", [
    ReferenceElementType<"ref", "result">,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "consuming pointer load operation";
  let description = [{
    Loads the data at the given pointer like `pop.load`, but consumes the
    ownership of the memory being accessed. This is used when performing MRValue
    to SRValue conversion (i.e. when consuming a `var` of `@register_passable`
    type), and is lowered to `pop.load`.

    Example:

    ```mlir
    %0 = lit.load.consume %ref : !lit.ref<scalar<f32>, #lifetime>
    ```
  }];

  let arguments = (ins Arg<LIT_RefType, "", [MemRead]>:$ref);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$ref attr-dict `:` qualified(type($ref))";
}

//===----------------------------------------------------------------------===//
// OwnershipUseOp
//===----------------------------------------------------------------------===//

def LIT_OwnershipUseOp : LITOp<"ownership.use"> {
  let summary = "A use of a value to keep the value alive";
  let description = [{
    This op looks like a use of a value (or its pointee) which keeps the value
    live until at least this point.  This operation is a noop that is removed by
    the LowerLIT pass.

    ```mlir
    lit.ownership.use %0 : !somestring
    ```
  }];
  let arguments = (ins AnyType:$value);
  let results = (outs);
  let assemblyFormat = "$value `:` type($value) attr-dict";
}

//===----------------------------------------------------------------------===//
// TransferMemOwnershipOp
//===----------------------------------------------------------------------===//

def LIT_TransferMemOwnershipOp : LITOp<"transfer_mem_ownership", [
          DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
          DeclareOpInterfaceMethods<ParamOpInterface>]>, OneParamDecl {
  let summary = "End the lifetime of a tracked value and start a new lifetime";
  let description = [{
    This op is spelled `x^` when `x` is a lifetime-tracked memory value.  It
    indicates that the lifetime of `x` ends here and a new value with a new
    lifetime is created.  This also defines the new lifetime used in the
    result reference.

    This works with both memory values.

    ```mlir
    %1 = lit.transfer_mem_ownership %0 : !lit.ref<T, lt1> -> !lit.ref<T, lt2>
    ```
  }];

  let arguments = (ins LIT_RefType:$value, KGEN_ParamDeclAttr:$paramDecl);
  let results = (outs LIT_RefType:$result);
  let assemblyFormat = "$value `:` type($value) `->` type($result) attr-dict";

  let builders = [
    /// Create a TransferMem that has a lifetime with the specified name.
    OpBuilder<(ins "Value":$srcValue, "StringAttr":$lifetimeName)>
  ];
}

//===----------------------------------------------------------------------===//
// OwnershipMarkInitialized
//===----------------------------------------------------------------------===//

def LIT_OwnershipMarkInitializedOp : LITOp<"ownership.mark_initialized"> {
  let summary = "Mark an uninitialized value as being fully initialized";
  let description = [{
    The `lit.ownership.mark_initialized` operation informs CheckLifetimes that
    the specified value is now fully initialized. Note that this marks all
    subelements of the value as being initialized too, not just the fully-object
    bit.

    ```mlir
    lit.ownership.mark_initialized %0 : !lit.ref<T, mut lt>
    ```

    The argument must be an indirectly lifetime-tracked value, because
    CheckLifetimes isn't field sensitive for register passable values.
  }];

  let arguments = (ins LIT_RefType:$value);
  let assemblyFormat = "$value `:` type($value) attr-dict";
}

//===----------------------------------------------------------------------===//
// OwnershipMarkConsumedOp
//===----------------------------------------------------------------------===//

def LIT_OwnershipMarkConsumedOp : LITOp<"ownership.mark_consumed"> {
  let summary = "Mark a value as completely consumed.";
  let description = [{
    This op informs CheckLifetimes that the specified value has had its whole
    value consumed, including all subfields.

    ```mlir
    lit.ownership.mark_consumed %0 : !somestring
    ```

    The argument must be an indirectly lifetime-tracked value.
  }];

  let arguments = (ins LIT_RefType:$value);
  let assemblyFormat = "$value `:` type($value) attr-dict";
}

//===----------------------------------------------------------------------===//
// OwnershipMarkDestroyedOp
//===----------------------------------------------------------------------===//

def LIT_OwnershipMarkDestroyedOp : LITOp<"ownership.mark_destroyed"> {
  let summary = "Mark a value as not need a dtor to run";
  let description = [{
    This op informs CheckLifetimes that the specified value does not need its
    full object destructor to run.  This corresponds to things like Rust's
    mem::forget and is synthesized in __del__ methods to indicate that self is
    destroyed.

    ```mlir
    lit.ownership.mark_destroyed %0 : !somestring
    ```

    The argument must be an indirectly lifetime-tracked value, because
    CheckLifetimes isn't field sensitive for register passable values.
  }];

  let arguments = (ins LIT_RefType:$value);
  let assemblyFormat = "$value `:` type($value) attr-dict";
}

//===----------------------------------------------------------------------===//
// RefImmutOp
//===----------------------------------------------------------------------===//

class RefImmutType<string lhs, string rhs> : TypesMatchWith<
    "reference as pointer type", lhs, rhs,
    "llvm::cast<RefType>($_self).getWithMutability(false)">;

def LIT_RefImmutOp : LITOp<"ref.immut", [RefImmutType<"ref", "result">]> {
  let summary = "Drop mutability from a reference";
  let description = [{
    This op drops mutability from a reference type: given a
    `!lit.ref<X, mut life>`, it returns `!lit.ref<X, imm life>`.

    ```mlir
    lit.ref.immut %0 : !lit.ref<@Foo, mut life>
    ```
  }];

  let hasFolder = 1;
  let arguments = (ins LIT_RefType:$ref);
  let results = (outs LIT_RefType:$result);
  let assemblyFormat = "$ref `:` type($ref) attr-dict";


  let extraClassDeclaration = [{
    /// If the specified value is a RefImmutOp, look through it to its operand.
    /// Otherwise just return the value.
    static Value strip(Value value) {
      if (auto cast = value.getDefiningOp<RefImmutOp>())
        return cast.getOperand();
      return value;
    }
  }];
}

//===----------------------------------------------------------------------===//
// RefToPointerOp
//===----------------------------------------------------------------------===//

// Given a `!lit.ref<X, life>`, return `!kgen.pointer<X>`.
class RefPointerType<string lhs, string rhs> : TypesMatchWith<
    "reference as pointer type", lhs, rhs,
    "llvm::cast<RefType>($_self).getAsPointerType()">;

def LIT_RefToPointerOp : LITOp<"ref.to_pointer",
                              [RefPointerType<"ref", "ptr">]> {
  let summary = "Extract a low-level pointer out of a reference";
  let description = [{
    This op extracts a kgen pointer out of a reference, exposing an untracked
    and unsafe inner pointer.  This is used when lowering lifetimes.

    ```mlir
    lit.ref.to_pointer %0 : !lit.ref<@Foo, life>
    ```
  }];

  let arguments = (ins LIT_RefType:$ref);
  let results = (outs PointerType:$ptr);
  let assemblyFormat = "$ref `:` type($ref) attr-dict";
}

//===----------------------------------------------------------------------===//
// RefFromPointerOp
//===----------------------------------------------------------------------===//

def LIT_RefFromPointerOp : LITOp<"ref.from_pointer",
                                 [RefPointerType<"ref", "ptr">]> {
  let summary = "Create an immortal reference from a pointer value";
  let description = [{
    This op converts a raw pointer to a reference with specified mutability and
    lifetime.  When `startsUninit` is true, the reference is assumed to be
    uninitialized but needs to be initialized before exit.  When false it is
    assumed to be initialized and must be initialized on exit.

    ```mlir
    lit.ref.from_pointer %0, startUninit=true : !lit.ref<mut @Foo, life>
    ```
  }];

  let arguments = (ins PointerType:$ptr, UnitAttr:$startUninit,
                       UnitAttr:$endUninit);
  let results = (outs LIT_RefType:$ref);
  let assemblyFormat = [{
    $ptr (`start_uninit` $startUninit^)? (`end_uninit` $endUninit^)?
    `:` type($ref) attr-dict
  }];

  let builders = [
    // Create with a pointer and lifetime, to infer result type.
    OpBuilder<(ins "Value":$pointer, "TypedAttr":$lifetime,
                   "bool":$startsUninit, "bool":$endsUninit)>,
  ];
}


//===----------------------------------------------------------------------===//
// RefFromPointerREPLOp
//===----------------------------------------------------------------------===//

def LIT_RefFromPointerREPLOp : LITOp<"ref.from_pointer.repl",
                                     [RefPointerType<"ref", "ptr">]> {
  let summary = "Create an untracked reference from a pointer value";
  let description = [{
    This op is like `lit.ref.from_pointer` is a HACK that is only intended to be
    used in one place in the REPL as a stop-gap, do not perpetuate this!
  }];

  let arguments = (ins PointerType:$ptr, StrAttr:$name);
  let results = (outs LIT_RefType:$ref);
  let assemblyFormat = "$ptr `:` type($ref) attr-dict";
}


//===----------------------------------------------------------------------===//
// RefStructGEROp
//===----------------------------------------------------------------------===//

def LIT_RefStructGEROp : LITOp<"ref.struct.ger", [Pure,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Get a reference to a struct field.";
  let description = [{
    The `lit.ref.struct.ger` operation computes a reference to a struct field by
    name from a parametric struct.  The result has the same lifetime as the
    source value.

    Example:

    ```mlir
    lit.struct.decl @MyStruct<T: dtype> {
      lit.struct.field a : !pop.scalar<T>
    }

    %0 = lit.ref.struct.ger %struct[a] : !lit.ref<@MyStruct<...>, mut life>
    ```
  }];

  let arguments = (ins StrAttr:$field, RefOf<LIT_StructType>:$container);
  let results = (outs LIT_RefType:$result);
  let assemblyFormat = [{
    $container `[` custom<KeywordAsString>($field) `]` attr-dict `:`
    custom<StructGERTypes>(type($result), type($container))
  }];

  let builders = [
    // Infer the result type based on the result of lookup.
    OpBuilder<(ins "Value":$structBasePtr, "StructFieldOp":$field)>
  ];

  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// Given a reference to a struct, return the reference type to the
    /// specified field, maintaining lifetime and mutability.
    static RefType getFieldType(RefType structRefTy, StructFieldOp field);
  }];

}

//===----------------------------------------------------------------------===//
// RefLoadOp
//===----------------------------------------------------------------------===//

def LIT_RefLoadOp : LITOp<"ref.load", [
    ReferenceElementType<"ref", "result">,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "reference load operation";
  let description = [{
    Loads the data at the given reference. Returns a scalar of the element type.
    The reference may either be mutable or immutable.

    Example:

    ```mlir
    %0 = lit.ref.load %ref0 : !lit.ref<scalar<f32>, lifetime>
    %1 = lit.ref.load %ref1 : !lit.ref<simd<4, type>, lifetime>
    %2 = lit.ref.load %ref0 : !lit.ref<scalar<f32>, mut lifetime>
    ```
  }];

  let arguments = (ins LIT_RefType:$ref);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$ref attr-dict `:` type($ref)";
}

//===----------------------------------------------------------------------===//
// RefStoreOp
//===----------------------------------------------------------------------===//

def LIT_RefStoreOp : LITOp<"ref.store", [
    ReferenceElementType<"ref", "arg">,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "Reference store operation";
  let description = [{
    Store the given value to the given reference of the same element type.  The
    reference must be mutable.

    Example:

    ```mlir
    lit.ref.store %0, %ref0 : !lit.ref<scalar<f32>, mut lifetime>
    lit.ref.store %1, %ref1 : !lit.ref<simd<4, type>, mut lifetime>
    ```
  }];

  let arguments = (ins AnyType:$arg, LIT_RefType:$ref);
  let assemblyFormat = "$arg `,` $ref attr-dict `:` type($ref)";
}

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

def LIT_LoopOp: LITOp<"loop"> {
  let summary = "loop operation";
  let description = [{
    The `lit.loop` operation defines a lexical single scoped control-flow
    region. The operation has three regions:
      - condition region
      - body region
      - else region

    Control-flow transfers from the operation to the beginning of the body
    region when the condition is true. The loop operation is the target of the
    `lit.continue` and `lit.break` terminators.


    ```mlir
    lit.loop cond {
      lit.loop.condition %0: i1
    } body {
      lit.loop.continue
    } else {
      lit.loop.yield
    }
    ```
  }];

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$condRegion,
                        SizedRegion<1>:$bodyRegion,
                        SizedRegion<1>:$elseRegion);

  let assemblyFormat = [{
    (`->` type($results)^)?
    `cond` $condRegion
    `body` $bodyRegion
    `else` $elseRegion attr-dict
  }];

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>
  ];
}

//===----------------------------------------------------------------------===//
// RefPackCreateOp
//===----------------------------------------------------------------------===//

def LIT_RefPackCreateOp : LITOp<"ref.pack.create", [Pure]> {
  let summary = "create a reference pack";
  let description = [{
    The `kgen.ref.pack.create` operation creates a value of `!lit.ref.pack`
    type, populated with the given SSA values.  The elements should all have
    reference type with matching lifetime and address space.

    Example:

    ```mlir
      %0 = lit.ref.pack.create(%arg0, %arg1) : ...
    ```
  }];

  let arguments = (ins Variadic<LIT_RefType>:$elements);
  let results = (outs LIT_RefPackType:$result);
  let assemblyFormat = [{
    `(` $elements `)` attr-dict `:`
    custom<RefPackCreateType>(type($result), type($elements))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// RefPackExtractOp
//===----------------------------------------------------------------------===//

def LIT_RefPackExtractOp : LITOp<"ref.pack.extract",
                      [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "reference pack get element";
  let description = [{
    The `lit.ref.pack.extract` operation returns the reference in the pack at the
    provided index.
  }];

  let arguments = (ins LIT_RefPackType:$pack, AttrOfType<Index>:$index);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $pack `[` custom<IndexParamValue>($index) `]` attr-dict `:` type($pack)
  }];
}
