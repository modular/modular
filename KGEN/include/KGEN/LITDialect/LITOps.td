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
include "KGEN/KGENDialect/KGENAttrs.td"
include "KGEN/KGENDialect/KGENEnums.td"
include "KGEN/KGENDialect/KGENInterfaces.td"
include "KGEN/POPDialect/POPTypes.td"
include "KGEN/HLCFDialect/HLCFAttrs.td"
include "KGEN/HLCFDialect/HLCFInterfaces.td"
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
    FunctionLike, FunctionOpInterface, Symbol, SymbolTable, SingleBlock,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
    DeclareOpInterfaceMethods<ParamOpInterface, ["collectParameterUses"]>,
    ASTDeclInterface, DeclInterface, FuncInterface, ExportInterface]> {
  let summary = "High level generator definition";
  let description = [{
    The `lit.func` operation models a def/fn declaration at the Lit source
    level.

    @staticmethod decorators in Lit correspond to the attribute `isStatic`.
    @adaptive is a decorator that the user can attach to an `fn` that tells
    the parser to add the `isAdaptive` flag, and to treat multiple
    implementations as an overload set.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<KGEN_ParamDeclAttr>:$paramDecl,
    TypeAttrOf<LIT_SignatureType>:$signature,
    TypeAttrOf<FunctionType>:$functionType,
    KGEN_ParamDeclArrayAttr:$inputParams,
    KGEN_ParamDeclArrayAttr:$resultParams,
    ConstraintArrayAttr:$constraints,
    KGEN_DecoratorsAttr:$decorators,
    UnitAttr:$isStatic,
    UnitAttr:$isAdaptive,
    UnitAttr:$isParametric,
    UnitAttr:$isDef,  // Present if 'def' absent if 'fn'.
    UnitAttr:$isInherited,
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
    OptionalAttr<LIT_DocStringAttr>:$docString,
    DefaultValuedAttr<DictionaryAttr, "std::nullopt">:$LLVMMetadata
  );
  let regions = (region SizedRegion<1>:$bodyRegion);

  let extraClassDeclaration = [{
    Block *getBody() { return &getFunctionBody().back(); }
    Region *getCallableRegion() { return &getBodyRegion(); }

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
    StringAttr getDeclName() { return getSymNameAttr(); }

    /// Return the full signature of this declaration, including parameters from
    /// enclosing struct declarations.
    LITSignatureType getFullSignature();

    /// Return a collection of the ParamDeclAttr's that are declared on this
    /// function and any enclosing unit, ordered as will be required by a call.
    SmallVector<ParamDeclAttr> collectAllInputParams();
  }];

  let builders = [
    // Create a func with a region and a body block.
    OpBuilder<(ins)>,
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
    DeclareOpInterfaceMethods<CallOpInterface>,
    DeclareOpInterfaceMethods<KGEN_CallOpInterface>,
    DeclareOpInterfaceMethods<ParamOpInterface, ["isImplicitlyParametric"]>]>,
    CallLikeOperation {
  let summary = "call a function";
  let description = [{
    The `lit.call` operation is a static call to a `lit.func`. It references its
    callee via symbol constant. It contains optional parameter declarations for
    any callee result parameters and bindings for implicit lifetime parameters.

    Example:

    ```mlir
    lit.call @callee[*"`a"]<1, :dtype f32 -> out: regtype>(%arg0) : (!Int) -> ()
    ```
  }];

  let arguments = (ins
    KGEN_SymbolConstantAttr:$callee,
    KGEN_ParameterExprArrayAttr:$implicitLifetimes,
    KGEN_ParamDeclArrayAttr:$paramDecls,
    Variadic<AnyType>:$operands
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    custom<CallOp>($callee, $implicitLifetimes, $paramDecls, $operands,
                   type($operands), type($results)) attr-dict
  }];

  let builders = [
    OpBuilder<(ins "TypeRange":$results, "SymbolConstantAttr":$callee,
                   "ValueRange":$operands), [{
      build($_builder, $_state, results, callee, /*implicitLifetimes=*/{},
            /*paramDecls=*/ArrayRef<ParamDeclAttr>(), operands);
    }]>
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CallParamOp
//===----------------------------------------------------------------------===//

def LIT_CallParamOp : LITOp<"call_param", [
    DeclareOpInterfaceMethods<KGEN_CallOpInterface>,
    DeclareOpInterfaceMethods<ParamOpInterface, ["isImplicitlyParametric"]>]>,
    CallLikeOperation {
  let summary = "call a parameter expresion";
  let description = [{
    The `lit.call_param` operation is a static call to a signature-typed
    parameter value. It references its callee via a parameter value. It contains
    optional parameter declarations for any callee result parameters and
    bindings for implicit lifetime parameters.

    Example:

    ```mlir
    lit.call_param[
      // Callee `!lit.signature`:
      !lit.signature<[1]<[] -> index>(!lit.ref<mut index, *[0,0]>) -> ()>:
      // Callee expresion:
      calleeFn
    // Implicit lifetime and result parameter bindings.
    ]<() -> out>[*"`x"](%x)
    ```
  }];

  let arguments = (ins
    AttrOfType<LIT_SignatureType>:$callee,
    KGEN_ParamDeclArrayAttr:$paramDecls,
    KGEN_ParameterExprArrayAttr:$implicitLifetimes,
    Variadic<AnyType>:$operands
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    `` custom<ParametricCallee>($callee, $paramDecls) ``
    custom<LifetimeParams>($implicitLifetimes) ``
    custom<CallOpTypes>(type($operands), type($results), ref($callee),
                        ref($implicitLifetimes))
    `(` $operands `)` attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CallSignatureOp
//===----------------------------------------------------------------------===//

def LIT_CallSignatureOp : LITOp<"call_signature"> {
  let summary = "indirect call";
  let description = [{
    The `lit.call_signature` operation is an indirect call to a signature-typed
    dynamic callee. It references its callee via an SSA operand. It contains
    optional bindings for implicit lifetime parameters.

    Example:

    ```mlir
    lit.call_signature %callee[*"`x"](%x) : (!lit.ref<mut index, *"`x">) -> ()
    ```
  }];

  let arguments = (ins
    LIT_SignatureType:$callee,
    KGEN_ParameterExprArrayAttr:$implicitLifetimes,
    Variadic<AnyType>:$arguments
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    $callee `` custom<LifetimeParams>($implicitLifetimes) `(` $arguments `)` `:`
    custom<Signature>(type($callee)) attr-dict ``
    custom<CallOpTypes>(type($arguments), type($results), ref(type($callee)),
                        ref($implicitLifetimes))
  }];

  let builders = [
    OpBuilder<(ins "TypeRange":$results, "Value":$callee,
                   "ValueRange":$args), [{
      build($_builder, $_state, results, callee,
            /*implicitLifetimes=*/ArrayRef<TypedAttr>(), args);
    }]>
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// FileModuleOp
//===----------------------------------------------------------------------===//

def FileModuleOp : LITOp<"file_module", [
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
    OptionalAttr<StrAttr>:$sourceName,
    OptionalAttr<LIT_DocStringAttr>:$docString
  );
  let regions = (region SizedRegion<1>:$bodyRegion);

  let extraClassDeclaration = [{
    /// Return the decl name of the file.
    StringAttr getDeclName() { return getSymNameAttr(); }
  }];

  let builders = [
    OpBuilder<(ins "StringAttr":$name, "StringAttr":$sourceName)>,
  ];
  let assemblyFormat = "$sym_name attr-dict-with-keyword $bodyRegion";
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// PackageOp
//===----------------------------------------------------------------------===//

def PackageOp : LITOp<"package", [
    ASTDeclInterface, IsolatedFromAbove, NoRegionArguments, SymbolTable, Symbol,
    DeclareOpInterfaceMethods<DeclInterface>,
  ] # GraphRegionNoTerminator.traits> {
  let summary = "mojo source package";
  let description = [{
    A `lit.package` models a mojo package. It contains a set of additional
    `lit.file_module` and `lit.package` operations.

    If the package was precompiled it may also contain:
    - The full pre-elaboration IR module, `preElaborationModule`.
    - The definitions used at compile time, `compiledEnv`.
    - Any and all precompiled, standalone archives of the package, `archives`.

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
    OptionalAttr<StrAttr>:$sourceName,
    OptionalAttr<LIT_DocStringAttr>:$docString,
    // The module containing the pre-elaboration IR of this package when it was
    // compiled.
    OptionalAttr<Builtin_DenseResourceElementsAttr>:$preElaborationModule,

    OptionalAttr<KGEN_EnvAttr>:$compiledEnv,
    // Zero or more precompiled archives for the package.
    KGEN_PackageArchiveArrayAttr:$archives
  );
  let regions = (region SizedRegion<1>:$bodyRegion);

  let extraClassDeclaration = [{
    /// Return the decl name of the package.
    StringAttr getDeclName() { return getSymNameAttr(); }
  }];

  let builders = [
    OpBuilder<(ins "StringAttr":$name, "StringAttr":$sourceName)>,
  ];
  let assemblyFormat = [{
    $sym_name `` custom<PackageArchiveArray>($archives) attr-dict-with-keyword
    $bodyRegion
  }];
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

    The parameters of the declaration can be constraints with optional
    constraints.

    Example:

    ```mlir
    lit.struct.decl @SmallString<size>
        constraints <[le(size, 8), "string must be small"]> {
      lit.struct.field data : !pop.array<size, scalar<si8>>
    }
    ```

    The `registerPassable` is used to indicate structs that are preferrably
    passed around as SSA values instead of being processed in memory.  This is
    true for simple values like F32 and Bool but not for complex aggregates like
    a SmallVector that can't be bitwise copied around or a Dictionary that
    shouldn't be.

    This includes a 'destructor' field if the type has a non-trivial destructor,
    allowing SSA passes to perform lifetime optimizations.  It includes a
    'moveInit' field if the type implements either the `__moveinit__` or
    `__takeinit__` special functions.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<LIT_TypeSignatureType>:$signature,
    KGEN_ParamDeclArrayAttr:$inputParams,
    KGEN_DecoratorsAttr:$decorators,
    TypeLineageArrayAttr:$parentTypes,
    OptionalAttr<TypeAttr>:$nonmaterializableTarget,
    OptionalAttr<KGEN_SymbolConstantAttr>:$destructor,
    OptionalAttr<KGEN_SymbolConstantAttr>:$moveInit,
    OptionalAttr<KGEN_SymbolConstantAttr>:$copyInit,
    OptionalAttr<TypeAttrOf<KGEN_SignatureType>>:$closureSignature,
    OptionalAttr<LIT_DocStringAttr>:$docString,
    EnumProperty<"TypeConvention">:$convention
  );
  let regions = (region SizedRegion<1>:$fields);
  let assemblyFormat = [{
    $sym_name ``
    custom<StructParameterSpec>($inputParams, $signature, $parentTypes)
    custom<TypeConvention>($convention) attr-dict-with-keyword
    custom<OptionalDecorators>($decorators) $fields
  }];

  let extraClassDeclaration = [{
    /// Get the field declarations.
    auto getFieldDecls() { return getFields().getOps<StructFieldOp>(); }
    auto field_begin() { return getFieldDecls().begin(); }
    auto field_end() { return getFieldDecls().end(); }

    /// Struct declarations have no result parameters.
    ArrayRef<ParamDeclAttr> getResultParams() { return {}; }

    TypeConvention getConvention() { return getProperties().convention; }
    void setConvention(TypeConvention value) {
      getProperties().convention = value;
    }
    bool isRegisterPassable() {
      return getConvention() != TypeConvention::MemoryOnly;
    }
    bool isRegisterPassableTrivial() {
      return getConvention() == TypeConvention::RegisterPassableTrivial;
    }

    /// Return the decl name of the struct.
    StringAttr getDeclName() { return getSymNameAttr(); }

    /// Get a bound reference to the struct type.
    DeclRefType bindReference(ArrayRef<TypedAttr> paramValues = {});
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
    StrAttr:$name, TypeAttr:$type, OptionalAttr<LIT_DocStringAttr>:$docString
  );
  let assemblyFormat = "custom<KeywordAsString>($name) attr-dict `:` $type";

  let builders = [
    OpBuilder<(ins "StringAttr":$name, "Type":$type)>,
    OpBuilder<(ins "const Twine &":$name, "Type":$type)>,
  ];

  let extraClassDeclaration = [{
    /// Given a struct base value type that may have bound parameters,
    /// substitute those bound parameters into our type and return the result.
    Type getReboundType(DeclRefType structSelfType);
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
      -> !kgen.declref<@MyStruct<T: dtype = type>>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands,
                       StringArrayAttr:$fields);
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
  let results = (outs PointerType:$result);
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
    lit.trait.decl @Trait<MT: regtype, T: !kgen.paramref<MT>> {
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
    KGEN_ParamDeclArrayAttr:$inputParams,
    TypeLineageArrayAttr:$parentTypes,
    OptionalAttr<TypeAttrOf<KGEN_SignatureType>>:$dtorSig,
    OptionalAttr<LIT_DocStringAttr>:$docString
  );

  let regions = (region SizedRegion<1>:$fields);

  let assemblyFormat = [{
    $sym_name ``
    custom<StructParameterSpec>($inputParams, $signature, $parentTypes)
    attr-dict-with-keyword $fields
  }];

  let extraClassDeclaration = [{
    /// Trait declarations have no result parameters.
    ArrayRef<ParamDeclAttr> getResultParams() { return {}; }

    /// Get a bound reference to the Trait type.
    Type bindReference();
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
      %0 = kgen.call @maybeRaises() : () throws -> !kgen.declref<@Int>
      lit.try.yield
    } except (%err: !kgen.declref<@Error>) {
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
    DefaultValuedOptionalAttr<BoolAttr, "false">:$suppressWarnings);
  let regions = (region SizedRegion<1>:$tryRegion, SizedRegion<1>:$exceptRegion,
                        SizedRegion<1>:$elseRegion,
                        VariadicRegion<SizedRegion<1>>:$finallyRegions);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    (`->` type($results)^)? $tryRegion `except`
    custom<ExceptRegion>($exceptRegion) `else` $elseRegion
    (`finally` $finallyRegions^)? attr-dict
  }];

  let builders = [
    OpBuilder<(ins CArg<"bool", "false">:$suppressWarnings),
      [{ build($_builder, $_state, {}, suppressWarnings, 1); }]>,
  ];

  let extraClassDeclaration = [{
    /// Get the finally region, assuming it is present.
    Region &getFinallyRegion() { return getFinallyRegions().front(); }

    /// Return true if the finally block is trivial.
    bool hasTrivialFinally();
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
    KGEN_ParamDeclAttr:$paramDecl, TypedAttrInterface:$value,
    OptionalAttr<LIT_DocStringAttr>:$docString
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
      build($_builder, $_state, decl, value, /*docString=*/{});
    }]>
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AliasForwardDeclOp
//===----------------------------------------------------------------------===//

// TODO(LLVM Issue #59877): Why isn't this in BuiltinAttributes.td?
def Builtin_LocationAttr : Builtin_Attr<"Location", []> {}

def LIT_AliasForwardDeclOp : LITOp<"alias.fwd_decl", [ASTDeclInterface]> {
  let summary = "`alias x: T` declaration for a callee return parameter";

  let description = [{
    This op represents an alias decl after its signature has been resolved and
    it was found to have no initializer.  It is therefore a forward declaration
    of a return parameter from a function call.

    `resultParamLoc` is filled in with the location of the result parameter that
    ultimately actually defines this.
  }];

  let arguments = (ins
    StrAttr:$name, TypeAttr:$type,
    OptionalAttr<Builtin_LocationAttr>:$resultParamLoc,
    OptionalAttr<LIT_DocStringAttr>:$docString
  );
  let assemblyFormat = "$name `:` $type attr-dict";
}

//===----------------------------------------------------------------------===//
// LetRegDeclOp
//===----------------------------------------------------------------------===//

def LIT_LetRegDeclOp : LITOp<"letreg.decl", [
    ASTDeclInterface,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    Pure, AllTypesMatch<["value", "result"]>
  ]> {
  let summary = "Register-primary local `let` declaration";
  let description = [{
    This declaration is used for register primary values assigned to a let
    declaration which have an inline initializer.  It does not expose an address
    for the value.

    This op consumes the ownership of its input value, and takes ownership.
  }];

  let arguments = (ins StrAttr:$name, AnyType:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$name `=` $value `:` type($result) attr-dict";

  let hasFolder = 1;

  let builders = [
    // Create a 'let' without a value, this is malformed until specified.
    OpBuilder<(ins "Type":$type, "StringAttr":$name)>,
  ];
}

//===----------------------------------------------------------------------===//
// VarLetDeclOp
//===----------------------------------------------------------------------===//

def LIT_VarLetDeclOp : LITOp<"varlet.decl", [
    ASTDeclInterface,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    DeclareOpInterfaceMethods<ParamOpInterface>]>, OneParamDecl {
  let summary = "Addressible local let or var declaration";
  let description = [{
    This declaration is used for var and let declarations with an exposed
    address.  `kind` encodes how this declaration was created, either via
    explicit `let` or `var` expressions, implicitly declared, or internally
    synthesized (which suppresses some warnings).  This type defines a lifetime
    and returns a mutable reference to its result in that lifetime.

    Example:

    ```mlir
    %foo = lit.varlet.decl "foo" var : !lit.ref<index, foo>
    ```
  }];

  let arguments = (ins StrAttr:$name, LIT_VarLetDeclKindAttr:$kind,
                       KGEN_ParamDeclAttr:$paramDecl,
                       OptionalAttr<LIT_DocStringAttr>:$docString);
  let results = (outs LIT_RefType:$result);

  let assemblyFormat = [{
    $name $kind `:` custom<VarLetDeclType>(type($result), $paramDecl) attr-dict
  }];

  let builders = [
    OpBuilder<(ins "Type":$elementType, "StringRef":$name,
                   "StringRef":$lifetimeName, "VarLetDeclKind":$kind)>
  ];
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
    lit.globalvar.decl @global_var : !kgen.declref<@String> isVar {
      %0 = lit.globalvar.ref @global_var : @String
      kgen.call @String::@__init__(%0)
    }, {
      %0 = lit.globalvar.ref @global_var : @String
      kgen.call @String::@__del__(%0)
    }
    ```

    `isVar` should be set if the global was declared with a `var` declaration
    and is mutable.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttr:$type,
    UnitAttr:$isVar,
    KGEN_ExportKindAttr:$exportKind,
    KGEN_DecoratorsAttr:$decorators,
    OptionalAttr<StrAttr>:$linkageName,
    OptionalAttr<LIT_DocStringAttr>:$docString
  );
  let regions = (region SizedRegion<1>:$ctor, SizedRegion<1>:$dtor);

  let assemblyFormat = [{
    `` custom<SymbolExport>($exportKind) $sym_name `:` $type
    attr-dict-with-keyword custom<OptionalDecorators>($decorators)
    (`isVar` $isVar^)?  custom<NoArgRegion>($ctor) `,`
    custom<NoArgRegion>($dtor)
  }];

  let extraClassDeclaration = [{
    /// Return the decl name of the variable.
    StringAttr getDeclName() { return getSymNameAttr(); }
  }];

  let builders = [
    OpBuilder<(ins "StringRef":$sym_name, "Type":$type, "bool":$isVar), [{
      build($_builder, $_state, sym_name, type, isVar, ExportKind::NotExported,
            DecoratorsAttr::get(type.getContext(), {}), {}, {});
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

    Example:

    ```mlir
    %0 = lit.globalvar.ref @global_var : <@Int>
    ```
  }];

  let arguments = (ins SymbolRefAttr:$global);
  let results = (outs PointerType:$result);

  let builders = [
    OpBuilder<(ins "GlobalVarDeclOp":$op), [{
      build($_builder, $_state, PointerType::get(op.getType()),
            getFullyResolvedSymbolRef(op));
    }]>,
  ];

  let assemblyFormat = "$global attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// AsyncCallOp
//===----------------------------------------------------------------------===//

def LIT_AsyncCallOp : LITOp<"async.call", [
    DeclareOpInterfaceMethods<KGEN_CallOpInterface>,
    DeclareOpInterfaceMethods<KGEN_GeneratorUserOpInterface>,
    DeclareOpInterfaceMethods<ParamOpInterface, ["isImplicitlyParametric"]>]>,
    CallLikeOperation {
  let summary = "Call to an async function.";
  let description = [{
    The `lit.async.call` operation calls an `async` parametric callable. It
    returns a `!pop.coroutine` of the callable's result types.

    Example:

    ```mlir
    // `%coro` has type `!pop.coroutine<index>`.
    %coro = lit.async.call<() -> bar>[:<() -> index>() -> index: @foo]()
    ```
  }];

  let arguments = (ins
    AttrOfType<KGEN_SignatureType>:$callee,
    KGEN_ParamDeclArrayAttr:$paramDecls,
    KGEN_ParameterExprArrayAttr:$implicitLifetimes,
    Variadic<AnyType>:$operands
  );
  let results = (outs POP_CoroutineType:$coroutine);

  let assemblyFormat = [{
    `` custom<ParametricCallee>($callee, $paramDecls) ``
    custom<LifetimeParams>($implicitLifetimes) ``
    custom<AsyncCallOpTypes>(type($operands), type($coroutine), ref($callee),
                             ref($implicitLifetimes))
     `(` $operands `)` attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AsyncExecuteOp
//===----------------------------------------------------------------------===//

def LIT_AsyncExecuteOp : LITOp<"async.execute", [SingleBlock,
    DeclareOpInterfaceMethods<FunctionLike>,
    DebugInfo_InlinedSubprogramScoped]> {
  let summary = "Execute an async region.";
  let description = [{
    The `lit.async.execute` operation asynchronously executes a region and
    returns a coroutine handle to the region. This operation can represent
    an inlined post-elaboration async function.

    Example:

    ```mlir
    %0 = kgen.param.constant: i32 = <3>
    %coroHdl = lit.async.execute <() -> (i32, i64)> {
      %1 = kgen.param.constant: i64 = <5>
      kgen.return %0, %1 : i32, i64
    }
    ```
  }];

  let arguments = (ins OptionalAttr<Builtin_LocationAttr>:$callLoc);
  let results = (outs Res<POP_CoroutineType, "", [MemAlloc]>:$coroutine);
  let regions = (region SizedRegion<1>:$bodyRegion);
  let assemblyFormat = [{
    type($coroutine) $bodyRegion attr-dict (`callLoc` `(` $callLoc^ `)`)?
  }];
}

//===----------------------------------------------------------------------===//
// ParamReturnOp
//===----------------------------------------------------------------------===//

def LIT_ParamReturnOp : LITOp<"param_return"> {
  let summary = "Bind result parameter values.";
  let description = [{
    The `lit.param_return` corresponds to a lexical `param_return` in Lit. It
    binds values to the result parameters of the enclosing function. Like
    `lit.return`, this operation is not a terminator.

    Example:

    ```mlir
    lit.func @foo<() -> r0: dtype, r1>() {
      lit.param_return<:dtype si32, 4>
      lit.end_func
    }
    ```
  }];

  let arguments = (ins KGEN_ParameterExprArrayAttr:$parameters);
  let assemblyFormat = "`` custom<ParameterValues>($parameters) attr-dict";

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
      %1 = lit.struct.create(a=%0) : (index) -> !kgen.declref<@Error>
      %2 = kgen.variant.create %1 :
        !kgen.declref<@Error> -> !kgen.variant<@Error, none>
      lit.error_return %2 : !kgen.variant<@Error, none>
    }
    ```
  }];

  let arguments = (ins KGEN_VariantType:$variant);
  let assemblyFormat = "$variant attr-dict `:` type($variant)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// RaiseOp
//===----------------------------------------------------------------------===//

def LIT_RaiseOp : LITOp<"raise"> {
  let summary = "Lexical raise statement.";
  let description = [{
    The `lit.raise` operation is not a terminator. It corresponds to a raise
    statement in Lit. It takes a single operand: the error to raise. This
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
// LoopConditionOp
//===----------------------------------------------------------------------===//

def LIT_LoopConditionOp : LITOp<"loop.condition", [
    ParentOneOf<["LoopOp"]>, Terminator]> {
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
    ParentOneOf<["LoopOp"]>, Terminator]> {

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
    ParentOneOf<["LoopOp"]>, Terminator]> {
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
// StoreBorrowOp
//===----------------------------------------------------------------------===//

def LIT_StoreBorrowOp : LITOp<"store.borrow",
                              [PointerElementType<"ptr", "arg">]> {
  let summary = "borrow register-passable value via memcpy";
  let description = [{
    The `lit.store.borrow` operation is a special store that creates a borrow
    of a register-passable value that lives in-memory.

    Example:

    ```mlir
    lit.store.borrow %arg0, %alloc : <index>
    ```
  }];

  let arguments = (ins AnyType:$arg, Arg<PointerType, "", [MemWrite]>:$ptr);
  let assemblyFormat = "$arg `,` $ptr attr-dict `:` type($ptr)";
}

//===----------------------------------------------------------------------===//
// LoadConsumeOp
//===----------------------------------------------------------------------===//

def LIT_LoadConsumeOp : LITOp<"load.consume",
                              [PointerElementType<"ptr", "result">]> {
  let summary = "consuming pointer load operation";
  let description = [{
    Loads the data at the given pointer like `pop.load`, but consumes the
    ownership of the memory being accessed. This is used when performing MRValue
    to SRValue conversion (i.e. when consuming a `var` of `@register_passable`
    type), and is lowered to `pop.load`.

    Example:

    ```mlir
    %0 = lit.load.consume %ptr0 : !kgen.pointer<scalar<f32>>
    ```
  }];

  let arguments = (ins Arg<PointerType, "", [MemRead]>:$ptr);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$ptr attr-dict `:` qualified(type($ptr))";
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
// OwnershipDefLValueOp
//===----------------------------------------------------------------------===//

def LIT_OwnershipDefLValueOp : LITOp<"ownership.def_lvalue"> {
  let summary = "A definition of a memory object, marking implicit mutation";
  let description = [{
    This op looks like a def of the pointee at the specified address, which
    allows CheckLifetimes to know that it must be mutable at this point.  This
    operation is a noop that is removed by the LowerLIT pass.

    ```mlir
    lit.ownership.def_lvalue %0 : !somestring
    ```
  }];
  let arguments = (ins AnyType:$value);
  let results = (outs);
  let assemblyFormat = "$value `:` type($value) attr-dict";
}

//===----------------------------------------------------------------------===//
// OwnershipEndLifetimeOp
//===----------------------------------------------------------------------===//

def LIT_OwnershipEndLifetimeOp : LITOp<"ownership.end_lifetime",
                                       [AllTypesMatch<["value", "newValue"]>]> {
  let summary = "End the lifetime of a tracked value and start a new lifetime";
  let description = [{
    This op is spelled `x^` when `x` is a lifetime-tracked value.  It indicates
    that the lifetime of `x` ends here and a new value is created.

    This works with both register and memory values.

    ```mlir
    %1 = lit.ownership.end_lifetime %0 : !somestring
    ```
  }];

  let arguments = (ins AnyType:$value, BoolAttr:$isReg);
  let results = (outs AnyType:$newValue);
  let assemblyFormat = "$value `:` type($value) attr-dict";
}


//===----------------------------------------------------------------------===//
// OwnershipMakePointerLValue
//===----------------------------------------------------------------------===//

def LIT_OwnershipMakePointerLValue : LITOp<"ownership.make_pointer_lvalue",
                                        [AllTypesMatch<["ptr", "valuePtr"]>]> {
  let summary = "Project a pointer address as an tracked LValue";
  let description = [{
    This op projects a !kgen.pointer value as an LValue, whose value is
    initialized on entry (or not) based on the first property, and initialized
    after last use (or not) depending on the second one.

    This works with memory values.

    ```mlir
    %1 = lit.ownership.make_pointer_lvalue %0 : !kgen.pointer<@MyStruct>
    ```
  }];

  let arguments = (ins PointerOf<KGEN_DeclRefType>:$ptr,
                       BoolAttr:$liveOnEntry,
                       BoolAttr:$liveOnExit);
  let results = (outs PointerOf<KGEN_DeclRefType>:$valuePtr);
  let assemblyFormat = [{
    $ptr `live_on_entry``=`$liveOnEntry`,` `exit``=`$liveOnExit
    `:` type($ptr) attr-dict
  }];
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

  let arguments = (ins PointerType:$value);
  let results = (outs);
  let assemblyFormat = "$value `:` type($value) attr-dict";
}

//===----------------------------------------------------------------------===//
// HandleVariantOp
//===----------------------------------------------------------------------===//

def LIT_HandleVariantOp : LITOp<"handle_variant",
    [DeclareOpInterfaceMethods<ControlFlowNode>]> {
  let summary = "handle the return of a throwing function";
  let description = [{
    The `lit.handle_variant` op represents an if statement paired with a set of
    conditionally initialized values. The success region of the
    `lit.handle_variant` op is executed upon a successful initialization of all
    values. The error region of the `lit.handle_variant` is executed otherwise.

    The result value is an owned value that is consumed by the enclosing
    lit.yield operation.

    Example:

    ```mlir
    %0 = kgen.call @throwing_callee(%y) :
      (!kgen.pointer<@MyStruct> byref_result) throws ->
      !kgen.variant<@Error, none>
    %1 = lit.handle_variant %0, %y : (!kgen.variant<@Error, none>,
      !kgen.pointer<@MyStruct>) -> !kgen.none {
      %7 = kgen.variant.get %0 :
        !kgen.variant<@Error, none> as !kgen.none
      lit.yield %7 : !kgen.none
    } else {
      %8 = kgen.variant.get %0, 0 : <@Error, none>
      %9 = kgen.variant.create %8, 0 : <@Error, none>
  	  kgen.return %9 : !kgen.variant<@Error, none>
    }
    ```
  }];
  let arguments = (ins
    KGEN_VariantType:$variant,
    Variadic<AnyType>:$maybeInitializedValues
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$successRegion,
                 SizedRegion<1>:$errorRegion);

  let assemblyFormat = [{
    $variant (`,` $maybeInitializedValues^)? `:`
    functional-type(operands, results) $successRegion `else` $errorRegion
    attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def LIT_YieldOp : LITOp<"yield", [Pure,
  Terminator, HasParent<"LIT::HandleVariantOp">,
  DeclareOpInterfaceMethods<ControlFlowTerminator>]> {
  let summary = "conditional yield operation";
  let description = [{
    The `lit.yield` operation is a control-flow terminator that transfers
    control-flow to after the parent `handle_variant` operation. The operation
    expects its operand types to match the types of the results of that
    operation, if there are any.

    This operation consumes ownership of the SSA value, which is returned
    through the enclosing HandleVariantOp.

    Example:

    ```mlir
    %1 = lit.handle_variant %0, %x : (!kgen.variant<@Error, none>,
      !kgen.pointer<@MyStruct>) -> !kgen.none {
      %7 = kgen.variant.get %0, 1 : <@Error, none>
      lit.yield %7 : !kgen.none
    } else {
      %8 = kgen.variant.get %0, 0 : <@Error, none>
      lit.try.raise %8 : !kgen.declref<@Error>
    }
    ```
    This operation can always be marked pure because the control-flow edge from
    this operation leads only to after the parent operation.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>];
  let assemblyFormat = "attr-dict ($operands `:` type($operands)^)?";
}

//===----------------------------------------------------------------------===//
// RefToPointerOp
//===----------------------------------------------------------------------===//

// Given a `!lit.ref<X, life>`, return `!kgen.pointer<X>`.
class RefPointerType<string lhs, string rhs> : TypesMatchWith<
    "reference as pointer type", lhs, rhs,
    "$_self.cast<RefType>().getAsPointerType()">;

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
// RefStructGEROp
//===----------------------------------------------------------------------===//

def LIT_RefStructGEROp : LITOp<"ref.struct.ger", [Pure,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Get a pointer to a struct field.";
  let description = [{
    The `lit.ref.struct.ger` operation computes a reference to a struct field by
    name from a parametric struct.  The result has the same lifetime as the
    source value.

    Example:

    ```mlir
    lit.struct.decl @MyStruct<T: dtype> {
      lit.struct.field a : !pop.scalar<T>
    }

    %0 = lit.ref.struct.ger %struct[a] : !lit.ref<mut @MyStruct<...>, life>
    ```
  }];

  let arguments = (ins StrAttr:$field, RefOf<KGEN_DeclRefType>:$container);
  let results = (outs LIT_RefType:$result);
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
// RefLoadOp
//===----------------------------------------------------------------------===//

def LIT_RefLoadOp : LITOp<"ref.load", [ReferenceElementType<"ref", "result">]> {
  let summary = "reference load operation";
  let description = [{
    Loads the data at the given reference. Returns a scalar of the element type.
    The reference may either be mutable or immutable.

    Example:

    ```mlir
    %0 = lit.ref.load %ref0 : !lit.ref<scalar<f32>, lifetime>
    %1 = lit.ref.load %ref1 : !lit.ref<simd<4, type>, lifetime>
    %2 = lit.ref.load %ref0 : !lit.ref<mut scalar<f32>, lifetime>
    ```
  }];

  let arguments = (ins LIT_RefType:$ref);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$ref attr-dict `:` type($ref)";
}

//===----------------------------------------------------------------------===//
// RefStoreOp
//===----------------------------------------------------------------------===//

def LIT_RefStoreOp : LITOp<"ref.store", [ReferenceElementType<"ref", "arg">]> {
  let summary = "Reference store operation";
  let description = [{
    Store the given value to the given pointer of the same element type.  The
    reference must be mutable.

    Example:

    ```mlir
    lit.ref.store %0, %ref0 : !lit.ref<mut scalar<f32>, lifetime>
    lit.ref.store %1, %ref1 : !lit.ref<mut simd<4, type>, lifetime>
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

  let arguments = (ins
    DefaultValuedAttr<HLCF_UnrollLevelAttr,
                      "HLCF::UnrollLevel::none()">:$unrollLevel
  );

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
    OpBuilder<(ins "HLCF::UnrollLevel":$level), [{
      build($_builder, $_state, {}, level);
    }]>
  ];
}
