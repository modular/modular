//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares types for the LIT dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LITDIALECT_LITTYPES_H
#define KGEN_LITDIALECT_LITTYPES_H

#include "KGEN/KGENDialect/KGENTypes.h"

namespace M::KGEN {
class ParameterExprArrayAttr;
namespace LIT {
class PogListAttr;
class FnMetadataAttr;
class RefPackType;
class SymbolAttr;
enum class PassingKind : uint32_t;
enum class VariadicKind : uint32_t;
} // namespace LIT
} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "KGEN/LITDialect/LITTypes.h.inc"

namespace M::KGEN::LIT {

//===----------------------------------------------------------------------===//
// LITGeneratorType
//===----------------------------------------------------------------------===//

class LITGeneratorType : public GeneratorType {
public:
  using GeneratorType::GeneratorType;
  LITGeneratorType(GeneratorType gen);

  /// Get the generator metadata.
  PogListAttr getMetadata();
  PogListAttr getParamListAttrs();

  /// Return the name for the parameter at the specified index.
  StringAttr getParamName(size_t idx);
};

//===----------------------------------------------------------------------===//
// FnType
//===----------------------------------------------------------------------===//

class FnType : public FuncType {
public:
  using FuncType::FuncType;
  FnType(FuncType sig);

  /// Get the signature metadata.
  FnMetadataAttr getMetadata();

  /// Get the argument list metadata.
  PogListAttr getArgListAttrs();

  /// Return the name for the argument at the specified index.
  StringAttr getArgName(size_t idx);

  /// Get the origin set of the capture lifetimes.
  TypedAttr getCaptureOrigins();

  /// Get whether nested lifetimes are excluded from exclusivity checking.
  bool getIsNestedOriginExclusivityCheckingDisabled();

  /// Get the function's default positional arguments.
  ArrayRef<TypedAttr> getDefaultPosArgs();

  /// Get the function's default keyword-only arguments.
  ArrayRef<TypedAttr> getDefaultKwOnlyArgs();

  /// Get the number of implicit origin decls this function type carries.
  size_t getNumImplicitOriginDecls();

  /// LIT-level signatures always have one result type.
  Type getResultType() { return getResults().front(); }

  /// Get the user result type of the signature.
  Type getUserResultType();

  /// Returns true if the argument at this index is any vararg or a pack.
  bool isAnyVarArg(size_t index);

  /// Returns true if the argument at this index is a positional vararg.
  bool isPosVarArg(size_t index);

  /// For a PosVarArg, return the declared ArgConvention of the elements. For
  /// example: fn x(inout *args: Int) is declared 'inout'.
  ArgConvention getPosVarArgConvention(size_t index);

  /// Returns true if the argument at this index is a keyword vararg.
  bool isKwVarArg(size_t index);

  /// Returns true if the argument at this index is a pack vararg.
  bool isPack(size_t index);

  /// For a PackVarArg, return the declared ArgConvention of the elements. For
  /// example: fn x[*Ts: AnyType](mut *pack: *Ts) is declared 'mut'.
  ArgConvention getPackVarArgConvention(size_t index);

  /// If the specified argument is a variadic pack, return the VariadicPack.
  Type getIfVariadicPack(size_t index);

  /// Returns true if the signature has has pack arguments.
  bool hasPackVarArgs();

  /// Returns true if the signature has keyword variadic arguments.
  bool hasKwVarArgs();

  /// Return the offset of the error slot argument from the back of the argument
  /// list, if the signature is raising.
  unsigned getErrorSlotOffset();

  /// Substitute the specified implicit origin references into the specified
  /// type, replacing them with `values` if they are at depth 0, or decrementing
  /// their depth if not.  This returns the resultant FunctionType on success,
  /// and invokes 'emitError'+returns null on error.
  FunctionType substituteImplicitOriginsIntoValues(
      ArrayRef<TypedAttr> values, function_ref<InFlightDiagnostic()> emitError);

  /// Return this signature with the specified capture lifetimes.
  FnType getWithCaptureOrigins(TypedAttr lifetimes);

  /// A `FuncType` is a LIT signature if it contains function metadata.
  static bool classof(FuncType type);
  static bool classof(Type type);

  static FnType get(MLIRContext *ctx, TypeRange inputs, TypeRange results,
                    size_t numImplicitOriginDecls);
};

//===----------------------------------------------------------------------===//
// FnTypeGeneratorType
//===----------------------------------------------------------------------===//

class FnTypeGeneratorType : public FuncTypeGeneratorType {
public:
  using FuncTypeGeneratorType::FuncTypeGeneratorType;
  FnTypeGeneratorType(LITGeneratorType gen);
  FnTypeGeneratorType(FuncTypeGeneratorType gen);

  FnType getBody();

  //===--------------------------------------------------------------------===//
  // Acting as a LITGeneratorType
  //===--------------------------------------------------------------------===//

  PogListAttr getMetadata();
  PogListAttr getParamListAttrs();

  /// Return the name for the parameter at the specified index.
  StringAttr getParamName(size_t idx);

  /// Reconstruct the generator using a list of named input parameters and info
  /// about what kind of variadic they are. These parameters are prepended to
  /// the current signature and references are remapped to index references.
  static FnTypeGeneratorType
  prependParams(FnTypeGeneratorType sig, ArrayRef<ParamDeclAttr> parentParams,
                ArrayRef<VariadicKind> parentVariadics);

  //===--------------------------------------------------------------------===//
  // Acting as a FnType
  //===--------------------------------------------------------------------===//

  FunctionType getValues() { return getBody().getValues(); }

  llvm::ArrayRef<ArgConvention> getArgConventions() {
    return getBody().getArgConventions();
  }
  FnEffects getFnEffects() { return getBody().getFnEffects(); }

  /// Helper to return the argument and result types.
  ArrayRef<Type> getArguments() { return getBody().getArguments(); }
  ArrayRef<Type> getResults() { return getBody().getResults(); }

  bool hasMemoryOnlyResult() { return getBody().hasMemoryOnlyResult(); }

  bool isThrows() { return getFnEffects().isThrows(); }
  bool isAsync() { return getFnEffects().isAsync(); }
  bool isCapturing() { return getFnEffects().isCapturing(); }
  bool isEscaping() { return getFnEffects().isEscaping(); }
  bool isRefResult() { return getFnEffects().isRefResult(); }

  /// Return the convention for the specified value argument.
  ArgConvention getArgConvention(size_t inputNo) {
    return getArgConventions()[inputNo];
  }

  size_t getNumArguments() { return getArguments().size(); }
  size_t getNumResults() { return getResults().size(); }

  size_t getNumAsyncReturnSlots() { return getBody().getNumAsyncReturnSlots(); }

  /// Get the signature metadata.
  FnMetadataAttr getFnMetadata();

  /// Get the argument list metadata.
  PogListAttr getArgListAttrs();

  /// Return the name for the argument at the specified index.
  StringAttr getArgName(size_t idx);

  /// Get the origin set of the capture lifetimes.
  TypedAttr getCaptureOrigins();

  /// Get whether nested lifetimes are excluded from exclusivity checking.
  bool getIsNestedOriginExclusivityCheckingDisabled();

  /// Get the function's default positional arguments.
  ArrayRef<TypedAttr> getDefaultPosArgs();

  /// Get the function's default keyword-only arguments.
  ArrayRef<TypedAttr> getDefaultKwOnlyArgs();

  /// Get the number of implicit origin decls this function type carries.
  size_t getNumImplicitOriginDecls();

  /// LIT-level signatures always have one result type.
  Type getResultType() { return getBody().getResults().front(); }

  /// Get the user result type of the signature.
  Type getUserResultType();

  /// Returns true if the argument at this index is any vararg or a pack.
  bool isAnyVarArg(size_t index);

  /// Returns true if the argument at this index is a positional vararg.
  bool isPosVarArg(size_t index);

  /// For a PosVarArg, return the declared ArgConvention of the elements. For
  /// example: fn x(inout *args: Int) is declared 'inout'.
  ArgConvention getPosVarArgConvention(size_t index);

  /// Returns true if the argument at this index is a keyword vararg.
  bool isKwVarArg(size_t index);

  /// Returns true if the argument at this index is a pack vararg.
  bool isPack(size_t index);

  /// For a PackVarArg, return the declared ArgConvention of the elements. For
  /// example: fn x[*Ts: AnyType](inout *pack: *Ts) is declared 'inout'.
  ArgConvention getPackVarArgConvention(size_t index);

  /// If the specified argument is a variadic pack, return the VariadicPack.
  Type getIfVariadicPack(size_t index);

  /// Returns true if the signature has has pack arguments.
  bool hasPackVarArgs();

  /// Returns the index of the pack variadic arg, or std::nullopt if none.
  std::optional<size_t> findPackVarArgIndex();

  /// Returns true if the signature has keyword variadic arguments.
  bool hasKwVarArgs();

  /// Return the offset of the error slot argument from the back of the argument
  /// list, if the signature is raising.
  unsigned getErrorSlotOffset();

  /// Substitute the specified implicit origin references into the specified
  /// type, replacing them with `values` if they are at depth 0, or decrementing
  /// their depth if not.  This returns the resultant FunctionType on success,
  /// and invokes 'emitError'+returns null on error.
  FunctionType substituteImplicitOriginsIntoValues(
      ArrayRef<TypedAttr> values, function_ref<InFlightDiagnostic()> emitError);

  /// Return this signature with the specified capture lifetimes.
  FnTypeGeneratorType getWithCaptureOrigins(TypedAttr lifetimes);

  /// This method replaces direct uses of NAMED implicit origin declarations
  /// with index-based references corresponding to the signature.  lifetimeDecls
  /// specifies the names of the implicit origin decls.
  FnTypeGeneratorType
  replaceImplicitOriginsWithIndexes(ArrayRef<ParamDeclAttr> lifetimeDecls);

  /// This method replaces direct uses of NAMED implicit origin declarations
  /// with index-based references.  lifetimeDecls specifies the names of the
  /// implicit origin decls to replace.
  static Type replaceImplicitOriginsWithIndexes(
      Type type, ArrayRef<ParamDeclAttr> lifetimeDecls, size_t indexOffset = 0);

  static bool classof(FuncTypeGeneratorType type);
  static bool classof(Type type);
};

//===----------------------------------------------------------------------===//
// MetaTypeOf
//===----------------------------------------------------------------------===//

template <typename T>
class MetaTypeOf : public MetaType {
public:
  using MetaType::MetaType;

  static MetaTypeOf<T> get(T type) {
    return llvm::cast<MetaTypeOf<T>>(MetaType::get(type));
  }

  T getType() const { return llvm::cast<T>(MetaType::getType()); }

  static bool classof(Type type) {
    auto metatype = llvm::dyn_cast<MetaType>(type);
    return metatype && llvm::isa_and_nonnull<T>(metatype.getType());
  }
};

class StructMetaType : public MetaTypeOf<LIT::StructType> {
private:
  using Base = MetaTypeOf<LIT::StructType>;

public:
  using Base::classof;
  using Base::get;
  using Base::MetaTypeOf;

  StructMetaType(Base base) : Base(base) {}

  SymbolRefAttr getSymbol() const;
  TypeSignatureType getSignature() const;
  ArrayRef<TypedAttr> getParamValues() const;

  /// Bind parameter values to the metatype, returning a new metatype.
  /// Expects the number of values to match the number of param values. Only
  /// positions that are currently unbound can be updated.
  StructMetaType bindAll(ArrayRef<TypedAttr> values) const;

  /// Bind parameter values to the metatype, returning a new metatype.
  /// Expects the number of values to match the number of unbound parameters
  /// in the current param values list.
  StructMetaType bindUnbound(ArrayRef<TypedAttr> values) const;
};

//===----------------------------------------------------------------------===//
// Type Utilities
//===----------------------------------------------------------------------===//

/// Returns the user-defined result type of a signature, looking through
/// implicit memory results and stripping off the variant from error throwing
/// results if needed.
Type getSignatureUserResultType(FnTypeGeneratorType sigType,
                                ArrayRef<Type> argTypes, Type resultType);

/// The Lit parser and KGEN have different semantics for binding function
/// argument and result types. The parser will evaluate 'apply' expressions, but
/// KGEN does not since it cannot always have access to a symbol table.
/// Specialize a signature type while rebinding the input parameter values to
/// the expected input parameter types.
std::pair<FnTypeGeneratorType, ParameterExprArrayAttr>
getUnboundSpecializedSignature(FnTypeGeneratorType type,
                               ParameterExprArrayAttr bindings,
                               ParameterEvaluationContext *evalContext);

} // namespace M::KGEN::LIT

#endif // KGEN_LITDIALECT_LITTYPES_H
