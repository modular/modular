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
#include "KGEN/LITDialect/LITAttrs.h"

namespace M::KGEN {
class ConstraintAttr;
class ParameterExprArrayAttr;
namespace LIT {
class RefPackType;
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

  /// Get the number of implicit origin decls this function type carries.
  size_t getNumImplicitOriginDecls();

  /// LIT-level signatures always have one result type.
  Type getResultType() { return getResults().front(); }

  /// Get the user result type of the signature.
  Type getUserResultType();

  /// Get the user thrown type for a raising function.
  Type getUserThrownType();

  /// Returns true if the argument at this index is any vararg or a pack.
  bool isAnyVarArg(size_t index);

  /// Returns true if the argument at this index is a positional vararg.
  bool isPosVarArg(size_t index);

  /// For a PosVarArg/PackVarArg, return the declared ArgConvention of the
  /// elements. For example: def x(mut *args: Int) is declared 'mut'.
  ArgConvention getVariadicConvention(size_t index);

  /// Returns true if the argument at this index is a keyword vararg.
  bool isKwVarArg(size_t index);

  /// Returns true if the argument at this index is a pack vararg.
  bool isPack(size_t index);

  /// If the specified argument is a variadic list/pack, return the
  /// VariadicList/VariadicPack, stripping RefType, otherwise return null.
  Type getIfVariadicListOrPack(size_t index);

  /// Returns the index of the pack variadic arg, or std::nullopt if none.
  std::optional<size_t> findPackVarArgIndex();

  /// Returns true if the signature has keyword variadic arguments.
  bool hasKwVarArgs();

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
// FnTypeWrapperGeneratorType
//===----------------------------------------------------------------------===//

// A CRTP base class for FnTypeGeneratorType and FnLiteralTypeGeneratorType that
// wraps a FnType.
template <typename SubClass, typename BaseClass>
class FnTypeWrapperGeneratorType : public BaseClass {
public:
  using BaseT = BaseClass;
  using BaseClass::BaseClass;

  FnType getBodyFnType() {
    return static_cast<SubClass *>(this)->getBodyFnType();
  }
  PogListAttr getMetadata() {
    return static_cast<SubClass *>(this)->getMetadata();
  }

  //===--------------------------------------------------------------------===//
  // Acting as a LITGeneratorType
  //===--------------------------------------------------------------------===//

  PogListAttr getParamListAttrs() { return getMetadata(); }
  StringAttr getParamName(size_t idx) { return getMetadata().getName(idx); }

  //===--------------------------------------------------------------------===//
  // Acting as a FnType
  //===--------------------------------------------------------------------===//

  FunctionType getValues() { return getBodyFnType().getValues(); }

  llvm::ArrayRef<ArgConvention> getArgConventions() {
    return getBodyFnType().getArgConventions();
  }
  FnEffects getFnEffects() { return getBodyFnType().getFnEffects(); }

  /// Helper to return the argument and result types.
  ArrayRef<Type> getArguments() { return getBodyFnType().getArguments(); }
  Type getArgument(size_t i) { return getArguments()[i]; }
  ArrayRef<Type> getResults() { return getBodyFnType().getResults(); }

  bool hasMemoryOnlyResult() { return getBodyFnType().hasMemoryOnlyResult(); }

  bool isThrows() { return getFnEffects().isThrows(); }
  bool isAsync() { return getFnEffects().isAsync(); }
  bool isCapturing() { return getFnEffects().isCapturing(); }
  bool isEscaping() { return getFnEffects().isEscaping(); }
  bool isExtern() { return getFnEffects().isExtern(); }
  bool isRefResult() { return getFnEffects().isRefResult(); }
  bool isUnified() { return getFnEffects().isUnified(); }
  bool isRegisterPassable() { return getFnEffects().isRegisterPassable(); }

  /// Return the convention for the specified value argument.
  ArgConvention getArgConvention(size_t inputNo) {
    return getArgConventions()[inputNo];
  }

  size_t getNumArguments() { return getArguments().size(); }
  size_t getNumResults() { return getResults().size(); }

  size_t getNumAsyncReturnSlots() {
    return getBodyFnType().getNumAsyncReturnSlots();
  }

  /// Get the signature metadata.
  FnMetadataAttr getFnMetadata() { return getBodyFnType().getMetadata(); }

  /// Get the argument list metadata.
  PogListAttr getArgListAttrs() { return getBodyFnType().getArgListAttrs(); }

  /// Return the name for the argument at the specified index.
  StringAttr getArgName(size_t idx) { return getArgListAttrs().getName(idx); }

  /// Get the origin set of the capture lifetimes.
  TypedAttr getCaptureOrigins() { return getBodyFnType().getCaptureOrigins(); }

  /// Get whether nested lifetimes are excluded from exclusivity checking.
  bool getIsNestedOriginExclusivityCheckingDisabled() {
    return getBodyFnType().getIsNestedOriginExclusivityCheckingDisabled();
  }

  /// Get the number of implicit origin decls this function type carries.
  size_t getNumImplicitOriginDecls() {
    return getBodyFnType().getNumImplicitOriginDecls();
  }

  /// LIT-level signatures always have one result type.
  Type getResultType() { return getBodyFnType().getResults().front(); }

  /// Get the user result type of the signature.
  Type getUserResultType() { return getBodyFnType().getUserResultType(); }

  /// Get the user thrown type for a raising function.
  Type getUserThrownType() { return getBodyFnType().getUserThrownType(); }

  /// Returns true if the argument at this index is any vararg or a pack.
  bool isAnyVarArg(size_t index) { return getBodyFnType().isAnyVarArg(index); }

  /// Returns true if the argument at this index is a positional vararg.
  bool isPosVarArg(size_t index) { return getBodyFnType().isPosVarArg(index); }

  /// For a PosVarArg/PackVarArg, return the declared ArgConvention of the
  /// elements. For example: def x(mut *args: Int) is declared 'mut'.
  ArgConvention getVariadicConvention(size_t index) {
    return getBodyFnType().getVariadicConvention(index);
  }

  /// Returns true if the argument at this index is a keyword vararg.
  bool isKwVarArg(size_t index) { return getBodyFnType().isKwVarArg(index); }

  /// Returns true if the argument at this index is a pack vararg.
  bool isPack(size_t index) { return getBodyFnType().isPack(index); }

  /// If the specified argument is a variadic list/pack, return the
  /// VariadicList/VariadicPack, stripping RefType, otherwise return null.
  Type getIfVariadicListOrPack(size_t index) {
    return getBodyFnType().getIfVariadicListOrPack(index);
  }

  /// Returns the index of the pack variadic arg, or std::nullopt if none.
  std::optional<size_t> findPackVarArgIndex() {
    return getBodyFnType().findPackVarArgIndex();
  }

  /// Returns true if the signature has keyword variadic arguments.
  bool hasKwVarArgs() { return getBodyFnType().hasKwVarArgs(); }
};

//===----------------------------------------------------------------------===//
// FnTypeGeneratorType
//===----------------------------------------------------------------------===//
class FnTypeGeneratorType
    : public FnTypeWrapperGeneratorType<FnTypeGeneratorType,
                                        FuncTypeGeneratorType> {
public:
  using FnTypeWrapperGeneratorType::FnTypeWrapperGeneratorType;
  FnTypeGeneratorType(LITGeneratorType gen);
  FnTypeGeneratorType(FuncTypeGeneratorType gen);

  // CRTP for FnTypeWrapperGeneratorType
  FnType getBodyFnType() { return getBody(); }
  PogListAttr getMetadata();

  FnType getBody();

  /// Reconstruct the generator using a list of named input parameters and info
  /// about what kind of variadic they are. These parameters are prepended to
  /// the current signature and references are remapped to index references.
  static FnTypeGeneratorType
  prependParams(FnTypeGeneratorType sig, ArrayRef<ParamDeclAttr> parentParams,
                ArrayRef<StringAttr> paramNames = {});

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
// FnLiteralType
//===----------------------------------------------------------------------===//

class FnLiteralType : public FuncLiteralType {
public:
  using FuncLiteralType::FuncLiteralType;
  FnLiteralType(FuncLiteralType fnLiteral);

  FnType getFnType() const { return cast<FnType>(getFuncLiteral().getType()); }

  /// True if this literal's signature is an LIT `FnType`.
  static bool classof(FuncLiteralType type);
  static bool classof(Type type);

  static FnLiteralType get(TypedAttr funcLiteral);
};

//===----------------------------------------------------------------------===//
// FnLiteralTypeGeneratorType
//===----------------------------------------------------------------------===//

class FnLiteralTypeGeneratorType
    : public FnTypeWrapperGeneratorType<FnLiteralTypeGeneratorType,
                                        FuncLiteralTypeGeneratorType> {
public:
  using FnTypeWrapperGeneratorType::FnTypeWrapperGeneratorType;
  FnLiteralTypeGeneratorType(LITGeneratorType gen);
  FnLiteralTypeGeneratorType(FuncLiteralTypeGeneratorType gen);

  // CRTP for FnTypeWrapperGeneratorType
  FnType getBodyFnType() { return getBody().getFnType(); }
  PogListAttr getMetadata();

  FnLiteralType getBody();

  static bool classof(FuncLiteralTypeGeneratorType type);
  static bool classof(Type type);
};

//===----------------------------------------------------------------------===//
// FnOrFnLiteralTypeGeneratorType
//===----------------------------------------------------------------------===//

// A simple wrapper around smart variant of FnTypeGeneratorType and
// FnLiteralTypeGeneratorType.
class FnOrFnLiteralTypeGeneratorType
    : public FnTypeWrapperGeneratorType<
          FnOrFnLiteralTypeGeneratorType,
          SmartVariant<FnTypeGeneratorType, FnLiteralTypeGeneratorType>> {

  using VariantT = FnTypeWrapperGeneratorType::BaseT;
  VariantT getAsVariant() const { return static_cast<VariantT>(*this); }

public:
  using FnTypeWrapperGeneratorType::FnTypeWrapperGeneratorType;
  FnOrFnLiteralTypeGeneratorType(FnTypeGeneratorType gen)
      : FnTypeWrapperGeneratorType(gen) {}
  FnOrFnLiteralTypeGeneratorType(FnLiteralTypeGeneratorType gen)
      : FnTypeWrapperGeneratorType(gen) {}

  // Delegates to SmartVariant
  FnTypeGeneratorType getIfFnTypeGenerator() {
    return dyn_cast<FnTypeGeneratorType>(getAsVariant());
  }

  FnLiteralTypeGeneratorType getIfFnLiteralTypeGenerator() {
    return dyn_cast<FnLiteralTypeGeneratorType>(getAsVariant());
  }

  // CRTP for FnTypeWrapperGeneratorType
  FnType getBodyFnType() {
    if (auto fnGen = getIfFnTypeGenerator())
      return fnGen.getBodyFnType();
    return getIfFnLiteralTypeGenerator().getBodyFnType();
  }

  PogListAttr getMetadata() {
    if (auto fnGen = getIfFnTypeGenerator())
      return fnGen.getMetadata();
    return getIfFnLiteralTypeGenerator().getMetadata();
  }
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

class StructMetaMetaType : public MetaTypeOf<StructMetaType> {
private:
  using Base = MetaTypeOf<StructMetaType>;

public:
  using Base::classof;
  using Base::get;
  using Base::MetaTypeOf;

  StructMetaMetaType(Base base) : Base(base) {}
  SymbolRefAttr getSymbol() const;
  TypeSignatureType getSignature() const;
  ArrayRef<TypedAttr> getParamValues() const;

  /// Bind parameter values to the metatype, returning a new metatype.
  /// Expects the number of values to match the number of param values. Only
  /// positions that are currently unbound can be updated.
  StructMetaMetaType bindAll(ArrayRef<TypedAttr> values) const;

  /// Bind parameter values to the metatype, returning a new metatype.
  /// Expects the number of values to match the number of unbound parameters
  /// in the current param values list.
  StructMetaMetaType bindUnbound(ArrayRef<TypedAttr> values) const;
};

class FnLiteralTypeGeneratorMetaType
    : public MetaTypeOf<FnLiteralTypeGeneratorType> {
private:
  using Base = MetaTypeOf<FnLiteralTypeGeneratorType>;

public:
  using Base::classof;
  using Base::get;
  using Base::MetaTypeOf;

  FnLiteralTypeGeneratorMetaType(Base base) : Base(base) {}
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

/// If this specified operation is a call-like operation, return the
/// FnTypeGeneratorType for the callee, otherwise return null.
LIT::FnTypeGeneratorType getFnTypeFromCall(Operation &op);

} // namespace M::KGEN::LIT

#endif // KGEN_LITDIALECT_LITTYPES_H
