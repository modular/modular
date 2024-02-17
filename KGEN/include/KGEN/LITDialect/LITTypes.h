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

//===----------------------------------------------------------------------===//
// SignatureType
//===----------------------------------------------------------------------===//

namespace M::KGEN::LIT {
class ArgParamListAttr;
class FnMetadataAttr;
class SymbolAttr;
enum class PassingKind : uint32_t;

class LITSignatureType : public SignatureType {
public:
  using SignatureType::SignatureType;
  LITSignatureType(SignatureType sig);

  /// Get the signature metadata.
  FnMetadataAttr getMetadata();

  /// Get the argument list metadata.
  ArgParamListAttr getArgListAttrs();

  /// Get the parameter list metadata.
  ArgParamListAttr getParamListAttrs();

  /// Get the function input argument names.
  ArrayRef<StringAttr> getArgNames();

  /// Return the name for the specified argument.
  StringAttr getArgName(size_t inputNo);

  /// Get the function argument passing kinds (e.g. keyword-only).
  ArrayRef<PassingKind> getArgPassingKinds();

  /// Get the function's default positional arguments.
  ArrayRef<TypedAttr> getDefaultPosArgs();

  /// Get the function's default keyword-only arguments.
  ArrayRef<TypedAttr> getDefaultKwOnlyArgs();

  /// Get the function's default positional parameters.
  ArrayRef<TypedAttr> getDefaultPosParams();

  /// Get the function's default keyword-only parameters.
  ArrayRef<TypedAttr> getDefaultKwOnlyParams();

  /// Get the function's (unmangled) parameter names.
  ArrayRef<StringAttr> getParamNames();

  /// Mojo only has input parameters.
  ArrayRef<Type> getParamTypes() { return getInputParamTypes(); }
  size_t getNumParams() { return getParamTypes().size(); }

  /// Get the function parameter passing kinds (e.g. keyword-only).
  ArrayRef<PassingKind> getParamPassingKinds();

  /// Get the number of implicit lifetime decls this function type carries.
  size_t getNumImplicitLifetimeDecls();

  /// LIT-level signatures always have one result type.
  Type getResultType() { return getResults().front(); }

  /// Return this signature with the input parameters dropped.
  LITSignatureType dropParamValues();

  /// Returns true if the argument at this index is a vararg.
  bool isVarArg(size_t index);

  /// Returns true if the argument at this index is a pack vararg.
  bool isPackVarArg(size_t index);

  /// Returns true if the argument at this index is a keyword vararg.
  bool isKwVarArg(size_t index);

  /// Returns true if the signature has variadic parameters.
  bool hasParamVarArgs();

  /// Returns true if the signature has has pack arguments.
  bool hasPackVarArgs();

  /// Substitute the specified implicit lifetime references into the specified
  /// type, replacing them with `values` if they are at depth 0, or decrementing
  /// their depth if not.  This returns the resultant FunctionType on success,
  /// and invokes 'emitError'+returns null on error.
  FunctionType substituteImplicitLifetimesIntoValues(
      ArrayRef<TypedAttr> values, function_ref<InFlightDiagnostic()> emitError);

  /// This method replaces direct uses of NAMED implicit lifetime declarations
  /// with index-based references corresponding to the signature.  lifetimeDecls
  /// specifies the names of the implicit lifetime decls.
  LITSignatureType
  replaceImplicitLifetimesWithIndexes(ArrayRef<ParamDeclAttr> lifetimeDecls);

  /// Get this signature with all the implicit lifetimes bound to #lit.lifetime
  /// and dropped from the signature.
  LITSignatureType getWithImplicitLifetimesBoundImmortal();

  /// This method replaces direct uses of NAMED implicit lifetime declarations
  /// with index-based references.  lifetimeDecls specifies the names of the
  /// implicit lifetime decls to replace.
  static Type replaceImplicitLifetimesWithIndexes(
      Type type, ArrayRef<ParamDeclAttr> lifetimeDecls, size_t indexOffset = 0);

  // Determine how many implicit lifetimes a signature with the specified input
  // values should have.
  static size_t countImplicitLifetimes(ArrayRef<ArgConvention> convs);

  /// A `SignatureType` is a LIT signature if it contains function metadata.
  static bool classof(SignatureType type);
  static bool classof(Type type);

  static LITSignatureType get(MLIRContext *ctx, TypeRange inputs,
                              TypeRange results,
                              size_t numImplicitLifetimeDecls);
  static LITSignatureType get(FunctionType values, ArrayRef<Type> paramTypes,
                              ArrayRef<ArgConvention> convs, FnEffects effects,
                              FnMetadataAttr metadata);
};
} // namespace M::KGEN::LIT

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "KGEN/LITDialect/LITTypes.h.inc"

#endif // KGEN_LITDIALECT_LITTYPES_H
