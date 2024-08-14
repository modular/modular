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
} // namespace LIT
} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "KGEN/LITDialect/LITTypes.h.inc"

namespace M::KGEN::LIT {

//===----------------------------------------------------------------------===//
// LITSignatureType
//===----------------------------------------------------------------------===//

/// Create an uninitialized TypedAttr instance of the type for symbolic
/// interpretation.
ErrorOr<TypedAttr> createUninitializedValueOf(Type type,
                                              InterpreterState &state);

class LITSignatureType : public SignatureType {
public:
  using SignatureType::SignatureType;
  LITSignatureType(SignatureType sig);

  /// Get the signature metadata.
  FnMetadataAttr getMetadata();

  /// Get the argument list metadata.
  PogListAttr getArgListAttrs();

  /// Get the parameter list metadata.
  PogListAttr getParamListAttrs();

  /// Return the name for the argument at the specified index.
  StringAttr getArgName(size_t idx);

  /// Get the function's default positional arguments.
  ArrayRef<TypedAttr> getDefaultPosArgs();

  /// Get the function's default keyword-only arguments.
  ArrayRef<TypedAttr> getDefaultKwOnlyArgs();

  /// Get the function's default positional parameters.
  ArrayRef<TypedAttr> getDefaultPosParams();

  /// Get the function's default keyword-only parameters.
  ArrayRef<TypedAttr> getDefaultKwOnlyParams();

  /// Return the name for the parameter at the specified index.
  StringAttr getParamName(size_t idx);

  /// Mojo only has input parameters.
  ArrayRef<Type> getParamTypes() { return getInputParamTypes(); }
  size_t getNumParams() { return getParamTypes().size(); }

  /// Get the number of implicit lifetime decls this function type carries.
  size_t getNumImplicitLifetimeDecls();

  /// LIT-level signatures always have one result type.
  Type getResultType() { return getResults().front(); }

  /// Get the user result type of the signature.
  Type getUserResultType();

  /// Return this signature with the input parameters dropped.
  LITSignatureType dropParamValues();

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
  bool isPackVarArg(size_t index);

  /// For a PackVarArg, return the declared ArgConvention of the elements. For
  /// example: fn x[*Ts: AnyType](inout *pack: *Ts) is declared 'inout'.
  ArgConvention getPackVarArgConvention(size_t index);

  /// If the specified argument is a variadic pack, return the VariadicPack.
  Type getIfVariadicPack(size_t index);

  //// Return true if the parameter at this index is a parameter vararg.
  bool isParamVarArg(size_t index);

  /// Returns true if the signature has variadic parameters.
  bool hasParamVarArgs();

  /// Returns true if the signature has has pack arguments.
  bool hasPackVarArgs();

  /// Returns true if the signature has keyword variadic arguments.
  bool hasKwVarArgs();

  /// Return the offset of the error slot argument from the back of the argument
  /// list, if the signature is raising.
  unsigned getErrorSlotOffset();

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

  /// Reconstruct the signature using a list of named input parameters and
  /// indices indicating which one of them are variadic. These parameters are
  /// prepended to the current signature and references are remapped to index
  /// references. An additional array of indices corresponding to variadic
  /// parameters of the prepended parameters is also required.
  static LITSignatureType prependParams(LITSignatureType sig,
                                        ArrayRef<ParamDeclAttr> parentParams,
                                        ArrayRef<bool> parentVariadicMask);
};

//===----------------------------------------------------------------------===//
// Type Utilities
//===----------------------------------------------------------------------===//

/// Returns the user-defined result type of a signature, looking through
/// implicit memory results and stripping off the variant from error throwing
/// results if needed.
Type getSignatureUserResultType(SignatureType sigType, ArrayRef<Type> argTypes,
                                Type resultType);

/// The Lit parser and KGEN have different semantics for binding function
/// argument and result types. The parser will evaluate 'apply' expressions, but
/// KGEN does not since it cannot always have access to a symbol table.
/// Specialize a signature type while rebinding the input parameter values to
/// the expected input parameter types.
std::pair<LITSignatureType, ParameterExprArrayAttr>
getUnboundSpecializedSignature(LITSignatureType type,
                               ParameterExprArrayAttr bindings);

/// This predicate returns true if a parameter of the specified type may only
/// expand into one parameter value (e.g. `!lit.lifetime<x>` that only expands
/// to a single #lit.lifetime value.  Such a parameter doesn't need elaboration.
static inline bool isSingletonParameter(Type type) {
  // TODO: Could support structs of lifetimes.
  return isa<LIT::LifetimeType>(type);
}

/// This returns the singleton value to use for a parameter value that
/// `isSingletonParameter` returns true on. This aborts on non-singleton types.
TypedAttr getSingletonParameterValue(Type type);

} // namespace M::KGEN::LIT

#endif // KGEN_LITDIALECT_LITTYPES_H
