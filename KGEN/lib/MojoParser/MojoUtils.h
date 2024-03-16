//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares common utilities shared by the parser implementation.
//
//===----------------------------------------------------------------------===//

#ifndef MOJOPARSER_MOJOUTILS_H
#define MOJOPARSER_MOJOUTILS_H

#include "Support/LLVMCompilerForwardDecls.h"
#include <cstddef>

namespace M {
class InflightDiag;
} // namespace M

namespace M::KGEN {
class SignatureType;
class PackType;
} // namespace M::KGEN

namespace M::KGEN::LIT {
class ASTType;
class LITSignatureType;
class SharedState;
enum class SpecialFunctionKind : uint8_t;

/// Given a number, return one string if the number is 1, otherwise return the
/// other. This is typically used to generate an "s" suffix, but can also be
/// used for things like `plural(count, "was", "were")`.
inline const char *plural(size_t value, const char *one = "",
                          const char *other = "s") {
  return value == 1 ? one : other;
}

/// If the argument at the given index is of pack type, returns that type.
/// therwise, returns null.
PackType getIfPackType(LITSignatureType sig, size_t index);

/// Returns whether the two signatures match, i.e. if they only differ in
/// argument or parameter names.
bool canZeroCostConvert(SharedState &shared, ASTType fromType, ASTType toType);

/// Returns a type if there is a shared supertype for the two specified types,
/// e.g. two derived classes may have the same base class even if neither is
/// convertible to the other.  This returns null if there is no common type.
ASTType getZeroCostCommonType(SharedState &shared, ASTType type1,
                              ASTType type2);

/// Certain special methods have type-specific restrictions or need special
/// handling. This function returns true if a given method can be synthesized
/// for a type with the given passability; if so an appropriate entry is added
/// to the given array of special function kinds.
bool canSynthesizeIfMissing(StringRef name, bool rpTrivial, bool regPassable);

/// Helper to delete code in a region and mark it as unreachable when it's
/// determined to be dead code.
void markRegionUnreachable(Region *deadRegion, Location unreachableLoc);

/// Return the expected type of variadic argument values based on the **kwargs
/// (dictionary) type from a signature.
Type getVariadicKwargsType(Type dictRefType);

//===----------------------------------------------------------------------===//
// Diagnostic utilities
//===----------------------------------------------------------------------===//

/// Helper to produce a consistent error message for incorrect argument and
/// parameter counts.
void emitWrongArgOrParamCount(InflightDiag &diag, size_t minRequired,
                              size_t maxAllowed, size_t numActual,
                              Twine argOrParam);

/// Helper to emit an error message for unknown keyword operands.
void emitUnknownKeywords(InflightDiag &diag,
                         ArrayRef<StringAttr> unknownKeywords,
                         StringRef argOrParam);

/// Helper to emit an error message for positional-only operands passed by
/// keyword.
void emitPosOnlyPassedByKw(InflightDiag &diag, ArrayRef<StringAttr> names,
                           StringRef argOrParam);

/// Helper to emit an error message for missing operands.
void emitMissing(InflightDiag &diag, ArrayRef<StringAttr> names,
                 const Twine &kindStr);

/// Helper to emit an error message for arguments/parameters passed both
/// positionally and by keyword.
void emitByPosAndKw(InflightDiag &diag, ArrayRef<StringAttr> names,
                    const Twine &kindStr);

/// Helper to emit an error message for too many positional arguments/params.
void emitTooManyPositional(InflightDiag &diag, size_t numMaxAllowed,
                           size_t numActual, const Twine &kindStr);

/// Return a printable name for an anonymous positional-only argument/parameter.
std::string nameForPosOnly(size_t idx, const Twine &argOrParam);

} // namespace M::KGEN::LIT

#endif // MOJOPARSER_MOJOUTILS_H
