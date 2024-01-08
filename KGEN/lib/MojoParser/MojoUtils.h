//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares common utilities shared by the parser implementation.
//
//===----------------------------------------------------------------------===//

#ifndef MOJOPARSER_UTILS_H
#define MOJOPARSER_UTILS_H

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
PackType getIfPackType(SignatureType sig, size_t index);

/// Returns whether the two signatures match, i.e. if they only differ in
/// argument or parameter names.
bool canZeroCostConvert(SharedState &shared, ASTType fromType, ASTType toType);

/// Returns a type if there is a shared supertype for the two specified types,
/// e.g. two derived classes may have the same base class even if neither is
/// convertible to the other.  This returns null if there is no common type.
ASTType getZeroCostCommonType(SharedState &shared, ASTType type1,
                              ASTType type2);

//===----------------------------------------------------------------------===//
// Diagnostic utilities
//===----------------------------------------------------------------------===//

/// Helper to produce a consistent error message for incorrect argument and
/// parameter counts.
void emitWrongArgOrParamCount(InflightDiag &diag, size_t minRequired,
                              size_t maxAllowed, size_t numActual,
                              Twine argOrParam);

/// Helper to emit an error message for unexpected keyword operands.
void emitUnexpectedKeywords(InflightDiag &diag,
                            SmallVectorImpl<StringRef> &&unknownKeywords,
                            StringRef argOrParam);

/// Helper to emit an error message for positional-only operands passed by
/// keyword.
void emitPosOnlyPassedByKw(InflightDiag &diag,
                           SmallVectorImpl<StringRef> &&names,
                           StringRef argOrParam);

/// Certain special methods have type-specific restrictions or need special
/// handling. This function returns true if a given method can be synthesized
/// for a type with the given passability; if so an appropriate entry is added
/// to the given array of special function kinds.
bool canSynthesizeIfMissing(
    StringRef name, bool rpTrivial, bool regPassable,
    std::optional<std::reference_wrapper<SmallVectorImpl<SpecialFunctionKind>>>
        specialFns = std::nullopt);

/// Helper to delete code in a region and mark it as unreachable when it's
/// determined to be dead code.
void markRegionUnreachable(Region *deadRegion, Location unreachableLoc);

} // namespace M::KGEN::LIT

#endif // MOJOPARSER_UTILS_H
