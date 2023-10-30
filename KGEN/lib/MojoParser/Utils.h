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

} // namespace M::KGEN::LIT

#endif // MOJOPARSER_UTILS_H
