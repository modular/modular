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

namespace M::KGEN {
enum class ArgConvention : uint32_t;
class ParamDeclAttr;
class FuncTypeGeneratorType;
} // namespace M::KGEN

namespace M::KGEN::LIT {
class MojoInflightDiag;
class StructMetaType;
class ASTType;
class OriginSetAttr;
class PogListAttr;
class SharedState;
enum class SpecialFunctionKind : uint8_t;

/// Given a number, return one string if the number is 1, otherwise return the
/// other. This is typically used to generate an "s" suffix, but can also be
/// used for things like `plural(count, "was", "were")`.
inline const char *plural(size_t value, const char *one = "",
                          const char *other = "s") {
  return value == 1 ? one : other;
}

/// This function takes a list of function parameter types and returns a
/// `OriginSetAttr` consisting of the origin parameters accessible through
/// the function parameters. For example, a function with the signature
///
///   fn[lt: MutOrigin, x: WeirdReference[lt]]() -> None
///
/// May access the origin set `{mut lt}` through its parameters. This function
/// takes the param decls, which means it returns the named origin parameter
/// references accessible through the type.
///
/// This function takes a `SharedState` instance to access the cached origin
/// finder instance.
TypedAttr getOriginsAccessibleByParams(PogListAttr paramList,
                                       ArrayRef<ParamDeclAttr> params,
                                       SharedState &shared,
                                       TypedAttr captureOrigins);

/// Helper to delete code in a region and mark it as unreachable when it's
/// determined to be dead code.
void markRegionUnreachable(Region *deadRegion, Location unreachableLoc);

//===----------------------------------------------------------------------===//
// Diagnostic utilities
//===----------------------------------------------------------------------===//

/// Helper to produce a consistent error message for incorrect argument and
/// parameter counts.
void emitWrongArgOrParamCount(MojoInflightDiag &diag, size_t minRequired,
                              size_t maxAllowed, size_t numActual,
                              const Twine &argOrParam);

/// Helper to emit an error message for unknown keyword operands.
void emitUnknownKeywords(MojoInflightDiag &diag,
                         ArrayRef<StringAttr> unknownKeywords,
                         StringRef argOrParam);

/// Helper to emit an error message for positional-only operands passed by
/// keyword.
void emitPosOnlyPassedByKw(MojoInflightDiag &diag, ArrayRef<StringAttr> names,
                           StringRef argOrParam);

/// Helper to emit an error message for explicitly-specified inferred parameters
/// passed out of order.
void emitOutOfOrderInferredKw(MojoInflightDiag &diag,
                              ArrayRef<StringAttr> names);

/// Helper to emit an error message for missing operands.
void emitMissing(MojoInflightDiag &diag, ArrayRef<StringAttr> names,
                 const Twine &kindStr);

/// Helper to emit an error message for arguments/parameters passed both
/// positionally and by keyword.
void emitByPosAndKw(MojoInflightDiag &diag, ArrayRef<StringAttr> names,
                    const Twine &kindStr);

/// Helper to emit an error message for too many positional arguments/params.
void emitTooManyPositional(MojoInflightDiag &diag, size_t numMaxAllowed,
                           size_t numActual, const Twine &kindStr);

/// Return a printable name for an anonymous positional-only argument/parameter.
std::string nameForPosOnly(size_t idx, const Twine &argOrParam);

/// Check if a name is for an internal decl or not.
bool isInternalName(StringRef name);

} // namespace M::KGEN::LIT

#endif // MOJOPARSER_MOJOUTILS_H
