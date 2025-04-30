//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_TRAITS_H
#define KGEN_MOJOPARSER_TRAITS_H

#include "KGEN/MojoParser/ASTDecl.h"
#include "Support/LLVMCompilerForwardDecls.h"

namespace M {
class InflightDiag;
} // namespace M

namespace M::KGEN::LIT {
class ASTDecl;
class SharedState;

/// Check conformance of a struct against a given trait decl. Inherited methods
/// are not checked, so to verify full conformance of an inheriting trait decl,
/// its full ancestor chain of decls must also be checked here too. This logic
/// is left for the caller so optimizations is possible when the struct is
/// already known to conform to certain ancestors.
/// On success, `witnessTable` will be populated.
using WitnessTable = SmallVector<std::pair<StringAttr, TypedAttr>>;
LogicalResult verifyConformance(ASTDecl &structDecl, SymbolRefAttr parent,
                                std::optional<InflightDiag> &diag,
                                WitnessTable &witnessTable);

/// Sort & deduplicate the list of symbols deterministically.
void sortAndDeduplicateSymbols(SmallVectorImpl<SymbolRefAttr> &symbols);

/// Canonicalize the list of symbols that form a trait composition.
void canonicalizeTraitCompositionSymbols(
    SharedState &shared, SmallVectorImpl<SymbolRefAttr> &symbols);
} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_TRAITS_H
