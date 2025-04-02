//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_TRAITS_H
#define KGEN_MOJOPARSER_TRAITS_H

#include "Support/LLVMCompilerForwardDecls.h"

namespace M {
class InflightDiag;
} // namespace M

namespace M::KGEN::LIT {
class ASTDecl;
class SharedState;
class TraitType;
class TypeLineageAttr;

/// Check conformance of a struct against a given trait type.
LogicalResult verifyConformance(ASTDecl &structDecl, TypeLineageAttr parent,
                                std::optional<InflightDiag> &diag);

/// Canonicalize the list of symbols that form a trait composition.
void canonicalizeTraitCompositionSymbols(
    SharedState &shared, SmallVectorImpl<SymbolRefAttr> &symbols);
} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_TRAITS_H
