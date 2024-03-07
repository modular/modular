//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_TRAITS_H
#define KGEN_MOJOPARSER_TRAITS_H

#include "Support/LLVMForwardDecls.h"

namespace M::KGEN::LIT {
class ASTDecl;
class SharedState;
class TraitType;
class TypeLineageAttr;

/// Check conformance of a struct against a given trait type.
LogicalResult verifyConformance(ASTDecl &structDecl, TypeLineageAttr trait,
                                SharedState &shared);
} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_TRAITS_H
