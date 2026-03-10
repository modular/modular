//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains utilities for working with constraints in the Mojo
// parser, including checking constraints and manipulating constraint
// expressions.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_CONSTRAINTS_H
#define KGEN_MOJOPARSER_CONSTRAINTS_H

#include "KGEN/LITDialect/LITAttrs.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SMLoc.h"

namespace M::KGEN {
class ParameterEvaluator;

namespace LIT {

using llvm::ArrayRef;
using llvm::SmallVectorImpl;
using llvm::SMLoc;

class ASTDecl;
class DeclResolver;
class MojoInflightDiag;
class PogListAttr;

/// Emit a note explaining why a constraint is inconclusive. The incoming
/// constraint is expected to be the folded form with all input parameters
/// already substituted.
void emitConstraintInconclusive(DeclResolver &resolver, MojoInflightDiag &diag,
                                ConstraintAttr constraint);

/// Result of checking constraints.
enum class ConstraintResult {
  Violated,   // Some constraints are violated.
  Unprovable, // No violated constraints but some constraints cannot be proven.
  Satisfied,  // All constraints are satisfied.
};

/// Check that the given constraints are satisfied under the given scope. An
/// optional callback can be provided to emit failures for constraint
/// violations. If provided, unprovableConstraints will be populated with any
/// unprovable constraints encountered. An optional ParameterEvaluator can be
/// provided to substitute parameters into the constraints. Additional
/// assumptions can be passed to consider alongside the scope's known
/// assumptions (e.g., a conformance constraint during trait checking).
ConstraintResult checkConstraints(
    ASTDecl &declScope, PogListAttr paramListAttr,
    ArrayRef<ConstraintAttr> constraints,
    ArrayRef<ConstraintAttr> origConstraints,
    llvm::function_ref<MojoInflightDiag &(std::optional<SMLoc> loc)> getDiag,
    SmallVectorImpl<ConstraintAttr> *unprovableConstraints,
    ParameterEvaluator *evaluator,
    ArrayRef<ConstraintAttr> additionalAssumptions = {});

/// Rewrite cond(a, b, a) patterns to and(a, b) for constraint propositions.
/// This breaks the "short-circuit" pattern of `and`/`or` operators, so is only
/// legal during constraint checking.
TypedAttr deShortCircuitCond(TypedAttr value);

/// Check if propA logically implies propB.
/// Uses canonicalization, weakening rules (A implies A OR B),
/// conjunction elimination ((A AND B) implies A), and set-containment
/// subsumption for TypeConformsToTraitAttr (relies on conforms_to attrs
/// being canonicalized at construction to include ancestor traits).
/// Returns true if propA implies propB.
bool constraintImplies(TypedAttr propA, TypedAttr propB);

/// Check if propA and propB are logically contradictory.
/// Two propositions contradict if their conjunction is necessarily false.
/// E.g., X and NOT(X) contradict, as do (X AND Y) and NOT(X).
/// Uses canonicalization and recursive decomposition of AND/NOT expressions.
/// Returns true if propA and propB contradict.
bool constraintsContradict(TypedAttr propA, TypedAttr propB);

} // namespace LIT
} // namespace M::KGEN

#endif // KGEN_MOJOPARSER_CONSTRAINTS_H
