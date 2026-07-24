//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_TRAITS_H
#define KGEN_MOJOPARSER_TRAITS_H

#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "Support/LLVMCompilerForwardDecls.h"

namespace M::KGEN {
class GetWitnessAttr;
} // namespace M::KGEN

namespace M::KGEN::LIT {
class ASTDecl;
class SharedState;
class FnTypeGeneratorType;
class MojoInflightDiag;

/// Check conformance of a struct against a given trait decl, build the
/// conformance op along the way. Inherited methods are not checked, so to
/// verify full conformance of an inheriting trait decl, its full ancestor chain
/// of decls must also be checked here too. This logic is left for the caller so
/// optimizations is possible when the struct is already known to conform to
/// certain ancestors. On success, the `ConformanceOp` will be populated.
LogicalResult verifyAndBuildConformance(ASTDecl &structDecl,
                                        SymbolRefAttr parent,
                                        std::optional<MojoInflightDiag> &diag,
                                        ConformanceOp op,
                                        ASTDecl &conformanceDecl);

/// Canonicalize the list of symbols that form a trait composition.
void canonicalizeTraitCompositionSymbols(
    SharedState &shared, SmallVectorImpl<SymbolRefAttr> &symbols);

/// Canonicalize the list of symbols that form a trait composition, but also
/// take constraints into account, `symbolConstraints` maps each symbol in the
/// provided `symbols` vector to its constraints.
SmallVector<ConstraintAttr> canonicalizeTraitSymbolsAndConstraints(
    SharedState &shared, SmallVectorImpl<SymbolRefAttr> &symbols,
    const DenseMap<SymbolRefAttr, ConstraintAttr> &symbolConstraints);

/// Reduce the list of trait composition symbols to the minimal set of symbols
/// that still implies the original trait composition.
SmallVector<SymbolRefAttr>
reduceTraitCompositionSymbols(SharedState &shared,
                              ArrayRef<SymbolRefAttr> symbols);

/// Given a type expression and scope-level assumptions, compute the effective
/// trait bound implied by any `conforms_to(type, Trait)` constraints.
TraitType getTraitBoundFromAssumptions(TypedAttr typeAttr, SharedState &shared,
                                       ArrayRef<ConstraintAttr> assumptions);

/// Fuse one or more constraints on the same symbol into a single constraint. A
/// single constraint is returned unchanged; multiple constraints are conjoined
/// -- their propositions are combined with `and` and their locations are fused
/// into a `FusedLoc`. The user message is dropped when fusing more than one,
/// since no single message describes a conjunction (see the definition for why
/// that drop is unobservable today).
ConstraintAttr fuseConstraints(SharedState &shared,
                               ArrayRef<ConstraintAttr> constraints);

/// Merge two meta-type values `typeA` and `typeB` into a single meta-type whose
/// trait bound is the common (intersection) bound of the two inputs.
/// Assumption: TypeA and TypeB must be the first level meta type, i.e., either
/// a trait type or a struct meta type.
Type mergeTwoMetaTypeBounds(SharedState &shared, ASTType typeA, ASTType typeB);

/// If `concreteType` fails to conform to `trait` because of one or more
/// conditional-conformance `where (cond, "message")` clauses, return the
/// failing constraints that carry a user message (so the caller can surface
/// the message). `callerAssumptions` are the enclosing scope's known
/// assumptions, so that a conformance the caller proved via its own `where`
/// clause is not reported as failing. Cold path: intended only to enrich the
/// "does not conform to trait" diagnostic, never for conformance decisions.
SmallVector<ConstraintAttr>
getFailedConformanceMessages(ASTType concreteType, TraitType trait,
                             SharedState &shared,
                             ArrayRef<ConstraintAttr> callerAssumptions);

/// Attach a note to `diag` for each conditional-conformance `where` message on
/// `srcType`'s struct that was not satisfied when converting to trait-typed
/// `targetType`. No-op if `targetType` is not a trait or `srcType` is null.
/// `scope` supplies the caller assumptions used to evaluate the constraints
/// (null means no assumptions). Centralizes the "unsatisfied conditional
/// conformance" note used at the value-conversion diagnostic sites.
void attachFailedConformanceNotes(MojoInflightDiag &diag, ASTType srcType,
                                  Type targetType, SharedState &shared,
                                  const ASTDecl *scope);

FnTypeGeneratorType specializeSignature(FnOp traitFn, ASTType newSelfType,
                                        DeclResolver &declResolver);

/// Emit a GetWitnessAttr that fetches a unique trait requirement if a type
/// conforms to it. The entry must be unique (non-overloaded) within the trait.
/// The GetWitnessAttr may be immediately evaluated if the type-value was
/// already resolved. If the type does not conform to the trait, returns an
/// empty TypedAttr.
FailureOr<TypedAttr> getUniqueWitnessForTypeIfConforms(
    SharedState &shared, ASTType type, TraitType trait, StringRef entryName,
    ArrayRef<ConstraintAttr> callerAssumptions, SMLoc errorLoc);
} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_TRAITS_H
