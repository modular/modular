//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// Helper for relating Values to properties of their underlying AsyncValues
// at runtime, including the impliesReady and impliedDead relations.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPTRANSFORMS_ASYNCVALUEDEPENDENCIES_H
#define GENERICML_MGPTRANSFORMS_ASYNCVALUEDEPENDENCIES_H

#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Transforms/IndexSet.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"

#include "GraphCompiler/MGPDialect/Transforms/RelationDataflowAnalysis.h"

namespace M::MGP {

/// This class helps recover implied control flow and liveness relations from
/// the asynchronous dataflow graph. There are four aspects to account for:
///  1. SSA values represent pointers to (reference counted) AsyncValues. There
///     may thus be many SSA names for the same underlying AsyncValue. When we
///     talk about liveness it is the AsyncValues we care about, not the SSA
///     names.
///  2. 'Primitive' ops (ie not control flow or output ops) BEGIN execution
///     when their 'gating' input AsyncValues are ready, and FINISH execution
///     when their 'indicating' result AsyncValues are ready. (See Utils.h for
///     the definition of 'gating' and 'indicating'.) The causal ordering of
///     ops depends on dataflow between these arguments and results. We avoid
///     reasoning about when an op RETURNS (binding its results), which may be
///     before it FINISHES, so ops that emplace results immediately and ops that
///     launch an asynchronous sub-task are treated alike.
///  3. Control flow ops only 'splice in' one sub-block at a time: a kBranch
///     takes one branch, and a kLoop waits for all results of a loop sub-block
///     before the next begins. So some ops may never execute, and two ops may
///     never overlap irrespective of dataflow between them.
///  4. Looping control flow ops may execute some ops more than once, and
///     results from one iteration may flow into the next. Thanks to 3, two
///     instances of the same call never execute concurrently.
///
/// We assume the model is 'overhang free': if all the results of the model
/// and any sub-block are ready then no ops can still be executing. Thus any
/// op which launches asynchronous computation must signal completion of that
/// computation via an indicating result, and that result cannot be ignored.
///
/// Ops are allowed to return results which 'lifetime extend' their arguments.
/// This means a result AsyncValue may contain a pointer to an argument
/// AsyncValue. Such ops must implement GRTExtendsLifetimeInterface.
///
/// Similarly, ops are allowed to return results which 'alias' their arguments.
/// This means a result may be a pointer to an argument AsyncValue, or some
/// part thereof. Such ops must implement GRTMayAliasInterface. Any possible
/// aliasing or sharing *within* AsyncValues can be considered as aliasing
/// of the AsyncValues themselves and signalled with the same interface. For
/// this analysis, GRTExtendsLifetimeInterface and GRTMayAliasInterface are
/// treated identically.
///
/// To make sense of all this we use:
///  - sameAs equivalence (two values always point to the same AsyncValue)
///  - maybeSameAs equivalence (two values may point to the same AsyncValue)
///  - isAlwaysReady predicate (the value is always a ready AsyncValue)
///  - impliesReady relation (if one value is ready then so is another)
///  - structurallyAtMostOne relation (two ops can never both be executed due
///    to being in distinct arms of a kBranch)
///  - structurallyCannotOverlapRelation (two ops can never both be executing
///    simultaneously due to being in distinct sub-blocks of a kBronch or
///    kLoop)
///  - runsLater relation (a later op can begin only after an earlier op
///    finishes)
///  - exclusive relation (the lifetimes of the AsyncValues pointed to by two
///    values never overlap)
///
/// In the following we'll write valueOf(%x) to mean the AsyncValue pointer
/// stored in %x at runtime.
class AsyncDataflowDependencies {
private:
  using ValueAnalysis = ForwardsRelationDataflowAnalysis;
  using OperationAnalysis = RelationDataflowAnalysisBase<Operation *>;

  static bool allOperationsFilter(Operation *op) { return true; }

public:
  using ValueSet = ValueAnalysis::Set;
  using ValueEquivalence = ValueAnalysis::Equivalence;
  using ValueRelation = ValueAnalysis::Relation;
  using OperationSet = OperationAnalysis::Set;
  using OperationRelation = OperationAnalysis::Relation;

  /// Helpers for working with domain maps and relations over both values and
  /// operations.
  ValueAnalysis valueAnalysis;
  OperationAnalysis operationAnalysis;

  /// What aspects of the analysis are required.
  enum AnalysisLevel {
    /// sameAs, maybeSameAs, isAlwaysReady, impliesReady
    kImpliesReady,
    /// areDisjoint
    kDisjoint,
    /// exclusiveSet, isNonSelfOverlapping
    kExclusive,
  };

  /// Performs the dataflow analysis to the given level.
  AsyncDataflowDependencies(IndexSet<Value> *valueIndex,
                            IndexSet<Operation *> *operationIndex,
                            AnalysisLevel analysisLevel,
                            llvm::unique_function<bool(Operation *)>
                                exclusiveOpsFilter = allOperationsFilter);

  /// The sameAs(%lhs, %rhs) equivalence is true only if at runtime:
  ///   valueOf(%lhs) == valueOf(%rhs)
  /// under all execution paths. The relation may not be complete. There is no
  /// causal ordering in this relation, and it is possible %lhs and %rhs are
  /// bound at different times.
  bool sameAs(Value lhs, Value rhs) const;

  /// The maybeSameAs(%lhs, %rhs) equivalence is *false* only if ot runtime:
  ///   valueOf(%lhs) != valueOf(%rhs), and
  ///   valueOf(%lhs) does not lifetime extend valueOf(%rhs)
  /// under all execution paths. The relation may not be complete. There is no
  /// causal ordering in this relation, and it is possible %lhs and %rhs are
  /// bound at different times.
  ///
  /// Note that we need the 'lifetime extends' property to support the
  /// 'exclusive' relation below.
  bool maybeSameAs(Value lhs, Value rhs) const;

  /// Returns the set { %rhs | maybeSameAs(%lhs, %rhs) }.
  ValueSet maybeSameAsSet(Value lhs) const;

  /// The isAlwaysReady(%value) predicate is true only if at runtime
  ///   valueOf(%value)->isReady()
  /// under all execution paths. For example, %value may be the result of a
  /// grt.chain.create (). The predicate may not be complete.
  bool isAlwaysReady(Value value) const;

  /// Returns the values which are always ready.
  const ValueSet &getAlwaysReady() const;

  /// The impliesReady(%lhs, %rhs) partial order is true only if at runtime
  ///   valueOf(%lhs)->isReady() =>
  ///     if %rhs is bound then valueOf(%rhs)->isReady()
  /// under all execution paths. The relation may not be complete, and
  /// impliesReady(%x, %x) may not be true.
  ///
  /// The additional 'is bound' qualification allows us to consider %lhs
  /// to imply %rhs as ready even though the block containing the binding for
  /// %rhs may never be executed (eg because it is within a kBronch control
  /// flow op sub-branch).
  bool impliesReady(Value lhs, Value rhs) const;

  /// Returns the set
  ///   { %lhs | impliesReady(%lhs, %rhs) or %lhs == %rhs }
  ///
  /// Here 'downwards' is in the sense of the usual top-to-bottom execution
  /// order of sequential programs, even though operations may be ordered
  /// arbitrarily within the block of the input model.
  ValueSet downwardsImpliesReadySet(Value rhs) const;

  /// Returns the set
  ///   { %rhs | impliesReady(%lhs, %rhs) }
  const ValueSet &strictUpwardsImpliesReadySet(Value lhs) const;

  /// Assuming
  ///   ... = some.op(values)
  /// (for strict some.op), removes as many values from 'values' without
  /// changing when some.op would be ready to execute under all execution paths.
  /// Prefers to remove later values above earlier values in the list. May
  /// remove all values.
  ///
  /// The rules are:
  ///  - Remove %x from values if isAlwaysReady(%x).
  ///  - Remove %x from values if %y in values and impliesReady(%y, %x).
  ///
  /// This can be used to simply the arguments to grt.chain.create ops. Since
  /// impliesReady includes the identity relation this will also remove outright
  /// duplicates.
  void simplify(ValueSet &values) const;

  /// The areDisjoint(lhsOp, rhsOp) relation is true only if lhsOp and rhsOp
  /// can never overlap in their execution over all execution paths. Here
  /// 'execution' is from BEGIN to FINISH (as opposed to BEGIN to RETURNED).
  ///
  /// Ops may be disjoint because lhsOp and rhsOp can never both be
  /// incorporated into the execution graph at the same time (eg because they
  /// are in distinct arms of an kBronch, or in distinct parts of conditioned
  /// kLoop), or because the execution of lhsOp and rhsOp are known to be
  /// strictly sequenced by AsyncValue dataflow dependencies. The relation
  /// may not be complete.
  bool areDisjoint(Operation *lhsOp, Operation *rhsOp) const;

  /// Returns the set of values who's underlying AsyncValue lifetimes can never
  /// overlap with that of indicator, under all execution paths. That is, a
  /// value is in the result only if it can never point to some AsyncValue
  /// at the same time as indicator points to some AsyncValue.
  ///
  /// Note we are using 'point to AsyncValues' here, rather than 'point to
  /// ready AsyncValues'. In particular, for:
  ///    %y = op %x
  /// with no other uses of %x and op strict, the 'point to AsyncValues'
  /// definition would consider %x and %y to be possibly overlapping, but the
  /// 'point to ready AsyncValues' would consider them exclusive. We prefer
  /// the former so that if %x is known to be exclusive to %y, %y can be
  /// created by the op itself without risk of conflict.
  ValueSet exclusiveSet(Value indicator) const;

  /// Returns true if value is bound at most once (because its defining
  /// op can only be executed at most once), or value is known to be dead
  /// by the time its defining op is (re-)executed (due to some enclosing
  /// kLoop control flow op).
  bool isNonSelfOverlapping(Value value) const;

  /// Prints a representation of all equivalences and relations to os.
  /// For testing only.
  void print(llvm::raw_ostream &os);

private:
  using ValueDomainMap = ValueAnalysis::DomainMap;

  /// Grows domainMap to capture the sameAs relation for operation.
  void flowSameAs(Operation *op, ValueDomainMap &domainMap);

  /// Sets the sameAsRelation for the root 'init' and 'execute' blocks.
  void setupSameAs();

  /// Grows domainMap to capture the maybeSameAs relation for operation.
  void flowMaybeSameAs(Operation *op, ValueDomainMap &domainMap);

  /// Sets the maybeSameAsRelation for the root 'init' and 'execute' blocks.
  void setupMaybeSameAs();

  /// Grows domainMap to capture the always-ready predicate for operation.
  void flowAlwaysReady(Operation *op, ValueDomainMap &domainMap);

  /// Sets the alwaysReadyRelation for the root 'init' and 'execute' blocks.
  void setupAlwaysReady();

  /// Grows domainMap to capture the impliesReady relation for operation.
  void flowImpliesReady(Operation *op, ValueDomainMap &domainMap);

  /// Sets the impliesReadyRelation and its inverse for the root 'init' and
  /// 'execute' blocks.
  void setupImpliesReady();

  /// Adds to relation all pairs (lhs, rhs) where lhs and rhs are drawn from
  /// distinct entries in opsPerSubBlock.
  void
  addCrossProduct(AsyncDataflowDependencies::OperationRelation &relation,
                  const SmallVector<SmallVector<Operation *>> &opsPerSubBlock);

  /// Sets the structurallyAtMostOneRelation and
  /// structurallyCannotOverlapRelation relations for all operations in the
  /// model.
  void setupStructurallyDisjoint();

  /// Returns the set:
  ///   { laterOp | laterOp can BEGIN execution only after earlierOp FINISHES }
  /// under all execution paths.
  OperationSet runsLaterSet(Operation *earlierOp) const;

  /// Sets the runsLaterRelation and its inverse for all operations in model.
  void setupRunsLater();

  /// Returns the op that effectively uses opOperand when finding all users of
  /// an SSA value, or null for a 'plumbing' use (control flow inputs and inner
  /// mef.output operands, already accounted for by maybeSameAs etc.). Returns:
  ///  - the model, for a top-level mef.output (a future use by the caller of
  ///    'init' or 'execute').
  ///  - the loop op, for a mef.output in a loop sub-block (a use by the next
  ///    iteration).
  ///  - otherwise, the owning operation.
  Operation *getEffectiveOwner(const OpOperand *opOperand) const;

  /// Returns the equivalence of values that could point to the same AsyncValue
  /// at op. Without loops this is just maybeSameAs. Under enclosing loops it
  /// is smaller, because:
  ///  - enclosing loop results are not bound yet, so no aliases escape via
  ///    them.
  ///  - op's result may be 'fresh', not an alias of its previous iteration's.
  ///
  /// The smaller equivalence lets getFullyConsumedWhenOpBegins find more
  /// consumed values and makes 'exclusiveSet' more complete.
  ValueEquivalence getEquivalenceAtOp(Operation *op) const;

  /// Returns true if every use of value in values (if any) is by an operation
  /// in ops. However the use must be in a gating operand position
  /// (otherwise we can't say anything about the actual lifetime of the value
  /// w.r.t. that operation).
  bool allUsesAreByOperationsInSet(const ValueSet &values,
                                   const OperationSet &ops) const;

  /// Returns all the values who's underlying AsyncValues are known to have
  /// zero references at the point op BEGINs execution.
  ///
  /// This means %x will be in the result only if:
  ///  - maybeSameAs(%x, %y), and %y appears as an operand in at least
  ///    one operation known to have FINISHED when op BEGINS.
  ///  - all such %y operands are in gating operand positions for a FINISHED op.
  ValueSet getFullyConsumedWhenOpBegins(Operation *op) const;

  /// Sets the exclusiveRelation for all values in the model.
  void setupExclusive();

  /// (%lhs, %rhs) is in sameAsRelation only if %lhs points to the same
  /// AsyncValue as %rhs when it is bound.
  ///
  /// Note that even if (%x, %z) and (%y, %z) are in the relation it may not be
  /// the case that (%x, %y) or (%y, %x) is in the relation. It is necessary to
  /// take the symmetric/transitive/identity closure to discover the equivalence
  /// class {%x, %y, %z} of SSA values all pointing to the same AsyncValue.
  ValueRelation sameAsRelation;
  /// The above closed under symmetry/transitivity/identity.
  ValueEquivalence sameAsEquivalence;

  /// (%lhs, %rhs) is in maybeSameAsRelation if %lhs could point to the same
  /// AsyncValue as %rhs when it is bound, or %lhs could lifetime extend %rhs.
  ValueRelation maybeSameAsRelation;
  /// The above closed under symmetry/transitivity/identity.
  ValueEquivalence maybeSameAsEquivalence;

  /// %x is in alwaysReadyPredicate only if %x points to an AsyncValue which is
  /// known to be always ready at runtime.
  ValueSet alwaysReadyPredicate;

  /// (%lhs, %rhs) is in impliesReady relation only if the AsyncValue pointed
  /// to by %rhs is known to be ready or never created if the AsyncValue pointed
  /// to by %lhs is ready at runtime.
  ValueRelation impliesReadyRelation;
  /// Inverse of above.
  ValueRelation inverseImpliesReadyRelation;

  /// (lhsOp, rhsOp) is in structurallyAtMostOneRelation if at most one
  /// can be executed for the lifetime of the entire graph (including both
  /// 'init' and 'execute' blocks). For kBranch sub-blocks outside kLoops,
  /// at most one arm is ever 'spliced in'.
  OperationRelation structurallyAtMostOneRelation;

  /// (lhsOp, rhsOp) is in structurallyCannotOverlapRelation if their execution
  /// can never overlap because they are in distinct sub-blocks of a control
  /// flow op: only one kBranch arm is 'spliced in' at a time, and the runtime
  /// waits for all outputs of a kLoop sub-block before splicing in the next.
  OperationRelation structurallyCannotOverlapRelation;

  /// (lhsOp, rhsOp) is in runsLaterRelation only if rhsOp can BEGIN execution
  /// only after either lhsOp has FINISHED execution, or rhsOp is never
  /// executed, under all execution paths.
  OperationRelation runsLaterRelation;
  /// Inverse of above.
  OperationRelation runsEarlierRelation;

  /// Indicates which operations are of interest for the exclusiveRelation.
  llvm::unique_function<bool(Operation *)> exclusiveOpsFilter;

  /// (%lhs, %rhs) is in exclusiveRelation only if %lhs and %rhs cannot
  /// simultaneously point to or otherwise alias the same AsyncValue over all
  /// execution paths.
  ///
  /// This can be because:
  ///  - by the time %rhs is created all uses of %lhs have been consumed
  ///  - vise versa
  ///  - both %lhs and %rhs can never both be created because their creating
  ///    ops are structurally disjoint.
  ValueRelation exclusiveRelation;
};

} // namespace M::MGP

#endif // GENERICML_MGPTRANSFORMS_ASYNCVALUEDEPENDENCIES_H
