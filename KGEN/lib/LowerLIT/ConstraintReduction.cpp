//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ConstraintSet.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_CONSTRAINTREDUCTION
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
class SignatureUnifier {
public:
  SignatureUnifier(GeneratorOp generatorOp);

  /// Add the constraints already on the generator to the constraint set,
  /// returning failure if a contradiction was detected.
  LogicalResult checkExistingConstraints();

  void reinstallConstraints();

public:
  GeneratorOp generatorOp;

  ConstraintSet constraints;

  /// This string is set to information indicating context about in inferred
  /// constraint or diagnostic, e.g. that this is happening with argument #0.
  std::string inferenceContext;
  Location inferenceLoc;
};
} // namespace

SignatureUnifier::SignatureUnifier(GeneratorOp generatorOp)
    : generatorOp(generatorOp), constraints(generatorOp),
      inferenceLoc(UnknownLoc::get(generatorOp.getContext())) {}

/// Add the constraints already on the generator to the constraint set,
/// returning failure if a contradiction was detected.
LogicalResult SignatureUnifier::checkExistingConstraints() {
  for (ConstraintAttr constraint : generatorOp.getConstraints())
    if (failed(constraints.addConstraint(constraint)))
      return failure();

  return success();
}

/// When we're done checking the conformance, this method reinstalls the
/// (possibly updated) constraint information on the generator declaration.
void SignatureUnifier::reinstallConstraints() {
  generatorOp.setConstraintsAttr(constraints.getConstraintsSpec());
}

namespace {
struct ConstraintReductionPass
    : public impl::ConstraintReductionBase<ConstraintReductionPass> {
  using ConstraintReductionBase::ConstraintReductionBase;

  /// Reduce the constraint set on a generator and detect potential
  /// contradictions by processing them through a constraint set.
  void runOnOperation() override {
    SignatureUnifier unifier(getOperation());

    // Verify that the constraints already imposed on the generator are
    // satisfiable.
    if (failed(unifier.checkExistingConstraints()))
      return signalPassFailure();

    // If this generator is not actually implementing an interface, just return
    // after successfully checking the existing constraints for contradictions.
    unifier.reinstallConstraints();
  }
};
} // namespace
