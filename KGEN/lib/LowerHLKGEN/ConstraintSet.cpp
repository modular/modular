//===- ConstraintSet.cpp --------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ConstraintSet.h"
#include "mlir/IR/Builders.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// ConstraintSet
//===----------------------------------------------------------------------===//

/// Add a single constraint with a single message.  This emits a diagnostic and
/// returns failure if a contradiction is detected.
LogicalResult ConstraintSet::addConstraint(TypedAttr constraint,
                                           StringAttr message, Location loc) {
  // If this is an equality constraint, handle it specially.
  if (auto oper = constraint.dyn_cast<ParamOperatorAttr>())
    if (oper.getOpcode() == POC::EQ)
      if (auto param = oper.getOperand(0).dyn_cast<ParamDeclRefAttr>())
        return addParamEqualityConstraint(param, oper.getOperand(1), message,
                                          loc);

  // Non-decodable attributes get added to the generalConstraints list so we can
  // properly maintain them when we regenerate the constraint spec.
  generalConstraints.push_back({constraint, message, loc});
  return success();
}

/// Re-encode this constraint set as a array of boolean conditions and
/// messages suitable for reinstalling on a generator.
std::pair<ArrayAttr, ArrayAttr> ConstraintSet::getConstraintsSpec() const {
  Builder b(decl->getContext());
  SmallVector<Attribute> values;
  SmallVector<Attribute> messages;

  // Turn pointwise parameters into constraints.  We iterate through this in
  // order of parameterOrder to make sure we produce a deterministically ordered
  // result.
  for (ParamDeclRefAttr param : parameterOrder) {
    auto it = pointwiseValues.find(param);
    assert(it != pointwiseValues.end());
    it->second.addConstraintSpec(param, values, messages);
  }

  // Flatten general constraints back into values/message array.
  for (auto [value, message, loc] : generalConstraints) {
    values.push_back(value);
    messages.push_back(message);
    // TODO: Handle locations.
  }

  return {b.getArrayAttr(values), b.getArrayAttr(messages)};
}

/// Add a constraint indicating the specified parameter is equal to the
/// specified value.  This emits a diagnostic and returns failure if a
/// contradiction is detected.
LogicalResult ConstraintSet::addParamEqualityConstraint(ParamDeclRefAttr param,
                                                        TypedAttr value,
                                                        StringAttr message,
                                                        Location loc) {
  assert(param.getType() == value.getType());

  auto newRecord = PointwiseValue::getSingleValue(value, message, loc);

  // Add the equality constraint if it doesn't exist already.
  auto [it, isNewEntry] = pointwiseValues.insert({param, newRecord});
  if (isNewEntry) {
    // If it didn't exist, just remember that we have an entry for this and we
    // are done.
    parameterOrder.push_back(param);
    return success();
  }

  // If it already existed, we need to merge in information and diagnose a
  // problem.
  return it->second.mergeIn(newRecord, decl);
}

//===----------------------------------------------------------------------===//
// PointwiseValue
//===----------------------------------------------------------------------===//

/// Lower this into a constraint spec for the specified parameter.
void PointwiseValue::addConstraintSpec(
    ParamDeclRefAttr param, SmallVectorImpl<Attribute> &values,
    SmallVectorImpl<Attribute> &messages) const {

  // We add pointwise equality constraints with equals.
  values.push_back(ParamOperatorAttr::get(POC::EQ, param, value));
  messages.push_back(message);
  // TODO: add loc.
}

/// Merge information from another pointwise value into this, emitting a
/// diagnostic on error or returning success if we are able to update.
LogicalResult PointwiseValue::mergeIn(PointwiseValue other,
                                      Operation *noteLoc) {
  // We only track "p == value" right now.  If this is telling us something we
  // already know, just accept and ignore it.
  if (value == other.value)
    return success();

  // Otherwise it must be a contradiction.
  auto diag = emitError(other.loc)
              << "constraint contradiction detected: " << other.message;
  diag.attachNote(loc) << "previously constrained " << message;
  diag.attachNote(noteLoc->getLoc()) << "generator declared here";
  return failure();
}
