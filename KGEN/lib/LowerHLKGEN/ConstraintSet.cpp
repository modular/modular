//===- ConstraintSet.cpp --------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ConstraintSet.h"
#include "mlir/IR/Builders.h"

using namespace M;
using namespace KGEN;

/// Add a list of boolean constraints with their specified messages to the
/// constraint set.
void ConstraintSet::addConstraints(ArrayAttr constraints,
                                   ArrayAttr constraintMessages) {
  for (auto [constraint, message] :
       llvm::zip(constraints.getValue(), constraintMessages.getValue()))
    generalConstraints.push_back({constraint, message.cast<StringAttr>()});
}

/// Add a single constraint with a single message.
void ConstraintSet::addConstraint(TypedAttr constraint, StringAttr message) {
  generalConstraints.push_back({constraint, message});
}

/// Add a constraint indicating the specified parameter is equal to the
/// specified value.
void ConstraintSet::addParamEqualityConstraint(ParamDeclRefAttr param,
                                               TypedAttr value, Twine message) {

  assert(param.getType() == value.getType());
  addConstraint(ParamOperatorAttr::get(POC::EQ, param, value),
                StringAttr::get(value.getContext(), message));
}

/// Re-encode this constraint set as a array of boolean conditions and
/// messages suitable for reinstalling on a generator.
std::pair<ArrayAttr, ArrayAttr> ConstraintSet::getConstraintsSpec() const {
  Builder b(context);
  SmallVector<Attribute> values;
  SmallVector<Attribute> messages;

  // Flatten our general constraints back into values/message array.
  for (auto [value, message] : generalConstraints) {
    values.push_back(value);
    messages.push_back(message);
  }

  return {b.getArrayAttr(values), b.getArrayAttr(messages)};
}