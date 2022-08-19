//===- ConstraintSet.cpp --------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ConstraintSet.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// ConstraintSet
//===----------------------------------------------------------------------===//

/// Add a single constraint with a single message.  This emits a diagnostic and
/// returns failure if a contradiction is detected.
LogicalResult ConstraintSet::addConstraint(ConstraintAttr constraint) {
  // If this is an equality constraint, handle it specially.
  if (auto oper = constraint.getExpr().dyn_cast<ParamOperatorAttr>()) {
    if (oper.getOpcode() == POC::EQ) {
      if (auto param = oper.getOperand(0).dyn_cast<ParamDeclRefAttr>()) {
        // 'param == 42' is equality comparable.
        if (isSimpleConstant(oper.getOperand(1))) {
          auto value = PointwiseValue::getSingleValue(
            oper.getOperand(1), constraint.getMessage(), constraint.getLoc());
          return addPointwiseParamConstraint(param, value);
        }

        // TODO: Handle linear expressions of relational values, e.g. x = y+4.
      }
    } else if (oper.getOpcode() == POC::IN) {
      // 'param in {1,2,3}' is equality comparable.
      if (auto param = oper.getOperand(0).dyn_cast<ParamDeclRefAttr>()) {
        if (llvm::all_of(oper.getOperands().drop_front(), isSimpleConstant)) {
          auto value = PointwiseValue::getSetValue(
              oper.getOperands().drop_front(), constraint.getMessage(),
              constraint.getLoc());
          return addPointwiseParamConstraint(param, value);
        }
      }
    }
  }

  // Non-decodable attributes get added to the generalConstraints list so we can
  // properly maintain them when we regenerate the constraint spec.
  generalConstraints.push_back(constraint);
  return success();
}

/// Re-encode this constraint set as a array of boolean conditions and
/// messages suitable for reinstalling on a generator.
ConstraintArrayAttr ConstraintSet::getConstraintsSpec() const {
  Builder b(decl->getContext());
  SmallVector<ConstraintAttr> constraints;

  // Turn pointwise parameters into constraints.  We iterate through this in
  // order of parameterOrder to make sure we produce a deterministically ordered
  // result.
  for (ParamDeclRefAttr param : parameterOrder) {
    auto it = pointwiseValues.find(param);
    assert(it != pointwiseValues.end());
    constraints.push_back(it->second.getAsConstraintSpec(param));
  }

  // Flatten general constraints back into values/message array.
  llvm::append_range(constraints, generalConstraints);

  return ConstraintArrayAttr::get(decl->getContext(), constraints);
}

/// Add a constraint indicating the specified parameter is equal to the
/// specified value.  This emits a diagnostic and returns failure if a
/// contradiction is detected.
LogicalResult ConstraintSet::addPointwiseParamConstraint(ParamDeclRefAttr param,
                                                         PointwiseValue value) {

  // Add the equality constraint if it doesn't exist already.
  auto [it, isNewEntry] = pointwiseValues.insert({param, value});
  if (isNewEntry) {
    // If it didn't exist, just remember that we have an entry for this and we
    // are done.
    parameterOrder.push_back(param);
    return success();
  }

  // If it already existed, we need to merge in information and diagnose a
  // problem.
  return it->second.mergeIn(value, decl);
}

//===----------------------------------------------------------------------===//
// PointwiseValue
//===----------------------------------------------------------------------===//

/// Return a Pointwise value indicating that the parameter is equal to one of
/// the members of (non-empty) set of values.
PointwiseValue PointwiseValue::getSetValue(ArrayRef<TypedAttr> values,
                                           StringAttr message, Location loc) {
  assert(!values.empty() && "Cannot get equality to empty set");
  if (values.size() == 1)
    return getSingleValue(values[0], message, loc);

  SmallVector<Attribute> valuesCopy(values.begin(), values.end());
  return PointwiseValue{ArrayAttr::get(message.getContext(), valuesCopy),
                        message, loc};
}

/// Lower this into a constraint spec for the specified parameter.
ConstraintAttr
PointwiseValue::getAsConstraintSpec(ParamDeclRefAttr param) const {
  // Set equivalence is handled with POC::IN.
  // TODO: This would be more convenient if POC::IN was a binary operation of
  // a value on the LHS and a typed set on the RHS.  This would make everything
  // smoother and more efficient.
  TypedAttr expr;
  if (auto valueArray = value.dyn_cast<ArrayAttr>()) {
    SmallVector<TypedAttr> operands;
    operands.push_back(param);
    llvm::append_range(operands, valueArray);
    expr = ParamOperatorAttr::get(POC::IN, operands);
  } else {
    // We add pointwise equality constraints with equals.
    expr = ParamOperatorAttr::get(POC::EQ, param, value);
  }
  return ConstraintAttr::get(expr, message, loc);
}

/// Merge information from another pointwise value into this, emitting a
/// diagnostic on error or returning success if we are able to update.
LogicalResult PointwiseValue::mergeIn(PointwiseValue other,
                                      Operation *noteLoc) {
  // Handle the case when this is merging a set into us.
  if (auto otherSet = other.value.dyn_cast<ArrayAttr>()) {
    // Simplify `x = [5,6,7,8]; x = [7,8,9]` to `x = [7,8]`.
    if (auto valueSet = value.dyn_cast<ArrayAttr>()) {
      SmallPtrSet<Attribute, 4> elements(otherSet.begin(), otherSet.end());
      SmallVector<TypedAttr> result;
      for (auto value : valueSet)
        if (elements.count(value))
          result.push_back(value);

      // If one set is a superset of the other, take the smaller set.
      if (result.size() == valueSet.size())
        return success();
      if (result.size() == otherSet.size())
        return *this = other, success();

      // If the result is non-empty then we build the new set.
      if (!result.empty()) {
        auto newMessage = StringAttr::get(message.getContext(),
                                          message.getValue() + ", and " +
                                              other.message.getValue());
        *this = PointwiseValue::getSetValue(
            result, newMessage,
            FusedLoc::get(message.getContext(), {loc, other.loc}));
        return success();
      }
      // `x = [5,6]; x = [7,8,9]` ==> unsatisfiable.
    } else {
      // Simplify `x = 42; x = [1,42, 59]` to `x = 42`.
      for (auto otherValue : otherSet)
        if (value == otherValue)
          return success();
      // `x = 42; x = [1, 59]` ==> unsatisfiable.
    }
  } else {
    // Ok, 'other' is a scalar value. If we have "param == value" that is
    // telling us something we already know, just accept and ignore it.
    if (value == other.value)
      return success();

    // If it is a set, check for set membership.
    if (auto valueSet = value.dyn_cast<ArrayAttr>()) {
      for (auto elt : valueSet) {
        // If we are saying something like `x = [5,6,7]; x = 7` simplify to x=7.
        if (elt == other.value) {
          value = other.value;
          return success();
        }
      }
    }

    // Otherwise, it is a contradiction.
  }

  // Otherwise it must be a contradiction.
  auto diag = emitError(other.loc)
              << "constraint contradiction detected: " << other.message;
  diag.attachNote(loc) << "previously constrained " << message;
  diag.attachNote(noteLoc->getLoc()) << "generator declared here";
  return failure();
}
