//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef CONSTRAINTSET_H
#define CONSTRAINTSET_H

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Location.h"

namespace M::KGEN {
class ParamDeclRefAttr;

/// This maintains information about a parameter value being tracked for
/// pointwise equivalence.  We track several possibilities for a pointwise
/// (equality comparable) parameter.  We could find that it is either:
///   1) an individual `isSimpleConstant` value like an DTypeConstantAttr, or
///   2) an set of values (stored as an ArrayAttr of `isSimpleConstant` values)
///   3) an equivalence constraint (stored as a ParamDeclRefAttr), which is the
///      parameter the value information is stored on.  This is a simple
///      implementation of Tarjan's union-find algorithm.
///
/// TODO: We eventually want to have != constraints.
class PointwiseValue {
public:
  static PointwiseValue getSingleValue(Attribute value, StringAttr message,
                                       Location loc) {
    assert(isSimpleConstant(value) &&
           "cannot get equality constraint with non-constant "
           "value");
    return PointwiseValue{value, message, loc};
  }

  /// Return a Pointwise value indicating that the parameter is equal to one of
  /// the members of (non-empty) set of values.
  static PointwiseValue getInSetValue(ArrayRef<TypedAttr> values,
                                      StringAttr message, Location loc);

  /// Return a pointwise value stating that this parameter is equivalent to some
  /// other parameter.
  static PointwiseValue getParamEquivalence(ParamDeclRefAttr otherParam,
                                            StringAttr message, Location loc) {
    return PointwiseValue{otherParam, message, loc};
  }

  /// Merge information from another pointwise value into this, emitting a
  /// diagnostic on error or returning success if we are able to update.
  LogicalResult mergeIn(PointwiseValue other, Operation *noteLoc);

  /// Lower this into a constraint spec for the specified parameter.
  ConstraintAttr getAsConstraintSpec(ParamDeclRefAttr param) const;

  /// Return true if this marks equality to a simple constant.
  bool isEquality() const { return isSimpleConstant(value); }

  /// Return true if the parameter is known to be within a set of values.
  bool isInSetValue() const { return value.isa<ArrayAttr>(); }

  /// Return true if this is an equivalence relationship.
  bool isEquivalence() const { return value.isa<ParamDeclRefAttr>(); }
  ParamDeclRefAttr getEquivalentParam() const {
    return value.cast<ParamDeclRefAttr>();
  }

  /// Append a message to this constraint entry.
  void appendMessage(Twine newSuffix) {
    message =
        StringAttr::get(message.getContext(), message.getValue() + newSuffix);
  }

private:
  PointwiseValue(Attribute value, StringAttr message, Location loc)
      : value(value), message(message), loc(loc) {}
  Attribute value;
  StringAttr message;
  Location loc;
};

/// This class maintains a decoded constraint specification list for a generator
/// or func.  It decomposes the list of constraints into a set of
/// per-parameter constraints along with a list of arbitrary undecodable
/// constraints.  When adding constraints to the set it can detect conflicts
/// which make a candidate impossible, and it generates a diagnostic when so.
/// It can also regenerate the constraint set after simplifying it.
///
/// We have two kinds of parameters: pointwise values that can only be compared
/// for equality (e.g. strings, dtypes, etc), and relationally comparable values
/// like integers and floating point.
///
/// TODO: We don't do anything with relationally comparable values.  We should
/// bring in something like the simple ABCD inequality graph to provide
/// something lightweight and predictable
/// (https://dl.acm.org/doi/10.1145/358438.349342).
///
class ConstraintSet {
public:
  /// Initialize an empty constraint set for the specified declaration.  Its
  /// location will be used to produce notes.
  ConstraintSet(Operation *decl) : decl(decl) {}

  /// Add a single constraint with a single message.  This emits a diagnostic
  /// and returns failure if a contradiction is detected.
  LogicalResult addConstraint(ConstraintAttr constraint);

  /// Add a constraint has the specified value.  This emits a diagnostic and
  /// returns failure if a contradiction is detected.
  LogicalResult addPointwiseParamConstraint(ParamDeclRefAttr param,
                                            PointwiseValue value);

  /// Add a constraint capturing that param1 and param2 are equivalent to each
  /// other.
  LogicalResult addParamEquivalenceConstraint(ParamDeclRefAttr param1,
                                              ParamDeclRefAttr param2,
                                              StringAttr message, Location loc);

  /// Re-encode this constraint set as a array of boolean conditions and
  /// messages suitable for reinstalling on a generator.
  ConstraintArrayAttr getConstraintsSpec() const;

  /// Return a set containing all of the parameters that are inferred.  This
  /// will also include some parameters that were in the initial constraint set
  /// as well.
  ArrayRef<ParamDeclRefAttr> getPotentiallyInferredParameters() const {
    return parameterOrder;
  }

private:
  Operation *decl;

  /// This stores information about attributes that are decoded into a pointwise
  /// representation.  Each entry in this map has an entry in parameterOrder.
  DenseMap<ParamDeclRefAttr, PointwiseValue> pointwiseValues;

  /// This vector includes an entry for every parameter with a decoded
  /// constraint, allowing us to rebuild the constraint list in a deterministic
  /// order.
  SmallVector<ParamDeclRefAttr> parameterOrder;

  /// This stores constraints that we can't decode into parameter relationships.
  SmallVector<ConstraintAttr> generalConstraints;

  ConstraintSet(const ConstraintSet &) = delete;
  ConstraintSet(ConstraintSet &&) = delete;
  void operator=(ConstraintSet &&) = delete;
};

} // namespace M::KGEN

#endif // CONSTRAINTSET_H
