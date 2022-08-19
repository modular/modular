//===- ConstraintSet.h ----------------------------------------------------===//
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
/// pointwise equivalence.
///
/// TODO: We eventually want to have 'x = y' equality, and != constraints.
class PointwiseValue {
public:
  static PointwiseValue getSingleValue(Attribute value, StringAttr message,
                                       Location loc) {
    return PointwiseValue{value, message, loc};
  }

  /// Return a Pointwise value indicating that the parameter is equal to one of
  /// the members of (non-empty) set of values.
  static PointwiseValue getSetValue(ArrayRef<TypedAttr> values,
                                    StringAttr message, Location loc);

  /// Merge information from another pointwise value into this, emitting a
  /// diagnostic on error or returning success if we are able to update.
  LogicalResult mergeIn(PointwiseValue other, Operation *noteLoc);

  /// Lower this into a constraint spec for the specified parameter.
  ConstraintAttr getAsConstraintSpec(ParamDeclRefAttr param) const;

private:
  PointwiseValue(Attribute value, StringAttr message, Location loc)
      : value(value), message(message), loc(loc) {}
  // This value is either an individual constant (e.g. an IntegerAttr,
  // FloatAttr, DTypeConstantAttr, etc) or an ArrayAttr which is a set of values
  // that it may equal.
  Attribute value;
  StringAttr message;
  Location loc;
};

/// This class maintains a decoded constraint specification list for a generator
/// or kernel.  It decomposes the list of constraints into a set of
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
/// TODO: We need to use a partial ordering + union-find like approach to unify
/// away and canonicalize equality constraints between parameters.  For example,
/// if we know `x == f32` and `y = x` then we can simplify all uses of 'y' to
/// use 'x'.  This is important when detecting conflicts (e.g. y is also known
/// to be f64.
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

  /// Re-encode this constraint set as a array of boolean conditions and
  /// messages suitable for reinstalling on a generator.
  ConstraintArrayAttr getConstraintsSpec() const;

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
