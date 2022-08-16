//===- ConstraintSet.h ----------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef CONSTRAINTSET_H
#define CONSTRAINTSET_H

#include "KGEN/KGENDialect/KGENAttrs.h"

namespace M::KGEN {

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
  ConstraintSet(MLIRContext *context) : context(context) {}

  // TODO: Generate error messages on contradictions.

  /// Add a list of boolean constraints with their specified messages to the
  /// constraint set.
  void addConstraints(ArrayAttr constraints, ArrayAttr constraintMessages);

  /// Add a single constraint with a single message.
  void addConstraint(TypedAttr constraint, StringAttr message);

  /// Add a constraint indicating the specified parameter is equal to the
  /// specified value.
  void addParamEqualityConstraint(ParamDeclRefAttr param, TypedAttr value,
                                  Twine message);

  /// Re-encode this constraint set as a array of boolean conditions and
  /// messages suitable for reinstalling on a generator.
  std::pair<ArrayAttr, ArrayAttr> getConstraintsSpec() const;

private:
  MLIRContext *context;

  /// This stores constraints that we can't decode into parameter relationships.
  SmallVector<std::pair<TypedAttr, StringAttr>> generalConstraints;

  ConstraintSet(const ConstraintSet &) = delete;
  ConstraintSet(ConstraintSet &&) = delete;
  void operator=(ConstraintSet &&) = delete;
};

} // namespace M::KGEN

#endif // CONSTRAINTSET_H
