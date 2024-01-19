//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Common attribute names shared between compile-time and runtime.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ML_COMMONATTRIBUTES_H
#define SUPPORT_ML_COMMONATTRIBUTES_H

namespace M {

//===----------------------------------------------------------------------===//
// MGP::ModelOp 'extra' attributes
//===----------------------------------------------------------------------===//

/// Binds to an array of strings of model tensor argument names.
constexpr const char *kMgpModelArgumentNames = "argument_names";

/// Binds to a string describing the name of dimensions in arguments.
/// Optional to appear in the graph.
constexpr const char *kMgpModelArgumentDimNames = "argument_dims";

/// Binds to an array of strings for model tensor result names.
constexpr const char *kMgpModelResultNames = "result_names";

} // namespace M

#endif // SUPPORT_ML_COMMONATTRIBUTES_H
