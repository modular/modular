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
// MGP::ModelOp constants
//===----------------------------------------------------------------------===//

/// Number of implicit init args that aren't device contexts.
constexpr int kMgpModelNumFixedImplicitInitArgs = 1;

/// Number of implicit execute args that aren't device contexts.
constexpr int kMgpModelNumFixedImplicitExecArgs = 3;

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

/// Binds to an array of indices referring to a device in `devices` attr used by
/// each argument.
constexpr const char *kMgpModelArgumentDeviceIndices =
    "argument_device_indices";

/// Binds to an array of indices referring to a device in `devices` attr used by
/// each result.
constexpr const char *kMgpModelResultDeviceIndices = "result_device_indices";

} // namespace M

#endif // SUPPORT_ML_COMMONATTRIBUTES_H
