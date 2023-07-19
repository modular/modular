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

/// Binds to a string in json syntax encoding the expected MachineInfoAttr
/// properties of the runtime host.
constexpr const char *kMgpModelTargetInfo = "target_info";
/// Binds to an array of strings of model tensor argument names.
constexpr const char *kMgpModelArgumentNames = "argument_names";
/// Binds to an array of strings for model tensor result names.
constexpr const char *kMgpModelResultNames = "result_names";

} // namespace M

#endif // SUPPORT_ML_COMMONATTRIBUTES_H
