//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMPILER_PROPERTIES_H
#define SUPPORT_COMPILER_PROPERTIES_H

#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/Builders.h"

namespace M {
namespace detail {
template <typename T>
using convertible_to_int_t =
    decltype(static_cast<int64_t>(static_cast<T>(int64_t())));
} // namespace detail

/// Generic converter to attributes for enum properties.
template <typename T, typename = std::enable_if_t<llvm::is_detected<
                          detail::convertible_to_int_t, T>::value>>
Attribute convertToAttribute(MLIRContext *ctx, T value) {
  return Builder(ctx).getI64IntegerAttr(static_cast<int64_t>(value));
}

/// Generic converter from attributes for enum properties.
template <typename T, typename = std::enable_if_t<llvm::is_detected<
                          detail::convertible_to_int_t, T>::value>>
LogicalResult
convertFromAttribute(T &value, Attribute attr,
                     function_ref<InFlightDiagnostic()> emitError) {
  auto intAttr = dyn_cast<IntegerAttr>(attr);
  if (!intAttr)
    return emitError() << "expected an integer attribute for enum property";
  value = static_cast<T>(intAttr.getInt());
  return success();
}
} // namespace M

#endif // SUPPORT_COMPILER_PROPERTIES_H
