//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/Nanobind/TypeCasters.h" // IWYU pragma: keep (type casters)
#include "nanobind/nanobind.h"
#include "llvm/Support/LogicalResult.h"

namespace nb = nanobind;

NB_MODULE(bindings, m) {
  m.def("return_logical_result_success",
        [] { return llvm::LogicalResult::success(); });
  m.def("return_logical_result_failure",
        [] { return llvm::LogicalResult::failure(); });

  m.def("return_error_or_success_success",
        []() -> M::ErrorOrSuccess { return M::ErrorOrSuccess(); });
  m.def("return_error_or_success_failure",
        []() -> M::ErrorOrSuccess { return M::Error("failed"); });

  m.def("return_error_or_success", []() -> M::ErrorOr<int> { return 42; });
  m.def("return_error_or_failure",
        []() -> M::ErrorOr<int> { return M::Error("failed"); });
}
