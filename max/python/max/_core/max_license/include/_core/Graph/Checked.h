//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef SDK_GRAPHAPI_PYTHON_CHECKED_H
#define SDK_GRAPHAPI_PYTHON_CHECKED_H

#include "_core/Graph/Bindings.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "nanobind/nanobind.h"

namespace M::Graph::Python {

/// RAII scope guard for registering an emit function to pass to verifiers.
/// If the emitter is used, this will raise a corresponding Python error.
struct RaisingScopedEmitFn {
  using EmitFn = llvm::function_ref<mlir::InFlightDiagnostic()>;
  std::string err;
  mlir::MLIRContext *ctx;
  mlir::ScopedDiagnosticHandler handler;
  llvm::unique_function<mlir::InFlightDiagnostic()> emitFn;
  RaisingScopedEmitFn() : RaisingScopedEmitFn(current_context()) {}
  RaisingScopedEmitFn(mlir::MLIRContext *ctx)
      : ctx(ctx),
        handler(ctx, [&](mlir::Diagnostic &diag) { err = diag.str(); }),
        emitFn(mlir::detail::getDefaultDiagnosticEmitFn(ctx)) {}
  ~RaisingScopedEmitFn() noexcept(false) {
    if (!err.empty()) {
      PyErr_Format(PyExc_TypeError, "%s", err.c_str());
      throw nanobind::python_error();
    }
  }
  operator EmitFn() { return emitFn; }
};

/// Function decorator which takes a verify- or getChecked-like function
/// which takens in an emit function (`mlir::InFlightDiagnostic()`)
/// and returns a function which omits that argument, using
/// `RaisingScopedEmitFn` to inject an appropriate emit function and
/// raise in python if it emits.
///
/// The implementation is a recursive template function which creates
/// curried wrappers until it either runs out of arguments, in which case
/// it returns the function, or finds the emit function in which case
/// it returns the injecting function.
template <typename T, typename Arg, typename... Rest>
constexpr auto checked(T (*f)(Arg, Rest...)) {
  using EmitFn = llvm::function_ref<mlir::InFlightDiagnostic()>;
  if constexpr (std::is_same_v<Arg, EmitFn>) {
    return [=](Rest &&...rest) {
      RaisingScopedEmitFn emitFn;
      return std::invoke(f, emitFn, std::forward<Rest>(rest)...);
    };
  } else {
    return [=](Arg &&arg, Rest &&...rest) {
      return checked([=](Rest &&...rest) {
        return std::invoke(f, std::forward<Arg>(arg),
                           std::forward<Rest>(rest)...);
      })(std::forward<Rest>(rest)...);
    };
  }
}

/// Base case for `checked` recursion
template <typename T>
constexpr auto checked(T (*f)()) {
  return f;
}

/// Wrap a `getChecked` call with a diagnostic handler which will raise a
/// python TypeError with the emitted error message.
template <typename T, typename... Args>
auto get_checked(Args &&...args) {
  using EmitFn = llvm::function_ref<mlir::InFlightDiagnostic()>;
  using Concrete =
      decltype(T::getChecked(std::declval<EmitFn>(), std::declval<Args>()...));
  using Overload = Concrete (*)(EmitFn, Args...);
  Overload getChecked = &T::getChecked;
  return checked(getChecked)(std::forward<Args>(args)...);
}

} // namespace M::Graph::Python

#endif // SDK_GRAPHAPI_PYTHON_CHECKED_H
