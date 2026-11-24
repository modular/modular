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
//
// Classes related to returning mlir::InFlightDiagnostic
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_DIAGNOSTIC_H
#define GENERICML_SUPPORT_DIAGNOSTIC_H

#include "Support/ErrorOr.h"
#include "Support/LogicalResult.h"
#include "mlir/IR/Diagnostics.h"
#include <variant>

namespace M {

/// Holds a Value or an already-emitted mlir::InFlightDiagnostic, which
/// receivers may amend. For functions that propagate an error or emit one on
/// an operation; use `CompileDiagnosticOr` when there's no operation.
template <typename Value>
class [[nodiscard]] CompileDiagnosticOr {
public:
  CompileDiagnosticOr(Value value) : data(std::move(value)) {}
  CompileDiagnosticOr(mlir::InFlightDiagnostic err) : data(std::move(err)) {}
  CompileDiagnosticOr(CompileDiagnosticOr &&other)
      : data(std::move(other.data)) {}
  CompileDiagnosticOr &operator=(CompileDiagnosticOr &&other) {
    if (&other != this) {
      this->~CompileDiagnosticOr();
      new (this) CompileDiagnosticOr(std::move(other));
    }
    return *this;
  }

  /// Move constructor from value.
  template <class OtherT, typename = std::enable_if_t<
                              std::is_convertible<OtherT, Value>::value>>
  CompileDiagnosticOr(OtherT &&val) : data(std::forward<OtherT>(val)) {}

  /// Move constructor from ErrorOr.
  template <class OtherT, typename = std::enable_if_t<
                              std::is_convertible<OtherT, Value>::value>>
  CompileDiagnosticOr(ErrorOr<OtherT> &&other) : data(other) {}

  bool isError() const { return data.index() == 0; }
  explicit operator bool() const { return isError(); }

  /// Move the error out of this object. Object is invalid afterwards.
  mlir::InFlightDiagnostic takeError() { return std::get<0>(std::move(data)); }

  /// Move the value out of this object. Object is invalid afterwards.
  Value takeValue() { return std::get<1>(std::move(data)); }

private:
  std::variant<mlir::InFlightDiagnostic, Value> data;
};

class [[nodiscard]] CompileDiagnosticOrSuccess
    : public CompileDiagnosticOr<Detail::Empty> {
public:
  using CompileDiagnosticOr::CompileDiagnosticOr;
  /*implicit*/ CompileDiagnosticOrSuccess(SuccessType success)
      : CompileDiagnosticOr(Detail::Empty()) {}

  CompileDiagnosticOrSuccess() : CompileDiagnosticOr(Detail::Empty()) {}
};

} // namespace M
#endif // GENERICML_SUPPORT_DIAGNOSTIC_H
