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

#ifndef SDK_GRAPHAPI_PYTHON_INFERCONTEXT_H
#define SDK_GRAPHAPI_PYTHON_INFERCONTEXT_H

#include "_core/Graph/Bindings.h"
#include "mlir/IR/MLIRContext.h"

namespace M::Graph::Python {

/// Decorator to infer the MLIRContext as the currently active Python MLIR
/// context. Returns a function which doesn't accept a context.
template <typename Return, typename Context, typename... Args,
          typename = std::enable_if_t<
              std::is_same_v<std::decay_t<Context>, mlir::MLIRContext *>, void>>
constexpr auto infer_context(Return (*func)(Context, Args...)) {
  return [=](Args... args) {
    return func(current_context(), std::forward<Args>(args)...);
  };
}

} // namespace M::Graph::Python

#endif // SDK_GRAPHAPI_PYTHON_INFERCONTEXT_H
