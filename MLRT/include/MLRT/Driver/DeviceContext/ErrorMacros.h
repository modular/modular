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
// Common error handling macros for DeviceContext implementations.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_DRIVER_DEVICECONTEXT_ERRORMACROS_H
#define MLRT_DRIVER_DEVICECONTEXT_ERRORMACROS_H

/// Return a formatted error with file/line/function context.
/// Usage: RETURN_ERROR("Something failed");
#define RETURN_ERROR(msg)                                                      \
  do {                                                                         \
    return M::Driver::DeviceContext::formatError(__FILE__, __LINE__,           \
                                                 __FUNCTION__, msg);           \
  } while (0)

/// Evaluate an ErrorOr expression. If error, return formatted error.
/// Otherwise, define a new variable with the extracted value.
/// Usage: DEFINE_OR_RETURN_ERROR(CUevent, event, cudaDevice.eventCachePop());
#define DEFINE_OR_RETURN_ERROR(type, var, expr)                                \
  ErrorOr<type> _err_or_##var = (expr);                                        \
  if (_err_or_##var.isError()) {                                               \
    return M::Driver::DeviceContext::formatError(                              \
        __FILE__, __LINE__, __FUNCTION__, _err_or_##var.getError());           \
  }                                                                            \
  type var = _err_or_##var.takeValue()

/// Evaluate an ErrorOrSuccess expression. If error, return formatted error.
/// Usage: RETURN_IF_ERROR(ctx.enqueueWaitForEvent(*srcEvent));
#define RETURN_IF_ERROR(expr)                                                  \
  do {                                                                         \
    auto _err_result_ = (expr);                                                \
    if (_err_result_.isError()) {                                              \
      return M::Driver::DeviceContext::formatError(                            \
          __FILE__, __LINE__, __FUNCTION__, _err_result_.getError());          \
    }                                                                          \
  } while (0)

/// Check for pending error on a DeviceContext.
/// Usage: RETURN_IF_PENDING_ERROR(*this);
#define RETURN_IF_PENDING_ERROR(ctx)                                           \
  do {                                                                         \
    M::ErrorOrSuccess _pending_err_ = (ctx).checkPendingError();               \
    if (_pending_err_.isError())                                               \
      return _pending_err_.takeError();                                        \
  } while (0)

/// Record pending error on context (used in async error handling).
/// This is for situations where we cannot return an error and must store it
/// for later reporting.
/// Usage: RECORD_PENDING_ERROR(ctx, someErrorOrCall());
#define RECORD_PENDING_ERROR(ctx, err)                                         \
  do {                                                                         \
    auto _record_err_result_ = (err);                                          \
    if (_record_err_result_.isError()) {                                       \
      M::Error _formatted_err_ = M::Driver::DeviceContext::formatError(        \
          __FILE__, __LINE__, __FUNCTION__, _record_err_result_.getError());   \
      (ctx).setPendingError(std::move(_formatted_err_));                       \
    }                                                                          \
  } while (0)

#endif // MLRT_DRIVER_DEVICECONTEXT_ERRORMACROS_H
