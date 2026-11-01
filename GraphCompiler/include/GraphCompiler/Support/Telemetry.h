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

#ifndef GENERICML_SUPPORT_TELEMETRY_H
#define GENERICML_SUPPORT_TELEMETRY_H

#include "Support/Context.h"
#include "Support/Error.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/STLExtras.h"
#include "Support/Telemetry/Telemetry.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/raw_ostream.h"

namespace M::GML {

enum class MaxEngineErrorCode : uint32_t {
  // Error in user supplied data.
  UserInputError = 0,

  // Separately tracked cases of the above.
  FileNotFoundError = 1,
  UnknownFormatError = 2,

  // Error importing the framework model to a graph representation.
  TFImportError = 100,
  TorchImportError = 101,

  // Error in lowering imported model to internal representation.
  GraphLoweringError = 1000,
  MissingOpError = 1001,

  // General case for compilation.
  CompilationError = 10000
};

std::string stringifyErrorCode(MaxEngineErrorCode code);

/// Telemetry counter for one error code, stored in a context and reused across
/// frameworks and models.
class ErrorCodeCounter {
public:
  ErrorCodeCounter(Telemetry::TelemetryContext &ctx)
      : counter(makeCounter(ctx)) {}

  template <MaxEngineErrorCode errorCode>
  void increment() {
    std::string category = stringifyErrorCode(errorCode);
    counter.add(1, {{category, to_underlying(errorCode)}});
  }

private:
  Telemetry::Counter<uint64_t> makeCounter(Telemetry::TelemetryContext &ctx) {
    return ctx.createUInt64Counter(
        "max.engine.error_code", M::Telemetry::Level::L1, {},
        "Category of failure emitted during model compilation.");
  }

  Telemetry::Counter<uint64_t> counter;
};

/// A convenience wrapper over TelemetryContext for MaxEngine-specific
/// telemetry.
class MaxEngineTelemetryContext {
public:
  MaxEngineTelemetryContext(M::ContextRef ctx)
      : context(ctx.getPointer()),
        telemetryContext(context->get<Telemetry::TelemetryContext>()) {
    assert(telemetryContext && "expected to be set up already");
  }

  ErrorCodeCounter &getErrorCounter() const {
    return context->emplaceIfMissing<ErrorCodeCounter>(*telemetryContext);
  }

  std::shared_ptr<Telemetry::Logs::Logger>
  getLogger(const std::string &eventName) const {
    return telemetryContext->getLogger(eventName);
  }

  const std::string &getAPILanguage() const { return apiLanguage; }

  void setAPILanguage(const std::string &newAPILanguage) {
    apiLanguage = newAPILanguage;
  }

  void flush() { telemetryContext->flush(); }

private:
  // Safe as a raw pointer: this object lives inside the parent M::Context.
  M::Context *context;
  Telemetry::TelemetryContext *telemetryContext;
  std::string apiLanguage;
};

/// Increments the session-scoped counter and emits metrics for an error code,
/// without emitting error logs.
template <MaxEngineErrorCode ErrorCode>
void incrementTelemetryErrorCount(MaxEngineTelemetryContext &ctx) {
  auto &counter = ctx.getErrorCounter();
  counter.increment<ErrorCode>();
  ctx.flush();
}

/// Increment the error count and emit error logs, under event domain
/// `compiler` with event name `compile.failure`.
template <MaxEngineErrorCode ErrorCode>
void emitErrorTelemetry(MaxEngineTelemetryContext &ctx, Error error) {
  incrementTelemetryErrorCount<ErrorCode>(ctx);

  std::shared_ptr<Telemetry::Logs::Logger> logger = ctx.getLogger("compiler");
  llvm::StringMap<Telemetry::Logs::AttributeValue> attributes = {
      {"error", error.get()}};
  std::string errorCodeStr = stringifyErrorCode(ErrorCode);
  attributes.insert({"max.engine.error_code", errorCodeStr});
  attributes.insert({"max.engine.api.language", ctx.getAPILanguage()});
  logger->emitL1Event("compile.failure", attributes);
}

/// Scoped handler that emits MLIR diagnostics to telemetry.
template <MaxEngineErrorCode ErrorCode>
struct MaxEngineTelemetryScope {
  MaxEngineTelemetryScope(mlir::MLIRContext *ctx,
                          GML::MaxEngineTelemetryContext *telemetryContext)
      : diagHandler(ctx, [telemetryContext](const mlir::Diagnostic &diag) {
          if (telemetryContext) {
            std::string errorMessage;
            llvm::raw_string_ostream os(errorMessage);
            diag.print(os);
            GML::emitErrorTelemetry<ErrorCode>(*telemetryContext,
                                               Error(errorMessage));
          }
          // Let the main handler process the diagnostic too.
          return failure();
        }) {}

  mlir::ScopedDiagnosticHandler diagHandler;
};

} // namespace M::GML

#endif // GENERICML_SUPPORT_TELEMETRY_H
