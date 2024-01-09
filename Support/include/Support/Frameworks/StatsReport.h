//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_FRAMEWORKS_STATSREPORT_H
#define SUPPORT_FRAMEWORKS_STATSREPORT_H

#include "Support/LLVMForwardDecls.h"
#include "Support/Telemetry/Telemetry.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringMap.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace M::Frameworks {

/// A helper API that can collect statistics about ops lowered vs. fallback
/// ops during an external framework translation pass. For the fallback ops,
/// counts for each separate signature are gathered as well.
///
/// The helper also has the functionality to record these statistics to a
/// configurable report file.
struct StatsReport {
  explicit StatsReport(llvm::StringRef framework, llvm::StringRef modelName)
      : framework(framework), modelName(modelName), numTotalOps(0),
        numFallbackOps(0) {}

  StatsReport(llvm::StringRef framework) : StatsReport(framework, "") {}

  /// Record one instance of an input op.
  void countOp(mlir::Operation &op);
  /// Record one instance of an op that was sent to fallback.
  void countFallback(mlir::Operation &op);
  /// Write the statistics to file.
  void writeToFile();

  // Emits telemetry info on the ops collected
  void emitTelemetry(M::Telemetry::TelemetryContext *telemetryCtx);

private:
  /// Framework name: pytorch, tf, onnx, etc.
  std::string framework;
  std::string modelName;
  size_t numTotalOps;
  size_t numFallbackOps;
  llvm::StringMap<size_t> fallbackHistogram;
  llvm::StringMap<size_t> opHistogram{};
};

} // namespace M::Frameworks

#endif // SUPPORT_FRAMEWORKS_STATSREPORT_H
