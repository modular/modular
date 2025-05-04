//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_FRAMEWORKS_STATSREPORT_H
#define SUPPORT_FRAMEWORKS_STATSREPORT_H

#include "Support/Telemetry/Telemetry.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringMap.h"

#include <string>

namespace M::Frameworks {

/// A helper API that can collect statistics about ops lowered vs. fallback
/// ops during an external framework translation pass. For the fallback ops,
/// counts for each separate *signature* (not just op name) are gathered as
/// well.
///
/// The helper also has the functionality to record these statistics to a
/// configurable report file.
struct StatsReport {
  explicit StatsReport(llvm::StringRef framework, llvm::StringRef modelName)
      : numLoweredOps(0), numFallbackOps(0), numFailedOps(0),
        framework(framework), modelName(modelName) {}

  StatsReport(llvm::StringRef framework) : StatsReport(framework, "") {}

  // NB: All `count_{Lowering, Fallback, Failure}` functions that take an
  // Operation as input will get a string representation of the op using
  // `StatsReport.cpp::getOpSignature`. To record a customized op representation
  // (e.g. one with more shape/dtype information), pass the string as input
  // instead.

  /// Record one instance of an input op lowered to MO.
  void countLowering(mlir::Operation &op);
  /// Record one instance of an op that was sent to fallback.
  void countFallback(mlir::Operation &op);
  /// Record one instance of an op lowering+fallback failure.
  void countFailure(mlir::Operation &op);

  /// Record one instance of an input op lowered to MO.
  void countLowering(std::string reconstructedType);
  /// Record one instance of an op that was sent to fallback.
  void countFallback(std::string reconstructedType);
  /// Record one instance of an op lowering+fallback failure.
  void countFailure(std::string reconstructedType);

  /// Write the statistics to file.
  void writeToFile();

  // Emits telemetry info on the ops collected.
  void emitTelemetry(M::Telemetry::TelemetryContext *telemetryCtx);

  std::string getJSON();

  size_t numLoweredOps;
  size_t numFallbackOps;
  size_t numFailedOps;
  llvm::StringMap<size_t> loweredHistogram;
  llvm::StringMap<size_t> fallbackHistogram;
  llvm::StringMap<size_t> failureHistogram;

private:
  /// framework name: pytorch, etc.
  std::string framework;
  std::string modelName;
};

} // namespace M::Frameworks

#endif // SUPPORT_FRAMEWORKS_STATSREPORT_H
