//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Frameworks/StatsReport.h"

#include "Support/Telemetry/Logs.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/ToolOutputFile.h"
#include <sstream>
#include <string>
#include <system_error>

using namespace M;
using namespace Frameworks;

/// Get just the op signature: name, in/out types, attributes.
static std::string getOpSignature(mlir::Operation &op) {
  std::string key;
  llvm::raw_string_ostream ss(key);
  ss << op.getName().stripDialect();
  ss << "(";
  bool more = false;
  for (auto t : op.getOperandTypes()) {
    if (more)
      ss << ", ";
    else
      more = true;
    t.print(ss);
  }

  ss << ") -> (";

  more = false;
  for (auto t : op.getResultTypes()) {
    if (more)
      ss << ", ";
    else
      more = true;
    t.print(ss);
  }

  ss << ") {";

  more = false;
  for (auto attr : op.getAttrs()) {
    if (more)
      ss << ", ";
    else
      more = true;
    ss << attr.getName() << "=";

    // Avoid writing excessively large attributes.
    std::string attrVal;
    llvm::raw_string_ostream attrs(attrVal);
    attr.getValue().print(attrs);
    if (attrVal.size() < 32)
      ss << attrVal;
    else
      ss << attrVal.substr(0, 14) << "..."
         << attrVal.substr(attrVal.size() - 14, 14);
  }
  ss << "}";
  return key;
}

void M::Frameworks::StatsReport::countLowering(mlir::Operation &op) {
  ++numLoweredOps;
  ++loweredHistogram[getOpSignature(op)];
}

void M::Frameworks::StatsReport::countFallback(mlir::Operation &op) {
  ++numFallbackOps;
  ++fallbackHistogram[getOpSignature(op)];
}

void M::Frameworks::StatsReport::countFailure(mlir::Operation &op) {
  ++numFailedOps;
  ++failureHistogram[getOpSignature(op)];
}

void M::Frameworks::StatsReport::writeToFile() {
  std::error_code ec{};
  // TODO: Unify with the crash reporting var?
  // TODO(#17500): Include some info about the model name.
  std::optional<std::string> optFileName =
      llvm::sys::Process::GetEnv("MODULAR_STATS_FILENAME");
  std::optional<std::string> optAlwaysRecord =
      llvm::sys::Process::GetEnv("MODULAR_STATS_ALWAYS_RECORD");
  // Only record if a file is configured, or "always writing" is enabled.
  llvm::SmallString<512> reportFileName;
  if (!optFileName) {
    if (optAlwaysRecord.has_value()) {
      std::error_code tempEc = llvm::sys::fs::createTemporaryFile(
          "modular", "coveragelog", reportFileName);
      if (tempEc) {
        llvm::errs() << "Note: could not create coverage stats file. Coverage "
                        "reporting will not be available. Error message: "
                     << tempEc.message() << "\n";
        return;
      }
      llvm::errs()
          << "MODULAR_STATS_ALWAYS_RECORD is on, writing coverage stats to "
          << reportFileName << "\n.";
    } else {
      // No file name configured, and "always record" not enabled, skipping.
      return;
    }
  } else {
    reportFileName = *optFileName;
  }

  llvm::ToolOutputFile reportFile = llvm::ToolOutputFile(
      reportFileName, ec, llvm::sys::fs::OpenFlags::OF_Append);
  if (ec) {
    llvm::errs() << "Note: could not create coverage stats file. Coverage "
                    "reporting will not be available. Error message: "
                 << ec.message() << "\n";
    return;
  }

  reportFile.keep();
  reportFile.os() << modelName << "\n";
  reportFile.os() << "TOTAL OPS\t"
                  << numLoweredOps + numFallbackOps + numFailedOps << "\n";
  reportFile.os() << "FALLBACK OPS\t" << numFallbackOps << "\n";

  reportFile.os() << "\nFALLBACK OP LIST\n";
  for (const auto &[key, value] : fallbackHistogram) {
    reportFile.os() << value << "\t" << key << "\n";
  }
  reportFile.os() << "--------------------------------------\n\n";
}

void M::Frameworks::StatsReport::emitTelemetry(
    M::Telemetry::TelemetryContext *telemetryCtx) {
// if MODULAR_ENABLE_TELEMETRY=OFF, we do not support event attributes that are
// vectors. ifdef on MODULAR_ENABLE_TELEMETRY to work around the problem.
#ifdef MODULAR_ENABLE_TELEMETRY
  auto logger = telemetryCtx->getLogger("engine");

  auto fillHistogram = [&](std::vector<std::string_view> &keys,
                           std::vector<uint32_t> &counts,
                           const llvm::StringMap<size_t> &histogram) {
    keys.reserve(histogram.size());
    counts.reserve(histogram.size());
    for (const auto &[op, count] : histogram) {
      keys.push_back(op);
      counts.emplace_back(count);
    }
  };
  // Add fallback histogram
  std::vector<std::string_view> loweredOps;
  std::vector<uint32_t> loweredCounts;
  fillHistogram(loweredOps, loweredCounts, loweredHistogram);

  // Add fallback histogram
  std::vector<std::string_view> fallbackOps;
  std::vector<uint32_t> fallbackCounts;
  fillHistogram(fallbackOps, fallbackCounts, fallbackHistogram);

  // Add failed histogram
  std::vector<std::string_view> failureOps;
  std::vector<uint32_t> failureCounts;
  fillHistogram(failureOps, failureCounts, failureHistogram);

  logger->emitL1Event(framework + ".stats",
                      {
                          {"lowered_op_count", numLoweredOps},
                          {"fallback_op_count", numFallbackOps},
                          {"failed_op_count", numFailedOps},
                          {"lowered_ops.histogram.keys", loweredOps},
                          {"lowered_ops.histogram.values", loweredCounts},
                          {"fallback_ops.histogram.keys", fallbackOps},
                          {"fallback_ops.histogram.values", fallbackCounts},
                          {"failed_ops.histogram.keys", failureOps},
                          {"failed_ops.histogram.values", failureCounts},
                      });
#endif
  // If the MODULAR_STATS_FILENAME env var is set, dump the telemetry to that
  // file.
  if (llvm::sys::Process::GetEnv("MODULAR_STATS_FILENAME"))
    writeToFile();
}
