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

void M::Frameworks::StatsReport::countLowering(mlir::Operation &op) {
  auto opName = op.getName().stripDialect();
  ++loweredHistogram[opName.str()];
  ++numLoweredOps;
}

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
    M::Telemetry::TelemetryContext *telemetryContext) {
  auto logger = telemetryContext->getLogger("engine");
  llvm::StringMap<Telemetry::Logs::AttributeValue> attributes = {};
  attributes["lowered_op_count"] = numLoweredOps;
  attributes["fallback_op_count"] = numFallbackOps;
  attributes["failed_op_count"] = numFailedOps;
  logger->emitL1Event(framework + ".stats", attributes);

  auto logAttributes =
      [&](llvm::StringMap<Telemetry::Logs::AttributeValue> &attributes,
          std::string name) {
        if (attributes.size() > 0)
          logger->emitL1Event(framework + ".stats." + name, attributes);
      };

  llvm::StringMap<Telemetry::Logs::AttributeValue> fallbackAttributes = {};
  for (const auto &[fallbackOp, count] : fallbackHistogram)
    fallbackAttributes[fallbackOp] = count;
  logAttributes(fallbackAttributes, "fallback");

  llvm::StringMap<Telemetry::Logs::AttributeValue> failedAttributes = {};
  for (const auto &[op, count] : failureHistogram)
    failedAttributes[op] = count;
  logAttributes(failedAttributes, "failed");
}
