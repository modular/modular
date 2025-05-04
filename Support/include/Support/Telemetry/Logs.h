//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TELEMETRY_LOGS_H
#define SUPPORT_TELEMETRY_LOGS_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/Telemetry/Common.h"
#include "Support/Telemetry/ForwardDecls.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#ifdef MODULAR_ENABLE_TELEMETRY
#include "opentelemetry/logs/event_logger.h"
#include "opentelemetry/logs/severity.h"
#endif // MODULAR_ENABLE_TELEMETRY
#include <unordered_map>
#include <variant>

namespace M::Telemetry::Logs {

/// Severity levels for logs.
/// See
/// https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/logs/data-model.md#field-severitynumber

#ifdef MODULAR_ENABLE_TELEMETRY

using Severity = opentelemetry::logs::Severity;
using AttributeValue = opentelemetry::common::AttributeValue;

#else

// If telemetry is disabled, this enum is only required for the NOOP API.
// Note: not including all of OTel's severity numbers here, only the ones
// we are going to use.
enum class Severity : uint8_t {
  kInvalid,
  kTrace,
  kDebug,
  kInfo,
  kWarn,
  kError,
  kFatal
};

struct AttributeValue
    : std::variant<bool, int32_t, int64_t, uint32_t, uint64_t, double,
                   llvm::StringRef, llvm::ArrayRef<bool>,
                   llvm::ArrayRef<int32_t>, llvm::ArrayRef<int64_t>,
                   llvm::ArrayRef<uint32_t>, llvm::ArrayRef<double>,
                   ArrayRef<uint64_t>, ArrayRef<uint8_t>,
                   llvm::ArrayRef<llvm::StringRef>> {
  using variant::variant;

  template <typename T>
  AttributeValue(T &&) : variant(false) {}
};

#endif // MODULAR_ENABLE_TELEMETRY

/// A Logger to emit logs. Logger's methods are thread-safe.
/// Usage examples:
/// - logger->getError("my-event") << "my log error message";
/// - logger->emitEvent("my-event", Severity::kError, "my log error message");
/// - logger->emitEvent("billing", Severity::kInfo);
class Logger : public std::enable_shared_from_this<Logger> {
public:
  virtual ~Logger() = default;

  virtual void
  emitL0Event(StringRef eventName,
              const llvm::StringMap<AttributeValue> &attributes = {}) {
    return emitEvent(eventName, Severity::kInfo, M::Telemetry::Level::L0,
                     attributes);
  }
  virtual void
  emitL1Event(StringRef eventName,
              const llvm::StringMap<AttributeValue> &attributes = {}) {
    return emitEvent(eventName, Severity::kInfo, M::Telemetry::Level::L1,
                     attributes);
  }
  virtual void
  emitL2Event(StringRef eventName,
              const llvm::StringMap<AttributeValue> &attributes = {}) {
    return emitEvent(eventName, Severity::kInfo, M::Telemetry::Level::L2,
                     attributes);
  }
  virtual void
  emitL0Error(StringRef eventName, const CodedErrorOrSuccess &codedError,
              const llvm::StringMap<AttributeValue> &attributes = {}) {
#ifdef MODULAR_ENABLE_TELEMETRY
    if (codedError.isError()) {
      llvm::StringMap<AttributeValue> attributesWithError{attributes};
      attributesWithError["error_component"] =
          StringRef(codedError.getComponentAsString());
      attributesWithError["error_id"] = StringRef(codedError.getIdAsString());
      attributesWithError["error"] = StringRef(codedError.getErrorAsString());
      return emitEvent(eventName, Severity::kInfo, M::Telemetry::Level::L0,
                       attributesWithError);
    }
#endif
  }

  /// Returns true if an event will be emitted based on its level and the
  /// configured telemetry level.
  bool eventEnabled(Level eventLevel) const {
    return eventLevel <= telemetryLevel;
  }

protected:
#ifdef MODULAR_ENABLE_TELEMETRY
  Logger(std::shared_ptr<opentelemetry::logs::EventLogger> logger,
         M::Telemetry::Level level)
      : logger(std::move(logger)), telemetryLevel(level) {}

  std::shared_ptr<opentelemetry::logs::EventLogger> logger;
#else
  Logger() {}
#endif // MODULAR_ENABLE_TELEMETRY

private:
  friend class M::Telemetry::TelemetryContext;

  /// Emit event with given name, severity, body and attributes.
  void emitEvent(StringRef eventName, Severity severity,
                 M::Telemetry::Level level,
                 const llvm::StringMap<AttributeValue> &attributes) {
#ifdef MODULAR_ENABLE_TELEMETRY
    if (eventEnabled(level)) {
      // Convert the attributes to unordered_map to pass to OTel.
      std::unordered_map<std::string, AttributeValue> attrs;
      for (auto &attr : attributes) {
        std::visit([&](auto v) { attrs[attr.first().str()] = v; }, attr.second);
      }
      // Use structured attributes rather than unstructured body
      logger->EmitEvent(eventName,
                        static_cast<opentelemetry::logs::Severity>(severity),
                        "", attrs);
    }
#endif
  }

  // Configured level for Telemetry.
  M::Telemetry::Level telemetryLevel;
};

} // namespace M::Telemetry::Logs

#endif // SUPPORT_TELEMETRY_LOGS_H
