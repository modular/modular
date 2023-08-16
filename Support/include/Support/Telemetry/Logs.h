//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TELEMETRY_LOGS_H
#define SUPPORT_TELEMETRY_LOGS_H

#include "Support/LLVMForwardDecls.h"
#include "Support/Telemetry/ForwardDecls.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#ifdef MODULAR_ENABLE_TELEMETRY
#include "opentelemetry/logs/event_logger.h"
#include "opentelemetry/logs/severity.h"
#endif // MODULAR_ENABLE_TELEMETRY
#include <variant>

namespace M::Telemetry::Logs {

/// Severity levels for logs.
/// See
/// https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/logs/data-model.md#field-severitynumber

#ifdef MODULAR_ENABLE_TELEMETRY

typedef opentelemetry::logs::Severity Severity;

using AttributeValue = opentelemetry::common::AttributeValue;

#else

// If telemetry is disabled, this enum is only required for the NOOP API.
// Note: not including all of Otel's severity numbers here, only the ones
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

using AttributeValue = std::variant<bool, int32_t, int64_t, uint32_t, double,
                                    const char *, std::string, uint64_t>;

#endif // MODULAR_ENABLE_TELEMETRY

/// A Logger to emit logs. Logger's methods are thread-safe.
/// Usage examples:
/// - logger->getError("my-event") << "my log error message";
/// - logger->emitEvent("my-event", Severity::kError, "my log error message");
/// - logger->emitEvent("billing", Severity::kInfo);
class Logger : public std::enable_shared_from_this<Logger> {
public:
  /// Emit event with given name, severity, body and attributes.
  void emitEvent(StringRef eventName, Severity severity, StringRef body = "",
                 const llvm::StringMap<AttributeValue> &attributes = {}) {
#ifdef MODULAR_ENABLE_TELEMETRY
    // Convert the attributes to unordered_map to pass to OTel.
    std::unordered_map<std::string, AttributeValue> attrs;
    for (auto &attr : attributes) {
      std::visit([&](auto v) { attrs[attr.first().str()] = v; }, attr.second);
    }
    logger->EmitEvent(eventName,
                      static_cast<opentelemetry::logs::Severity>(severity),
                      body, attrs);
#endif
  }

  /// LogStream is a llvm::raw_ostream wrapper around Logger::emitEvent().
  /// Logger has convenience methods that return a LogStream with a preset
  /// severity level (e.g. Logger::getWarn() returns a LogStream with severity
  /// of warn). LogStream is implemented by having a subclass of raw_ostream
  /// where event emission happens when the stream goes out of scope. The stream
  /// needs a pointer to the logger to emit the event. The pointer is shared and
  /// obtained from logger with shared_from_this() to avoid situations where the
  /// stream might outlive the logger.
  struct LogStream : public llvm::raw_string_ostream {

    virtual ~LogStream() {
      logger->emitEvent(eventName, severity, body, attributes);
    }

    /// Provide an explicit overload for strings that escapes special
    /// characters.
    friend LogStream &operator<<(LogStream &os, StringRef str) {
      os.write_escaped(str);
      return os;
    }

  private:
    friend class Logger;

    LogStream(const Twine &eventName, Severity severity,
              const llvm::StringMap<AttributeValue> &attrs,
              std::shared_ptr<Logger> logger)
        : raw_string_ostream(body), eventName(eventName.str()),
          severity(severity), logger(logger) {
      for (auto &attr : attrs) {
        std::visit([&](auto v) { attributes[attr.first().str()] = v; },
                   attr.second);
      }
    }

    std::string body;
    std::string eventName;
    Severity severity;
    // We convert the attributes to unordered_map to pass to OTel.
    std::unordered_map<std::string, AttributeValue> attributes;
    // TODO: timestamp.
    std::shared_ptr<Logger> logger;
  };

  /// Get raw_ostream to write a log with a severity of trace.
  LogStream getTrace(const Twine &eventName,
                     const llvm::StringMap<AttributeValue> &attributes = {}) {
    return LogStream(eventName, Severity::kTrace, attributes,
                     shared_from_this());
  }

  /// Get raw_ostream to write a log with a severity of debug.
  LogStream getDebug(const Twine &eventName,
                     const llvm::StringMap<AttributeValue> &attributes = {}) {
    return LogStream(eventName, Severity::kDebug, attributes,
                     shared_from_this());
  }

  /// Get raw_ostream to write a log with a severity of info.
  LogStream getInfo(const Twine &eventName,
                    const llvm::StringMap<AttributeValue> &attributes = {}) {
    return LogStream(eventName, Severity::kInfo, attributes,
                     shared_from_this());
  }

  /// Get raw_ostream to write a log with a severity of warn.
  LogStream getWarn(const Twine &eventName,
                    const llvm::StringMap<AttributeValue> &attributes = {}) {
    return LogStream(eventName, Severity::kWarn, attributes,
                     shared_from_this());
  }

  /// Get raw_ostream to write a log with a severity of error.
  LogStream getError(const Twine &eventName,
                     const llvm::StringMap<AttributeValue> &attributes = {}) {
    return LogStream(eventName, Severity::kError, attributes,
                     shared_from_this());
  }

  /// Get raw_ostream to write a log with a severity of fatal.
  LogStream getFatal(const Twine &eventName,
                     const llvm::StringMap<AttributeValue> &attributes = {}) {
    return LogStream(eventName, Severity::kFatal, attributes,
                     shared_from_this());
  }

private:
  friend class M::Telemetry::TelemetryContext;

#ifdef MODULAR_ENABLE_TELEMETRY
  Logger(std::shared_ptr<opentelemetry::logs::EventLogger> logger)
      : logger(logger) {}

  std::shared_ptr<opentelemetry::logs::EventLogger> logger;
#else
  Logger() {}
#endif // MODULAR_ENABLE_TELEMETRY

  /// Emit event with given name, severity, body and attributes. This method is
  /// private and used only by LogStream to log the event with OTel without
  /// additional copies or conversions of the attributes.
  void
  emitEvent(StringRef eventName, Severity severity, StringRef body,
            const std::unordered_map<std::string, AttributeValue> &attributes) {
#ifdef MODULAR_ENABLE_TELEMETRY
    logger->EmitEvent(eventName,
                      static_cast<opentelemetry::logs::Severity>(severity),
                      body, attributes);
#endif
  }
};

} // namespace M::Telemetry::Logs

#endif // SUPPORT_TELEMETRY_LOGS_H
