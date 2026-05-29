//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// MLOG is a macro that takes a variable number of arguments and logs them to
// a file or the console.

// If the single arg is a string, it is printed to the INFO level with a
// newline. MLOG(""); // "\n" is printed to the INFO level.
// MLOG("hello");
// "hello" is printed to the INFO level.
//
// Two args or more is a format string and a value, or values
// MLOG("{}", "hello"); // hello
// MLOG("{} {}", "hello", "world"); // hello world
// MLOG("{} {} {}", "hello", "world", "!"); // hello world!
//
// ...unless the first arg is a LogLevel, then everything above
// shifts left
// MLOG(LogLevel::DEBUG, "hi"); // "hi" printed at debug level
// MLOG(LogLevel::DEBUG, "{}", "hello"); // hello printed at debug level
// MLOG(LogLevel::DEBUG, "{} {}", "hello", 42); // hello 42

#ifndef SUPPORT_LOG_H
#define SUPPORT_LOG_H

#include "LogChannels.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"

#define FMT_EXCEPTIONS 0
#include <fmt/base.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string_view>

#define MLOG(...) M::Log::log(__VA_ARGS__)

#define MLOG_DEBUG(...) MLOG(::M::Log::LogLevel::DEBUG, __VA_ARGS__)
#define MLOG_INFO(...) MLOG(::M::Log::LogLevel::INFO, __VA_ARGS__)
#define MLOG_WARN(...) MLOG(::M::Log::LogLevel::WARN, __VA_ARGS__)
#define MLOG_ERROR(...) MLOG(::M::Log::LogLevel::ERROR, __VA_ARGS__)
#define MLOG_FATAL(...)                                                        \
  do {                                                                         \
    MLOG(::M::Log::LogLevel::FATAL, __VA_ARGS__);                              \
    std::abort();                                                              \
  } while (0)

namespace M::Log {

enum class LogLevel : uint8_t {
  // Logs detailed debugging information for development and troubleshooting.
  DEBUG = 0,
  // Logs general informational messages about normal program operation.
  INFO = 1,
  // Logs warning messages about potential issues that don't prevent execution.
  WARN = 2,
  // Logs error messages about problems that affect functionality but allow
  // continuation.
  ERROR = 3,
  // Logs critical error messages that indicate severe problems requiring
  // immediate attention.
  FATAL = 4,
};

// Holds the pre-formatted data for a log event. The tag and the union should be
// kept in sync.
struct LogArg {
  enum class Type : uint8_t {
    Bool,
    Int64,
    UInt64,
    Fp32,
    Fp64,
    SmallString,
    String,
    Pointer
  };

  // Plain POD string view: pointer + length with a guaranteed, stable layout
  // that does not depend on any string_view-like implementation.
  struct StrView {
    const char *ptr;
    size_t len;
  };

  union {
    bool b;
    int64_t i64;
    uint64_t ui64;
    float fp32;
    double fp64;
    // for short strings, stored inline (16 chars max)
    std::array<char, sizeof(StrView)> ssoStr;
    StrView str{};
    const void *ptr;
  } data;
  Type tag;
};

namespace Detail {
template <typename T>
LogArg toLogArg(T &&val) {
  using D = std::decay_t<T>;
  using enum LogArg::Type;
  if constexpr (std::is_same_v<D, bool>) {
    return LogArg{.data = {.b = val}, .tag = Bool};
  } else if constexpr (std::is_integral_v<D> && std::is_signed_v<D> &&
                       sizeof(D) <= sizeof(int64_t)) {
    return LogArg{.data = {.i64 = static_cast<int64_t>(val)}, .tag = Int64};
  } else if constexpr (std::is_integral_v<D> && std::is_unsigned_v<D> &&
                       sizeof(D) <= sizeof(uint64_t)) {
    return LogArg{.data = {.ui64 = static_cast<uint64_t>(val)}, .tag = UInt64};
  } else if constexpr (std::is_same_v<D, float>) {
    return LogArg{.data = {.fp32 = val}, .tag = Fp32};
  } else if constexpr (std::is_same_v<D, double>) {
    return LogArg{.data = {.fp64 = val}, .tag = Fp64};
  } else if constexpr (std::is_convertible_v<D, std::string_view>) {
    std::string_view sv(val);
    if (sv.size() <= sizeof(LogArg::data.ssoStr)) {
      LogArg arg;
      arg.tag = LogArg::Type::SmallString;
      std::copy(sv.data(), sv.data() + sv.size(), arg.data.ssoStr.data());
      // Null-terminate only when there is room; a full 16-byte string is
      // identified by the absence of a null within the buffer.
      if (sv.size() < sizeof(LogArg::data.ssoStr))
        arg.data.ssoStr[sv.size()] = '\0';
      return arg;
    } else {
      return LogArg{.data = {.str = {sv.data(), sv.size()}}, .tag = String};
    }
  } else if constexpr (std::is_pointer_v<D>) {
    return LogArg{.data = {.ptr = val}, .tag = Pointer};
  } else {
    static_assert(!std::is_same_v<T, T>, "Unsupported log argument type.");
  }
}
} // namespace Detail

struct LogRecord {
  constexpr static size_t maxArgs = 8;
  using Timestamp = std::chrono::time_point<std::chrono::system_clock>;
  // Mojo FFI mirrors Timestamp as Int64. If this fires, the platform's
  // system_clock uses a different rep type and the Mojo bindings need
  // revisiting.
  static_assert(std::is_same_v<Timestamp::clock::duration::rep, int64_t>);
  Timestamp timestamp;
  std::string_view fmtString;
  std::array<LogArg, maxArgs> args;
  uint8_t argCount;
  LogLevel level;
  Channel::Channels channel;

  template <typename... Args>
  LogRecord(Timestamp ts, LogLevel lvl, Channel::Channels c,
            std::string_view fmt, Args &&...args)
      : timestamp(ts), fmtString(fmt),
        args{Detail::toLogArg(std::forward<Args>(args))...},
        argCount(sizeof...(Args)), level(lvl), channel(c) {
    static_assert(sizeof...(Args) <= maxArgs, "Too many log arguments");
  }

  // Constructs from a pre-built args array. Used by the C FFI shim
  // (LogFFI.cpp), which serializes typed arguments across the boundary itself.
  LogRecord(Timestamp ts, LogLevel lvl, Channel::Channels c,
            std::string_view fmt, std::array<LogArg, maxArgs> prebuiltArgs,
            uint8_t count)
      : timestamp(ts), fmtString(fmt), args(std::move(prebuiltArgs)),
        argCount(count), level(lvl), channel(c) {}
};

// Receives formatted log lines and writes to destinations (stdout, file &c).
class Sink;

class Logger {
  struct LogFormatState {
    bool useEnhancedFormat = true;
    bool showTimeStamp = true;
    bool showColors = true;
    bool showLogLevel = true;
    bool useIsoTimestamps = false;
    bool showMicroseconds = false;
    bool emitJSON = false;
  } formatState;
  std::atomic<LogLevel> level = LogLevel::WARN;
  std::vector<std::unique_ptr<Sink>> sinks;
  ChannelState channelsEnabled;

  llvm::SmallString<32> buildTimestampString(LogRecord::Timestamp);
  llvm::SmallString<128> buildLogPrefix(LogLevel, Channel::Channels,
                                        LogRecord::Timestamp);

public:
  Logger();
  ~Logger();
  void log(LogRecord record);

  void enableChannel(Channel::Channels c) { channelsEnabled.enable(c); }
  void disableChannel(Channel::Channels c) { channelsEnabled.disable(c); }
  void enableAllChannels() { channelsEnabled.enableAll(); }
  void disableAllChannels() { channelsEnabled.disableAll(); }

  bool isEnabled(Channel::Channels c) const {
    return channelsEnabled.isEnabled(c);
  }

  LogLevel getLogLevel() const {
    return level.load(std::memory_order::acquire);
  }

  void setLogLevel(LogLevel newLevel) {
    level.store(newLevel, std::memory_order::release);
  }
};

inline Logger &getDefaultLog() {
  static Logger defaultLog;
  return defaultLog;
}

inline void setLogLevel(LogLevel level) { getDefaultLog().setLogLevel(level); }

inline void enableChannel(Channel::Channels c) {
  getDefaultLog().enableChannel(c);
}

inline void disableChannel(Channel::Channels c) {
  getDefaultLog().disableChannel(c);
}

inline void enableAllChannels() { getDefaultLog().enableAllChannels(); }

inline void disableAllChannels() { getDefaultLog().disableAllChannels(); }

inline void logWrite(Logger &log, LogRecord record) {
  log.log(std::move(record));
}

// Checks the log level to see if it should emit a message, and, if so,
// dispatches to the Logger object.
template <typename... Args>
inline void logWriteDispatch(Logger &log, LogLevel level,
                             Channel::Channels channel,
                             fmt::format_string<Args...> fmt, Args &&...args) {
  if (log.getLogLevel() > level || !log.isEnabled(channel))
    return;

  LogRecord record(std::chrono::system_clock::now(), level, channel,
                   {fmt.get().data(), fmt.get().size()},
                   std::forward<Args>(args)...);
  logWrite(log, std::move(record));
}

template <typename... Args>
void log(fmt::format_string<Args...> fmt, Args &&...args) {
  logWriteDispatch(getDefaultLog(), LogLevel::INFO, Channel::Default, fmt,
                   std::forward<Args>(args)...);
}

template <typename... Args>
void log(LogLevel level, fmt::format_string<Args...> fmt, Args &&...args) {
  logWriteDispatch(getDefaultLog(), level, Channel::Default, fmt,
                   std::forward<Args>(args)...);
}

template <typename... Args>
void log(Channel::Channels channel, fmt::format_string<Args...> fmt,
         Args &&...args) {
  logWriteDispatch(getDefaultLog(), LogLevel::INFO, channel, fmt,
                   std::forward<Args>(args)...);
}

template <typename... Args>
void log(LogLevel level, Channel::Channels channel,
         fmt::format_string<Args...> fmt, Args &&...args) {
  logWriteDispatch(getDefaultLog(), level, channel, fmt,
                   std::forward<Args>(args)...);
}

} // namespace M::Log

#endif // SUPPORT_LOG_H
