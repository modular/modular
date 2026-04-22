//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// MLOG is a macro that takes a variable number of arguments and logs them to
// a file or the console.
//
// Zero args prints a newline to default INFO level.
// MLOG(); // "\n" is printed.
//
// If the single arg is a string, it is printed to the INFO level with a
// newline. MLOG(""); // "\n" is printed to the INFO level.
// MLOG("hello");
// "hello" is printed to the INFO level.
//
// If the single arg is a log level, a newline is printed at the given level.
// MLOG(LogLevel::DEBUG); // "\n" is printed at the DEBUG level.
//
// Two args or more is a format string and a value, or values
// MLOG("{}", "hello"); // hello
// MLOG("{} {}", "hello", "world"); // hello world
// MLOG("{} {} {}", "hello", "world", "!"); // hello world!
//
// ...unless the first arg is a LogLevel, then everything above
// shifts left
// MLOG(LogLevel::DEBUG); // new line printed at debug level
//
// MLOG(LogLevel::DEBUG, "hi"); // "hi" printed at debug level
// MLOG(LogLevel::DEBUG, "{}", "hello"); // hello printed at debug level
// MLOG(LogLevel::DEBUG, "{} {}", "hello", 42); // hello 42

#ifndef SUPPORT_LOG_H
#define SUPPORT_LOG_H

#include "llvm/ADT/StringRef.h"

#define FMT_EXCEPTIONS 0
#include <fmt/format.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>

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

class Logger;

Logger &getDefaultLog();

void setLogLevel(LogLevel level);

LogLevel getLogLevel(const Logger &log = getDefaultLog());

void logWrite(Logger &log, LogLevel level, llvm::StringRef msg);

// Forwards on most arguments unchanged, but adds special handling for pointer
// types to differentiate between string-like pointers (which should format as
// strings) and other pointers (which should format as their address).
template <typename... Args>
constexpr auto transformArgs(Args &&...args) {
  return std::make_tuple([](auto &&arg) {
    using T = std::decay_t<decltype(arg)>;
    if constexpr (std::is_pointer_v<T>) {
      // Don't cast string-like pointers - let them format as strings.
      if constexpr (std::is_same_v<T, const char *> ||
                    std::is_same_v<T, char *> ||
                    std::is_same_v<T, const wchar_t *> ||
                    std::is_same_v<T, wchar_t *>) {
        return std::forward<decltype(arg)>(arg);
      } else {
        return fmt::format("{}", static_cast<const void *>(arg));
      }
    } else {
      return std::forward<decltype(arg)>(arg);
    }
  }(args)...);
}

// Formats the provided message. If no args, returns the message as-is.
// Otherwise, fmt::vformat to all the arguments in the tuple.
template <typename... Args>
inline std::string safeFormat(std::string_view format_str, Args &&...args) {
  if constexpr (sizeof...(args) == 0) {
    return std::string(format_str);
  } else {
    auto transformedArgs = transformArgs(std::forward<Args>(args)...);

    // Use the provided format string with all arguments.
    return std::apply(
        [format_str](auto &&...transformed) {
          return fmt::vformat(format_str,
                              fmt::make_format_args(transformed...));
        },
        transformedArgs);
  }
}

// Checks the log level to see if it should emit a message, and, if so,
// writes the formatted message.
template <typename... Args>
inline void logWriteDispatch(Logger &log, LogLevel level, Args &&...args) {
  if (getLogLevel(log) > level)
    return;

  if constexpr (sizeof...(args) == 0)
    logWrite(log, level, "");
  else if constexpr (sizeof...(args) == 1)
    logWrite(log, level, safeFormat("{}", std::forward<Args>(args)...));
  else
    logWrite(log, level, safeFormat(std::forward<Args>(args)...));
}

namespace Detail {
// Helper to check if the type at a given index in the parameter pack is a
// specific type.
template <typename T, int Index, typename... Args>
static constexpr auto is_same_at_index_v = std::is_same_v<
    std::decay_t<std::tuple_element_t<Index, std::tuple<Args...>>>, T>;
} // namespace Detail

// Main log entry point (called from the user macros). Checks to see what
// type the arguments being passed to the macros are, to determine how to
// route to logWriteDispatch. Users aren't expected to pass in their own
// Logger instances, but this can still handle the different types if this
// is added later.
template <typename... Args>
inline void log(Args &&...args) {
  if constexpr (sizeof...(args) == 0)
    logWriteDispatch(getDefaultLog(), LogLevel::INFO);
  else if constexpr (sizeof...(args) == 1) {
    if constexpr (Detail::is_same_at_index_v<LogLevel, 0, Args...>)
      logWriteDispatch(getDefaultLog(), std::forward<Args>(args)...);
    else if constexpr (Detail::is_same_at_index_v<Logger, 0, Args...>)
      logWriteDispatch(std::forward<Args>(args)..., LogLevel::INFO);
    else
      logWriteDispatch(getDefaultLog(), LogLevel::INFO,
                       std::forward<Args>(args)...);
  } else {
    if constexpr (Detail::is_same_at_index_v<LogLevel, 0, Args...>)
      logWriteDispatch(getDefaultLog(), std::forward<Args>(args)...);
    else
      logWriteDispatch(getDefaultLog(), LogLevel::INFO,
                       std::forward<Args>(args)...);
  }
}

} // namespace M::Log

#endif // SUPPORT_LOG_H
