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

void setLogLevel(LogLevel level);

LogLevel getLogLevel();

void logWrite(LogLevel level, llvm::StringRef msg);

// This helper function to transform arguments: apply fmt::ptr to non-string
// pointer types and displays them as hex pointers.
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

template <typename... Args>
inline void logWriteDispatch(LogLevel level, Args &&...args) {
  if (static_cast<uint8_t>(getLogLevel()) > static_cast<uint8_t>(level))
    return;

  if constexpr (sizeof...(args) == 0)
    logWrite(level, "");
  else if constexpr (sizeof...(args) == 1)
    logWrite(level, safeFormat("{}", std::forward<Args>(args)...));
  else
    logWrite(level, safeFormat(std::forward<Args>(args)...));
}

template <typename... Args>
inline void log(Args &&...args) {
  if constexpr (sizeof...(args) == 0)
    logWriteDispatch(LogLevel::INFO);
  else if constexpr (std::is_same_v<std::decay_t<std::tuple_element_t<
                                        0, std::tuple<Args...>>>,
                                    LogLevel>)
    logWriteDispatch(std::forward<Args>(args)...);
  else
    logWriteDispatch(LogLevel::INFO, std::forward<Args>(args)...);
}

} // namespace M::Log

#endif // SUPPORT_LOG_H
