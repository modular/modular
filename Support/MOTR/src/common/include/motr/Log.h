//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_LOG_H
#define MOTR_LOG_H

#define FMT_HEADER_ONLY
#include "fmt/format.h"
#include <cstdint>
#include <string>
#include <unistd.h>

enum class LogLevel : uint8_t {
  DEBUG,
  INFO,
  WARNING,
  ERROR,
  FATAL,
};

#define MOTR_LOG_WITH_LEVEL(level, format_str, ...)                            \
  do {                                                                         \
    std::string __msg = fmt::format(format_str, __VA_ARGS__);                  \
    [[maybe_unused]] auto unused0000 = write(1, __msg.c_str(), __msg.size());  \
    [[maybe_unused]] auto unused0001 = write(1, "\n", 1);                      \
    [[maybe_unused]] auto unused0002 = fsync(1);                               \
  } while (0)

#define MOTR_LOG(format_str, ...)                                              \
  MOTR_LOG_WITH_LEVEL(LogLevel::INFO, format_str, __VA_ARGS__)

namespace M::motr {

inline std::string substitute(std::string_view sv, std::string_view from,
                              std::string_view to) {
  if (from.empty())
    return std::string(sv);

  std::string result;
  size_t pos = 0;
  size_t last = 0;

  while ((pos = sv.find(from, last)) != std::string_view::npos) {
    result.append(sv.substr(last, pos - last));
    result.append(to);
    last = pos + from.size();
  }

  result.append(sv.substr(last));
  return result;
}

inline std::string shortString(std::string_view sv, size_t max_len) {
  std::string str = substitute(sv.substr(0, max_len), "\n", "\\n");
  if (sv.size() <= max_len)
    return str;
  return str.substr(0, max_len - 3) + "...";
}

inline std::string summaryString(std::string_view sv, size_t max_len) {
  max_len = max_len < 10 ? 10 : max_len;
  std::string str = substitute(sv, "\n", "\\n");
  if (sv.size() <= max_len)
    return str;
  size_t half_len = (max_len - 5) / 2;

  std::string_view newsv{str};
  std::string_view first_half = newsv.substr(0, half_len);
  std::string_view second_half = newsv.substr(newsv.size() - half_len);

  return fmt::format("{} ... {}", first_half, second_half);
}

} // namespace M::motr

#endif
