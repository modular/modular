//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_JSON_H
#define MOTR_JSON_H

inline const char *&getGlobalJsonTextRef() {
  static const char *global_json_text = "";

  return global_json_text;
}

template <typename T>
inline void my_json_error_handler(const T &e);

#define JSON_THROW_USER(__exception__)                                         \
  do {                                                                         \
    const auto &e = __exception__;                                             \
    my_json_error_handler(e);                                                  \
  } while (0)

#include "motr/Log.h"

// The JSON_THROW_USER macro continues execution, but the nlohmann/json
// library expects it to never return. This is a workaround to suppress the
// warning.

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
#include "nlohmann/json.hpp"
#pragma clang diagnostic pop

// Build the marker line (spaces + caret '^' + newline)
inline std::string build_error_marker(size_t error_pos_in_slice) {
  std::string marker;
  marker.append(error_pos_in_slice, ' '); // spaces up to error
  marker.push_back('^');                  // caret at error position
  marker.push_back('\n');                 // newline
  return marker;
}

// Main error handler
template <typename T>
inline void my_json_error_handler(const T &e) {
  const char *json_text = getGlobalJsonTextRef();
  MOTR_LOG("JSON error[{}]: {}", e.id, e.what());

  // Check if the exception is a parse error
  if (e.id == 101 || e.id == 102 || e.id == 103) {
    // Assuming parse_error_ is a struct with a byte field
    const auto &parse_error =
        reinterpret_cast<const nlohmann::json::parse_error &>(e);
    if (json_text && parse_error.byte != static_cast<std::size_t>(-1)) {
      const size_t context = 20; // how many chars before and after error
      size_t text_len = strlen(json_text);

      size_t error_pos = parse_error.byte;
      size_t start_pos = (error_pos > context) ? (error_pos - context) : 0;
      size_t end_pos =
          (error_pos + context < text_len) ? (error_pos + context) : text_len;

      // Write around error
      std::string_view json_text_view(json_text + start_pos,
                                      end_pos - start_pos);
      MOTR_LOG("Around error:\n{}", json_text_view);

      // Build and write marker
      std::string marker = build_error_marker(error_pos - start_pos);
      MOTR_LOG("Marker:\n{}", marker);
    }
  }

  // abort();
}

namespace M::motr {
inline std::string strip_comments_sv(std::string_view input) {
  std::string output;
  output.reserve(input.size());

  size_t i = 0;
  while (i < input.size()) {
    if (input[i] == '/' && i + 1 < input.size()) {
      if (input[i + 1] == '/') {
        // Skip to end of line
        i += 2;
        while (i < input.size() && input[i] != '\n')
          ++i;
        continue;
      } else if (input[i + 1] == '*') {
        // Skip block comment
        i += 2;
        while (i + 1 < input.size() &&
               !(input[i] == '*' && input[i + 1] == '/'))
          ++i;
        i += 2; // Skip closing */
        continue;
      }
    }
    output += input[i++];
  }
  return output;
}
} // namespace M::motr
#endif // MOTR_JSON_H
