//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_COMMON_HASH_H
#define MOTR_COMMON_HASH_H

#include "motr/Macros.h"

#include <algorithm>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <vector>

// Define REPO_ROOT, defaulting to an empty string if not defined
#ifndef MOTR_REPO_ROOT
#define MOTR_REPO_ROOT ""
#endif

namespace M::motr::detail {
// Compile-time check for string literals
template <typename T>
constexpr bool is_string_literal(T &&) {
  return false;
}

// Overload that only matches string literals
template <std::size_t N>
constexpr bool is_string_literal(const char (&)[N]) {
  return true;
}
} // namespace M::motr::detail

// MOTR_TOKEN_HASH(TOKEN_STRING_LITERAL)
//
// Computes a compile-time hash of:
//  1. the TOKEN_STRING_LITERAL string literal
//  2. __FILE__ (stripped of MOTR_REPO_ROOT)
//  3. __LINE__
//
// Requirements (statically checked at compile time)
// - TOKEN_STRING_LITERAL must be a string literal
// __FILE__ must start with MOTR_REPO_ROOT
//
// Purpose:
//   Used to compute a deterministic hash for any given TOKEN_STRING_LITERAL
//   at a source code location.
//
//   This hash an be used for separate tools to identify the same token
//   by keeping a reverse mapping of hashes to back to TOKEN_STRING_LITERAL,
//   __FILE__, and __LINE__
//
//   This is useful for performace critical code where we want to identify
//   mark specific locations in the code with a lightweight (64-bit) unique
//   identifier
//

constexpr const uint64_t fnv1aInit = 0xcbf29ce484222325ull;
constexpr const uint64_t fnv1aMul = 0x100000001b3ull;

constexpr uint64_t ConstExprHashCString(const char *str_param) {
  uint64_t h = fnv1aInit;
  while (*str_param) {
    h ^= static_cast<uint8_t>(*str_param++);
    h *= fnv1aMul;
  }
  return h;
}

constexpr uint64_t ConstExprHashCString2(const char *str_param, size_t size) {
  uint64_t h = fnv1aInit;
  for (size_t i = 0; i < size; ++i) {
    h ^= static_cast<uint8_t>(str_param[i]);
    h *= fnv1aMul;
  }
  return h;
}

namespace M::motr::Hash {

struct Value {
  uint64_t v;
  constexpr Value() : v(0) {}
  constexpr Value(uint64_t v) : v(v) {}
  constexpr Value(std::string_view str)
      : v(ConstExprHashCString2(str.data(), str.size())) {}

  constexpr bool operator<(const Value &other) const { return v < other.v; }
  constexpr bool operator==(const Value &other) const { return v == other.v; }
  constexpr bool operator!=(const Value &other) const { return v != other.v; }
  constexpr bool operator>(const Value &other) const { return v > other.v; }
  constexpr bool operator<=(const Value &other) const { return v <= other.v; }
  constexpr bool operator>=(const Value &other) const { return v >= other.v; }

  constexpr bool operator==(const uint64_t &other) const { return v == other; }
  constexpr bool operator!=(const uint64_t &other) const { return v != other; }
  constexpr bool operator<(const uint64_t &other) const { return v < other; }
  constexpr bool operator<=(const uint64_t &other) const { return v <= other; }
  constexpr bool operator>(const uint64_t &other) const { return v > other; }
  constexpr bool operator>=(const uint64_t &other) const { return v >= other; }
};

struct VectorHash {
  uint64_t v{};
  template <typename T>
  VectorHash(const std::vector<T> &vec)
      : v(ConstExprHashCString2(reinterpret_cast<const char *>(vec.data()),
                                vec.size() * sizeof(T))) {}
};

struct SetFingerprint {
  uint64_t x0r = 0;
  int64_t sum = 0;
  void add(uint64_t value) {
    x0r ^= value;
    sum += value;
  }

  void add(Value value) { add(value.v); }

  uint64_t get() const { return (x0r + sum) ^ sum; }
};

struct DeterministicHash {
  uint64_t v;
  template <typename T>
  DeterministicHash(const std::vector<T> &unsorted_vec) {
    std::vector<T> sorted_vec = unsorted_vec;
    std::sort(sorted_vec.begin(), sorted_vec.end());
    v = VectorHash(sorted_vec).v;
  }
  Value asValue() const { return Value{v}; }
};

} // namespace M::motr::Hash

namespace std {
template <>
struct hash<M::motr::Hash::Value> {
  size_t operator()(const M::motr::Hash::Value &h) const { return h.v; }
};
} // namespace std

// MOTR_TOKEN_HASH_SINGLE is used to calculate source location free hash
// equivalent to MOTR_TOKEN_HASH_3(TOKEN_STRING_LITERAL, "", 0)
// todo: Since this does not use __FILE__ and __LINE__, it is not
//       necessary to be macro and should be converted to a pure constexpr
#define MOTR_TOKEN_HASH_SINGLE(TOKEN_STRING_LITERAL)                           \
  []() constexpr {                                                             \
    static_assert(                                                             \
        M::motr::detail::is_string_literal(TOKEN_STRING_LITERAL),              \
        "\n\n\n*** MOTR_TOKEN_HASH ERROR***\nMOTR_TOKEN_HASH requires a "      \
        "string literal.\nTOKEN_STRING_LITERAL=" #TOKEN_STRING_LITERAL         \
        "\n\n\n");                                                             \
    constexpr uint64_t hash = []() constexpr {                                 \
      uint64_t h = fnv1aInit;                                                  \
      const char *str_param = TOKEN_STRING_LITERAL;                            \
      while (*str_param) {                                                     \
        h ^= static_cast<uint8_t>(*str_param++);                               \
        h *= fnv1aMul;                                                         \
      }                                                                        \
      return h;                                                                \
    }();                                                                       \
    return hash;                                                               \
  }()

#define MOTR_TOKEN_HASH_3(TOKEN_STRING_LITERAL, FILE, LINE)                    \
  []() constexpr {                                                             \
    static_assert(                                                             \
        M::motr::detail::is_string_literal(TOKEN_STRING_LITERAL),              \
        "\n\n\n*** MOTR_TOKEN_HASH ERROR***\nMOTR_TOKEN_HASH requires a "      \
        "string literal.\nTOKEN_STRING_LITERAL=" #TOKEN_STRING_LITERAL         \
        "\n\n\n");                                                             \
    static_assert(                                                             \
        []() constexpr {                                                       \
          const char *str = TOKEN_STRING_LITERAL;                              \
          while (*str) {                                                       \
            if (*str++ == '\n')                                                \
              return false;                                                    \
          }                                                                    \
          return true;                                                         \
        }(),                                                                   \
        "\n\n\n*** MOTR_TOKEN_HASH ERROR***\nTOKEN_STRING_LITERAL must not "   \
        "contain newline "                                                     \
        "characters.\nTOKEN_STRING_LITERAL=" #TOKEN_STRING_LITERAL "\n\n\n");  \
    static_assert(                                                             \
        []() constexpr {                                                       \
          const char *file = bool(FILE) ? FILE : ""; /* FILE is optional */    \
          const char *prefix = MOTR_REPO_ROOT;                                 \
          while (*prefix) {                                                    \
            if (*file++ != *prefix++)                                          \
              return false;                                                    \
          }                                                                    \
          return true;                                                         \
        }(),                                                                   \
        "\n\n\n*** MOTR_TOKEN_HASH ERROR***\n__FILE__ must start with "        \
        "MOTR_REPO_ROOT\n__FILE__      : " __FILE__                            \
        "\nMOTR_REPO_ROOT: " MOTR_REPO_ROOT "\n\n\n");                         \
    constexpr uint64_t hash = []() constexpr {                                 \
      uint64_t h = fnv1aInit;                                                  \
      if (LINE != 0) {                                                         \
        /* Don't hash the line number if it is 0 */                            \
        /* Allows matching simple fnv1a(str) hash behavior */                  \
        /* When called via MOTR_TOKEN_HASH_3(str, "", 0) */                    \
        h ^= static_cast<uint64_t>(LINE);                                      \
        h *= fnv1aMul;                                                         \
      }                                                                        \
      { /* Remove MOTR_REPO_ROOT from FILE */                                  \
        const char *file = FILE;                                               \
        const char *prefix = MOTR_REPO_ROOT;                                   \
        while (*file && *prefix && *file == *prefix) {                         \
          ++file;                                                              \
          ++prefix;                                                            \
        }                                                                      \
        /* Hash the remaining characters in FILE */                            \
        while (*file) {                                                        \
          h ^= static_cast<uint8_t>(*file++);                                  \
          h *= fnv1aMul;                                                       \
        }                                                                      \
      }                                                                        \
      { /* Hash the characters in TOKEN_STRING_LITERAL */                      \
        const char *str_param = TOKEN_STRING_LITERAL;                          \
        while (*str_param) {                                                   \
          h ^= static_cast<uint8_t>(*str_param++);                             \
          h *= fnv1aMul;                                                       \
        }                                                                      \
      }                                                                        \
      return h;                                                                \
    }();                                                                       \
    return hash;                                                               \
  }()

#define MOTR_TOKEN_HASH(TOKEN_STRING_LITERAL)                                  \
  MOTR_TOKEN_HASH_3(TOKEN_STRING_LITERAL, __FILE__, __LINE__)

#endif // MOTR_TOKEN_HASH_H
