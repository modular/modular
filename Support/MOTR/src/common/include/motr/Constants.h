//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_COMMON_CONSTANTS_H
#define MOTR_COMMON_CONSTANTS_H

#include "motr/Hash.h"
#include <array>
#include <string_view>

namespace M::motr::Constants {

// The string constants are constrained by C++ identifier rules:
// constraints are the same as for any C++ identifier:
// - must start with a letter or underscore
// - can only contain letters, digits, and underscores
// - no spaces

// This is a macro that applies a macro to each string in the Constants
// namespace So you only need to add the string to this macro once to make it
// available in the Constants namespace
#define MOTR_STRING_CONSTANTS_APPLY_XMACRO(APPLY)                              \
  APPLY(FlagName)                                                              \
  APPLY(ProcessId)                                                             \
  APPLY(ProgramName)                                                           \
  APPLY(SourceFile)                                                            \
  APPLY(SourceLine)                                                            \
  APPLY(ThreadId)                                                              \
  APPLY(TraceName)                                                             \
  APPLY(argc)                                                                  \
  APPLY(argv)                                                                  \
  APPLY(name)                                                                  \
  APPLY(color)                                                                 \
  APPLY(source_loc)                                                            \
  APPLY(__rpc_fingerprint__)                                                   \
  APPLY(__rpc_request_id__)                                                    \
  APPLY(__rpc_call_name__)                                                     \
  APPLY(UNKNOWN)
// add more strings here and update the count below

constexpr size_t count = 16;

// Define a struct for each key containing the C-string, string_view, and hash
// eg for the
#define MOTR_STRING_CONSTANTS_STRUCT_XMACRO(__STRING__)                        \
  struct __STRING__ {                                                          \
    static constexpr const char *cstr = #__STRING__;                           \
    static constexpr std::string_view sv{#__STRING__};                         \
    static constexpr Hash::Value hash{#__STRING__};                            \
  };

MOTR_STRING_CONSTANTS_APPLY_XMACRO(MOTR_STRING_CONSTANTS_STRUCT_XMACRO);
#undef MOTR_STRING_CONSTANTS_STRUCT_XMACRO

// Define an array of all string views
#define MOTR_STRING_CONSTANTS_ALL_STRING_VIEWS_XMACRO(__STRING__)              \
  std::string_view{#__STRING__},

constexpr std::array<std::string_view, count> allStringViews = {
    MOTR_STRING_CONSTANTS_APPLY_XMACRO(
        MOTR_STRING_CONSTANTS_ALL_STRING_VIEWS_XMACRO)};
#undef MOTR_STRING_CONSTANTS_ALL_STRING_VIEWS_XMACRO

// Undefine the apply macro to cleanup the macro namespace
#undef MOTR_STRING_CONSTANTS_APPLY_XMACRO

} // namespace M::motr::Constants

#endif // MOTR_COMMON_CONSTANTS_H
