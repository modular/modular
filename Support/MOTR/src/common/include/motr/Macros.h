//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_MACROS_H
#define MOTR_MACROS_H

// MOTR Macros
//   All macros are prefixed with MOTR_
//
// Flags:
//   MOTR_FlagInt(varname)  Used to create a synchronized atomic integer flag
//
// Tags:
//   MOTR_TagStr(key, value)  Used to tag key string literals at compile time
//   MOTR_TagStrView(key, value)  Used to tag a string literal and string_view
//   MOTR_TagStrViews(key, value)  Used to tag string value at runtime
//   MOTR_TagStrOnce(key, value)  Used to tag a string literals once per run
//   MOTR_TagIntVar(varname, key, default_value)  Used create a tag integer
//   value MOTR_TagInt(key, value)  Used to tag an integer value immediately
//
// Tracing:
//   MOTR_Trace(varname)  Create a trace span named `varname`
//   MOTR_TraceProgram(varname, "program name")  Used to trace a program
//   MOTR_TraceProgramArgs(varname, "program name", argc, argv)
//        Used to trace a program and log all command line arguments
//
// Other:
//   MOTR_ALWAYS_INLINE  Used to mark a function as always inline
//   MOTR_STRINGIFY(x)  Used to convert a string to a string literal
//   MOTR_TOSTRING(x)  Used to convert a string to a string literal
//   MOTR_LINESTR  Used to get the current line number as a string

#if MOTR_ENABLED != 1
#error "motr/Macros.h is included when MOTR_ENABLED is not defined or is not 1"
#endif

#define MOTR_ALWAYS_INLINE __attribute__((always_inline)) inline

#define MOTR_STRINGIFY(x) #x
#define MOTR_TOSTRING(x) MOTR_STRINGIFY(x)
#define MOTR_LINESTR MOTR_TOSTRING(__LINE__)

// MOTR_FlagInt(varname)
//
// Used to create a synchronized atomic integer flag
//
// "Synchronized" means that the flag is stored in shared memory
// and is synchronized across all processes on the same machine
//
//    Note: there can be higher level synchornization that happens between
//    multiple machines e.g. `motr server` will synchronize flags with motr gui
//    web clients via websockets
//
// varname is the both the name of the variable
// and "varname" is the string name of the flag
// uint64_t is the type of the flag
//
// The flag is stored in shared memory
// and is synchronized across processes
#define MOTR_FlagInt(varname)                                                  \
  static constexpr const char __FlagName_##varname[] = #varname;               \
  namespace Key = ::M::motr::Constants;                                        \
  MOTR_TagStrOnce(Key::FlagName::cstr, #varname);                              \
  MOTR_TagStrOnce(Key::SourceFile::cstr, __FILE__);                            \
  MOTR_TagInt(Key::SourceLine::cstr, __LINE__);                                \
  [[maybe_unused]] M::motr::Flags::FlagT<__FlagName_##varname, uint64_t> varname

#define MOTR_static_FlagInt(varname)                                           \
  static constexpr const char __FlagName_##varname[] = #varname;               \
  namespace Key = ::M::motr::Constants;                                        \
  MOTR_TagStrOnce(Key::FlagName::cstr, #varname);                              \
  MOTR_TagStrOnce(Key::SourceFile::cstr, __FILE__);                            \
  MOTR_TagInt(Key::SourceLine::cstr, __LINE__);                                \
  [[maybe_unused]] static M::motr::Flags::FlagT<__FlagName_##varname,          \
                                                uint64_t> varname

#define MOTR_TagStrView(__key_str, __val_sv)                                   \
  do {                                                                         \
    namespace motr = ::M::motr;                                                \
    constexpr const std::string_view __key_sv{__key_str};                      \
    constexpr const motr::Hash::Value __key_hash{__key_sv};                    \
    constexpr const motr::Hash::Value __val_hash{__val_sv};                    \
    motr::TagStr(__key_hash, __val_hash);                                      \
    motr::ServerOutboxString::send({__key_sv, __val_sv});                      \
  } while (0)

#define MOTR_TagStrViews(__key_str, __val_str)                                 \
  do {                                                                         \
    namespace motr = ::M::motr;                                                \
    const std::string_view __key_sv{__key_str};                                \
    const std::string_view __val_sv{__val_str};                                \
    const motr::Hash::Value __key_hash{__key_sv};                              \
    const motr::Hash::Value __val_hash{__val_sv};                              \
    motr::TagStr(__key_hash, __val_hash);                                      \
    motr::ServerOutboxString::send({__key_sv, __val_sv});                      \
  } while (0)

#define MOTR_TagStr(__key_str, __val_str)                                      \
  do {                                                                         \
    namespace motr = ::M::motr;                                                \
    constexpr std::string_view __key_sv{__key_str};                            \
    constexpr std::string_view __val_sv{__val_str};                            \
    constexpr motr::Hash::Value __key_hash{__key_sv};                          \
    constexpr motr::Hash::Value __val_hash{__val_sv};                          \
    motr::TagStr(__key_hash, __val_hash);                                      \
    motr::ServerOutboxString::send({__key_sv, __val_sv});                      \
  } while (0)

#define MOTR_TagStrOnce(__key_str, __val_str)                                  \
  do {                                                                         \
    namespace motr = ::M::motr;                                                \
    constexpr std::string_view __key_sv{__key_str};                            \
    constexpr std::string_view __val_sv{__val_str};                            \
    constexpr motr::Hash::Value __key_hash{__key_sv};                          \
    constexpr motr::Hash::Value __val_hash{__val_sv};                          \
    motr::TagStrOnce<__key_hash.v, __val_hash.v>{};                            \
    motr::ServerOutboxString::send({__key_sv, __val_sv});                      \
  } while (0)

#define MOTR_TraceProgram(__varname, __program_name_str)                       \
  [[maybe_unused]] ::M::motr::TraceSpan __varname;                             \
  do {                                                                         \
    namespace motr = ::M::motr;                                                \
    namespace Key = ::M::motr::Constants;                                      \
    MOTR_TagStrOnce(Key::TraceName::cstr, #__varname);                         \
    MOTR_TagStrOnce(Key::ProgramName::cstr, __program_name_str);               \
    MOTR_TagInt(Key::SourceLine::cstr, __LINE__);                              \
    MOTR_TagStrOnce(Key::SourceFile::cstr, __FILE__);                          \
  } while (0)

#define MOTR_TraceProgramArgs(__varname, __program_name_str, __argc, __argv)   \
  [[maybe_unused]] ::M::motr::TraceSpan __varname;                             \
  do {                                                                         \
    namespace Key = ::M::motr::Constants;                                      \
    MOTR_TagStrOnce(Key::TraceName::cstr, #__varname);                         \
    MOTR_TagStrOnce(Key::ProgramName::cstr, __program_name_str);               \
    MOTR_TagInt(Key::SourceLine::cstr, __LINE__);                              \
    MOTR_TagStrOnce(Key::SourceFile::cstr, __FILE__);                          \
    MOTR_TagInt(Key::argc::cstr, __argc);                                      \
    for (int i = 0; i < __argc; i++) {                                         \
      std::string key = fmt::format("argv[{}]", i);                            \
      MOTR_TagStrViews(key, __argv[i]);                                        \
    }                                                                          \
  } while (0)

#define MOTR_Trace(__varname)                                                  \
  [[maybe_unused]] ::M::motr::TraceSpan __varname;                             \
  do {                                                                         \
    namespace Key = ::M::motr::Constants;                                      \
    MOTR_TagStr(Key::TraceName::cstr, #__varname);                             \
    MOTR_TagInt(Key::SourceLine::cstr, __LINE__);                              \
    MOTR_TagStr(Key::SourceFile::cstr, __FILE__);                              \
  } while (0)

#define MOTR_Trace2(__varname, __name)                                         \
  [[maybe_unused]] ::M::motr::TraceSpan __varname;                             \
  do {                                                                         \
    namespace Key = ::M::motr::Constants;                                      \
    MOTR_TagStrView(Key::TraceName::cstr, __name);                             \
    MOTR_TagInt(Key::SourceLine::cstr, __LINE__);                              \
    MOTR_TagStr(Key::SourceFile::cstr, __FILE__);                              \
  } while (0)

#define MOTR_TagIntVar(__varname, __key_str, __default_value)                  \
  constexpr ::M::motr::Hash::Value __varname##_key_hash{__key_str};            \
  [[maybe_unused]] ::M::motr::TagIntOnExit<uint64_t, __varname##_key_hash.v>   \
      __varname;                                                               \
  ::M::motr::ServerOutboxString::send({__key_str});                            \
  __varname = __default_value

#define MOTR_TagInt(__key_str, __value)                                        \
  do {                                                                         \
    MOTR_TagIntVar(tmpvar, __key_str, __value);                                \
  } while (0)

#endif
