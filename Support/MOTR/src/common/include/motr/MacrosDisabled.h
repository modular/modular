//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_MACROS_DISABLED_H
#define MOTR_MACROS_DISABLED_H

// Disabled MOTR_ Macros
//   This file is is only used when MOTR is disabled and provides empty
//   definitions for public MOTR_ macros detailed in Macros.h

#if defined(MOTR_ENABLED)
#error "cannot #include motr/MacrosDisabled.h when MOTR_ENABLED""
#endif

#define MOTR_LOG(...)

#define MOTR_FlagInt(varname) [[maybe_unused]] uint64_t varname = 0

#define MOTR_TagStr(key, value)
#define MOTR_TagStrView(key, value)
#define MOTR_TagStrViews(key, value)
#define MOTR_TagStrOnce(key, value)

#define MOTR_TagIntVar(varname, key, default_value)                            \
  [[maybe_unused]] uint64_t varname = default_value
#define MOTR_TagInt(key, value)

#define MOTR_Trace(varname)                                                    \
  [[maybe_unused]] uint64_t varname {}
#define MOTR_TraceProgram(varname, program_name)                               \
  [[maybe_unused]] uint64_t varname {}
#define MOTR_TraceProgramArgs(varname, program_name, argc, argv)               \
  [[maybe_unused]] uint64_t varname {}
#endif
