//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_COMMON_H
#define MOTR_COMMON_H

#include "motr/Macros.h"

#if defined(__APPLE__)
#define MOTR_PLATFORM_MACOS
#elif defined(__linux__)
#define MOTR_PLATFORM_LINUX
#elif defined(EMSCRIPTEN)
#define MOTR_PLATFORM_EMSCRIPTEN
#endif

#if defined(MOTR_PLATFORM_LINUX)
#ifndef PSHMNAMLEN
// PSHMNAMLEN defines the maximum length for POSIX shared memory names.
// MacOS provides this constant in its system headers, but Linux does not.
// Defining it manually here ensures that our code behaves consistently across
// platforms.
#define PSHMNAMLEN 31 // Define as needed, 31 is common on macOS.
#endif
#endif

#if defined(MOTR_PLATFORM_EMSCRIPTEN)
#ifndef PSHMNAMLEN
// PSHMNAMLEN defines the maximum length for POSIX shared memory names.
#define PSHMNAMLEN 31 // Define as needed, 31 is common on macOS.
#endif
#endif

#if !defined(MOTR_VERSION_MAJOR)
#define MOTR_VERSION_MAJOR 0
#endif

#if !defined(MOTR_VERSION_MINOR)
#define MOTR_VERSION_MINOR 2
#endif

#if !defined(MOTR_VERSION_PATCH)
#define MOTR_VERSION_PATCH 0
#endif

#if !defined(MOTR_VERSION_STRING)
#define MOTR_VERSION_STRING                                                    \
  MOTR_TOSTRING(MOTR_VERSION_MAJOR)                                            \
  "." MOTR_TOSTRING(MOTR_VERSION_MINOR) "." MOTR_TOSTRING(MOTR_VERSION_PATCH)
#endif

#endif // MOTR_COMMON_H
