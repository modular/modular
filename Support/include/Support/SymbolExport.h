//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines macros for exporting symbols.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_EXPORT_H
#define SUPPORT_EXPORT_H

#if (defined(_WIN32) || defined(__CYGWIN__))
#define MODULAR_VISIBILITY_EXPORT __declspec(dllexport)
#else
#define MODULAR_VISIBILITY_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
#define MODULAR_EXPORT extern "C" MODULAR_VISIBILITY_EXPORT
#else
#define MODULAR_EXPORT MODULAR_VISIBILITY_EXPORT
#endif

#define MODULAR_CXX_EXPORT MODULAR_VISIBILITY_EXPORT

// For CompilerRT we need the runtime entry points to have unmangled names,
// but currently do not wish to give them default visibility in any dylib
// they end up within.
#define COMPILERRT_EXPORT extern "C"

#endif // SUPPORT_EXPORT_H
