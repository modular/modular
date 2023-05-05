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
#ifdef MODULAR_BUILDING_LIBRARY
#define MODULAR_VISIBILITY_EXPORT __declspec(dllexport)
#else
#define MODULAR_VISIBILITY_EXPORT __declspec(dllimport)
#endif
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

#if (defined(_WIN32) || defined(__CYGWIN__))
#ifdef MODULAR_BUILDING_COMPILERRT
#define COMPILERRT_VISIBILITY_EXPORT __declspec(dllexport)
#else
#define COMPILERRT_VISIBILITY_EXPORT __declspec(dllimport)
#endif
#else
#define COMPILERRT_VISIBILITY_EXPORT __attribute__((visibility("default")))
#endif

#if (defined(_WIN32) || defined(__CYGWIN__))
#ifdef MODULAR_BUILDING_DRIVER
#define DRIVER_VISIBILITY_EXPORT __declspec(dllexport)
#else
#define DRIVER_VISIBILITY_EXPORT __declspec(dllimport)
#endif
#else
#define DRIVER_VISIBILITY_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
#define MODULAR_DRIVER_EXPORT extern "C" DRIVER_VISIBILITY_EXPORT
#else
#define MODULAR_DRIVER_EXPORT DRIVER_VISIBILITY_EXPORT
#endif

#endif // SUPPORT_EXPORT_H
