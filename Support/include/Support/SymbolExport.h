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

#define MODULAR_EXPORT extern "C" MODULAR_VISIBILITY_EXPORT
#define MODULAR_CXX_EXPORT MODULAR_VISIBILITY_EXPORT

#endif // SUPPORT_EXPORT_H