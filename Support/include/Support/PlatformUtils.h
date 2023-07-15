//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares abstractions for platform specific macros.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_PLATFORM_UTILS_H
#define SUPPORT_PLATFORM_UTILS_H

#if defined(__x86_64__) || defined(__x86_64) || defined(_M_AMD64) ||           \
    defined(_M_X64)
#define MODULAR_X86_64 1
#elif defined(__ARM_NEON__) || defined(__ARM_NEON)
#define MODULAR_ARM_NEON 1
#endif

#endif // SUPPORT_PLATFORM_UTILS_H
