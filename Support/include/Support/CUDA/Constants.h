//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Defines constants used by CUDA Support
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_CUDA_CONSTANTS_H
#define SUPPORT_CUDA_CONSTANTS_H

#include <string_view>

constexpr std::string_view kCUDADriverPath =
#ifdef __linux__
    "/usr/lib/x86_64-linux-gnu/libcuda.so";
#else
    "";
#endif

#endif // SUPPORT_CUDA_CONSTANTS_H
