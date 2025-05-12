//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef ASYNCRT_NIXL_UTILS_COMMON_STATUS_H
#define ASYNCRT_NIXL_UTILS_COMMON_STATUS_H

#include "AsyncRT/NIXL/utils/common/nixl_log.h"
#include "llvm/Support/FormatVariadic.h"
#include <iostream>

#define NIXL_LOG_AND_RETURN_IF_ERROR(status, message)                          \
  do {                                                                         \
    if ((status) != nixl_status_t::NIXL_SUCCESS && (status) != NIXL_IN_PROG) { \
      llvm::errs() << llvm::formatv("Error: {0} - {1}", (status), (message));  \
      return (status);                                                         \
    }                                                                          \
  } while (0)

#define NIXL_RETURN_IF_NOT_IN_PROG(status)                                     \
  do {                                                                         \
    if ((status) != NIXL_IN_PROG) {                                            \
      NIXL_LOG_AND_RETURN_IF_ERROR(                                            \
          status, " Received handle with pre-existing error");                 \
      return (status);                                                         \
    }                                                                          \
  } while (0)

#endif // ASYNCRT_NIXL_UTILS_COMMON_STATUS_H
