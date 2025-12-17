//===----------------------------------------------------------------------===//
// Copyright (c) 2025, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef MAX_C_COMMON_H
#define MAX_C_COMMON_H

#include "max/c/symbol_export.h"
#include "max/c/types.h"

/// Gets the MAX Engine version.
///
/// @returns A string containing the semantic version of the MAX Engine.
MODULAR_API_EXPORT const char *M_version();

/// Creates a new status object.
///
/// This is required as an argument for several functions, such as
/// `M_newRuntimeContext()` and  `M_compileModel()`. They will update the status
/// object and you can check for errors with `M_isError()` and get the status
/// message with `M_getError()`. For example:
///
/// ```c
/// M_Status *status = M_newStatus();
/// M_RuntimeConfig *runtimeConfig = M_newRuntimeConfig();
/// M_RuntimeContext *context = M_newRuntimeContext(runtimeConfig, status);
/// if (M_isError(status)) {
///   logError(M_getError(status));
///   return EXIT_FAILURE;
/// }
/// ```
///
/// @returns A pointer to the new status object.  You are responsible for
///   the memory associated with the pointer returned. You can
///   deallocate the memory by calling `M_freeStatus()`.
MODULAR_API_EXPORT M_Status *M_newStatus();

/// Gets an error message from the `M_Status` parameter.
///
/// You should call this only if `M_isError()` is true.
///
/// @param status The status object for reporting errors and other messages.
/// @returns A pointer to a null-terminated string containing the error
///   message.
MODULAR_API_EXPORT const char *M_getError(const M_Status *status);

/// Checks if status holds an error value.
///
/// @param status The status object for reporting errors and other messages.
/// @returns `0` if there is no error, `1` otherwise.
MODULAR_API_EXPORT int M_isError(const M_Status *status);

/// Deallocates the memory for the status object.  No-op if `status` is `NULL`.
///
/// @param status The status object for reporting errors and other messages.
MODULAR_API_EXPORT void M_freeStatus(M_Status *status);

#endif // MAX_C_COMMON_H
