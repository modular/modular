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

#ifndef MAX_C_CONTEXT_H
#define MAX_C_CONTEXT_H

#include "max/c/symbol_export.h"
#include "max/c/types.h"

#ifndef __cplusplus
#include <stdbool.h>
#endif

/// Creates a new runtime config.
///
/// This configures runtime details such as the number of threads and log level.
///
/// By default, the config object's number of threads will be set to `0`, which
/// is internally used to refer to the number of physical processors in the
/// first socket in the system. You can change this with `M_setNumThreads()`.
///
/// You need this as an argument for `M_newRuntimeContext()`.
///
/// @returns A pointer to the new runtime config.  You are responsible for
/// the memory associated with the pointer returned. You can deallocate the
/// memory by calling `M_freeRuntimeConfig()`.
MODULAR_API_EXPORT M_RuntimeConfig *M_newRuntimeConfig();

/// Deallocates the memory for a runtime config.  No-op if `config` is `NULL`.
///
/// @param config The runtime config.
MODULAR_API_EXPORT void M_freeRuntimeConfig(M_RuntimeConfig *config);

/// Adds a device to be accessible from the runtime.
///
/// @param config The runtime config.
/// @param device The device to add to the runtime config.
MODULAR_API_EXPORT void M_runtimeConfigAddDevice(M_RuntimeConfig *config,
                                                 M_Device *device);

/// Creates a runtime context.
///
/// The context is an application-level object that sets up various resources
/// such as threadpool and allocators during inference. You need this
/// before you can call `M_compileModel()`.
///
/// It's expected that there's only one runtime context active in an
/// inference session at a time. We recommended you create
/// one context and use it throughout your application.
///
/// For example:
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
/// @param config The runtime config, from `M_newRuntimeConfig()`.
/// @param status The status object for reporting errors.  It is filled with
///   an error message if construction of the runtime context fails.
///
/// @returns A pointer to the runtime context object.  On success, this is a
///   valid pointer.  On failure, this is a `NULL` pointer with an error message
///   in the status. You are responsible for the memory associated with the
///   pointer returned.  You can deallocate the memory by calling
///   `M_freeRuntimeContext()`.
MODULAR_API_EXPORT M_RuntimeContext *
M_newRuntimeContext(const M_RuntimeConfig *config, M_Status *status);

/// Deallocates the memory for a runtime context.  No-op if `context` is `NULL`.
///
/// @param context The runtime context.
MODULAR_API_EXPORT void M_freeRuntimeContext(M_RuntimeContext *context);

/// Set the options for debugging printing of tensors when executing a model.
///
/// @param context The runtime context.
/// @param style The way the data will be printed.
/// @param precision The floating point print out precision.
/// @param directory The directory to store binary output.
MODULAR_API_EXPORT void M_setDebugPrintOptions(M_RuntimeContext *context,
                                               M_ResultOutputStyle style,
                                               unsigned int precision,
                                               const char *directory);

/// Sets a mojo compile-time define with an boolean value.
///
/// @param context The runtime context.
/// @param key The name of the define.
/// @param value The boolean to set the define to.
MODULAR_API_EXPORT void M_setMojoDefineBool(M_RuntimeContext *context,
                                            const char *key, bool value);

/// Sets a mojo compile-time define with an integer value.
///
/// @param context The runtime context.
/// @param key The name of the define.
/// @param value The integer to set the define to.
MODULAR_API_EXPORT void M_setMojoDefineInt(M_RuntimeContext *context,
                                           const char *key, int value);

/// Sets a mojo compile-time define with an string value.
///
/// @param context The runtime context.
/// @param key The name of the define.
/// @param value The string to set the define to.
MODULAR_API_EXPORT void M_setMojoDefineString(M_RuntimeContext *context,
                                              const char *key,
                                              const char *value);
#endif // MAX_C_CONTEXT_H
