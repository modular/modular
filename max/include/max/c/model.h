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

#ifndef MAX_C_MODEL_H
#define MAX_C_MODEL_H

#include "max/c/symbol_export.h"
#include "max/c/types.h"

/// Creates an object you can use to configure model compilation.
///
/// You need `M_CompileConfig` as an argument for several functions, including
/// `M_setModelPath()` and `M_compileModel()`.
///
/// @returns A pointer to a new compilation configuration. You are responsible
/// for the memory associated with the pointer returned. You can deallocate the
/// memory by calling `M_freeCompileConfig()`. This compilation configuration
/// can only be used for a single compilation call. Any subsequent compilations
/// must be passed a new `M_CompileConfig` (created by calling
/// `M_newCompileConfig()` again).
MODULAR_API_EXPORT M_CompileConfig *M_newCompileConfig();

/// Sets the path to a model.
///
/// You must call this before you call `M_compileModel()`.
/// Otherwise, `M_compileModel()` returns an error in `status`.
///
///
/// @param compileConfig The compilation configuration for your model, from
/// `M_newCompileConfig()`.
/// @param path The path to your model. The model does not need to exist on the
/// filesystem at this point. This follows the same semantics and expectations
/// as `std::filesystem::path`.
MODULAR_API_EXPORT void M_setModelPath(M_CompileConfig *compileConfig,
                                       const char *path);

/// Compiles a model.
///
/// This immediately returns an `M_AsyncCompiledModel`, with compilation
/// happening asynchronously. If you need to block to await compilation, you can
/// then call `M_waitForCompilation()`.
///
/// You must call `M_setModelPath()` before you call this. For example:
///
/// ```c
/// M_CompileConfig *compileConfig = M_newCompileConfig();
/// M_setModelPath(compileConfig, modelPath);
/// M_AsyncCompiledModel *compiledModel =
///     M_compileModel(context, &compileConfig, status);
/// if (M_isError(status)) {
///   logError(M_getError(status));
///   return EXIT_FAILURE;
/// }
/// ```
///
/// The `M_AsyncCompiledModel` returned here is not ready for inference yet.
/// You need to then initialize the model with `M_initModel()`.
///
/// @param context The runtime context, from `M_newRuntimeContext()`.
/// @param compileConfig Address of compilation configuration for your
/// model created with `M_newCompileConfig()`, and with the model set via
/// `M_setModelPath()`. Ownership of configuration is handed over to API.
/// @param status The status used to report errors in the case of failures
/// during model compilation.
///
/// @returns A pointer to an `M_AsyncCompiledModel`. You are responsible
/// for the memory associated with the pointer returned. You can deallocate the
/// memory by calling `M_freeCompiledModel()`. If the config is invalid, it
/// returns a `NULL` pointer.  If the model compilation fails, the
/// pointer is `NULL` and the `status` parameter contains an error message.
/// `compileConfig`  will be reset to `NULL` after this call irrespective of
/// status and cannot be reused, and any subsequent calls must
/// take a new `M_CompileConfig`.
MODULAR_API_EXPORT M_AsyncCompiledModel *
M_compileModel(const M_RuntimeContext *context, M_CompileConfig **compileConfig,
               M_Status *status);

/// Blocks execution until the model is compiled.
///
/// This waits for the async compiled model to be complete after calling
/// `M_compileModel()`.  When this function returns, the model is resolved to
/// either a compiled model or an error.
///
/// @param compiledModel The model received from `M_compileModel()`.
/// @param status The status used to report errors in the case of failures.
MODULAR_API_EXPORT void
M_waitForCompilation(M_AsyncCompiledModel *compiledModel, M_Status *status);

/// Synchronously compiles a model.
///
/// Unlike `M_compileModel()`, this blocks until model compilation is complete.
/// It returns an `M_AsyncCompiledModel` without needing to call
/// `M_waitForCompilation()`. All other setup and usage is identical to
/// `M_compileModel()`.
///
/// @param context The runtime context, from `M_newRuntimeContext()`.
/// @param compileConfig Address of compilation configuration for your
/// model created with `M_newCompileConfig()`, and with the model set via
/// `M_setModelPath()`. Ownership of configuration is handed over to API.
/// @param status The status used to report errors in the case of failures
/// during model compilation.
///
/// @returns A pointer to an `M_AsyncCompiledModel`. You are responsible
/// for the memory associated with the pointer returned. You can deallocate the
/// memory by calling `M_freeCompiledModel()`. If the config is invalid, it
/// returns a `NULL` pointer.  If the model compilation fails, the
/// pointer is `NULL` and the `status` parameter contains an error message.
/// `compileConfig`  will be reset to `NULL` after this call irrespective of
/// status and cannot be reused, and any subsequent calls must take a new
/// `M_CompileConfig`.
MODULAR_API_EXPORT M_AsyncCompiledModel *
M_compileModelSync(const M_RuntimeContext *context,
                   M_CompileConfig **compileConfig, M_Status *status);

/// Sets up a model for execution.
///
/// You can call this immediately after `M_compileModel()`—you don't need to
/// wait for the async compilation.
///
/// This function also returns immediately with model initialization happening
/// asynchronously. For example:
///
/// ```c
/// M_AsyncModel *model = M_initModel(
///   context, compiledModel, weightsRegistry, status);
/// if (M_isError(status)) {
///   logError(M_getError(status));
///   return EXIT_FAILURE;
/// }
/// ```
///
/// If you want to block until `M_AsyncModel` is initialized, you can call
/// `M_waitForModel()`, but that's not necessary and you can immediately call
/// `M_executeModelSync()`.
///
/// @param context The runtime context, from `M_newRuntimeContext()`.
/// @param compiledModel The compiled model, from `M_compileModel()`.
/// @param weightsRegistry A mapping from weights' names to their data.
/// The weights registry is used to update weights or otherwise pass weights to
/// the model init block at runtime, without recompiling the model graph.
/// If the model doesn't use the weights registry, it is safe to pass as NULL
/// @param status The status used to report errors in the case of failures. The
/// status contains an error only if the given context or compiled model is
/// invalid.  Other errors will not surface until the next synchronization
/// point.
///
/// @returns A pointer to an `M_AsyncModel` that holds an async value to
/// a compiled model. You are responsible for the memory associated with
/// the pointer returned. You can deallocate the memory by calling
/// `M_freeModel()`.  If model initialization fails, the `status` parameter
/// contains an error message.
MODULAR_API_EXPORT M_AsyncModel *
M_initModel(const M_RuntimeContext *context,
            const M_AsyncCompiledModel *compiledModel,
            const M_WeightsRegistry *weightsRegistry, M_Status *status);

/// Blocks execution until the model is initialized.
///
/// This waits for the model setup to finish in `M_initModel()`.
///
/// @param model The model.
/// @param status The status used to report errors in the case of failures.
MODULAR_API_EXPORT void M_waitForModel(M_AsyncModel *model, M_Status *status);

/// Executes a model synchronously.
///
/// The inputs and outputs are `M_AsyncTensorMap` objects to allow chaining of
/// inference. This operation is blocking and waits until the output results are
/// ready.
///
/// @param context The runtime context.
/// @param initializedModel The model to execute, from `M_initModel()`. Although
/// that function is async, you can pass the `M_AsyncModel` here immediately.
/// @param inputs The tensor inputs.
/// @param status The status used to report errors in the case of failures.
/// This includes failures encountered while running the model; there is no
/// need for an explicit synchronization point.
///
/// @returns A pointer to an `M_AsyncTensorMap` that holds the output tensors.
/// These tensors are in a resolved state. You are responsible for the memory
/// associated with the pointer returned.  You can deallocate the memory by
/// calling `M_freeAsyncTensorMap()`. In the case that executing the model
/// fails, the `status` parameter contains an error message.
MODULAR_API_EXPORT M_AsyncTensorMap *
M_executeModelSync(const M_RuntimeContext *context,
                   M_AsyncModel *initializedModel, M_AsyncTensorMap *inputs,
                   M_Status *status);

/// Deallocates the memory for the model.  No-op if `model` is `NULL`.
///
/// @param model The model to deallocate.
MODULAR_API_EXPORT void M_freeModel(M_AsyncModel *model);

/// Deallocates the memory for the compiled model.  No-op if `model` is `NULL`.
///
/// @param model The compiled model to deallocate.
MODULAR_API_EXPORT void M_freeCompiledModel(M_AsyncCompiledModel *model);

/// Deallocates the memory for the compile config.  No-op if `config` is `NULL`.
///
/// @param config The compilation configuration to deallocate.
MODULAR_API_EXPORT void M_freeCompileConfig(M_CompileConfig *config);

#endif // MAX_C_MODEL_H
