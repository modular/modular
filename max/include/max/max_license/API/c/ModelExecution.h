//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef MAX_C_INTERNAL_MODEL_H
#define MAX_C_INTERNAL_MODEL_H

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/ErrorOr.h"
#include "Support/ML/TensorSpec.h"
#include "Support/SymbolExport.h"
#include "max/c/symbol_export.h"
#include "max/c/types.h"
#include "max/max_license/API/c/EngineContext.h"
#include "llvm/ADT/SmallVector.h"

namespace M::GraphCompiler::Driver {

/// A single model output: the device buffer holding it, the spec describing
/// its shape and dtype, and the device it is resident on. Aliasing outputs
/// share one buffer, so consumers that wrap these can dedup on the buffer
/// pointer.
struct ExecResult {
  M::Driver::DeviceBufferRef buffer;
  TensorSpec spec;
  DeviceSpecRef device;
};

/// Container holding the result of device execution.
using ExecResultType = llvm::SmallVector<ExecResult>;

/// Validates that a provided tensor spec matches the expected spec.
/// This matches Python API validation behavior:
/// - Checks dtype matches
/// - Checks rank matches
/// - Checks static dimensions match (skips dynamic/symbolic dimensions)
///
/// @param providedSpec The tensor spec provided by the user
/// @param expectedSpec The tensor spec expected by the model
/// @returns Error if validation fails, success otherwise
MODULAR_CXX_EXPORT
ErrorOrSuccess validateInputSpec(const TensorSpec &providedSpec,
                                 const TensorSpec &expectedSpec);

/// Executes a model with Python DeviceTensor inputs and/or Mojo values.
///
/// @param context The runtime context.
/// @param executableModel The model to execute.
/// @param inputValues Input values, representing the backing memory for
/// device-resident tensors or Mojo values.
/// @param inputCount Number of inputs.
///
/// @returns A list of driver tensor memories and/or Mojo values.
/// The caller owns the results and should destroy and free them.
MODULAR_CXX_EXPORT
ErrorOr<ExecResultType>
extractDriverResults(const M::Engine::Context &context,
                     M::AsyncExecutableModel executableModel,
                     MutableArrayRef<AnyAsyncValueRef> results);

/// Executes a model with Python DeviceTensor inputs and/or Mojo values.
///
/// @param context The runtime context.
/// @param executableModel The model to execute.
/// @param inputValues Input values, representing the backing memory for
/// device-resident tensors or Mojo values.
/// @param inputCount Number of inputs.
///
/// @returns A list of driver tensor memories and/or Mojo values.
/// The caller owns the results and should destroy and free them.
MODULAR_CXX_EXPORT
ErrorOr<ExecResultType>
executeFromDriver(const M::Engine::Context &context,
                  M::AsyncExecutableModel executableModel,
                  SmallVectorImpl<AnyAsyncValueRef> &inputValues);
} // namespace M::GraphCompiler::Driver

#endif // MAX_C_INTERNAL_MODEL_H
