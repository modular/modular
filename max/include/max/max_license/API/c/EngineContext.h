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

#ifndef GENERICML_API_ENGINECONTEXT_H
#define GENERICML_API_ENGINECONTEXT_H

#include "AsyncRT/ForwardDecls.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "Kernels/max_license/WeightsRegistry.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/SymbolExport.h"
#include "max/max_license/API/c/CompiledModels.h"
#include "max/max_license/API/c/EngineConfig.h"
#include "max/max_license/API/c/TensorMap.h"
#include "llvm/Support/CrashRecoveryContext.h"

#include <memory>

namespace M {
using namespace M::AsyncRT;
using AsyncExecutableModel = AsyncValueRef<std::unique_ptr<M::ExecutableModel>>;
using ExecutableModels = SmallVector<std::unique_ptr<M::ExecutableModel>>;
namespace Engine {
/// Context object used for program compilation and execution. The context
/// object is responsible for owning any objects that are needed globally
/// and is agnostic to the program being executed.
class MODULAR_CXX_EXPORT Context {
public:
  /// Create a RuntimeContext object. This will create 'numThreads'
  /// threads. If \p config has no max context, checks for a global Max context
  /// and reuses it when options match (supporting concurrent
  /// InferenceSessions); otherwise obtains the process context via
  /// Init::getOrCreateContext.
  static ErrorOr<Context> create(const Config &config);

  // TODO: (Joe) come back and document all of these internal functions
  AsyncValueRef<CompiledModels> compileModel(CompilationRequest config) const;

  ErrorOr<CompiledModels> compileModelSync(CompilationRequest config) const;

  static ErrorOrSuccess await(AnyAsyncValueRef value);

  /// Initialize all models in \p compiledModels asynchronously, returning an
  /// async value that resolves to one ExecutableModel per model in MEF order.
  AsyncValueRef<ExecutableModels>
  initModel(AsyncValueRef<CompiledModels> compiledModels,
            const WeightsRegistryRef &weightsRegistry) const;

  /// Initialize a single model asynchronously, returning an async value that
  /// resolves to the initialized ExecutableModel.
  ///
  /// The model pointer must remain valid until the returned async value is
  /// resolved (i.e., the CompiledModels that owns it must outlive this call).
  AsyncExecutableModel
  initModel(const M::Model *model,
            const WeightsRegistryRef &weightsRegistry) const;

  // This returns inputSpecs in internal tensor order which differs from source
  // model tensor order.
  static ErrorOr<llvm::ArrayRef<QuantTensorSpec>>
  getInputSpecs(AsyncValueRef<CompiledModels> compiledModels,
                size_t modelIndex = 0);

  // This returns outputSpecs in internal tensor order which differs from source
  // model tensor order.
  static ErrorOr<llvm::ArrayRef<QuantTensorSpec>>
  getOutputSpecs(AsyncValueRef<CompiledModels> compiledModels,
                 size_t modelIndex = 0);

  static ErrorOr<llvm::ArrayRef<std::string>>
  getInputNames(AsyncValueRef<CompiledModels> compiledModels,
                size_t modelIndex = 0);

  static ErrorOr<llvm::ArrayRef<std::string>>
  getOutputNames(AsyncValueRef<CompiledModels> compiledModels,
                 size_t modelIndex = 0);

  static ErrorOr<std::vector<std::string>>
  getKernelSummaries(const AsyncValueRef<CompiledModels> &compiledModels,
                     size_t modelIndex = 0);

  CPUDevice &getCPUDevice() const { return cpuDevice; }

  AsyncValueRef<TensorMap>
  execute(AsyncExecutableModel executableModel,
          AsyncValueRef<TensorMap> inputTensorMap) const;

  const Config &getConfig() const { return config; }

  /// Resolve a device reference to a device object.
  ErrorOr<ConfiguredDevice> resolveDevice(M::DeviceRef deviceRef) const;

  // Requires handling for the cpuDevice.
  Context(Context &&other);
  Context &operator=(Context &&other) = delete;

  M::ContextRef maxContext;

private:
  explicit Context(const Config &config, M::ContextRef context);

  Config config;
  Context(const Context &) = delete;
  void operator=(const Context &other) = delete;

  // The cpuDevice here either refers to an external cpuDevice, or to the value
  // of ownedRuntime inside context.
  AsyncRT::CPUDevice &cpuDevice;

  llvm::CrashRecoveryContext crashRecoveryContext;
};
} // namespace Engine

} // namespace M

#endif // GENERICML_API_ENGINECONTEXT_H
