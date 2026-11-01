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

#ifndef GENERICML_MGPPRIMITIVES_MGPPRIMITIVES_H
#define GENERICML_MGPPRIMITIVES_MGPPRIMITIVES_H

#include "GraphCompiler/Support/CompiledKernelsContext.h"
#include "Mojo/ExecutionEngine/ExecutionEngine.h"
#include "Support/ReferenceCounted.h"

namespace M {
class PrimitiveRegistry;

namespace MGP {

/// Storage of the loaded kernel modules for CPU/host kernel execution.
struct KernelExecutionState : public ReferenceCounted<KernelExecutionState> {
  // Execution engine held by the cpuDevice and shared by all loaded modules.
  KGEN::ExecutionEngine *engine;

  // The kernel context for each loaded module. Modules are referenced by
  // index into this vector.
  SmallVector<RCRef<GML::CompiledKernelsContext>> modules;

  KernelExecutionState(KGEN::ExecutionEngine *engine) : engine(engine) {}

  ~KernelExecutionState() {
    if constexpr (ProfilerEntry<Trace::EnableTrace(Trace::kMojo, 1),
                                Trace::kMojo>::isEnabled()) {
      // The profiler borrows pointers to Mojo static strings (on
      // `Trace.__enter__()`/`__exit__()`) that live in the jitted object files
      // this dtor unloads. Intern them so `TimeTraceProfiler::write()` still
      // sees valid strings.
      if (auto cpuDevice = AsyncRT::CPUDevice::getCurrentCPUDeviceOrNull()) {
        if (std::optional<TimeTraceProfiler> &profiler =
                cpuDevice->getProfiler())
          profiler->intern();
      }
    }
  }

  /// Returns the compiled kernel at the given module and kernel indexes.
  GML::CompiledKernelFn *getKernel(size_t moduleIndex,
                                   size_t kernelIndex) const {
    ASSERT_STREAM(moduleIndex < modules.size(),
                  << "invalid module index " << moduleIndex);
    return modules[moduleIndex]->getKernel(kernelIndex);
  }
  /// Returns the compiled kernelAsyncABI at the given index.
  GML::CompiledKernelAsyncABIFn *getKernelAsyncABI(size_t moduleIndex,
                                                   size_t kernelIndex) const {
    ASSERT_STREAM(moduleIndex < modules.size(),
                  << "invalid module index " << moduleIndex);
    return modules[moduleIndex]->getKernelAsyncABI(kernelIndex);
  }
};

} // namespace MGP

/// Registers the device-generic subset of the 'mgp' dialect primitives for
/// "cpu" devices.
void registerMGPCPUPrimitives(PrimitiveRegistry &registry);

/// Registers all the 'mgp' dialect primitives.
void registerMGPPrimitives(PrimitiveRegistry &registry);

} // namespace M

#endif // GENERICML_MGPPRIMITIVES_MGPPRIMITIVES_H
