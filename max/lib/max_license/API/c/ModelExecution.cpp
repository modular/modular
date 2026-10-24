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

#include "max/max_license/API/c/ModelExecution.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "GraphCompiler/Driver/Device/DeviceExecution.h"
#include "Support/Context.h"
#include "Support/ML/DebugPrint.h"
#include "Support/ML/TensorShape.h"
#include "Support/SpanGuard.h"
#include "Support/max_license/Tracy/Tracy.h"
#include "max/max_license/API/c/EngineContext.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"

using namespace M::MLRT;

static std::string formatTensorSpec(const M::TensorSpec &spec) {
  return llvm::formatv(
      "[({}), {}]",
      llvm::join(llvm::map_range(spec,
                                 [](size_t dim) {
                                   return mlir::ShapedType::isDynamic(dim)
                                              ? std::string("*")
                                              : std::to_string(dim);
                                 }),
                 ", "),
      spec.getEltType().getAsString());
}

namespace M::GraphCompiler::Driver {

bool inputSpecsMatch(const TensorSpec &providedSpec,
                     const TensorSpec &expectedSpec) {
  if (providedSpec.getEltType() != expectedSpec.getEltType())
    return false;

  if (providedSpec.getRank() != expectedSpec.getRank())
    return false;

  // Validate static dimensions
  // - allow any value for symbolic and algebraic dimensions
  // - any constraints are validated in the model
  for (auto [expectedDim, providedDim] :
       llvm::zip_equal(expectedSpec, providedSpec))
    if (!mlir::ShapedType::isDynamic(expectedDim) && providedDim != expectedDim)
      return false;

  return true;
}

ErrorOrSuccess validateInputSpec(const TensorSpec &providedSpec,
                                 const TensorSpec &expectedSpec) {
  if (inputSpecsMatch(providedSpec, expectedSpec))
    return success();
  else
    return Error("Buffer of type " + formatTensorSpec(providedSpec) +
                 " does not match expected type " +
                 formatTensorSpec(expectedSpec));
}

namespace {
/// Hierarchical map to allow easy caching of device buffers with Tensor keys.
/// Since TensorSpec is not hashable (and could be slow to hash anyway), but at
/// the same time, it's uncommon to have many tensors with identical buffers
/// but different specs, this is okay.
struct DeviceBufferCache {
  M::Driver::DeviceBufferRef &findOrInsertNull(const Tensor &tensor) {
    auto [it, _] = map.try_emplace(
        KeyTy{tensor.getBuffer(), tensor.getSizeInBytes()}, SpecMapTy());
    SpecMapTy &specMap = it->getSecond();
    for (auto &[spec, buffer] : specMap)
      if (spec == tensor.getSpec())
        return buffer;

    return specMap.emplace_back(tensor.getSpec(), M::Driver::DeviceBufferRef())
        .second;
  }

private:
  using SpecMapTy =
      SmallVector<std::pair<TensorSpec, M::Driver::DeviceBufferRef>>;
  using KeyTy = std::tuple<const void *, size_t>;
  llvm::DenseMap<KeyTy, SpecMapTy> map;
};
} // namespace

ErrorOr<ExecResultType>
extractDriverResults(const M::Engine::Context &context,
                     M::AsyncExecutableModel executableModel,
                     MutableArrayRef<AnyAsyncValueRef> results) {
  TRACY_ZONE_SCOPED_NC("extractDriverResults", TRACY_COLOR_TEAL);

  // Await async model so we can execute synchronously.
  M::AsyncExecutableModel asyncModel = executableModel.copy();

  if (auto err = awaitOrError(asyncModel))
    return err.takeError();

  std::unique_ptr<ExecutableModel> &model = asyncModel.get();
  if (!model)
    return Error::getStaticString("Executable model not available");

  // While we wait for the results (which can take a while), we collect the
  // result devices. This is a local DeviceRef -> Device lookup (distinct from
  // Driver::DeviceMap, which now maps to device contexts, not full Devices).
  llvm::DenseMap<DeviceRef, const M::Engine::ConfiguredDevice *> deviceMap;
  for (const auto &device : context.getConfig().getDevices())
    deviceMap.insert({device.spec->ref, &device});

  SmallVector<const M::Engine::ConfiguredDevice *> resultDevices;
  for (const auto &deviceRef : model->getModel()->getOutputDevices()) {
    if (auto it = deviceMap.find(deviceRef); it != deviceMap.end()) {
      resultDevices.emplace_back(it->second);
    } else {
      return Error("DeviceRef=" + deviceRef.toString() +
                   " needed for output result of execution is not "
                   "configured in InferenceSession.");
    }
  }

  // Also copy some stuff we will need later.
  ContextRef maxContext = context.maxContext.copy();

  if (auto err = awaitOrError(results)) {
    Error error = err.takeError();
    // If the error doesn't already include source notes (Python stack traces),
    // append a hint about how to enable them. See GEX-2678.
    std::string msg = error.get();
    if (msg.find("Source Traceback:") == std::string::npos)
      return Error(msg + "\n\nTo see the Python source location of this op, "
                         "run again with the `max-debug.source-tracebacks` "
                         "config key enabled (for example, "
                         "`Graph.debug.source_tracebacks = True` or "
                         "`MODULAR_DEBUG=source-tracebacks`).");
    return error;
  }

  // Marshal the execution results. NOTE:
  // - Tensors may be aliasing, so we try to unique them.
  // - We only count non-opaque or Tensor values as a result.
  // - We drop the chain, which is the first result of the execute block.
  DeviceBufferCache bufferCache;
  Driver::ExecResultType execResults;
  execResults.reserve(results.size() - 1);
  size_t resultDeviceIndex = 0;
  for (AnyAsyncValueRef &resultRef : results.drop_front()) {
    if (isa<Tensor>(resultRef)) {
      AsyncValueRef<Tensor> tensorRef = std::move(resultRef);
      Tensor &tensor = tensorRef.get();
      // AIPIPE-115: Need an output tensor <--> device mapping.
      const M::Engine::ConfiguredDevice *resultDevice =
          resultDevices[resultDeviceIndex];
      M::Driver::DeviceBufferRef &buffer = bufferCache.findOrInsertNull(tensor);
      if (!buffer) {
        // If we haven't seen this tensor yet, we construct its buffer. Note
        // that we must take a shallow copy of the tensor, otherwise aliasing
        // outputs will cause a segfault due to double move.
        ErrorOr<M::Driver::DeviceBufferRef> bufferOr =
            Driver::constructDeviceMemoryFromTensor(
                maxContext.copy(), resultDevice->context, tensor.borrow());
        if (bufferOr.isError())
          return bufferOr.takeError();
        buffer = bufferOr.takeValue();
      }

      execResults.push_back(
          Driver::ExecResult{buffer, tensor.getSpec(), resultDevice->spec});
      ++resultDeviceIndex;
    } else {
      return Error::getStaticString("unexpected result from model execution");
    }
  }

  return execResults;
}

/// Executes a model with inputs using the "driver" path (device tensors are
/// the only supported kind of tensor).
ErrorOr<ExecResultType>
executeFromDriver(const M::Engine::Context &context,
                  M::AsyncExecutableModel executableModel,
                  SmallVectorImpl<AnyAsyncValueRef> &inputs) {
  TRACY_ZONE_SCOPED_NC("executeFromDriver", TRACY_COLOR_TEAL);
  // Every way into the runtime funnels through here, so one
  // guard covers a forward pass whoever the caller is.
  M::Log::SpanGuard executeSpan("model_execute");

  // Await async model so we can execute synchronously.
  M::AsyncExecutableModel asyncModel = executableModel.copy();

  if (auto err = awaitOrError(asyncModel))
    return err.takeError();

  std::unique_ptr<ExecutableModel> &model = asyncModel.get();
  if (!model)
    return Error::getStaticString("Executable model not available");

  // Pass inputs to model and start the execution.
  SmallVector<AnyAsyncValueRef> results = model->execute(inputs);
  return extractDriverResults(context, std::move(executableModel), results);
}

} // namespace M::GraphCompiler::Driver
