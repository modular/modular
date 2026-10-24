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

#include "GraphCompiler/MGPDialect/Primitives/MGPPrimitives.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/AsyncValue.h"
#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Runtime/WorkQueue.h"
#include "GraphCompiler/MGPDialect/Primitives/KernelLogger.h"
#include "GraphCompiler/Support/CompiledKernelsContext.h"
#include "GraphCompiler/Support/List.h"
#include "Init/Init.h"
#include "MLRT/Core/Tensor/StateContext.h"
#include "MLRT/Core/Tensor/StateSlot.h"
#include "MLRT/Core/Tensor/Tensor.h"
#include "MLRT/Driver/DeviceContext/DeviceAffinity.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/GraphRT/MEFExecutor/PrimitiveRegistry.h"
#include "Mojo/Support/Configuration.h"
#include "Support/AssertStream.h"
#include "Support/Context.h"
#include "Support/DeviceSpecs.h"
#include "Support/MArchTarget/MArchTarget.h"
#include "Support/ML/DebugPrint.h"
#include "Support/ML/SizeUtils.h"
#include "Support/ML/TensorShape.h"
#include "Support/Threading/SpinWaiter.h"
#include "Support/max_license/Tracy/Tracy.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "mgp"

// Set to 1 (here or in BUILD.bazel) to log each MGP kernel launch and
// completion to stderr.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmacro-redefined"
#define ENABLE_MGP_KERNEL_LOGGING 0
#pragma clang diagnostic pop

#if ENABLE_MGP_KERNEL_LOGGING
#define MAYBE_LOG_ASYNC_KERNEL(summary, operands, results)                     \
  MGP::KernelAsyncLogger<decltype(operands)> asyncLogger(summary, operands,    \
                                                         results)
#define MAYBE_LOG_SYNC_KERNEL(summary, operands, results)                      \
  MGP::KernelSyncLogger<RemainingArguments> logger(summary, operands, results)
#else
#define MAYBE_LOG_ASYNC_KERNEL(summary, operands, results) ((void)(summary))
#define MAYBE_LOG_SYNC_KERNEL(summary, operands, results) ((void)(summary))
#endif // ENABLE_MGP_KERNEL_LOGGING

using namespace M;
using namespace M::MLRT;
using namespace M::AsyncRT;
using namespace M::GML;
using namespace M::MGP;

/// Indicates that a work item has no device preference for execution.
static constexpr int kNoDevicePreference = -1;

//===----------------------------------------------------------------------===//
// mgp.context.create
//===----------------------------------------------------------------------===//

/// Fulfills the result with a StateContext initialized with `numPrimContexts`
/// primitive contexts.
static void mgp_context_create(Result<StateContext> result,
                               MEFAttr<uint64_t> numSlots) {
  std::move(result).emplace(*numSlots);
}

//===----------------------------------------------------------------------===//
// mgp.generic.execute
//===----------------------------------------------------------------------===//

/// Packs pointers to the input and output async values (then the constant
/// resources) into the array the kernel ABI expects.
template <typename OperandTy, typename ResultTy>
llvm::SmallVector<void *> getKernelInputs(OperandTy &operands,
                                          ResultTy &results,
                                          MEFListAttr constantResources) {
  TRACY_ZONE_SCOPED_NC("getKernelInputs", TRACY_COLOR_TEAL);

  llvm::SmallVector<void *> kernelInputs;
  kernelInputs.reserve(results.size() + operands.size() +
                       constantResources.size());

  auto appendInput = [&kernelInputs](const AnyAsyncValueRef &ref) {
    kernelInputs.push_back(
        const_cast<void *>(reinterpret_cast<const void *>(&ref)));
  };

  for (const AnyAsyncValueRef &res : operands)
    appendInput(res);
  for (const AnyAsyncValueRef &res : results)
    appendInput(res);

  for (MEFDynamicAttr attr : constantResources) {
    kernelInputs.push_back(
        const_cast<void *>(reinterpret_cast<const void *>(attr.get())));

    size_t byteCount = 1;
    if (attr.getAttrKind() == MEF::AttributeKind::kString) {
      byteCount = M::decodeMEFArrayLength(attr.get());
    } else if (((uint8_t)attr.getAttrKind()) &
               ((uint8_t)MEF::AttributeKind::mArrayType)) {
      byteCount = M::decodeMEFArrayLength(attr.get()) *
                  M::MEF::getWidthInBits(attr.getAttrKind()) / 8;
    } else {
      ASSERT_STREAM(attr.getAttrKind() != MEF::AttributeKind::kBool &&
                        attr.getAttrKind() != MEF::AttributeKind::kDictionary &&
                        attr.getAttrKind() != MEF::AttributeKind::kList &&
                        attr.getAttrKind() != MEF::AttributeKind::kType &&
                        attr.getAttrKind() != MEF::AttributeKind::kSymbolRef,
                    "unhandled type of resources.");
    }
    kernelInputs.push_back(reinterpret_cast<void *>(byteCount));
  }

  return kernelInputs;
}

/// Returns the accelerator device all operand device contexts share, and one
/// such context for pinning; otherwise kNoDevicePreference and null.
static std::pair<int, M::Driver::DeviceContextRef>
deviceHintAndContextFor(const RemainingArguments &operands) {
  TRACY_ZONE_SCOPED_NC("deviceHintAndContextFor", TRACY_COLOR_TEAL);

  llvm::SmallVector<M::Driver::DeviceContextRef, 8> acceleratorContexts;

  for (const AnyAsyncValueRef &operand : operands) {
    if (operand.isType<M::Driver::DeviceContextRef>()) {
      const auto &deviceCtxRef =
          *static_cast<const M::Driver::DeviceContextRef *>(
              operand.getPointerToData());
      if (deviceCtxRef->device().api() != M::Driver::Device::cpuAPI) {
        acceleratorContexts.push_back(
            M::Driver::DeviceContextRef::copy(deviceCtxRef.getPointer()));
      }
    }
  }

  // Several contexts on one device occur in side-stream regions (the
  // side-stream view plus its base context); the task still belongs on that
  // device's worker.
  if (!acceleratorContexts.empty()) {
    int deviceId = acceleratorContexts.front()->device().id();
    if (llvm::all_of(acceleratorContexts, [&](const auto &ctx) {
          return ctx->device().id() == deviceId;
        }))
      return {deviceId, std::move(acceleratorContexts.front())};
  }
  return {kNoDevicePreference, M::Driver::DeviceContextRef{}};
}

static void
mgp_generic_execute(StateContext &ctx, RemainingArguments operands,
                    RemainingResults outResults, MEFAttr<uint64_t> aSlot,
                    MEFAttr<uint64_t> bModuleIndex,
                    MEFAttr<uint64_t> cFunctionIndex,
                    MEFAttr<StringRef> dSummary, MEFAttr<bool> eIsAsync,
                    MEFListAttr fConstantResources, CPUDevice &cpuDevice) {
  TRACY_ZONE_SCOPED_NCT("mgp_generic_execute", TRACY_COLOR_TEAL,
                        "(Summary: " + dSummary.get().str() + ")");

  M::MGP::KernelExecutionState &kernelExecutionState =
      ctx.getStateSlotPayload<M::MGP::KernelExecutionState>(
          static_cast<size_t>(*aSlot));
  CompiledKernelAsyncABIFn *kernelPtr = kernelExecutionState.getKernelAsyncABI(
      static_cast<size_t>(*bModuleIndex), static_cast<size_t>(*cFunctionIndex));

  // deviceCtx is non-null only when deviceHint != kNoDevicePreference.
  auto [deviceHint, deviceCtx] = deviceHintAndContextFor(operands);

  AsyncRT::WorkQueue *workQueue = cpuDevice.getWorkQueue();
  int taskId =
      (deviceHint == kNoDevicePreference)
          ? kDefaultTaskId
          : MLRT::taskIdForDevice(deviceHint, workQueue->getParallelismLevel());

  // Async kernels always run out of line. Sync kernels run inline unless they
  // have task affinity and this is the wrong worker.
  if (eIsAsync.get() || !workQueue->shouldRunInlineForTask(taskId)) {
    TRACY_ZONE_SCOPED_NCT("-> mgp_generic_execute (async)", TRACY_COLOR_TEAL,
                          "(Summary: " + dSummary.get().str() + ")");

    // GraphRT expects allocated results; only needed here because the task
    // outlives this function.
    outResults.allocateIndirectResults();

    // Captures keep the inputs, outputs, and pinned device context alive until
    // the kernel completes.
    auto runKernelAndExtendLifetimes =
        [outResults = outResults.copyAsVector(),
         takenOperands = operands.takeAsVector(),
         constantResources = std::move(fConstantResources), kernelPtr, dSummary,
         deviceCtx = std::move(deviceCtx)]() mutable {
          TRACY_ZONE_SCOPED_NCT("runKernelAndExtendLifetimes", TRACY_COLOR_TEAL,
                                "(Summary: " + dSummary.get().str() + ")");

          // Results are published into `outResults` as the task's last action
          // so no downstream work starts before `kernelPtr` fully completes.
          llvm::SmallVector<AnyAsyncValueRef> localResults(outResults.size());

          llvm::SmallVector<void *> kernelInputs =
              getKernelInputs(takenOperands, localResults, constantResources);
          MAYBE_LOG_ASYNC_KERNEL(*dSummary, takenOperands, localResults);

          if (deviceCtx.getPointer() != nullptr)
            (void)deviceCtx->bindToCurrentThread();

          {
            TRACY_ZONE_SCOPED_NCT("Invoke Mojo function", TRACY_COLOR_ORANGE,
                                  "(Summary: " + dSummary.get().str() + ")");
            kernelPtr(reinterpret_cast<void **>(kernelInputs.data()));
          }

          if (deviceCtx.getPointer() != nullptr)
            M::Driver::DeviceContext::unbindFromCurrentThread();

          // Must have all been released/reset by the compiled kernel.
          assert(llvm::all_of(takenOperands, [](const AnyAsyncValueRef &ref) {
            return !ref.getPointer();
          }));

          // Release everything context-reachable BEFORE the driver can observe
          // completion: operand slots (already reset by the kernel) and the
          // pinned device context reference.
          takenOperands.clear();
          deviceCtx = M::Driver::DeviceContextRef();

          for (const auto &[local, output] :
               llvm::zip_equal(localResults, outResults)) {
            assert(local && "compiled kernel did not fulfill a result slot");
            std::move(output).resolveIndirect(std::move(local));
          }
        };

    AsyncRT::WorkItem workItem(std::move(runKernelAndExtendLifetimes));
    workQueue->addTask(std::move(workItem), taskId);
  } else {
    TRACY_ZONE_SCOPED_NCT("-> mgp_generic_execute (sync kernel)",
                          TRACY_COLOR_TEAL,
                          "(Summary: " + dSummary.get().str() + ")");

    if (deviceCtx.getPointer() != nullptr)
      (void)deviceCtx->bindToCurrentThread();

    llvm::SmallVector<void *> kernelInputs =
        getKernelInputs(operands, outResults, fConstantResources);
    MAYBE_LOG_SYNC_KERNEL(*dSummary, operands, outResults.get());
    {
      TRACY_ZONE_SCOPED_NCT("Invoke Mojo function", TRACY_COLOR_ORANGE,
                            "(Summary: " + dSummary.get().str() + ")");

      kernelPtr(reinterpret_cast<void **>(kernelInputs.data()));
    }

    if (deviceCtx.getPointer() != nullptr)
      M::Driver::DeviceContext::unbindFromCurrentThread();

    // Must have all been released/reset by the compiled kernel.
    assert(llvm::all_of(operands, [](const AnyAsyncValueRef &ref) {
      return !ref.getPointer();
    }));
  }
}

//===----------------------------------------------------------------------===//
// mgp.device_graph.execute
//===----------------------------------------------------------------------===//

/// Returns the data pointer and byte size of a graph input operand, or nullopt
/// for context operands.
///
/// Must stay in step with `DeviceGraphCollectiveCreateOp::emitMojo`, which
/// picks the operands that get an `add_input` stable location, or inputs pair
/// with the wrong stable buffers.
static std::optional<std::pair<void *, size_t>>
graphInputData(const AnyAsyncValueRef &operand) {
  if (auto *tensor = dyn_cast<MLRT::Tensor>(operand))
    return std::pair{tensor->getMutableBuffer(), tensor->getSizeInBytes()};
  if (auto *buffer = dyn_cast<TensorBufferRef>(operand))
    return std::pair{buffer->getBuffer(), buffer->getSize()};
  return std::nullopt;
}

/// Copies the live `inputs` into `graph`'s stable input buffers. `inputs` must
/// be exactly this graph's slice of the operands.
static M::ErrorOrSuccess
copyDeviceGraphInputs(M::Driver::DeviceGraph &graph,
                      ArrayRef<AnyAsyncValueRef> inputs) {
  ArrayRef<M::Driver::DeviceBufferRef> stableInputs = graph.getInputs();
  if (stableInputs.empty())
    return M::success();

  size_t inputIdx = 0;
  for (const AnyAsyncValueRef &operand : inputs) {
    std::optional<std::pair<void *, size_t>> data = graphInputData(operand);
    if (!data)
      continue;

    if (inputIdx >= stableInputs.size())
      return Error("device graph input count mismatch");

    if (!stableInputs[inputIdx]) {
      ++inputIdx;
      continue;
    }

    M::Driver::DeviceBuffer &dst = *stableInputs[inputIdx++];
    auto [srcPtr, srcSize] = *data;

    M::Driver::DeviceContextRef ctx =
        M::Driver::DeviceContextRef::copy(&dst.context());
    M::Driver::DeviceBufferRef srcView =
        dst.isHost() ? ctx->createHostBufferOwning(srcPtr, srcSize,
                                                   /*owning=*/false)
                     : ctx->createBufferOwning(srcPtr, srcSize,
                                               /*owning=*/false);

    if (ErrorOrSuccess err = srcView->copyTo(dst))
      return err.takeError();
  }

  if (inputIdx != stableInputs.size())
    return Error("device graph input count mismatch");
  return M::success();
}

//===----------------------------------------------------------------------===//
// mgp.device_graph.collective.execute
//===----------------------------------------------------------------------===//

/// Writes every stable address slot the `graphs` publish into, and waits for
/// the writes to land. A slot published by one device's graph is read by
/// another device's graph as soon as that graph launches, and nothing orders
/// the two streams, so every slot write must complete before any graph in the
/// collective starts.
static M::ErrorOrSuccess
patchCollectiveStableAddresses(ArrayRef<M::Driver::DeviceGraph *> graphs) {
  llvm::SmallVector<M::Driver::DeviceContext *, 8> patchedContexts;
  patchedContexts.reserve(graphs.size());

  for (M::Driver::DeviceGraph *graph : graphs) {
    if (graph->getStableAssignments().empty())
      continue;
    if (M::ErrorOrSuccess err = graph->patchStableAddresses())
      return err;
    patchedContexts.push_back(&graph->context());
  }

  for (M::Driver::DeviceContext *ctx : patchedContexts)
    if (M::ErrorOrSuccess err = ctx->synchronize())
      return err;
  return M::success();
}

// Replays the graphs built by `mgp.device_graph.collective.create`. Operands
// are the per-device `DeviceGraphRef`s, then every graph's live inputs in graph
// order; each graph's recorded signature sets how many it consumes.
static void mgp_device_graph_collective_execute(RemainingArguments operands,
                                                Result<Chain> outChain,
                                                RemainingResults outputs,
                                                EncodedLocation loc) {
  TRACY_ZONE_SCOPED_NC("mgp_device_graph_collective_execute", TRACY_COLOR_TEAL);

  SmallVector<M::Driver::DeviceGraph *, 8> graphs;

  for (const AnyAsyncValueRef &operand : operands) {
    RCRef<M::Driver::DeviceGraph> *graph =
        dyn_cast<M::Driver::DeviceGraphRef>(operand);
    if (!graph)
      break;

    graphs.push_back(graph->getPointer());
  }

  MutableArrayRef<AnyAsyncValueRef> remainingInputs =
      operands.get().slice(graphs.size());

  outputs.allocateIndirectResults();

  auto fail = [&](Error msg) {
    std::move(outChain).setToError(msg.copy());
    std::move(outputs).setAllToError(std::move(msg), loc);
  };

  // Slots first, so the writes have landed before any graph launches.
  if (M::ErrorOrSuccess err = patchCollectiveStableAddresses(graphs))
    return fail(err.takeError());

  size_t inputOffset = 0;
  size_t outputIdx = 0;

  for (M::Driver::DeviceGraph *graph : graphs) {
    // The operand list carries every value the graph's region captured, so
    // non-input values (`!mgp.context`, scalars) ride along between the data
    // operands. The graph's stable-slot count only counts data operands, so
    // grow this graph's window until it holds that many, carrying the riders
    // for `copyDeviceGraphInputs` to skip.
    size_t numStableInputs = graph->getInputs().size();
    size_t numDataInputs = 0;
    size_t windowEnd = inputOffset;
    while (numDataInputs < numStableInputs) {
      if (windowEnd == remainingInputs.size())
        return fail("device graph collective execute: input operand count "
                    "does not match graph signatures");
      if (graphInputData(remainingInputs[windowEnd]))
        ++numDataInputs;
      ++windowEnd;
    }

    ArrayRef<AnyAsyncValueRef> graphInputs =
        remainingInputs.slice(inputOffset, windowEnd - inputOffset);

    if (M::ErrorOrSuccess err = copyDeviceGraphInputs(*graph, graphInputs))
      return fail(err.takeError());

    inputOffset = windowEnd;

    if (ErrorOrSuccess err = graph->replay())
      return fail(err.takeError());

    ArrayRef<AnyAsyncValueRef> graphOutputs = graph->getOutputs();
    if (outputIdx + graphOutputs.size() > outputs.size())
      return fail("device graph collective execute: output count does not "
                  "match graph signatures");

    for (const AnyAsyncValueRef &graphOutput : graphOutputs) {
      AnyAsyncValueRef &out = outputs[outputIdx++];
      out.copy().resolveIndirect(graphOutput.copy());
    }
  }

  if (outputIdx != outputs.size())
    return fail("device graph collective execute: output count does not "
                "match graph signatures");

  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// mgp.list.append
//===----------------------------------------------------------------------===//

static void mgp_list_append(List &inputList, AnyAsyncValueRef element,
                            Result<List> output) {
  TRACY_ZONE_SCOPED_NC("mgp_list_append", TRACY_COLOR_TEAL);

  if (element.isType<Tensor>()) {
    if (!inputList.getElements().empty() &&
        inputList.getElements().front().getValue().getPointer()->getTypeID() !=
            element.getPointer()->getTypeID()) {
      std::move(output).setToError("Tensor must be from the same device");
      return;
    }
  }

  List newList = inputList.copy();
  newList.getElements().push_back(element.copy());
  std::move(output).emplace(std::move(newList));
}

//===----------------------------------------------------------------------===//
// mgp.list.concat
//===----------------------------------------------------------------------===//

static void mgp_list_concat(List &lhs, List &rhs, Result<List> output) {
  TRACY_ZONE_SCOPED_NC("mgp_list_concat", TRACY_COLOR_TEAL);

  List resultList = lhs.copy();
  resultList.getElements().insert(resultList.getElements().end(),
                                  rhs.getElements().begin(),
                                  rhs.getElements().end());

  std::move(output).emplace(std::move(resultList));
}

//===----------------------------------------------------------------------===//
// mgp.list.create
//===----------------------------------------------------------------------===//

static void mgp_list_create(RemainingArguments operands,
                            Result<List> outputList) {
  TRACY_ZONE_SCOPED_NC("mgp_list_create", TRACY_COLOR_TEAL);

  std::move(outputList).emplace(List{operands.copyAsVector()});
}

//===----------------------------------------------------------------------===//
// mgp.list.get
//===----------------------------------------------------------------------===//

static void mgp_list_get(List &inputList, Tensor &idxTensor,
                         RemainingResults outputs, EncodedLocation loc) {
  TRACY_ZONE_SCOPED_NC("mgp_list_get", TRACY_COLOR_TEAL);

  // We must unbox the index value from the tensor.
  // TODO: Should be ssize_t.
  idxTensor.getEltType()
      .dispatch<void>(idxTensor.getBuffer())
      .template when<DType::si32, DType::si64>([&](auto *idxDataPtr) {
        auto idx = *idxDataPtr;

        int64_t listSize = inputList.getElements().size();
        if (idx < -listSize || listSize <= idx) {
          return std::move(outputs).setAllToError(
              Twine("index ") + Twine(idx) +
                  " is out of bound for list of size " + Twine(listSize),
              loc);
        }

        if (idx < 0)
          idx += listSize;

        // Elements have arbitrary types, so no typed `Result<...>`; sharing
        // the `AsyncValueRef` also avoids deep-copying values like `Tensor`.
        *outputs.begin() = inputList.getElements()[idx].getValue().copy();
      });
}

//===----------------------------------------------------------------------===//
// mgp.list.insert
//===----------------------------------------------------------------------===//

static void mgp_list_insert(List &inputList, AnyAsyncValueRef element,
                            Tensor &idxTensor, Result<List> output) {
  TRACY_ZONE_SCOPED_NC("mgp_list_insert", TRACY_COLOR_TEAL);

  // We must unbox the index value from the tensor.
  // TODO: Should be ssize_t.
  idxTensor.getEltType()
      .dispatch<void>(idxTensor.getBuffer())
      .template when<DType::si32, DType::si64>([&](auto *idxDataPtr) {
        auto idx = *idxDataPtr;

        int64_t listSize = inputList.getElements().size();
        if (idx < -listSize || listSize < idx) {
          return std::move(output).setToError(
              Twine("index ") + Twine(idx) +
              " is out of bound for list of size " + Twine(listSize));
        }

        if (idx < 0)
          idx += listSize;

        List newList = inputList.copy();
        newList.getElements().insert(newList.getElements().begin() + idx,
                                     element.copy());
        std::move(output).emplace(std::move(newList));
      });
}

//===----------------------------------------------------------------------===//
// mgp.list.slice
//===----------------------------------------------------------------------===//

static void mgp_list_slice(List &inputList, Tensor &startTensor,
                           Tensor &endTensor, Tensor &stepTensor,
                           Result<List> output) {
  TRACY_ZONE_SCOPED_NC("mgp_list_slice", TRACY_COLOR_TEAL);

  auto sliceFunctor = [&](auto start, auto end, auto step) {
    List resultList;

    auto &elems = inputList.getElements();

    ssize_t length = elems.size();

    if (step == 0) {
      return std::move(output).setToError(
          "The slice step argument must be a non-zero");
    }

    const bool stepIsPositive = step > 0;

    if (start < 0) {
      start += length;
    }
    if (end < 0) {
      end += length;
    }

    if (stepIsPositive) {
      start = std::clamp<decltype(start)>(start, 0, length - 1);
      end = std::clamp<decltype(end)>(end, 0, length);
    } else {
      start = std::clamp<decltype(start)>(start, 0, length - 1);
      end = std::clamp<decltype(end)>(end, -1, length);
    }

    if (stepIsPositive) {
      for (ssize_t i = start; i < end && i < length && i >= 0; i += step) {
        resultList.getElements().push_back(inputList.getElements()[i]);
      }
    } else {
      for (ssize_t i = start; i > end; i += step) {
        resultList.getElements().push_back(inputList.getElements()[i]);
      }
    }

    std::move(output).emplace(std::move(resultList));
  };

  startTensor.getEltType()
      .dispatch<void>(startTensor.getBuffer())
      .template when<DType::si32, DType::si64>([&](auto *startDataPtr) {
        auto start = *startDataPtr;

        endTensor.getEltType()
            .dispatch<void>(endTensor.getBuffer())
            .template when<DType::si32, DType::si64>([&](auto *endDataPtr) {
              auto end = *endDataPtr;

              stepTensor.getEltType()
                  .dispatch<void>(stepTensor.getBuffer())
                  .template when<DType::si32, DType::si64>(
                      [&](auto *stepDataPtr) {
                        auto step = *stepDataPtr;
                        sliceFunctor(start, end, step);
                      });
            });
      });
}

//===----------------------------------------------------------------------===//
// mgp.guard
//===----------------------------------------------------------------------===//

static void mgp_guard(Chain inChain, RemainingArguments inputs,
                      RemainingResults results) {
  TRACY_ZONE_SCOPED_NC("mgp_guard", TRACY_COLOR_TEAL);

  assert(inputs.size() == results.size());

  auto inputAsyncVals = inputs.get();
  auto resultAsyncVals = results.get();

  for (const auto &[input, result] :
       llvm::zip(inputAsyncVals, resultAsyncVals)) {
    result = std::move(input);
  }
}

//===----------------------------------------------------------------------===//
// mgp.model
//===----------------------------------------------------------------------===//

static void mgp_model() {
  TRACY_ZONE_SCOPED_NC("mgp_model", TRACY_COLOR_TEAL);

  ASSERT_STREAM(false, << "mgp.model should never be invoked");
}

//===----------------------------------------------------------------------===//
// mgp.extend.lifetime
//===----------------------------------------------------------------------===//

// Runs once all operands are ready, keeping them alive until then.
static void mgp_nop(RemainingArguments operands, Result<Chain> outChain) {
  TRACY_ZONE_SCOPED_NC("mgp_nop", TRACY_COLOR_TEAL);

  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// mgp.runtime.create
//===----------------------------------------------------------------------===//

void mgp_runtime_create(Chain inChain, StateContext &theContext,
                        Result<Chain> outChain, MEFAttr<uint64_t> aRuntimeSlot,
                        CPUDevice &cpuDevice) {
  TRACY_ZONE_SCOPED_NC("mgp_runtime_create", TRACY_COLOR_TEAL);

  // M::Context is not in the primitive invocation ABI; callers (engine, mt,
  // mo-opt) create it and it is retrieved here.
  M::ContextRef ctxRef = Init::getContext();

  // One execution engine is shared by all models using this cpuDevice.
  //
  // TODO(#30647): Currently there's no garbage collection of dylibs loaded into
  //               this engine.
  ErrorOr<KGEN::ExecutionEngine *> engineOr =
      ctxRef->createIfMissing<KGEN::ExecutionEngine>(
          [&]() -> ErrorOr<std::unique_ptr<KGEN::ExecutionEngine>> {
            llvm::InitializeNativeTarget();
            llvm::InitializeNativeTargetAsmParser();
            llvm::InitializeNativeTargetAsmPrinter();
            auto targetMachineOr = getTargetMachineForHost();
            if (targetMachineOr.isError())
              return targetMachineOr.takeError();

            auto cfgOr = KGEN::MojoConfig::open();
            if (cfgOr.isError())
              return cfgOr.takeError();
            KGEN::MojoConfig cfg = std::move(*cfgOr);

            KGEN::ExecutionEngineOptions options;
            options.registerPerfPlugins =
                cpuDevice.getProfilerDebuginfo() ==
                M::AsyncRT::CPUDeviceOptions::ProfilerDebuginfo::kPerfProfiler;
            options.libraryPaths.push_back(cfg.getMGPRTPath());
            return KGEN::ExecutionEngine::createWithStandardLayers(
                std::move(options), **targetMachineOr);
          });

  if (engineOr.isError())
    return std::move(outChain).setToError(engineOr.takeError());

  theContext.emplaceStateSlot<KernelExecutionState>(
      static_cast<size_t>(*aRuntimeSlot), *engineOr);

  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// mgp.device_context.select_stream
//===----------------------------------------------------------------------===//

static ErrorOr<M::Driver::DeviceContextRef>
mgp_device_context_select_stream(M::Driver::DeviceContextRef &deviceContext,
                                 MEFAttr<uint64_t> streamId) {
  TRACY_ZONE_SCOPED_NC("mgp_device_context_select_stream", TRACY_COLOR_TEAL);
  return deviceContext->selectStream(static_cast<unsigned>(*streamId));
}

//===----------------------------------------------------------------------===//
// mgp.buffer.cache
//===----------------------------------------------------------------------===//

static void mgp_buffer_cache(Chain inChain, StateContext &theContext,
                             AsyncValueRef<TensorBufferRef> buffer,
                             Result<Chain> outChain,
                             MEFAttr<uint64_t> bBufferSlot,
                             CPUDevice &cpuDevice) {
  TRACY_ZONE_SCOPED_NC("mgp_buffer_cache", TRACY_COLOR_TEAL);

  auto &tensorStateSlot = theContext.emplaceStateSlot<TensorStateSlot>(
      static_cast<size_t>(*bBufferSlot));

  // Always borrow the buffer.
  ErrorOrSuccess errOr = tensorStateSlot.setBuffer(buffer.get().borrow());

  if (errOr.isError())
    return std::move(outChain).setToError(errOr.takeError());
  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// registerMGPPrimitives
//===----------------------------------------------------------------------===//

void M::registerMGPPrimitives(PrimitiveRegistry &registry) {
  ADD_PRIMITIVE(registry, "mgp.runtime.create", mgp_runtime_create);
  ADD_PRIMITIVE(registry, "mgp.device_context.select_stream",
                mgp_device_context_select_stream);
  ADD_PRIMITIVE(registry, "mgp.buffer.cache", mgp_buffer_cache);
  ADD_PRIMITIVE(registry, "mgp.context.create", mgp_context_create);
  ADD_PRIMITIVE(registry, "mgp.generic.execute", mgp_generic_execute);
  ADD_PRIMITIVE(registry, "mgp.device_graph.collective.execute",
                mgp_device_graph_collective_execute);
  ADD_PRIMITIVE(registry, "mgp.list.append", mgp_list_append);
  ADD_PRIMITIVE(registry, "mgp.guard", mgp_guard);
  ADD_PRIMITIVE(registry, "mgp.list.concat", mgp_list_concat);
  ADD_PRIMITIVE(registry, "mgp.list.create", mgp_list_create);
  ADD_PRIMITIVE(registry, "mgp.list.get", mgp_list_get);
  ADD_PRIMITIVE(registry, "mgp.list.insert", mgp_list_insert);
  ADD_PRIMITIVE(registry, "mgp.list.slice", mgp_list_slice);
  ADD_PRIMITIVE(registry, "mgp.model", mgp_model);

  ADD_PRIMITIVE(registry, "mgp.extend.lifetime", mgp_nop);

  registerMGPCPUPrimitives(registry);
}
