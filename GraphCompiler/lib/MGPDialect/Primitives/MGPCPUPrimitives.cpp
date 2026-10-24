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

#include "AsyncRT/Support/Chain.h"
#include "GraphCompiler/MGPDialect/Primitives/MGPPrimitives.h"
#include "GraphCompiler/Support/CompiledKernelsContext.h"
#include "MLRT/Core/MojoValue.h"
#include "MLRT/Core/Tensor/StateContext.h"
#include "MLRT/Core/Tensor/Tensor.h"
#include "MLRT/Core/Tensor/TensorBufferRef.h"
#include "MLRT/GraphRT/MEF/MEFAttr.h"
#include "MLRT/GraphRT/MEFExecutor/PrimitiveRegistry.h"
#include "Support/MArchTarget/MArchTarget.h"
#include "Support/ML/SizeUtils.h"
#include "Support/ML/TensorSpec.h"
#include "mojo_string_c_api.h"
#include "weights_registry_c_api.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "mgp"

using namespace M;
using namespace M::MLRT;
using namespace M::AsyncRT;
using namespace M::GML;
using namespace M::MGP;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

TensorBufferRef createBorrowedSlice(TensorBufferRef &borroweBufferRef,
                                    size_t offset, size_t size) {
  const void *newBuffer = borroweBufferRef.getBuffer<uint8_t>() + offset;
  return TensorBufferRef::create(const_cast<void *>(newBuffer), size,
                                 borroweBufferRef.getMemStorageHandle());
}

//===----------------------------------------------------------------------===//
// mgp.module.load.cpu
//===----------------------------------------------------------------------===//

static void
mgp_module_load_cpu(Chain inChain, StateContext &theContext,
                    Result<Chain> outChain, MEFAttr<uint64_t> aRuntimeSlot,
                    MEFDynamicAttr bImage, MEFAttr<StringRef> cFunctionSymbols,
                    MEFAttr<StringRef> dConstantSymbols,
                    MEFAttr<uint64_t> eDeviceId, MEFAttr<uint64_t> fModuleIndex,
                    MEFAttr<bool> debuginfo, CPUDevice &cpuDevice) {
  KernelExecutionState &kernelExecutionState =
      theContext.getStateSlotPayload<KernelExecutionState>(
          static_cast<size_t>(*aRuntimeSlot));
  if (!dConstantSymbols->empty() && !(dConstantSymbols->size() == 0) &&
      !(dConstantSymbols->size() == 1 && *dConstantSymbols->begin() == '\0')) {
    return std::move(outChain).setToError(
        "mgp.module.load.cpu does not support constants");
  }

  if (fModuleIndex.get() >= kernelExecutionState.modules.size())
    kernelExecutionState.modules.resize(fModuleIndex.get() + 1);
  kernelExecutionState.modules[fModuleIndex.get()] =
      RCRef<CompiledKernelsContext>::create(
          kernelExecutionState.engine,
          debuginfo.get() ||
              cpuDevice.getProfilerDebuginfo() ==
                  CPUDeviceOptions::ProfilerDebuginfo::kSOProfiler);
  CompiledKernelsContext &module =
      *kernelExecutionState.modules[fModuleIndex.get()];

  ArrayRef<uint8_t> image;
  if (bImage.decode(&image).isError()) {
    return std::move(outChain).setToError(
        "mgp.module.load.cpu requires image to be a byte array");
  }
  if (image.empty()) {
    return std::move(outChain).setToError(
        "mgp.module.load.cpu requires a non-empty image to load");
  }

  // M.multiline strings are explicitly null-terminated when encoded into MEF.
  SmallVector<StringRef> functionSymbols;
  StringRef combined = *cFunctionSymbols;
  size_t start = 0;
  size_t end = combined.size();
  while (start < end) {
    size_t next = combined.find('\n', start);
    if (next == std::string::npos) {
      assert(start + 1 == end && combined[start] == '\0');
      break;
    } else {
      size_t len = next - start;
      if (len)
        functionSymbols.emplace_back(combined.data() + start, len);
      start = next + 1;
    }
  }
  if (functionSymbols.empty()) {
    return std::move(outChain).setToError(
        "expecting at least one function symbol to load");
  }

  ErrorOrSuccess errOr = module.load(image, functionSymbols);
  if (errOr)
    return std::move(outChain).setToError(errOr.takeError());

  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// mgp.buffer.constant.external.cpu
//===----------------------------------------------------------------------===//

// Implemented in C++ to enable subgraph inlining after JitCompileKernels.
// TODO(GEX-2312): Revisit.
static void mgp_buffer_constant_external_cpu(
    Chain inChain, AsyncValueRef<MojoValue> weights,
    AsyncValueRef<MojoValue> name, Result<TensorBufferRef> outBuffer,
    MEFAttr<uint64_t> aSize, MEFAttr<uint64_t> bAlign,
    MEFAttr<uint64_t> cDeviceId, MEFAttr<bool> dHasAlias,
    MEFAttr<bool> eIsPlaceholder) {

  if (bAlign.get() == 0) {
    return std::move(outBuffer).setToError(
        "align must be a positive integer value");
  }

  void *data;
  static constexpr int kResultSuccess = 0;

  if (weights_registry_lookup_mojo_string(&data, weights->getData(),
                                          name->getData()) != kResultSuccess) {
    auto mojoStringPtr = name->getData();
    auto bytes = (char *)get_mojo_string_data(mojoStringPtr);
    auto len = get_mojo_string_length(mojoStringPtr);
    std::string nameCppStr(bytes, bytes + len);

    return std::move(outBuffer).setToError(
        Twine("unable to look up weight by name: ") + Twine(nameCppStr));
  }

  const std::byte *weightPtr = llvm::bit_cast<const std::byte *>(data);

  if ((reinterpret_cast<uintptr_t>(weightPtr) % bAlign.get()) != 0) {
    return std::move(outBuffer).setToError(
        Twine("invalid alignment for address ") +
        Twine(reinterpret_cast<uintptr_t>(weightPtr)) + " and align " +
        Twine(bAlign.get()));
  }

  // The weights registry owns the data.
  std::move(outBuffer).emplace(TensorBufferRef::createWithNonTrackedMemory(
      const_cast<void *>(reinterpret_cast<const void *>(weightPtr)),
      aSize.get(),
      /*alignment=*/bAlign.get()));
}

//===----------------------------------------------------------------------===//
// mgp.string.create
//===----------------------------------------------------------------------===//

static void mgp_string_create(Result<MojoValue> outString,
                              MEFAttr<StringRef> value) {
  StringRef strValue = value.get();

  void *mojoStringPtr =
      create_mojo_string(const_cast<char *>(strValue.data()),
                         static_cast<ssize_t>(strValue.size()));

  std::move(outString).emplace(MojoValue::createOwningRef(
      mojoStringPtr,
      reinterpret_cast<MojoValue::DestructorFn>(destroy_mojo_string)));
}

//===----------------------------------------------------------------------===//
// mgp.string.concat
//===----------------------------------------------------------------------===//

static void mgp_string_concat(AsyncValueRef<MojoValue> prefix,
                              AsyncValueRef<MojoValue> end,
                              Result<MojoValue> outString) {
  void *mojoStringPtr = concat_mojo_strings(prefix->getData(), end->getData());

  std::move(outString).emplace(MojoValue::createOwningRef(
      mojoStringPtr,
      reinterpret_cast<MojoValue::DestructorFn>(destroy_mojo_string)));
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void M::registerMGPCPUPrimitives(PrimitiveRegistry &registry) {
  ADD_PRIMITIVE(registry, "mgp.module.load.cpu", mgp_module_load_cpu);
  ADD_PRIMITIVE(registry, "mgp.buffer.constant.external.cpu",
                mgp_buffer_constant_external_cpu);
  ADD_PRIMITIVE(registry, "mgp.buffer.constant.external.gpu",
                mgp_buffer_constant_external_cpu);
  ADD_PRIMITIVE(registry, "mgp.string.create", mgp_string_create);
  ADD_PRIMITIVE(registry, "mgp.string.concat", mgp_string_concat);
}
