//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
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
//
// TargetBackend for NVIDIA GPU (nvptx) targets. Emits PTX assembly via llc,
// then compiles it to a cubin through the CUDA compilation device. Assembly
// emission (PTX) is inherited from the host path.
//
//===----------------------------------------------------------------------===//

#include "Target/Host/HostBackend.h"
#include "Target/max_license/NVPTX/NVPTXTraits.h"

#include "MLRT/Driver/DeviceContext/CompilationDevice.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Mojo/ToolCommon/Debug.h"

#include "LLVM/Transforms/SetFunctionAttributes.h"

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"

#include <optional>

#define DEBUG_TYPE "object-compiler"
#define KGEN_DEBUG_TYPE "object-compiler"

namespace M::KGEN {
namespace {

class NVPTXBackend final : public HostBackend {
public:
  const TargetTraits *traits() const override { return &NVPTXTraits::get(); }

  // NVPTX codegen is inter-procedural (e.g. for argument alignment when calling
  // a function), which disables parallel/per-function llc.
  bool isCodegenInterprocedural() const override { return true; }

  bool isOffload() const override { return true; }

  std::optional<unsigned> sharedMemoryAddressSpace() const override {
    return static_cast<unsigned>(mlir::NVVM::NVVMMemorySpace::Shared);
  }

  // MCLinker reorders functions, but NVPTX needs the original order to emit
  // function declarations before use to avoid illegal use-before-def.
  bool requiresOriginalFunctionOrder() const override { return true; }

  // NVPTX shares a single llvm::Module across llvm-opt and llc so its stale
  // AnnotationCache is cleaned correctly, so MCLinker must keep one split.
  bool requiresSingleModuleSplit() const override { return true; }

  // GPU branches are far costlier than on CPU, so raise the SimplifyCFG bonus.
  llvm::SimplifyCFGOptions
  adjustSimplifyCFGOptions(llvm::SimplifyCFGOptions options) const override {
    return options.bonusInstThreshold(2);
  }

  // LLVM's sanitizers are not supported on NVPTX.
  void addSanitizers(llvm::ModulePassManager &,
                     const CompilationOptions &) const override {}

  // Sort the linked module's functions back to their original order so
  // definitions precede uses (avoids illegal use-before-def in PTX).
  void reorderLinkedModuleFunctions(
      llvm::Module &module,
      const llvm::StringMap<unsigned> &originalOrder) const override {
    module.getFunctionList().sort([&](const auto &lhs, const auto &rhs) {
      if (lhs.isDeclaration() && rhs.isDeclaration())
        return true;
      if (lhs.isDeclaration())
        return false;
      if (rhs.isDeclaration())
        return true;
      auto iter1 = originalOrder.find(lhs.getName());
      if (iter1 == originalOrder.end())
        return true;
      auto iter2 = originalOrder.find(rhs.getName());
      if (iter2 == originalOrder.end())
        return true;
      return iter1->second < iter2->second;
    });
  }

  // Recursion is not supported for GPU kernel entry functions (same as clang).
  void attachCodegenAttributes(llvm::Function *kernelEntry) const override {
    kernelEntry->addFnAttr(llvm::Attribute::NoRecurse);
  }

  // NVPTX augments (rather than replaces) the standard pipeline with a leading
  // SetFunctionAttributes pass.
  void addPipelineStartPasses(llvm::ModulePassManager &mpm,
                              const CompilationOptions &) const override {
    mpm.addPass(SetFunctionAttributes());
  }

  void registerPipelinePasses(llvm::PassBuilder &passBuilder) const override {
    passBuilder.registerPipelineParsingCallback(
        [](llvm::StringRef name, llvm::ModulePassManager &mpm,
           llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) -> bool {
          if (name == "kgen-set-function-attrs") {
            mpm.addPass(SetFunctionAttributes());
            return true;
          }
          return false;
        });
  }

  // HACK HACK HACK https://github.com/modularml/modular/issues/27478
  // Using LineTables for NVPTX disables optimizations in the cuda JIT. Use
  // DebugDirectives instead for equivalent performance to no-debug.
  void
  prepareModuleForLowering(mlir::Operation *module,
                           const CompilationOptions &options) const override {
    if (options.getDIEmissionKind() != DebugInfo::EmissionKind::LineTablesOnly)
      return;
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement(
        [](DebugInfo::DICompileUnitAttr CU) -> std::optional<mlir::Attribute> {
          if (CU.getEmissionKind() == DebugInfo::EmissionKind::LineTablesOnly) {
            return DebugInfo::DICompileUnitAttr::get(
                CU.getSourceLanguage(), CU.getFile(), CU.getProducer(),
                CU.getIsOptimized(),
                DebugInfo::EmissionKind::DebugDirectivesOnly,
                CU.getNameTableKind());
          }
          return std::nullopt;
        });
    replacer.recursivelyReplaceElementsIn(module, /*replaceAttrs=*/false,
                                          /*replaceLocs=*/true);
  }

  /// Emits PTX via llc, then compiles it to a cubin via the CUDA compilation
  /// device.
  ErrorOr<BufferRef> emitObject(llvm::Module &module,
                                EmitContext &ctx) const override {
    // Compile to PTX first.
    WriteableBufferRef codeBuf = WriteableBuffer::get();
    if (ErrorOrSuccess err =
            ctx.runLlc(module, *codeBuf, /*createObjectFile=*/false)) {
      return Error(Twine(err.getError()) +
                   ", llc failed to codegen LLVM IR to object code");
    }
    llvm::StringRef ptx(codeBuf->getBufferStart(), codeBuf->getBufferSize());

    // A cubin must be assembled for the same arch as the `.target` directive of
    // the PTX above. `ctx.tm` is the TargetMachine that emitted that PTX, and
    // NVPTX takes `.target` from its subtarget, so that machine's CPU is the
    // arch to assemble for. A CPU that is not an SM arch cannot name one.
    llvm::StringRef targetArch = ctx.tm.getTargetCPU();
    if (!targetArch.starts_with("sm_")) {
      return Error(Twine("expected an NVIDIA SM target CPU (e.g. 'sm_90a') to "
                         "compile PTX to a cubin, but got '") +
                   targetArch + "'");
    }

    // Create CompilationDevice to compile from PTX to cubin.
    ErrorOr<std::unique_ptr<Driver::CompilationDevice>> errCD =
        Driver::CompilationDevice::create(Driver::Device::cudaAPI, targetArch);
    if (errCD.isError()) {
      return Error(
          Twine(errCD.getError()) +
          ", failed to create cuda compilation device to compile to cubin");
    }

#if MODULAR_LINUX
    if (!llvm::sys::Process::GetEnv("MODULAR_NVPTX_COMPILER_PATH")) {
      // We don't use NVPTXCompiler when MODULAR_NVPTX_COMPILER_PATH is set as a
      // way to fallback to lower version of CUDA.
      LLVM_DEBUG(llvm::dbgs()
                 << "Using the NVPTXCompiler API to compile PTX to CUBIN.\n");
      KGEN_DEBUG(0, {
        llvm::dbgs()
            << "Using the NVPTXCompiler API to compile PTX to CUBIN.\n";
      });
    }
    KGEN_DEBUG(0, {
      if (auto path =
              llvm::sys::Process::GetEnv("MODULAR_NVPTX_COMPILER_PATH")) {
        llvm::dbgs() << "MODULAR_NVPTX_COMPILER_PATH: " << path.value();
      }
    });
#endif

    return (*errCD)->compileFunction(ptx, ctx.options.getDebugLevelString(),
                                     ctx.options.optimizationLevel);
  }

  // This is max only target
  bool isBaseTarget() const override { return false; }
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
RegisterTargetBackend<NVPTXBackend> registerNVPTXBackend;
#pragma GCC diagnostic pop

} // namespace
} // namespace M::KGEN
