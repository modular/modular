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
// TargetBackend for AMD GPU (amdgcn) targets. Shares the host
// llc -> object -> link emission path to produce an hsaco binary (an ELF
// shared object), and adds AMD-specific device-library linking and
// shared-object link customization.
//
//===----------------------------------------------------------------------===//

#include "Target/Host/HostBackend.h"
#include "Target/max_license/AMDGPU/AMDGPUTraits.h"

#include "Mojo/ToolCommon/CompilationOptions.h"

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVM/ROCDL/Utils.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Triple.h"

namespace M::KGEN {
namespace {

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

/// Override ROCDL module serializer to access pre-lowering bitcode linking
/// logic.
class AMDGPUModuleLinker : public mlir::ROCDL::SerializeGPUModuleBase {
public:
  /// The module passed in does not matter as we do not use this serializer to
  /// perform MLIR to LLVM translation.
  AMDGPUModuleLinker(mlir::Operation &module,
                     mlir::ROCDL::ROCDLTargetAttr target,
                     const mlir::gpu::TargetOptions &targetOptions)
      : SerializeGPUModuleBase(module, target, targetOptions) {}

  /// Link the set of `amdLibs` into the LLVM module, along with any other libs
  /// explicitly specified in `targetOptions` when creating this object.
  mlir::LogicalResult link(llvm::Module &llvmModule,
                           mlir::ROCDL::AMDGCNLibraries amdLibs) {
    // This is required to set control variables during prelink.
    deviceLibs = amdLibs;
    handleModulePreLink(llvmModule);

    // This is required to actually find the .bc files.
    if (deviceLibs != mlir::ROCDL::AMDGCNLibraries::None &&
        failed(appendStandardLibs(deviceLibs)))
      return mlir::failure();

    // Load and link AMD-specific libraries from librariesToLink.
    llvm::SmallVector<std::unique_ptr<llvm::Module>> libs;
    if (failed(loadBitcodeFilesFromList(llvmModule.getContext(),
                                        librariesToLink, libs, true)))
      return mlir::failure();

    if (!libs.empty())
      if (failed(linkFiles(llvmModule, std::move(libs))))
        return mlir::failure();

    handleModulePostLink(llvmModule);
    return mlir::success();
  }
};

class AMDGPUBackend final : public HostBackend {
public:
  const TargetTraits *traits() const override { return &AMDGPUTraits::get(); }

  bool isCodegenInterprocedural() const override { return true; }

  bool isOffload() const override { return true; }

  std::optional<unsigned> sharedMemoryAddressSpace() const override {
    return mlir::ROCDL::ROCDLDialect::kSharedMemoryAddressSpace;
  }

  // GPU branches are far costlier than on CPU, so raise the SimplifyCFG bonus.
  llvm::SimplifyCFGOptions
  adjustSimplifyCFGOptions(llvm::SimplifyCFGOptions options) const override {
    return options.bonusInstThreshold(2);
  }

  // AddressSanitizer is supported, but GPUs don't run the host ThreadSanitizer.
  void addSanitizers(llvm::ModulePassManager &mpm,
                     const CompilationOptions &options) const override {
    addAddressSanitizerPass(mpm, options);
  }

  // Recursion is not supported for GPU kernel entry functions (same as clang).
  void attachCodegenAttributes(llvm::Function *kernelEntry) const override {
    kernelEntry->addFnAttr(llvm::Attribute::NoRecurse);
  }

  /// Links AMD-specific device/runtime bitcode (ROCm device libraries) into
  /// `module`.
  ErrorOrSuccess
  linkRuntimeLibraries(mlir::Location loc, llvm::Module &module,
                       const CompilationOptions &options) const override {
    mlir::MLIRContext *ctx = loc.getContext();
    mlir::OpBuilder b(ctx);
    mlir::OwningOpRef<mlir::ModuleOp> mlirModule(
        mlir::ModuleOp::create(b, loc, "dummy"));
    mlir::ROCDL::ROCDLTargetAttr target = mlir::ROCDL::ROCDLTargetAttr::get(
        ctx, options.optimizationLevel, options.targetTriple, options.targetCpu,
        options.targetFeatures);

    mlir::ROCDL::AMDGCNLibraries standardLibs =
        mlir::ROCDL::AMDGCNLibraries::None;
    llvm::SmallVector<mlir::Attribute> otherLibs;
    if (options.sanitizers.has(Sanitizers::kAddress)) {
      // Both ocml & ockl libs are required for asan.
      standardLibs = mlir::ROCDL::AMDGCNLibraries::Ockl |
                     mlir::ROCDL::AMDGCNLibraries::Ocml;
      // TODO(billyz): Add "asanrtl.bc" to the upstream standard set of
      // libraries.
      otherLibs.push_back(
          b.getStringAttr("/opt/rocm/amdgcn/bitcode/asanrtl.bc"));
    }

    // Create targetOptions with only AMD-specific libraries (no custom
    // bitcode).
    mlir::gpu::TargetOptions targetOptions(
        /*toolkitPath=*/"/opt/rocm", otherLibs);
    AMDGPUModuleLinker moduleLinker(**mlirModule, target, targetOptions);
    if (failed(moduleLinker.link(module, standardLibs)))
      return Error("failed to link AMD-specific bitcode libraries");
    return {};
  }

  void appendLinkArgs(llvm::SmallVectorImpl<llvm::StringRef> &args,
                      const CompilationOptions &options) const override {
    // Add --no-undefined to catch undefined symbols at link time instead of
    // getting a generic hipErrorNoBinaryForGpu error at runtime.
    args.push_back("--no-undefined");
  }

  // This is max only target
  bool isBaseTarget() const override { return false; }
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
RegisterTargetBackend<AMDGPUBackend> registerAMDGPUBackend;
#pragma GCC diagnostic pop

} // namespace
} // namespace M::KGEN
