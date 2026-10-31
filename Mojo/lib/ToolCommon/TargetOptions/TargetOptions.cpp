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

#include "Mojo/ToolCommon/TargetOptions.h"
#include "Support/DeviceSpecs.h"
#include "Support/MArchTarget/MArchTarget.h"
#include "Support/MArchTarget/MArchTargetMinimal.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"

using namespace M;
using namespace KGEN;
using llvm::StringRef;

ErrorOrSuccess M::parseTargetOptions(const TargetArgs &targetArgs,
                                     CompilationOptions &compilationOptions,
                                     MLIRContext &ctx, TargetInfoAttr &target) {
  StringRef targetTriple = targetArgs.triple;
  StringRef targetCpu = targetArgs.cpu;
  StringRef targetFeatures = targetArgs.features;
  StringRef targetAbi = targetArgs.abi;
  StringRef mArch = targetArgs.march;
  StringRef mCpu = targetArgs.mcpu;
  StringRef mTune = targetArgs.mtune;

  // Validate that LLVM-style options (--target-cpu, --target-features) are not
  // mixed with GCC/Clang-style options (--march, --mcpu, --mtune).
  // These are two separate option families that should not be mixed.
  bool hasGccStyleOptions = !mArch.empty() || !mCpu.empty();
  if (hasGccStyleOptions) {
    if (!targetCpu.empty()) {
      return Error("--target-cpu cannot be used with --march or --mcpu; "
                   "use either --target-cpu/--target-features or "
                   "--march/--mcpu/--mtune");
    }
    if (!targetFeatures.empty()) {
      return Error(
          "--target-features cannot be used with --march or --mcpu; "
          "use --target-cpu with --target-features, or use --march/--mcpu with "
          "extension syntax (e.g., --march=skylake-avx512)");
    }
  }

  StringRef targetAccelerator = targetArgs.accelerator;
  StringRef mcmodel = targetArgs.mcmodel;
  StringRef largeDataThreshold = targetArgs.largeDataThreshold;

  // If the user specified the triple, the target CPU, or the target feature
  // set, use those to override the defaults.
  if (!targetTriple.empty())
    compilationOptions.targetTriple = targetTriple.str();
  if (!targetCpu.empty())
    compilationOptions.targetCpu = targetCpu.str();
  else if (!mCpu.empty()) {
    // Use -mcpu to set targetCpu when --target-cpu is not specified.
    // Strip any extensions (e.g., "haswell+avx512f" -> "haswell").
    compilationOptions.targetCpu = mCpu.split('+').first.str();
  } else
    compilationOptions.setDefaultCPU();

  if (!targetAbi.empty())
    compilationOptions.targetAbi = targetAbi.str();

  if (!targetFeatures.empty())
    compilationOptions.targetFeatures = targetFeatures.str();
  else if (mArch.empty() && mCpu.empty()) {
    // Only compute features here if not using -march/-mcpu, since those will
    // be handled later by getMArchFeatures() which computes the correct
    // features for the specified architecture/CPU.
    //
    // TODO: Reconsider whether `mojo build` with no target CPU/arch should
    // default to the host CPU at all. Clang defaults to the architecture
    // baseline (e.g. x86-64 with SSE2 only), which is conservative and
    // portable, and relies on -march=native to opt into host-specific
    // features. Mojo currently defaults to the host CPU, which is aggressive
    // and matches the common case of compiling and running on the same machine,
    // but diverges from Clang's convention.
    //
    // Short-term fix: when the default CPU is the host CPU, route through
    // getHostTargetInfo() so the runtime CPUID cross-check is applied. Without
    // this, Clang's CPU model database would enable features (e.g. AVX-512 on
    // znver4) that the OS may not have made available.
    if (compilationOptions.targetCpu == llvm::sys::getHostCPUName()) {
      ErrorOr<TargetInfo> hostInfoOr = M::getHostTargetInfo();
      if (hostInfoOr)
        return hostInfoOr.takeError();
      compilationOptions.targetFeatures = encodeFeatures(*hostInfoOr);
    } else {
      ErrorOr<std::vector<std::string>> featuresOr = M::getFeatures(
          compilationOptions.targetTriple, compilationOptions.targetCpu);
      if (featuresOr)
        return featuresOr.takeError();
      compilationOptions.targetFeatures =
          encodeFeatures(TargetInfo({}, {}, std::move(*featuresOr)));
    }
  }
  if (!targetAccelerator.empty()) {
    compilationOptions.targetAccelerator = targetAccelerator.str();
    compilationOptions.isCrossCompilation = true;
  }
#if MLRT_ACCELERATOR_SUPPORT
  else {
    compilationOptions.targetAccelerator =
        M::Driver::Device::getAcceleratorArchOrEmpty();
  }
#endif

  if (!mcmodel.empty()) {
    if (!llvm::is_contained({"small", "medium", "large"}, mcmodel)) {
      return Error("invalid mcmodel'" + mcmodel +
                   "', expected one of: `small`, `medium` or `large`");
    }
    if (mcmodel == "small")
      compilationOptions.mcmodel = llvm::CodeModel::Small;
    else if (mcmodel == "medium")
      compilationOptions.mcmodel = llvm::CodeModel::Medium;
    else if (mcmodel == "large")
      compilationOptions.mcmodel = llvm::CodeModel::Large;
  }

  if (!largeDataThreshold.empty()) {
    uint64_t value;
    if (!llvm::to_integer(largeDataThreshold, value)) {
      return Error("invalid large-data-threshold'" + largeDataThreshold +
                   "', expected a positive integer number");
    }
    compilationOptions.largeDataThreshold = value;
  }

  StringRef relocationModel = targetArgs.relocationModel;
  if (!relocationModel.empty()) {
    ErrorOr<llvm::Reloc::Model> model =
        M::symbolizeRelocationModel(relocationModel);
    if (model.isError())
      return model.takeError();

    compilationOptions.relocModel = *model;
  }

  // Initialize targets first - we rely on this for getTargetInfo as well as for
  // the ExecutionEngine.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  // Construct a target specification using the command line options.
  ErrorOr<TargetInfoAttr> targetOr = nullptr;
  if (!mArch.empty() || !mCpu.empty()) {
    // Use `-march` to determine the feature set.
    targetOr = getMArchFeatures(
        &ctx, compilationOptions.targetTriple, mArch, mCpu, mTune,
        compilationOptions.targetAccelerator, compilationOptions.relocModel,
        compilationOptions.targetAbi);
  } else {
    // Use the full triple, specific CPU, and manually specified features to
    // get the target info.
    targetOr = getTargetInfoFor(
        &ctx, compilationOptions.targetTriple, compilationOptions.targetCpu,
        compilationOptions.targetFeatures, /*tuneCpu=*/mTune,
        /*acceleratorArch=*/compilationOptions.targetAccelerator,
        compilationOptions.relocModel, compilationOptions.targetAbi);
  }
  if (targetOr.isError())
    return targetOr.takeError();
  target = targetOr.takeValue();

  return success();
}
