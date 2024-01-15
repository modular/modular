//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/MArchTarget/MArchTarget.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/AArch64TargetParser.h"
#include "llvm/TargetParser/ARMTargetParser.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/X86TargetParser.h"

using namespace M;

/// Returns the feature set according to clang for the given options, which
/// should include the Triple and CPU.
static ErrorOr<std::vector<std::string>>
getFeaturesFromClang(std::shared_ptr<clang::TargetOptions> opts) {
  // Intercept diagnostics from Clang and then bundle them up in an `Error` if
  // something bad happens.
  struct DiagInterceptor : public clang::DiagnosticConsumer {
    void HandleDiagnostic(clang::DiagnosticsEngine::Level level,
                          const clang::Diagnostic &info) override {
      if (level >= clang::DiagnosticsEngine::Level::Error) {
        // Keep the last message.
        msg.clear();
        info.FormatDiagnostic(msg);
      }
    }

    SmallString<64> msg;
  };

  // Instantiate the Clang diagnostic engine. Pass in our interceptor.
  clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> ids(
      new clang::DiagnosticIDs());
  clang::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagOpts(
      new clang::DiagnosticOptions());
  DiagInterceptor interceptor;
  clang::DiagnosticsEngine diags(std::move(ids), std::move(diagOpts),
                                 &interceptor, /*ShouldOwnClient=*/false);

  // Ask Clang to create the target info for the architecture and CPU. This will
  // populate `opts` with the full target triple and feature set.
  auto targetInfo = std::unique_ptr<clang::TargetInfo>(
      clang::TargetInfo::CreateTargetInfo(diags, opts));
  if (!targetInfo)
    return Error("failed to create target info: " + interceptor.msg);

  // Concat the features together, only keeping included '+' features.
  std::vector<std::string> features;
  for (StringRef feature : opts->Features) {
    if (feature.front() == '+')
      features.emplace_back(feature.drop_front());
  }
  llvm::sort(features);

  return features;
}

/// Returns feature set for host, falling back to clang using triple and cpu
/// options if the native LLVM helper fails.
static ErrorOr<std::vector<std::string>> getHostFeatures(StringRef triple,
                                                         StringRef cpu) {
  auto opts = std::make_shared<clang::TargetOptions>();
  opts->Triple = triple;
  opts->CPU = cpu;
  return getFeaturesFromClang(opts);
}

ErrorOr<TargetInfo> M::getHostTargetInfo() {
  std::string hostTriple = llvm::sys::getDefaultTargetTriple();
  std::string hostCpu(llvm::sys::getHostCPUName());
  auto featuresOr = getHostFeatures(hostTriple, hostCpu);
  if (featuresOr)
    return featuresOr.takeError();
  return TargetInfo(llvm::Triple(hostTriple), hostCpu, *featuresOr);
}

std::string M::getHostCPUFeatures() {
  auto targetInfoOr = getHostTargetInfo();
  if (targetInfoOr)
    return "";
  return encodeFeatures(targetInfoOr->features);
}

ErrorOr<std::unique_ptr<llvm::TargetMachine>>
M::getTargetMachineForHost(bool isJIT, llvm::CodeGenOptLevel optLevel) {
  std::string hostTriple = llvm::sys::getDefaultTargetTriple();
  std::string hostCpu(llvm::sys::getHostCPUName());
  std::string targetFeatures = getHostCPUFeatures();

  std::string errorMessage;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(hostTriple, errorMessage);
  if (!target)
    return Error("no target exists for '" + hostTriple + "': " + errorMessage);

  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      hostTriple, hostCpu, targetFeatures,
      /*Options=*/{},
      /*RM=*/llvm::Reloc::Model::PIC_,
      /*CM=*/std::nullopt, /*OL=*/optLevel, /*JIT=*/isJIT));
  if (!machine)
    return Error("unable to create target machine");

  return machine;
}

ErrorOr<TargetInfo> M::getMArchTargetInfo(StringRef march, StringRef mcpu,
                                          StringRef mtune) {
  using namespace llvm;

  // Handle -march=native.
  if (march == "native")
    return getHostTargetInfo();

  // `-march` has different meaning depending on the architecture. Determine the
  // LLVM target triple and CPU from it.
  Triple triple;
  auto opts = std::make_shared<clang::TargetOptions>();

  auto processExts = [&opts](StringRef &m) {
    StringRef exts;
    std::tie(m, exts) = m.split("+");
    while (!exts.empty()) {
      StringRef ext;
      std::tie(ext, exts) = exts.split("+");
      if (ext.starts_with("no"))
        opts->FeatureMap[ext.drop_front(2)] = false;
      else
        opts->FeatureMap[ext] = true;
    }
  };
  processExts(march);
  processExts(mcpu);

  if (!mtune.empty())
    opts->TuneCPU = mtune;

  auto tryParseX86 = [&](StringRef cpuName) {
    // Check for a 64-bit one first.
    if (X86::CPUKind x86_64Cpu = X86::parseArchX86(cpuName, /*Only64Bit=*/true);
        x86_64Cpu != X86::CK_None) {
      triple.setArch(Triple::x86_64);
      opts->CPU = cpuName;
      return true;
    }

    // Otherwise, see if it is a 32-bit one.
    if (X86::CPUKind x86Cpu = X86::parseArchX86(cpuName, /*Only64Bit=*/false);
        x86Cpu != X86::CK_None) {
      triple.setArch(Triple::x86_64);
      opts->CPU = cpuName;
      return true;
    }

    return false;
  };

  // Try to parse an X86 architecture from either -march or -mcpu.
  if (tryParseX86(march) || tryParseX86(mcpu)) {
    if (mcpu == "generic")
      opts->CPU = "";

    // Check for an AArch64 CPU.
  } else if (std::optional<AArch64::CpuInfo> aarch64Cpu =
                 AArch64::parseCpu(mcpu)) {
    triple.setArchName(march);
    triple.setArch(Triple::aarch64, triple.getSubArch());
    opts->CPU = mcpu;

    // Check for an ARM CPU.
  } else if (ARM::ArchKind armArch = ARM::parseCPUArch(mcpu);
             armArch != ARM::ArchKind::INVALID) {
    triple.setArchName(ARM::getArchName(armArch));
    opts->CPU = mcpu;

    // Check for an AArch64 arch.
  } else if (std::optional<AArch64::ArchInfo> aarch64Arch =
                 AArch64::parseArch(march)) {
    triple.setArchName(march);
    opts->CPU = "generic";

    // Check for an ARM arch.
  } else if (ARM::ArchKind armArch = ARM::parseArch(march);
             armArch != ARM::ArchKind::INVALID) {
    triple.setArchName(march);
    // If -mcpu was not specified, use a default CPU for the architecture.
    if (mcpu.empty())
      opts->CPU = ARM::getDefaultCPU(triple.getArchName());
    else
      opts->CPU = mcpu;

  } else {
    triple.setArchName(march);
    opts->CPU = "generic";
  }

  // Reset the vendor name if it's not one known to LLVM. This can occur when
  // the triple arch name is set to a value containing hyphens, such as
  // "armv8.2-a". In this case, the vendor is set to "a", which is unknown.
  if (triple.getVendor() == Triple::UnknownVendor)
    triple.setVendor(Triple::VendorType::UnknownVendor);

    // Set the OS name (see #17241).
#ifdef __linux__
  triple.setOS(llvm::Triple::OSType::Linux);
#elif __APPLE__
  triple.setOS(llvm::Triple::OSType::MacOSX);
#elif _WIN32
  triple.setOS(Triple::Triple::OSType::Win32);
#else
#error "unsupported operating system."
#endif

  opts->Triple = triple.str();

  // Gather features from clang.
  auto featuresOr = getFeaturesFromClang(opts);
  if (featuresOr)
    return featuresOr.takeError();

  return TargetInfo(std::move(triple), std::move(opts->CPU),
                    std::move(*featuresOr));
}

ErrorOr<TargetInfoAttr> M::getMArchFeatures(MLIRContext *ctx, StringRef march,
                                            StringRef mcpu, StringRef mtune) {
  auto runtimeTargetInfoOr = getMArchTargetInfo(march, mcpu, mtune);
  if (runtimeTargetInfoOr)
    return runtimeTargetInfoOr.takeError();

  return getTargetInfoFor(ctx, runtimeTargetInfoOr->triple.str(),
                          runtimeTargetInfoOr->arch,
                          encodeFeatures(runtimeTargetInfoOr->features), mtune);
}
