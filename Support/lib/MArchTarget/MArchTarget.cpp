//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/MArchTarget/MArchTarget.h"
#include "Support/Target.h"
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

ErrorOr<TargetInfoAttr> M::getMArchFeatures(MLIRContext *ctx, StringRef march,
                                            StringRef mcpu) {
  using namespace llvm;

  // Handle -march=native.
  if (march == "native") {
    return getTargetInfoFor(ctx, sys::getDefaultTargetTriple(),
                            sys::getHostCPUName(), getHostCPUFeatures());
  }

  // `-march` has different meaning depending on the architecture. Determine the
  // LLVM target triple and CPU from it.
  Triple triple;
  auto opts = std::make_shared<clang::TargetOptions>();

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
    triple.setArchName(aarch64Cpu->Arch.Name);
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
  clang::TargetInfo *targetInfo =
      clang::TargetInfo::CreateTargetInfo(diags, opts);
  if (!targetInfo)
    return Error("failed to create target info: " + interceptor.msg);

  // Concat the features together, only keeping included '+' features.
  std::string featureStr;
  llvm::raw_string_ostream os(featureStr);
  for (StringRef feature : opts->Features)
    if (feature.front() == '+')
      os << feature << ',';
  // Drop the extra comma.
  if (!featureStr.empty())
    featureStr.pop_back();

  // Use this to create the target info.
  return getTargetInfoFor(ctx, opts->Triple, opts->CPU, featureStr);
}
