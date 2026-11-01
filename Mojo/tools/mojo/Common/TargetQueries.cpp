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

#include "TargetQueries.h"

#include "Mojo/Compiler/Target/TargetBackend.h"
#include "Support/MDialect/MAttrs.h"
#include "Target/TargetTraits.h"

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Triple.h"

#include <algorithm>

using namespace M;
using namespace KGEN;

/// Normalize a target triple string to canonical form with all components.
/// First applies LLVM's normalization (handles reordering), then fills in
/// missing components with "unknown" for clearer user output.
/// Example: "aarch64" -> "aarch64-unknown-unknown"
static std::string normalizeTriple(StringRef tripleStr) {
  // First let LLVM handle reordering (e.g., "-pc-i386" -> "i386-pc-unknown")
  llvm::Triple triple(llvm::Triple::normalize(tripleStr));
  StringRef vendorName = triple.getVendorName();
  StringRef osName = triple.getOSName();
  StringRef envName = triple.getEnvironmentName();
  // Then fill in missing components with "unknown" for clarity
  return (llvm::Twine(triple.getArchName()) + "-" +
          (vendorName.empty() ? "unknown" : vendorName) + "-" +
          (osName.empty() ? "unknown" : osName) +
          (envName.empty() ? "" : "-" + envName))
      .str();
}

namespace {
/// Simple diagnostic consumer that prints errors to stderr.
class StderrDiagConsumer : public clang::DiagnosticConsumer {
public:
  void HandleDiagnostic(clang::DiagnosticsEngine::Level level,
                        const clang::Diagnostic &info) override {
    if (level >= clang::DiagnosticsEngine::Error) {
      SmallString<128> message;
      info.FormatDiagnostic(message);
      llvm::errs() << "error: " << message << "\n";
    }
  }
};
} // namespace

/// Get the list of valid CPUs for a target triple using clang.
/// Returns an empty vector if the target is invalid.
static std::vector<std::string> getValidCPUsForTarget(StringRef triple) {
  clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagIDs(
      new clang::DiagnosticIDs());
  clang::DiagnosticOptions diagOpts;
  StderrDiagConsumer diagConsumer;
  clang::DiagnosticsEngine diags(diagIDs, diagOpts, &diagConsumer,
                                 /*ShouldOwnClient=*/false);

  auto targetOpts = std::make_shared<clang::TargetOptions>();
  targetOpts->Triple = triple;

  std::unique_ptr<clang::TargetInfo> targetInfo(
      clang::TargetInfo::CreateTargetInfo(diags, *targetOpts.get()));

  std::vector<std::string> cpus;
  if (targetInfo) {
    SmallVector<StringRef, 128> cpuRefs;
    targetInfo->fillValidCPUList(cpuRefs);
    for (StringRef cpu : cpuRefs)
      cpus.push_back(cpu.str());
  }
  return cpus;
}

int M::printSupportedTargets() {
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();

  // An LLVM backend being linked in is necessary but not sufficient: emission
  // additionally requires a registered `TargetBackend`. Filter through the same
  // registry `--emit` consults so this list cannot advertise a target that then
  // fails to build.
  std::vector<std::pair<std::string, std::string>> targets;
  for (const llvm::Target &tgt : llvm::TargetRegistry::targets()) {
    llvm::Triple triple;
    triple.setArch(llvm::Triple::getArchTypeForLLVMName(tgt.getName()));
    if (TargetBackendRegistry::get().lookup(triple).isError())
      continue;
    targets.emplace_back(tgt.getName(), tgt.getShortDescription());
  }
  // Sort alphabetically for consistent, scannable output.
  llvm::sort(targets);

  llvm::outs() << "Registered Targets:\n";

  // If no targets found, something is wrong with the LLVM build. Handle
  // gracefully.
  if (targets.empty()) {
    llvm::outs() << "  No targets found.\n";
    return EXIT_SUCCESS;
  }

  for (const auto &[name, desc] : targets)
    llvm::outs() << "  " << name << " - " << desc << "\n";

  return EXIT_SUCCESS;
}

int M::printSupportedCpus(StringRef userTriple) {
  if (userTriple.empty()) {
    llvm::errs() << "error: --print-supported-cpus requires --target-triple "
                    "to be specified\n";
    llvm::errs() << "Use --print-supported-targets to see available "
                    "architectures.\n";
    return EXIT_FAILURE;
  }

  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();

  std::string normalized = normalizeTriple(userTriple);
  std::vector<std::string> cpus = getValidCPUsForTarget(normalized);

  if (cpus.empty()) {
    // This likely means invalid target.
    // Clang already printed an error via StderrDiagConsumer, add guidance.
    llvm::errs() << "Use --print-supported-targets to see available "
                    "architectures.\n";
    return EXIT_FAILURE;
  }

  llvm::outs() << "Available CPUs for target " << normalized << ":\n";
  for (const std::string &cpu : cpus)
    llvm::outs() << "  " << cpu << "\n";
  return EXIT_SUCCESS;
}

int M::printSupportedAccelerators() {
  // The '#' characters below are intentional.
  // These delimiters allow Mojo tests to extract the architecture list
  // without needing to know specific architecture prefixes.
  llvm::outs() << "Supported Accelerator Architectures:\n\n#\n";

  // Print one section per target, name-sorted.
  llvm::SmallVector<const TargetTraits *> targets;
  for (const std::unique_ptr<TargetTraits> &traits :
       TargetTraitsRegistry::get().targets())
    if (!traits->acceleratorSectionTitle().empty())
      targets.push_back(traits.get());
  llvm::sort(targets, [](const TargetTraits *lhs, const TargetTraits *rhs) {
    return lhs->name() < rhs->name();
  });

  llvm::ListSeparator sectionSep("\n");
  for (const TargetTraits *traits : targets) {
    llvm::outs() << sectionSep << traits->acceleratorSectionTitle() << "\n";
    // Pad the arch column so the descriptions line up.
    size_t width = 12;
    for (const TargetTraits::AcceleratorArch &arch :
         traits->supportedAcceleratorArchs())
      width = std::max(width, arch.arch.size() + 2);
    for (const TargetTraits::AcceleratorArch &arch :
         traits->supportedAcceleratorArchs())
      llvm::outs() << "  " << llvm::left_justify(arch.arch, width) << "- "
                   << arch.description << "\n";
  }

  llvm::outs() << "#\n";

  llvm::outs() << "\nUsage: mojo build --target-accelerator <arch> file.mojo\n";

  return EXIT_SUCCESS;
}

int M::printEffectiveTarget(TargetInfoAttr targetInfo) {
  std::string normalized = normalizeTriple(targetInfo.getTripleStr());
  StringRef cpu = targetInfo.getArch();
  StringRef features = targetInfo.getFeatures();
  StringRef abi = targetInfo.getAbi();
  StringRef accelerator = targetInfo.getAcceleratorArch();

  llvm::outs() << "Effective target configuration:\n";

  llvm::outs() << "  --target-triple " << normalized << "\n";
  llvm::outs() << "  --target-cpu " << cpu << "\n";
  if (!features.empty())
    llvm::outs() << "  --target-features " << features << "\n";
  if (!abi.empty())
    llvm::outs() << "  --target-abi " << abi << "\n";
  if (!accelerator.empty())
    llvm::outs() << "  --target-accelerator " << accelerator << "\n";

  return EXIT_SUCCESS;
}
