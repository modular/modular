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

#include "GraphCompiler/MOGGDialect/Support/MojoStubCompiler.h"

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

namespace M::MOGG {

// Builds the vendor-prefixed accelerator string ("nvidia:sm_90a",
// "amdgpu:gfx942") from an {arch, triple} override. The override carries the
// default device, which may be a CPU (e.g. "graniterapids"); a non-GPU triple
// maps to empty so the caller probes the host GPU. The triple, not the arch
// mnemonic, decides GPU-ness and the prefix, which consumers (e.g. the ptxas
// invocation) split on ':'.
static std::string overrideAcceleratorArch(StringRef arch,
                                           StringRef tripleStr) {
  if (arch.empty() || tripleStr.empty())
    return "";
  llvm::Triple triple(tripleStr);
  if (!KGEN::isGPUTriple(triple))
    return ""; // non-GPU target (e.g. a CPU default device)
  if (arch.contains(':'))
    return arch.str(); // already vendor-qualified (e.g. Metal's "metal:3")
  if (triple.isNVPTX())
    return ("nvidia:" + arch).str();
  if (triple.isAMDGCN())
    return ("amdgpu:" + arch).str();
  if (KGEN::isMetalTriple(triple))
    return ("metal:" + arch).str();
  return arch.str(); // GPU triple, unrecognized vendor: pass the arch through
}

static ErrorOr<std::string> resolveVirtualCpuArch(StringRef requested,
                                                  StringRef triple) {
  if (requested.empty())
    return std::string();
  if (requested == "native")
    return Error("virtual CPU target 'native' re-leaks the host CPU; pass an "
                 "explicit arch (e.g. 'x86-64-v3') or 'generic'.");
  if (requested != "generic")
    return requested.str();

  llvm::Triple t(triple);
  if (t.getArch() == llvm::Triple::x86_64)
    return std::string("x86-64");
  if (t.isAArch64())
    return std::string("generic"); // LLVM's AArch64 baseline CPU name
  return Error("virtual CPU target 'generic' is only supported for x86_64 and "
               "aarch64 hosts.");
}

ErrorOr<VirtualCpuTargetSpec> parseVirtualCpuTarget(StringRef knob,
                                                    StringRef hostTriple) {
  // No '=' => legacy bare-arch form.
  if (!knob.contains('=')) {
    ErrorOr<std::string> archOr = resolveVirtualCpuArch(knob, hostTriple);
    if (archOr.isError())
      return archOr.takeError();
    VirtualCpuTargetSpec spec;
    spec.cpu = archOr.takeValue();
    return spec;
  }

  VirtualCpuTargetSpec spec;
  bool haveCpu = false;
  llvm::SmallVector<StringRef, 4> fields;
  knob.split(fields, ';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef field : fields) {
    auto [key, value] = field.split('=');
    key = key.trim();
    value = value.trim();
    if (key == "triple") {
      spec.triple = value.str();
    } else if (key == "cpu") {
      spec.cpu = value.str();
      haveCpu = !value.empty();
    } else if (key == "features") {
      spec.features = value.str();
      spec.hasExplicitFeatures = true;
    } else {
      return Error("unknown virtual CPU target field '" + key.str() +
                   "'; expected 'triple', 'cpu', or 'features'.");
    }
  }
  if (!haveCpu)
    return Error("virtual CPU target descriptor must specify a non-empty "
                 "'cpu=' field.");

  // De-sugar "generic" against the descriptor's triple (else host) so it
  // resolves to the target arch's baseline, not the host's.
  StringRef effTriple =
      spec.triple.empty() ? hostTriple : StringRef(spec.triple);
  ErrorOr<std::string> archOr = resolveVirtualCpuArch(spec.cpu, effTriple);
  if (archOr.isError())
    return archOr.takeError();
  spec.cpu = archOr.takeValue();
  return spec;
}

ErrorOr<TargetInfoAttr> getTargetInfo(MLIRContext *ctx,
                                      StringRef acceleratorArchOverride,
                                      StringRef acceleratorTripleOverride,
                                      bool forHostExecution) {
  std::string triple = llvm::sys::getDefaultTargetTriple();
  std::string cpu = llvm::sys::getHostCPUName().str();
  std::string features = getHostCPUFeatures();

  // A GPU override wins over probing the host, so a GPU can be targeted from a
  // machine without one; a non-GPU override falls back to host detection. Only
  // the accelerator is overridden; host triple/CPU/features stay as-is.
  auto acceleratorArch = [&]() -> std::string {
    if (std::string gpu = overrideAcceleratorArch(acceleratorArchOverride,
                                                  acceleratorTripleOverride);
        !gpu.empty())
      return gpu;
    return M::Driver::Device::getAcceleratorArchOrEmpty();
  };

  std::string virtualCpu =
      forHostExecution ? std::string() : M::Driver::getVirtualCpuTarget();
  if (!virtualCpu.empty()) {
    ErrorOr<VirtualCpuTargetSpec> specOr =
        parseVirtualCpuTarget(virtualCpu, triple);
    if (specOr.isError())
      return specOr.takeError();
    VirtualCpuTargetSpec spec = specOr.takeValue();
    if (!spec.triple.empty())
      triple = spec.triple;
    cpu = spec.cpu;
    // Name-only (empty) features unless the descriptor pins them verbatim;
    // name-only derives from the CPU model with no host CPUID leak.
    features = spec.hasExplicitFeatures ? spec.features : "";
  }

  ErrorOr<TargetInfoAttr> targetOr =
      getTargetInfoFor(ctx, triple, cpu, features, /*tuneCpu=*/"",
                       /*acceleratorArch=*/acceleratorArch());
  if (targetOr.isError())
    return targetOr;

  TargetInfoAttr target = targetOr.takeValue();

  // Completes the target info with fields derived from the spec's core fields.
  ErrorOr<TargetInfoAttr> targetInfoOr = getTargetInfoFor(
      ctx, target.getTripleStr(), target.getArch(), target.getFeatures(),
      target.getTuneCpu(), acceleratorArch());
  return targetInfoOr;
}

DenseResourceElementsAttr createArchiveAttr(MLIRContext *ctx, BufferRef buffer,
                                            StringRef name) {
  auto resourceManager = DenseResourceElementsHandle::getManagerInterface(ctx);

  // Pretend this is a "tensor" of data.
  auto attrType =
      RankedTensorType::get({(int64_t)buffer->getBufferSize()},
                            IntegerType::get(ctx, 8, IntegerType::Unsigned));
  std::string attrName = "archive_" + name.str();
  mlir::AsmResourceBlob blob(
      {buffer->getBufferStart(), buffer->getBufferSize()}, /*dataAlignment=*/8,
      /*deleter=*/
      [buffer = std::move(buffer)](void *, size_t, size_t) mutable {
        buffer.reset();
      },
      /*isMutable=*/false);
  return DenseResourceElementsAttr::get(
      attrType, resourceManager.insert(attrName, std::move(blob)));
}

} // namespace M::MOGG
