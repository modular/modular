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

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_TRANSFORMS_MOGGPASSES_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_TRANSFORMS_MOGGPASSES_H

#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"

#include "Mojo/HLCFDialect/HLCFDialect.h"
#include "Mojo/LITDialect/LITAttrs.h"
#include "Mojo/LITDialect/LITDialect.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Support/AlignedAlloc.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

namespace M::MO {
class GraphOp;
} // namespace M::MO

namespace M::MOGG {
class MojoRegistry;
} // namespace M::MOGG

namespace M::MOGG {

enum class KernelBuildType {
  kRelease,
  kRelWithDebInfo,
  kDebug,
};

inline bool hasDebuginfo(KernelBuildType kernelBuildType) {
  return kernelBuildType == KernelBuildType::kRelWithDebInfo ||
         kernelBuildType == KernelBuildType::kDebug;
}

inline std::string buildTypeString(KernelBuildType bt) {
  if (bt == KernelBuildType::kRelease)
    return "release";
  if (bt == KernelBuildType::kRelWithDebInfo)
    return "relwithdebinfo";
  if (bt == KernelBuildType::kDebug)
    return "debug";
  llvm_unreachable("All cases should be covered");
}

/// Thread-safe one-time LLVM initialization.
void initLLVM();

/// Probes the host and builds the `TargetInfoAttr` that feeds the cache key,
/// so key computation does no hardware probing and one result can be reused.
///
/// The cache key and codegen must resolve the target the same way, so a
/// caller that overrides the codegen target must pass the identical overrides
/// here, including `forHostExecution` (see `MOGG::getTargetInfo`). Otherwise a
/// host-compiled artifact cached under the virtual target's key poisons
/// lookups by compiles that really target it.
ErrorOr<TargetInfoAttr>
getHostTargetInfoAttr(mlir::MLIRContext *ctx,
                      llvm::StringRef acceleratorArchOverride = "",
                      llvm::StringRef acceleratorTripleOverride = "",
                      bool forHostExecution = false);

KGEN::CompilationOptions
createCompilationOptions(TargetInfoAttr target, KernelBuildType kernelBuildType,
                         bool debugKGENStubs, StringRef saveTempsPrefix);

/// Writes the graph compiler configuration (target, optimization level,
/// versions) as a cache key. `target` is typically from
/// `getHostTargetInfoAttr`.
ErrorOrSuccess
graphCompilerCacheKey(llvm::raw_ostream &keyBuf, TargetInfoAttr target,
                      KernelBuildType buildType = KernelBuildType::kRelease,
                      bool debug = false, llvm::StringRef saveTempsPrefix = "");

struct GraphCompilerCacheKey {
  mlir::MLIRContext *ctx;

  KernelBuildType buildType = KernelBuildType::kRelease;

  bool debug = false;

  std::string saveTempsPrefix = "";

  void print(llvm::raw_ostream &os);
};

//===----------------------------------------------------------------------===//
// Generated Pass Classes and Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

void mergeIdenticalKernelStubs(mlir::ModuleOp module);

/// Lift inline `index.constant` ops feeding `mgp.buffer.get_cached`'s slot
/// operand from kernel-stub bodies into new kernel arguments. See GEX-3685.
void liftBufferCacheSlots(mlir::ModuleOp module);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_TRANSFORMS_MOGGPASSES_H
