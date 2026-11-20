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

#ifndef GENERICML_COMPILEDRIVER_COMPILETOBINARY_H
#define GENERICML_COMPILEDRIVER_COMPILETOBINARY_H

#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "Support/Config.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Support/SourceMgr.h"

namespace M {

/// A context for the `M_compileToBinary()` driver method.
struct CompileToBinaryCtx {
  CompileToBinaryCtx(M::ContextRef ctx)
      : prevRuntimeProfilingTypeMask(Trace::kFullyEnabled),
        context(std::move(ctx)), cpuDevice(context->get<CPUDevice>()),
        mlirCtx(mlir::MLIRContext::Threading::DISABLED),
        diagHandler(sourceMgr, &mlirCtx) {
    assert(cpuDevice && "expected to have a cpuDevice");
    registerContext(mlirCtx, context);
    configureMLIRContext(mlirCtx);

    std::optional<TimeTraceProfiler> &profilerOr = cpuDevice->getProfiler();
    if (profilerOr) {
      // Restored on destruction.
      prevRuntimeProfilingTypeMask = profilerOr->runtimeProfilingTypeMask();
      // `kCompiler` events bloat the trace to several GB during compilation.
      profilerOr->setRuntimeProfilingTypeMask(Trace::disableMaskType(
          prevRuntimeProfilingTypeMask, Trace::kCompiler));
    }
  }

  /// Relies on reverse-declaration-order member destruction.
  ~CompileToBinaryCtx() {
    std::optional<TimeTraceProfiler> &profilerOr = cpuDevice->getProfiler();
    if (profilerOr)
      profilerOr->setRuntimeProfilingTypeMask(prevRuntimeProfilingTypeMask);
  }

  uint64_t prevRuntimeProfilingTypeMask;
  M::ContextRef context;
  AsyncRT::CPUDevice *cpuDevice;
  mlir::MLIRContext mlirCtx;
  mlir::OwningOpRef<mlir::ModuleOp> module;

  // Displays MLIR diagnostics from downstream compilation errors.
  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler diagHandler;
};

} // namespace M

#endif // GENERICML_COMPILEDRIVER_COMPILETOBINARY_H
