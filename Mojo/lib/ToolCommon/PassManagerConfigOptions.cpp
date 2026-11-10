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

#include "Mojo/ToolCommon/PassManagerConfigOptions.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "Support/Compiler/TimeProfilerTimingManager.h"
#include "Support/Config.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;

ErrorOrSuccess
PassManagerConfigOptions::configurePassManager(mlir::PassManager &pm) const {
  return configureWithTiming(pm, timingScope);
}

ErrorOrSuccess PassManagerConfigOptions::configurePassManager(
    mlir::PassManager &pm, mlir::TimingScope &timing) const {
  return configureWithTiming(pm, timing ? &timing : nullptr);
}

mlir::TimingScope
PassManagerConfigOptions::nestTimingScope(llvm::StringRef name) const {
  if (!timingScope)
    return mlir::TimingScope();
  mlir::TimingScope scope = timingScope->nest(name);
  scope.hide();
  return scope;
}

ErrorOrSuccess
PassManagerConfigOptions::configureWithTiming(mlir::PassManager &pm,
                                              mlir::TimingScope *timing) const {
  M::configurePassManager(pm);

  if (applyPassManagerCLOptions) {
    if (failed(mlir::applyPassManagerCLOptions(pm)))
      return Error("applyPassManagerCLOptions failed during configuring");
  }

  if (timing)
    pm.enableTiming(*timing);
  else if constexpr (KGEN::kIsTracingEnabled)
    pm.enableTiming(std::make_unique<TimeProfilerTimingManager>());
  else if (enableTiming)
    pm.enableTiming();

  if (crashReproducerOptions.enable) {
    pm.enableCrashReproducerGeneration(
        crashReproducerOptions.inputFileName + ".repro.mlir",
        crashReproducerOptions.enableLocalMLIRReproducer);
  }

  if (irPrintingOptions.enable) {
    pm.enableIRPrinting(
        [&](mlir::Pass *pass, mlir::Operation *) -> bool {
          return pass->getName() == irPrintingOptions.passName;
        },
        [&](mlir::Pass *, mlir::Operation *) -> bool {
          return irPrintingOptions.shouldPrintAfterPass;
        },
        irPrintingOptions.printModuleScope,
        irPrintingOptions.printAfterOnlyOnChange,
        irPrintingOptions.printAfterOnlyOnFailure, *irPrintingOptions.out,
        irPrintingOptions.opPrintingFlags);
  }

  return {};
}
