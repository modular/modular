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

#ifndef KGEN_TOOLCOMMON_PASSMANAGERCONFIGOPTIONS_H
#define KGEN_TOOLCOMMON_PASSMANAGERCONFIGOPTIONS_H

#include "Support/ErrorOr.h"
#include "Support/LogicalResult.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include <string>

namespace M::KGEN {
struct PassManagerConfigOptions {
  struct CrashReproducerOptions {
    bool enable = false;
    bool enableLocalMLIRReproducer = false;
    std::string inputFileName;
  };

  struct IRPrintingOptions {
    bool enable = false;
    std::string passName;
    bool shouldPrintAfterPass = false;
    bool printModuleScope = false;
    bool printAfterOnlyOnChange = false;
    bool printAfterOnlyOnFailure = false;
    llvm::raw_ostream *out;
    mlir::OpPrintingFlags opPrintingFlags = mlir::OpPrintingFlags();
  };

  CrashReproducerOptions crashReproducerOptions;
  bool enableTiming = false;
  mlir::TimingScope *timingScope = nullptr;
  IRPrintingOptions irPrintingOptions;
  bool applyPassManagerCLOptions = false;
  std::optional<std::string> operationName;

  /// Configures `pm` and reports pass times into `timingScope`. Only safe when
  /// `pm` is the sole pass manager under that scope; otherwise give each pass
  /// manager its own `nestTimingScope` and use the overload below.
  ErrorOrSuccess configurePassManager(mlir::PassManager &pm) const;

  /// Configures `pm` as above, but reports pass times into `timing` instead of
  /// `timingScope`. `timing` must outlive `pm`.
  ErrorOrSuccess configurePassManager(mlir::PassManager &pm,
                                      mlir::TimingScope &timing) const;

  /// Returns a hidden scope named `name` under `timingScope`, or an empty scope
  /// when the timing is off. MLIR keys pass timers by the address of the pass,
  /// so a pass manager that reports into a scope after another one has been
  /// destroyed can land a pass on a freed pass's address and take over its
  /// timer and name. Give each pass manager a scope of its own. A hidden scope
  /// prints no row of its own in the tree display.
  mlir::TimingScope nestTimingScope(llvm::StringRef name) const;

private:
  ErrorOrSuccess configureWithTiming(mlir::PassManager &pm,
                                     mlir::TimingScope *timing) const;
};

} // namespace M::KGEN

#endif // KGEN_TOOLCOMMON_PASSMANAGERCONFIGOPTIONS_H
