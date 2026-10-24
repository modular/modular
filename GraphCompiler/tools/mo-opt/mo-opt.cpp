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

#include "AsyncRT/CompilerSupport/Context.h"
#include "GraphCompiler/GraphCompiler.h"
#include "Init/Init.h"
#include "MLRT/GraphRT/MEFDialect/MEFDialect.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "Support/Context.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Support/Timing.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PrettyStackTrace.h"

using namespace M;

int main(int argc, char **argv) {
  // No llvm::InitLLVM here: `mlir::MlirOptMain` creates the only one.
  llvm::setBugReportMsg(M::MO::BugReportMessage.data());

  DialectRegistry registry;
  MO::loadDialects(registry);
  registry.insert<mlir::func::FuncDialect, MEF::MEFDialect>();

  MO::registerPasses();
  mlir::registerDefaultTimingManagerCLOptions();

  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "mo-opt", Init::Options().withCPUDeviceOptions(
                    AsyncRT::CPUDeviceOptions().withLeakCheckedAllocator()));
  if (ctxOr.isError()) {
    llvm::errs() << "failed to create context: " << ctxOr.getError() << "\n";
    return 1;
  }
  registerContext(registry, *ctxOr);

  static llvm::cl::opt<bool> timeTrace{
      "time-trace",
      llvm::cl::desc("Turn on time profiler. Generates JSON file "
                     "called kgen.trace.json in the derived directory. "
                     "Currently integrated mainly for KGEN compilation.")};

  static llvm::cl::opt<int> timeTraceGranularity{
      "time-trace-granularity",
      llvm::cl::desc("Minimum time granularity (in microseconds) "
                     "traced by time profiler."),
      llvm::cl::init(0)};

  std::string inputFilename, outputFilename;
  std::tie(inputFilename, outputFilename) =
      registerAndParseCLIOptions(argc, argv, "mo optimizer driver", registry);
  KGEN::TraceProfiler tracer(timeTrace, timeTraceGranularity);
  return failed(
      MlirOptMain(argc, argv, inputFilename, outputFilename, registry));
}
