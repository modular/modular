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
//
// This is a command line driver to test GraphRT Primitives.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/RuntimeCLOptions.h"
#include "AsyncRT/Runtime/Utils.h"
#include "Init/Init.h"
#include "MLRT/GraphRT/GRTTestcase.h"
#include "MLRT/GraphRT/MEFExecutor/MEFFile.h"
#include "MLRT/GraphRT/MEFExecutor/MEFInvokable.h"
#include "MLRT/GraphRT/MEFExecutor/PrimitiveRegistry.h"
#include "MLRT/GraphRT/Primitives.h"
#include "Support/CommandLine.h"
#include "Support/CommonCLOptions.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/ML/SizeUtils.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace M::AsyncRT;
using mlir::MLIRContext;

/// This method can be used to print the results of `invokable` in the
/// standard format.
static void printResults(MEFInvokable *invokable,
                         ArrayRef<AnyAsyncValueRef> results,
                         MLIRContext *context, raw_ostream &os) {
  // Check all the results for errors, and if we find any of them, emit them
  // through MLIR error handling.  We use MLIR here just because it makes
  // diagnostic verification easy.  We do this before printing the results
  // so the diagnostics and result printing don't get intermixed.
  for (auto &result : results) {
    if (auto *diag = result.getDiagnosticIfPresent()) {
      auto loc = diag->decodeLocation();
      auto mlirLoc =
          FileLineColLoc::get(context, loc.filename, loc.line, loc.column);
      mlir::emitError(mlirLoc) << diag->getMessage();
    }
  }

  // Print out the invokable results that we know about.
  // Go ahead and print out the invokable results that we know about.
  if (!results.empty()) {
    llvm::interleaveComma(results, os, [&](const AnyAsyncValueRef &result) {
      // Handle error value by printing it inline, and emitting it as an error
      // through SourceMgr so we can validate it.
      if (auto *diag = result.getDiagnosticIfPresent()) {
        os << "<error: " << diag->getMessage() << '>';
        return;
      }

      // Otherwise we support a set of standard types for convenience.
#define HANDLE_TYPE(TYPE)                                                      \
  if (result.isType<TYPE>()) {                                                 \
    os << result.get<TYPE>();                                                  \
    return;                                                                    \
  }
      HANDLE_TYPE(bool)
      HANDLE_TYPE(int8_t)
      HANDLE_TYPE(int16_t)
      HANDLE_TYPE(int32_t)
      HANDLE_TYPE(int64_t)
      HANDLE_TYPE(uint8_t)
      HANDLE_TYPE(uint16_t)
      HANDLE_TYPE(uint32_t)
      HANDLE_TYPE(uint64_t)
      HANDLE_TYPE(float)
      HANDLE_TYPE(double)
      HANDLE_TYPE(std::string)
      if (result.isType<Chain>())
        os << "chain";
      else
        os << "<unknown type>";
    });
    os << '\n';
    os.flush();
  }
}

//===----------------------------------------------------------------------===//
// MEFExecutorMain implementation
//===----------------------------------------------------------------------===//

/// Run the specified mef.invokable as a testcase - this means passing in test
/// input and printing results if we can.
static ErrorOrSuccess runTestcase(ContextRef ctx, const GRTTestcase &test,
                                  MLIRContext *context) {
  // In this test driver, we want to make sure that every invokable completes
  // all execution before moving on to the next one.  This makes the leak
  // checker work better in the face of side effecting kernels that aren't
  // properly chained together (which is useful for testing).  As such, we run
  // each testcase with its own cpuDevice instance and own leak checker.
  runWithLeakCheckedRuntime(test.getName().str().c_str(), [&]() {
    // Actually run the invokable.  We use printf here because it has atomic
    // output behavior w.r.t. the printfs in various TestPrimitives.
    printf("--- Running '%.*s':\n", (int)test.getName().size(),
           test.getName().data());
    fflush(stdout);

    AsyncRT::CPUDevice &cpuDevice = *ctx->get<AsyncRT::CPUDevice>();
    auto results = test.getBody()->invoke(/*arguments=*/{}, cpuDevice);

    // Block until the invokable results are fully resolved.
    await(results);

    // If anything printed to stdout, force it out to a stream, since we
    // will be printing with raw_ostream.
    fflush(stdout);

    auto &os = llvm::outs();
    if (!results.empty())
      os << "'" << test.getName() << "' returned ";

    printResults(test.getBody(), results, context, os);
  });

  // FIXME: this function currently does not handle errors.
  return success();
}

namespace {
class MEFExecutorCLOptions : public CommonOptions, public CPUDeviceOptions {
public:
  CommonCLOptions parser;
  RuntimeCLOptions rtParser;
  MEFExecutorCLOptions(int argc, char **argv, bool skipInitLLVM = false)
      : parser(argc, argv, *this, skipInitLLVM), rtParser(*this) {}

  M::cl::MListOpt<std::string> testcases{
      "tests",
      cl::desc("Specify testcases to run, defaults to all of them. Provide a "
               "comma-separated list like --tests=foo,bar."),
      llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated};

  M::cl::MOpt<bool> verbose{
      "verbose", cl::desc("Print configuration information at startup"),
      cl::init(false)};
};
} // namespace

static LogicalResult
runTestcases(SmallVectorImpl<GRTTestcase> &testcasesToExecute,
             const MEFExecutorCLOptions &options, MLIRContext *context) {
  // Create a copy of the cpuDevice options from the MEFExecutorCLOptions
  // options.
  CPUDeviceOptions cpuDeviceOptions = options.copy();

  // Create our execution context. Note that we limit the scope of this
  // context in order to ensure that the full cpuDevice is torn-down before
  // the mefFile is reset, since there may still be operations outstanding.
  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "mef-executor", Init::Options().withCPUDeviceOptions(cpuDeviceOptions));
  if (ctxOr.isError())
    return ctxOr.takeError();

  registerContext(*context, *ctxOr);

  for (auto &fn : testcasesToExecute) {
    if (auto err = runTestcase(ctxOr->copy(), fn, context)) {
      options.reportError(err.getError());
      return failure();
    }
  }
  return success();
}

int main(int argc, char **argv) {
  MEFExecutorCLOptions options(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "mef-executor test driver\n");

  // Set up all our primitives.
  PrimitiveRegistry primitives;
  registerAllGraphRTPrimitives(primitives);
  primitives.finalize();

  verifyNoLiveReferenceCountedObjects("No objects should start allocated");

  if constexpr (AsyncValue::isAllocationTrackingEnabled())
    assert(AsyncValue::getNumAllocatedInstances() == 0 &&
           "We have async values allocated before we started to do anything");

  // Read the input file from disk into a memory buffer.
  auto fileBuffer = options.openInputFileOrExit(maxAttributeAlignment());

  // Print some informative messages about allocator and threadpool config.
  if (options.verbose.getValue()) {
    printf("%s ", argv[0]);
    options.printRuntimeConfig();
    fflush(stdout);
  }

  auto mefFileOrError = MEFFile::open(fileBuffer->getBuffer(), primitives);
  if (failed(mefFileOrError))
    return options.reportError(Twine("error opening '") +
                               options.inputFilename +
                               "': " + mefFileOrError.getError());

  auto mefFile = mefFileOrError.takeValue();

  // Figure out which testcases we want to execute.
  SmallVector<GRTTestcase, 1> testcasesToExecute;

  auto *moduleInv = mefFile->getTopLevelInvokable();
  if (auto err = moduleInv->walkPrimitives<GRTTestcase>([&](GRTTestcase test) {
        // If no testcases are specified, run them all.
        if (options.testcases.empty() ||
            llvm::is_contained(options.testcases, test.getName()))
          testcasesToExecute.push_back(test);
      })) {
    return options.reportError(err.getError());
  }

  // Run each test with a MLIRContext set up for the right kind of diagnostic
  // emission.
  LogicalResult hadError = options.configureMLIRContextAndExecute(
      std::move(fileBuffer), [&](MLIRContext *context) -> LogicalResult {
        return runTestcases(testcasesToExecute, options, context);
      });

  // We're done with the MEF, we can safely deallocate (see above).
  mefFile.reset();

  if constexpr (AsyncValue::isAllocationTrackingEnabled())
    assert(AsyncValue::getNumAllocatedInstances() == 0 &&
           "All async values should be cleaned up at the end");
  verifyNoLiveReferenceCountedObjects(
      "All refcounted objects should be deallocated");
  return failed(hadError);
}
