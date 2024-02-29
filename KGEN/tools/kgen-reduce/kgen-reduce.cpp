//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/ToolCommon/CLOptions.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/ToolOutputFile.h"

#include <chrono>
#include <queue>

using namespace M;
using namespace KGEN;

namespace {
struct IRState {
  IRState(OwningOpRef<ModuleOp> ir) : ir(std::move(ir)), symtab(*this->ir) {}

  OwningOpRef<ModuleOp> ir;
  SymbolTable symtab;
};
} // namespace

static uint64_t getCurTimeMs() {
  using namespace std::chrono;
  auto ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
  return ms.count();
}

static std::string getTempName() {
  return ("kgen-reduce." + Twine(getCurTimeMs())).str();
}

static ErrorOr<std::unique_ptr<llvm::ToolOutputFile>>
getTempFile(ModuleOp module, const Twine &fileName) {
  std::string err;
  std::unique_ptr<llvm::ToolOutputFile> output =
      mlir::openOutputFile((fileName + ".mlirbc").str());
  if (!output)
    return Error(err);
  if (failed(mlir::writeBytecodeToFile(module, output->os())))
    return Error("failed to write bytecode");
  return std::move(output);
}

static ErrorOrSuccess stashFile(ModuleOp module, const Twine &fileName) {
  auto err = getTempFile(module, fileName);
  if (err.isError())
    return err.takeError();
  err.takeValue()->keep();
  return success();
}

/// HACK: Unkeep a file and mark it for deletion. Access private members of the
/// type.
static void unkeep(llvm::ToolOutputFile &file) {
  struct DirtyHack {
    struct CleanupInstaller {
      std::string filename;
      bool keep;
    } installer;

    std::optional<llvm::raw_fd_ostream> osHolder;
    llvm::raw_fd_ostream *os;
  };
  ((DirtyHack *)&file)->installer.keep = false;
}

static ErrorOrSuccess reduceLoop(OwningOpRef<ModuleOp> origModule,
                                 StringRef pipeline) {
  llvm::errs() << "[kgen-reduce] Pipeline: " << pipeline << "\n";

  MLIRContext *ctx = origModule->getContext();

  mlir::PassManager pm(ctx);
  std::string err;
  llvm::raw_string_ostream errOs(err);
  if (failed(mlir::parsePassPipeline(pipeline, pm, errOs)))
    return Error(err);

  mlir::PassManager dce(ctx);
  dce.addPass(createEliminateDeadSymbols());

  std::string expectedDiag;
  llvm::raw_string_ostream expectedDiagOs(expectedDiag);
  {
    mlir::ScopedDiagnosticHandler handler(
        ctx, [&](Diagnostic &diag) { diag.print(expectedDiagOs); });
    OwningOpRef<ModuleOp> testModule = origModule->clone();
    if (auto err = stashFile(*testModule, "kgen-reduce.base"))
      return err;
    if (succeeded(pm.run(*testModule)))
      return Error("original input IR does not fail the provided pipeline");
  }

  llvm::errs() << "[kgen-reduce] expected diagnostic:\n"
               << expectedDiag << "\n\n";

  llvm::errs() << "[kgen-reduce] trimming functions\n";

  std::vector<StringAttr> funcNames;
  bool anyKgenFunc = false;
  for (auto func : origModule->getOps<KGEN::FuncOp>()) {
    anyKgenFunc = true;
    // Ignore functions that are already stubbed.
    if (isa<KGEN::UnreachableOp>(func.getBody()->front()))
      continue;
    funcNames.push_back(func.getSymNameAttr());
  }
  if (!anyKgenFunc) {
    return Error("zero 'kgen.func' operations found, 'kgen-reduce' only works "
                 "on post-elaboration IR right now");
  }

  IRState curModule(std::move(origModule));

  size_t funcNum = 0;
  const size_t totalNumFuncs = funcNames.size();
  llvm::errs() << "[kgen-reduce] attempt to stub " << totalNumFuncs
               << " functions\n";

  // Persist the last N successfully reduce reproducers. This is so that if the
  // tool crashes, is interrupted, or whatever, the user still has access to the
  // last 10 snapshotted files.
  constexpr size_t keepCount = 10;
  std::queue<std::unique_ptr<llvm::ToolOutputFile>> lastNAttempts;
  uint64_t lastSnapshotTimeMs = getCurTimeMs();
  // Save a snapshot every 2 seconds to avoid bottlenecking on the filesystem.
  constexpr uint64_t snapshotDelta = 2000;
  auto maybeSnapshotModule = [&](ModuleOp module) -> ErrorOrSuccess {
    uint64_t curTime = getCurTimeMs();
    if (curTime - lastSnapshotTimeMs < snapshotDelta)
      return success();
    lastSnapshotTimeMs = curTime;

    llvm::errs() << "[kgen-reduce] snapshotting (!!) current function\n";

    auto fileOr = getTempFile(module, getTempName());
    if (fileOr.isError())
      return fileOr.takeError();
    auto file = fileOr.takeValue();
    file->keep();
    llvm::sys::DontRemoveFileOnSignal(file->getFilename());
    lastNAttempts.push(std::move(file));
    // Pop the oldest file off and unkeep it.
    if (lastNAttempts.size() > keepCount) {
      auto drop = std::move(lastNAttempts.front());
      lastNAttempts.pop();
      unkeep(*drop);
    }
    return success();
  };

  // TODO: This is a linear search, which gets faster as more functions are
  // stubbed but it would be even faster to do a bisect search.
  //
  // Starting with N functions, stub out N/2. If repro, continue. Else, bisect
  // only the first half and then repeat until repro. Then repeat on remaining
  // functions.
  while (!funcNames.empty()) {
    IRState nextModule(curModule.ir->clone());
    if (auto err = maybeSnapshotModule(*nextModule.ir))
      return err.takeError();

    StringAttr toRemove = funcNames.back();
    funcNames.pop_back();

    auto func = nextModule.symtab.lookup<KGEN::FuncOp>(toRemove);
    if (!func)
      continue;

    llvm::errs() << "[stubbing function " << (funcNum++) << "/" << totalNumFuncs
                 << "] " << toRemove << "\n\n";

    func.getBody()->clear();
    OpBuilder b(func.getBody(), func.getBody()->begin());
    b.create<KGEN::UnreachableOp>(func.getLoc());

    std::string nextDiag;
    llvm::raw_string_ostream nextDiagOs(nextDiag);
    bool repro;
    {
      mlir::ScopedDiagnosticHandler handler(
          ctx, [&](Diagnostic &diag) { diag.print(nextDiagOs); });
      OwningOpRef<ModuleOp> testModule = nextModule.ir->clone();
      if (succeeded(pm.run(*testModule))) {
        llvm::errs() << "[kgen-reduce] succeeded\n";
        repro = false;
      } else if (nextDiag != expectedDiag) {
        llvm::errs() << "[kgen-reduce] different failure:\n"
                     << nextDiag << "\n\n";
        repro = false;
      } else {
        llvm::errs() << "[kgen-reduce] same failure\n";
        repro = true;
      }
    }

    if (repro)
      curModule = std::move(nextModule);
  }

  if (auto err = stashFile(*curModule.ir, "kgen-reduce.functions"))
    return err.takeError();

  return success();
}

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerAllKGENDialects(registry);

  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"),
      llvm::cl::init("-"));

  static llvm::cl::opt<std::string> pipeline(
      "pipeline", llvm::cl::desc("Repro pipeline string"));

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);

  MLIRContext ctx;
  ctx.appendDialectRegistry(registry);

  OwningOpRef<ModuleOp> module = mlir::parseSourceFile<ModuleOp>(
      inputFilename.getValue(), mlir::ParserConfig(&ctx));
  if (!module) {
    llvm::errs() << "failed to parse input MLIR file\n";
    return -1;
  }

  LLCL::RuntimeOptions llclOpts;
  llclOpts.withLeakCheckedAllocator();
  std::unique_ptr<LLCL::Runtime> runtime = LLCL::createUniqueRuntime(llclOpts);
  KGEN::registerDefaultKGENPasses(*runtime);

  if (ErrorOrSuccess err = reduceLoop(std::move(module), pipeline);
      err.isError()) {
    llvm::errs() << "error: " << err.getError() << "\n";
    return -1;
  }

  return 0;
}
