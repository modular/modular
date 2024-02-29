//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Helpers.h"
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
/// This struct tracks the current IR state of the reducer.
struct IRState {
  IRState(OwningOpRef<ModuleOp> ir) : ir(std::move(ir)), symtab(*this->ir) {}

  OwningOpRef<ModuleOp> ir;
  SymbolTable symtab;
};

/// This is the reducer class that keeps track of all state during reduction.
struct Reducer {
  llvm::cl::opt<std::string> inputFilename{llvm::cl::Positional,
                                           llvm::cl::desc("<input file>"),
                                           llvm::cl::init("-")};

  llvm::cl::opt<std::string> pipeline{"pipeline",
                                      llvm::cl::desc("Repro pipeline string")};

  cl::opt<unsigned> numSnapshots{"num-snapshots",
                                 llvm::cl::desc("number of snapshots to keep"),
                                 llvm::cl::init(10)};

  cl::opt<unsigned> snapshotDelta{
      "snapshot-delta",
      llvm::cl::desc("delta between snapshots in millseconds"),
      llvm::cl::init(2000)};

  Reducer(MLIRContext *ctx) : ctx(ctx), reproPm(ctx), dcePm(ctx) {
    dcePm.addPass(createEliminateDeadSymbols());
  }

  ErrorOrSuccess run();

  /// Run the pass pipeline on a clone of the module and return the diagnostics
  /// if any were emitted.
  std::optional<std::string> attemptRepro(ModuleOp ir);

  /// Maybe save a snapshot of the current module if enough time has passed
  /// since the last.
  ErrorOrSuccess maybeSnapshot(ModuleOp ir);

  ErrorOrSuccess reduceFunctions(IRState &curState);

  /// The MLIR context.
  MLIRContext *ctx;

  /// The pass manager containing to run to generate the desired error.
  mlir::PassManager reproPm;
  /// The pass manager used to run symbol DCE.
  mlir::PassManager dcePm;

  /// The expected error to reproduce.
  std::string expectedDiag;

  /// The set of module snapshots to keep.
  std::queue<std::unique_ptr<llvm::ToolOutputFile>> snapshots;

  /// Time since the last snapshot.
  uint64_t lastSnapshotTime;

  /// Logging output.
  llvm::raw_ostream &log = llvm::outs();
};
} // namespace

ErrorOrSuccess Reducer::run() {
  OwningOpRef<ModuleOp> inputModule = mlir::parseSourceFile<ModuleOp>(
      inputFilename.getValue(), mlir::ParserConfig(ctx));
  if (!inputModule)
    return Error("failed to parse input file: " + inputFilename.getValue());

  log << "[[===============================================================]]\n"
      << "[[======================== KGEN ⚜️ REDUCER =======================]]\n"
      << "[[===============================================================]]"
         "\n\n";

  log << "[kgen-reduce] " << inputFilename.getValue() << "\n";
  log << "[kgen-reduce] " << pipeline.getValue() << "\n";

  // Parse the pass pipeline.
  {
    std::string err;
    llvm::raw_string_ostream os(err);
    if (failed(mlir::parsePassPipeline(pipeline.getValue(), reproPm, os)))
      return Error(err);
  }

  std::optional<std::string> initDiag = attemptRepro(*inputModule);
  if (!initDiag)
    return Error("original input IR does not fail the provided pipeline");
  expectedDiag = std::move(*initDiag);

  llvm::errs() << "[kgen-reduce] expected diagnostic:\n"
               << expectedDiag << "\n\n";

  IRState curModule(std::move(inputModule));
  if (auto err = stashFile(*curModule.ir, "kgen-reduce.base"))
    return err;

  if (auto err = reduceFunctions(curModule))
    return err;
  if (auto err = stashFile(*curModule.ir, "kgen-reduce.functions"))
    return err;

  return success();
}

std::optional<std::string> Reducer::attemptRepro(ModuleOp ir) {
  std::string diag;
  llvm::raw_string_ostream os(diag);
  mlir::ScopedDiagnosticHandler handler(
      ctx, [&](Diagnostic &diag) { diag.print(os); });
  OwningOpRef<ModuleOp> tmpModule = ir.clone();
  if (succeeded(reproPm.run(*tmpModule)))
    return {};
  return std::move(diag);
}

ErrorOrSuccess Reducer::maybeSnapshot(ModuleOp module) {
  uint64_t curTime = getCurTimeMs();
  if (curTime - lastSnapshotTime < snapshotDelta.getValue())
    return success();
  lastSnapshotTime = curTime;

  auto fileOr = getTempFile(module, getTempFileName());
  if (fileOr.isError())
    return fileOr.takeError();
  auto file = fileOr.takeValue();

  llvm::errs() << "[kgen-reduce] snapshotting IR to " << file->getFilename()
               << "\n";

  file->keep();
  llvm::sys::DontRemoveFileOnSignal(file->getFilename());
  snapshots.push(std::move(file));

  // Pop the oldest file off and unkeep it.
  if (snapshots.size() > numSnapshots.getValue()) {
    auto drop = std::move(snapshots.front());
    snapshots.pop();
    unkeepToolOutputFile(*drop);
  }

  return success();
}

ErrorOrSuccess Reducer::reduceFunctions(IRState &curState) {
  std::vector<KGEN::FuncOp> funcs;
  bool anyKgenFunc = false;
  for (auto func : curState.ir->getOps<KGEN::FuncOp>()) {
    anyKgenFunc = true;
    // Ignore functions that are already stubbed.
    if (isa<KGEN::UnreachableOp>(func.getBody()->front()))
      continue;
    funcs.push_back(func);
  }
  if (!anyKgenFunc) {
    return Error("zero 'kgen.func' operations found, 'kgen-reduce' only works "
                 "on post-elaboration IR right now");
  }

  size_t funcNum = 0;
  const size_t totalNumFuncs = funcs.size();
  llvm::errs() << "[kgen-reduce] attempt to stub " << totalNumFuncs
               << " functions\n";

  // TODO: This is a linear search, which gets faster as more functions are
  // stubbed but it would be even faster to do a bisect search.
  //
  // Starting with N functions, stub out N/2. If repro, continue. Else, bisect
  // only the first half and then repeat until repro. Then repeat on remaining
  // functions.
  while (!funcs.empty()) {
    if (auto err = maybeSnapshot(*curState.ir))
      return err.takeError();

    KGEN::FuncOp func = funcs.back();
    funcs.pop_back();

    llvm::errs() << "[stubbing function " << (funcNum++) << "/" << totalNumFuncs
                 << "] " << func.getSymName() << "\n";

    // Remove the body of the function and store it here.
    Region owner;
    owner.takeBody(func.getBodyRegion());

    // Create a new block with the same argument kinds.
    func.getBodyRegion().push_back(new Block);
    for (BlockArgument arg : owner.getArguments())
      func.getBody()->addArgument(arg.getType(), arg.getLoc());

    // Stub the function with an unreachable.
    OpBuilder b(func.getBody(), func.getBody()->begin());
    b.create<KGEN::UnreachableOp>(func.getLoc());

    std::optional<std::string> nextDiag = attemptRepro(*curState.ir);
    if (nextDiag && nextDiag == expectedDiag) {
      llvm::errs() << "[kgen-reduce] same failure 🛑\n\n";
      continue;
    }

    if (!nextDiag) {
      llvm::errs() << "[kgen-reduce] succeeded 🟢\n\n";
    } else {
      llvm::errs() << "[kgen-reduce] different failure 🟠:\n"
                   << *nextDiag << "\n\n";
    }

    // Revert the transformation.
    func.getBodyRegion().takeBody(owner);
  }

  return success();
}

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerAllKGENDialects(registry);
  MLIRContext ctx;
  ctx.appendDialectRegistry(registry);

  Reducer reducer(&ctx);

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);

  LLCL::RuntimeOptions llclOpts;
  llclOpts.withLeakCheckedAllocator();
  std::unique_ptr<LLCL::Runtime> runtime = LLCL::createUniqueRuntime(llclOpts);
  KGEN::registerDefaultKGENPasses(*runtime);

  if (auto err = reducer.run()) {
    llvm::errs() << "ERROR: " << err.getError() << "\n";
    return -1;
  }
}
