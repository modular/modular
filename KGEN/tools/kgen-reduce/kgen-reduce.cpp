//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Helpers.h"
#include "KGEN/HLCFDialect/HLCFInterfaces.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/ToolCommon/CLOptions.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "KGEN/TransformUtils/Walkers.h"
#include "LLCL/CompilerSupport/Context.h"
#include "PreOrderRegionIterator.h"
#include "Support/Context.h"
#include "Support/Init/Init.h"
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
  IRState(OwningOpRef<ModuleOp> ir) : ir(std::move(ir)) {}

  OwningOpRef<ModuleOp> ir;
};

/// This is the reducer class that keeps track of all state during reduction.
struct Reducer {
  llvm::cl::opt<std::string> inputFilename{llvm::cl::Positional,
                                           llvm::cl::desc("<input file>"),
                                           llvm::cl::init("-")};

  mlir::PassPipelineCLParser passPipeline{"", "pass pipeline to run"};
  mlir::PassReproducerOptions reproOptions;

  cl::opt<unsigned> numSnapshots{"num-snapshots",
                                 llvm::cl::desc("number of snapshots to keep"),
                                 llvm::cl::init(10)};

  cl::opt<unsigned> snapshotDelta{
      "snapshot-delta",
      llvm::cl::desc("delta between snapshots in millseconds"),
      llvm::cl::init(2000)};

  cl::opt<unsigned> startAt{"start-at",
                            llvm::cl::desc("the reducer phase to start at"),
                            llvm::cl::init(0)};

  Reducer(MLIRContext *ctx) : ctx(ctx), reproPm(ctx), dcePm(ctx) {
    dcePm.addPass(createEliminateDeadSymbols());
  }

  ErrorOrSuccess run();
  ErrorOrSuccess reduceFunctions(IRState &curState);
  ErrorOrSuccess reduceRegions(IRState &curState);
  ErrorOrSuccess reduceOps(IRState &curState);
  ErrorOrSuccess tryDCE(IRState &curState);

  /// Run the pass pipeline on a clone of the module and return the diagnostics
  /// if any were emitted.
  std::optional<std::string> attemptRepro(ModuleOp ir);
  /// Return true if the supplied module repros the error.
  bool doesRepro(ModuleOp ir);

  /// Maybe save a snapshot of the current module if enough time has passed
  /// since the last.
  ErrorOrSuccess maybeSnapshot(ModuleOp ir);

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
  mlir::ParserConfig parserConfig(ctx);
  if (!passPipeline.hasAnyOccurrences())
    reproOptions.attachResourceParser(parserConfig);

  OwningOpRef<ModuleOp> inputModule =
      mlir::parseSourceFile<ModuleOp>(inputFilename.getValue(), parserConfig);
  if (!inputModule)
    return Error("failed to parse input file: " + inputFilename.getValue());

  log << "[[===============================================================]]\n"
      << "[[======================== KGEN ⚜️ REDUCER =======================]]\n"
      << "[[===============================================================]]"
         "\n\n";

  log << "[kgen-reduce] " << inputFilename.getValue() << "\n";

  // Parse the pass pipeline.
  {
    std::string err;
    llvm::raw_string_ostream os(err);
    if (passPipeline.hasAnyOccurrences()) {
      if (failed(passPipeline.addToPipeline(reproPm, [&](const Twine &msg) {
            os << msg;
            return failure();
          })))
        return Error(err);
    } else if (failed(reproOptions.apply(reproPm))) {
      return Error("failed to read pass reproducer");
    }
  }

  log << "[kgen-reduce] ";
  reproPm.printAsTextualPipeline(log);
  log << "\n";

  std::optional<std::string> initDiag = attemptRepro(*inputModule);
  if (!initDiag)
    return Error("original input IR does not fail the provided pipeline");
  expectedDiag = std::move(*initDiag);

  log << "[kgen-reduce] expected diagnostic:\n" << expectedDiag << "\n\n";

  IRState curModule(std::move(inputModule));
  if (auto err = stashFile(*curModule.ir, "kgen-reduce.base"))
    return err;

  if (startAt <= 0) {
    if (auto err = reduceFunctions(curModule))
      return err;
    if (auto err = tryDCE(curModule))
      return err;
    if (auto err = stashFile(*curModule.ir, "kgen-reduce.functions"))
      return err;
  }

  if (startAt <= 1) {
    if (auto err = reduceRegions(curModule))
      return err;
    if (auto err = tryDCE(curModule))
      return err;
    if (auto err = stashFile(*curModule.ir, "kgen-reduce.regions"))
      return err;
  }

  if (startAt <= 2) {
    if (auto err = reduceOps(curModule))
      return err;
    if (auto err = tryDCE(curModule))
      return err;
    if (auto err = stashFile(*curModule.ir, "kgen-reduce.ops"))
      return err;
  }

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

bool Reducer::doesRepro(ModuleOp ir) {
  std::optional<std::string> nextDiag = attemptRepro(ir);
  if (nextDiag && nextDiag == expectedDiag) {
    log << "[kgen-reduce] same failure 🛑\n\n";
    return true;
  }

  if (!nextDiag) {
    log << "[kgen-reduce] succeeded 🟢\n\n";
  } else {
    log << "[kgen-reduce] different failure 🟠:\n" << *nextDiag << "\n\n";
  }
  return false;
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

  log << "[kgen-reduce] snapshotting IR to " << file->getFilename() << "\n";

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
    if (isStubbed(func.getBodyRegion()))
      continue;
    funcs.push_back(func);
  }
  if (!anyKgenFunc) {
    return Error("zero 'kgen.func' operations found, 'kgen-reduce' only works "
                 "on post-elaboration IR right now");
  }

  size_t funcNum = 0;
  const size_t totalNumFuncs = funcs.size();
  log << "[kgen-reduce] attempt to stub " << totalNumFuncs << " functions\n";

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

    log << "[stubbing function " << (funcNum++) << "/" << totalNumFuncs << "] "
        << func.getSymName() << "\n";

    // Stub the function with an unreachable.
    Region owner;
    stubRegion(func.getBodyRegion(), owner);

    if (doesRepro(*curState.ir))
      continue;

    // Revert the transformation.
    func.getBodyRegion().takeBody(owner);
  }

  return success();
}

ErrorOrSuccess Reducer::reduceRegions(IRState &curState) {
  std::vector<KGEN::FuncOp> funcs;
  for (auto func : curState.ir->getOps<KGEN::FuncOp>()) {
    // Ignore stubbed functions.
    if (isStubbed(func.getBodyRegion()))
      continue;
    funcs.push_back(func);
  }

  size_t funcNum = 0;
  const size_t totalNumFuncs = funcs.size();
  log << "[kgen-reduce] reducing regions in " << totalNumFuncs
      << " functions\n";

  while (!funcs.empty()) {
    KGEN::FuncOp func = funcs.back();
    funcs.pop_back();

    log << "[reducing regions " << (funcNum++) << "/" << totalNumFuncs << "] "
        << func.getSymName();

    size_t regionNum = 0;
    auto it = PreOrderRegionIterator::begin(func);
    auto end = PreOrderRegionIterator::end(func);
    for (; it != end; ++it) {
      Region &region = *it;

      // Skip the region if we don't understand its semantics.
      if (!isa<HLCF::ControlFlowNode>(region.getParentOp()))
        continue;

      if (auto err = maybeSnapshot(*curState.ir))
        return err;

      log << "[region #" << regionNum++ << "]\n";

      Region owner;
      stubRegion(region, owner);

      if (doesRepro(*curState.ir))
        continue;
      region.takeBody(owner);
    }
  }

  return success();
}

ErrorOrSuccess Reducer::reduceOps(IRState &curState) {
  std::vector<KGEN::FuncOp> funcs;
  for (auto func : curState.ir->getOps<KGEN::FuncOp>()) {
    // Ignore stubbed functions.
    if (isStubbed(func.getBodyRegion()))
      continue;
    funcs.push_back(func);
  }

  size_t funcNum = 0;
  const size_t totalNumFuncs = funcs.size();
  log << "[kgen-reduce] reducing operations in " << totalNumFuncs
      << " functions\n";

  while (!funcs.empty()) {
    KGEN::FuncOp func = funcs.back();
    funcs.pop_back();

    log << "[reducing operations " << (funcNum++) << "/" << totalNumFuncs
        << "] " << func.getSymName();

    reversePostOrderWalk(func, [&](Operation *op) {
      if (op == func || op->hasTrait<OpTrait::IsTerminator>())
        return;
      Operation *next = op->getNextNode();
      Operation *stub = nullptr;
      assert(next && "unexpected terminator");
      if (!op->use_empty()) {
        OperationState state(op->getLoc(), "kgen-reduce.stub", {},
                             op->getResultTypes());
        OpBuilder b(op);
        stub = b.create(state);
        op->replaceAllUsesWith(stub->getResults());
      }
      op->remove();

      if (doesRepro(*curState.ir)) {
        op->erase();
        return;
      }

      OpBuilder(next).insert(op);
      if (stub) {
        stub->replaceAllUsesWith(op->getResults());
        stub->erase();
      }
    });
  }

  return success();
}

ErrorOrSuccess Reducer::tryDCE(IRState &curState) {
  // Clone the module and attempt to run DCE.
  IRState nextState(curState.ir->clone());
  if (failed(dcePm.run(*nextState.ir)))
    return Error("DCE failed");

  // If the failure still reproduces after running DCE, swap the module.
  if (doesRepro(*nextState.ir)) {
    log << "[kgen-reduce] fails with DCE\n";
    curState.ir = std::move(nextState.ir);
    return success();
  }
  log << "[kgen-reduce] does not fail with DCE\n";

  return success();
}

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerAllKGENDialects(registry);
  MLIRContext mlirCtx{MLIRContext::Threading::DISABLED};
  mlirCtx.allowUnregisteredDialects();
  mlirCtx.appendDialectRegistry(registry);

  Reducer reducer(&mlirCtx);

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Create our context.
  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "kgen-reduce", Init::Options().withRuntimeOptions(
                         LLCL::RuntimeOptions().withLeakCheckedAllocator()));
  if (ctxOr.isError()) {
    llvm::errs() << "failed to create context: " << ctxOr.getError() << "\n";
    return 1;
  }
  registerContext(mlirCtx, *ctxOr);

  KGEN::registerDefaultKGENPasses();

  if (auto err = reducer.run()) {
    llvm::errs() << "ERROR: " << err.getError() << "\n";
    return -1;
  }
}
