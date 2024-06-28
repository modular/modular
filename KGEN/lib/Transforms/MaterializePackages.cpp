//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CustomDialect/CustomDialect.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITDialect.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "Support/Compiler/BytecodeReaderWriter.h"
#include "Support/Compiler/MLIRDenseAttr.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_MATERIALIZEPACKAGES
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// PackageState
//===----------------------------------------------------------------------===//

namespace {
/// This class represents the state of a package that is being materialized. It
/// handles the logic for materializing external generators, and inflating
/// operations into the main module.
class PackageState {
public:
  /// Create a new package state instance.
  static std::unique_ptr<PackageState>
  create(SymbolTable &symtab, StringAttr moduleRef, mlir::ParserConfig &config,
         const PackageGenLibraryFn &packageGenLibraryFn, Operation *ctx,
         const std::shared_ptr<llvm::SourceMgr> &bufferOwnerRef);

  /// Materialize a new extern generator. This enqueues operations to inflate,
  /// which should be processed as part of `processOperationsToInflate`.
  LogicalResult materializeExternGenerator(ExternGeneratorOp func,
                                           SymbolTable &symtab);

  /// Process the current set of operations to inflate.
  LogicalResult processOperationsToInflate(
      SymbolTable &symtab, function_ref<LogicalResult(Operation *)> processFn);

  /// Finalize the state.
  LogicalResult finalize() {
    // Finalize the bytecode reader, dropping anything that wasn't materialized.
    return reader.finalize([&](Operation *) { return false; });
  }

private:
  PackageState(BufferRef buffer, const mlir::ParserConfig &config,
               const std::shared_ptr<llvm::SourceMgr> &bufferOwnerRef)
      : bytecodeBuffer(std::move(buffer)),
        reader(bytecodeBuffer->getMemBufferRef(), config,
               /*lazyLoad=*/true, bufferOwnerRef) {}

  /// A set of operations that need to be inflated.
  SmallVector<Operation *> operationsToInflate;

  /// The bytecode parser state.
  BufferRef bytecodeBuffer;
  Block block;
  mlir::BytecodeReader reader;
  std::unique_ptr<SymbolTable> bytecodeSymtab;
};
} // namespace

std::unique_ptr<PackageState> PackageState::create(
    SymbolTable &symtab, StringAttr moduleRef, mlir::ParserConfig &config,
    const PackageGenLibraryFn &packageGenLibraryFn, Operation *ctx,
    const std::shared_ptr<llvm::SourceMgr> &bufferOwnerRef) {
  auto packageLink = symtab.lookup<PackageLinkOp>(moduleRef);
  if (!packageLink) {
    ctx->emitOpError("unable to find the link for preCompiledModuleRef");
    return nullptr;
  }

  ErrorOr<BufferRef> bytecodeOr = packageGenLibraryFn(packageLink);
  if (bytecodeOr.isError()) {
    mlir::emitError(packageLink.getLoc(),
                    "failed to load precompiled module and its dependencies "
                    "for this package");
    return nullptr;
  }

  // Get the data for the imported module body.
  std::unique_ptr<PackageState> state(new PackageState(
      std::move(*bytecodeOr), config, std::make_shared<llvm::SourceMgr>()));

  // Parse in the top-level module.
  if (failed(state->reader.readTopLevel(&state->block)) ||
      failed(state->reader.materialize(&state->block.front()))) {
    (void)state->reader.finalize();
    return nullptr;
  }
  state->bytecodeSymtab = std::make_unique<SymbolTable>(&state->block.front());
  return state;
}

LogicalResult PackageState::materializeExternGenerator(ExternGeneratorOp func,
                                                       SymbolTable &symtab) {
  StringAttr name = func.getSymNameAttr();
  auto result = bytecodeSymtab->lookup<GeneratorOp>(name);
  if (!result) {
    return mlir::emitError(func.getLoc(), "unable to find ")
           << name.getValue() << " in imported package bytecode";
  }
  operationsToInflate.push_back(result);
  result->moveAfter(func);

  // Replace the original function with the parsed KGEN Func.
  symtab.erase(func);
  symtab.insert(result);
  return success();
}

LogicalResult PackageState::processOperationsToInflate(
    SymbolTable &symtab, function_ref<LogicalResult(Operation *)> processFn) {
  bool processFailed = false;
  auto insertFn = [&](Operation *op, Operation *after) {
    op->moveAfter(after);
    symtab.insert(op);

    processFailed |= failed(processFn(op));
  };
  auto existsFn = [&](StringAttr name) -> bool { return symtab.lookup(name); };

  // Process and clear the current set of operations.
  SmallVector<Operation *> toInflate = std::move(operationsToInflate);
  for (Operation *op : toInflate)
    if (failed(loadSymbolsFromBytecode(op, reader, existsFn, insertFn,
                                       *bytecodeSymtab)))
      return failure();
  return failure(processFailed);
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
class MaterializePackagesPass
    : public impl::MaterializePackagesBase<MaterializePackagesPass> {
public:
  explicit MaterializePackagesPass(
      PackageGenLibraryFn packageGenLibraryFn = nullptr)
      : MaterializePackagesBase(),
        packageGenLibraryFn(std::move(packageGenLibraryFn)) {}

  LogicalResult initialize(MLIRContext *context) override {
    if (!packageGenLibraryFn)
      packageGenLibraryFn = [](PackageLinkOp packageLink) {
        return Error("package link handler is null");
      };
    return success();
  }

  void runOnOperation() override;

private:
  PackageGenLibraryFn packageGenLibraryFn;
};
} // namespace

void MaterializePackagesPass::runOnOperation() {
  auto theModule = cast<ModuleOp>(getOperation());
  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();
  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  mlir::ParserConfig parserConfig(&getContext(), /*verifyAfterParse=*/false);
  llvm::MapVector<StringAttr, std::unique_ptr<PackageState>> states;

  /// Materialize the given extern generator.
  auto materializeGenerator = [&](ExternGeneratorOp func) {
    StringAttr refAttr = func.getPreCompiledModuleRefAttr().getAttr();
    std::unique_ptr<PackageState> &state = states[refAttr];
    if (!state) {
      state = PackageState::create(symtab, refAttr, parserConfig,
                                   packageGenLibraryFn, func, sourceMgr);
      if (!state) {
        states.erase(refAttr);
        return failure();
      }
    }
    return state->materializeExternGenerator(func, symtab);
  };
  auto cleanupState = [&] {
    for (auto &state : states)
      if (failed(state.second->finalize()))
        signalPassFailure();
  };

  // Materialize the initial set of extern generators.
  SmallVector<ExternGeneratorOp> generatorsToProcess(
      theModule.getOps<ExternGeneratorOp>());

  // Inflate all of the extern generators and their dependencies in a fixed
  // point until we've processed all of the imported operations.
  auto processExternGenerators = [&](Operation *op) {
    if (auto func = dyn_cast<ExternGeneratorOp>(op))
      generatorsToProcess.emplace_back(func);
    return mlir::success();
  };
  while (!generatorsToProcess.empty()) {
    // Materialize the generators.
    for (ExternGeneratorOp func : generatorsToProcess)
      if (failed(materializeGenerator(func))) {
        signalPassFailure();
        return cleanupState();
      }
    generatorsToProcess.clear();

    // Process any new operations to inflate.
    for (auto &state : states) {
      if (failed(state.second->processOperationsToInflate(
              symtab, processExternGenerators)))
        return cleanupState();
    }
  }

  // Finalize all of the package states.
  cleanupState();
}

std::unique_ptr<mlir::Pass>
M::KGEN::createMaterializePackages(PackageGenLibraryFn packageGenLibraryFn) {
  return std::make_unique<MaterializePackagesPass>(
      std::move(packageGenLibraryFn));
}
