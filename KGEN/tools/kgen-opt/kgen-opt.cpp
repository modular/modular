//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// The kgen-opt driver implementation.
//
//===----------------------------------------------------------------------===//

#include "Cache/CacheDialect/CacheDialect.h"
#include "KGEN/InitAllDialects.h"
#include "KGEN/KGENPasses.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/HLCFDialect/HLCFDialect.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

using namespace M;

int main(int argc, char **argv) {
  DialectRegistry registry;

  // Register MLIR stuff
  registerAllKGENDialects(registry);
  registry.insert<DebugInfo::DebugInfoDialect, Cache::CacheDialect,
                  HLCF::HLCFDialect, mlir::index::IndexDialect,
                  mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect>();
  // The elaborator requires LLVM lowering to run the generated functions.
  mlir::registerLLVMDialectTranslation(registry);

  // Register the standard passes we want.
  mlir::registerCSEPass();
  mlir::registerSCCPPass();
  mlir::registerInlinerPass();
  mlir::registerCanonicalizerPass();
  mlir::registerReconcileUnrealizedCasts();
  mlir::registerConvertIndexToLLVMPass();

  LLCL::Runtime runtime(
      LLCL::createLeakCheckAllocator(LLCL::createMallocAllocator()),
      LLCL::createSingleThreadWorkQueue());
  // Register the EmitLLVM pass with the runtime instance.
  KGEN::registerEmitLLVMPass(runtime);

  KGEN::registerPasses();
  KGEN::registerLowerToLLVMPipeline();

  return failed(mlir::MlirOptMain(argc, argv, "kgen optimizer driver", registry,
                                  /*preloadDialectsInContext=*/true));
}
