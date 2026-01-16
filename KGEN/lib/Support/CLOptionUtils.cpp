//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Support/CLOptionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"

using namespace llvm;
using namespace llvm::codegen;

static llvm::ManagedStatic<llvm::codegen::RegisterCodeGenFlags> codegenFlagsOpt;

void M::registerCommandFlags() {
  // Register llvm::codegen::RegisterCodegenFlags flags.
  // E.g. we want to use denormal-fp-math-f32
  *codegenFlagsOpt;

  // Remove duplicated flags that are conflicting with other mojo and kgen
  // options.
  llvm::DenseMap<llvm::StringRef, llvm::cl::Option *> &options =
      llvm::cl::getRegisteredOptions();
  options["march"]->removeArgument();
  options["mcpu"]->removeArgument();
  options["mattr"]->removeArgument();
  options["large-data-threshold"]->removeArgument();
}
