//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#include "SetFunctionAttributes.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/TypedPointerType.h"

using namespace llvm;
using namespace llvm::codegen;
using namespace M::KGEN;

//===----------------------------------------------------------------------===//
// SetFunctionAttributes
//===----------------------------------------------------------------------===//

PreservedAnalyses SetFunctionAttributes::run(Module &module,
                                             ModuleAnalysisManager &MAM) {

  llvm::DenseMap<llvm::StringRef, llvm::cl::Option *> &options =
      llvm::cl::getRegisteredOptions();

  runImpl(module, options);
  return PreservedAnalyses::none();
}

static std::optional<DenormalMode::DenormalModeKind> getDenormalKind(
    const llvm::DenseMap<llvm::StringRef, llvm::cl::Option *> &options) {

  auto denormIter = options.find("denormal-fp-math-f32");
  if (denormIter == options.end())
    return std::nullopt;

  auto *denormalIntVal = static_cast<llvm::cl::opt<int> *>(denormIter->second);
  if (!denormalIntVal || denormalIntVal->getNumOccurrences() == 0)
    return std::nullopt;

  return (DenormalMode::DenormalModeKind)denormalIntVal->getValue();
}

void SetFunctionAttributes::runImpl(
    llvm::Module &module,
    const llvm::DenseMap<llvm::StringRef, llvm::cl::Option *> &options) {
  // Set function denormal-fp-math-f32 attributes based on cl option value.
  // Clang does similar thing for `-fdenormal-fp-math-f32`
  // https://github.com/llvm/llvm-project/blob/cc271437553452ede002d871d32abc02084341a8/clang/lib/CodeGen/CGCall.cpp#L1940-L1948
  std::optional<DenormalMode::DenormalModeKind> denormalKind =
      getDenormalKind(options);

  if (denormalKind.has_value()) {
    for (Function &func : module) {
      if (!func.hasFnAttribute("denormal-fp-math-f32")) {
        func.addFnAttr("denormal-fp-math-f32",
                       DenormalMode(*denormalKind, *denormalKind).str());
      }
    }
  }
}
