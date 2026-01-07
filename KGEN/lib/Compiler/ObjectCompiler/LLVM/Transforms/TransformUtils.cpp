//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// LLVM IR Transform Utils
//
//===----------------------------------------------------------------------===//

#include "TransformUtils.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace M::KGEN;
using namespace llvm;

/// Update all functions and their callsites that were previously identified.
void CallGraphUpdater::update() {
  assert(!functionsToUpdate.empty() &&
         "No functions to update. Run analyze() first ?.");

  DenseMap<llvm::Function *, llvm::Function *> funcMap;

  for (llvm::Function *f : functionsToUpdate) {
    llvm::Function *newF = updateFunction(*f);
    funcMap[f] = newF;
  }

  // Replace call sites with calls that include thread parameters
  for (llvm::Function *func : functionsToUpdate) {
    llvm::Function *newFunc = funcMap[func];
    SmallVector<llvm::CallInst *> calls;
    for (llvm::BasicBlock &bb : *newFunc) {
      for (llvm::Instruction &inst : bb) {
        if (llvm::CallInst *call = llvm::dyn_cast<llvm::CallInst>(&inst)) {
          llvm::Function *callee = call->getCalledFunction();
          if (!functionsToUpdate.contains(callee))
            continue;
          calls.push_back(call);
        }
      }
    }

    llvm::LLVMContext &ctx = module.getContext();
    llvm::IRBuilder<> builder(ctx);
    for (llvm::CallInst *call : calls) {
      llvm::Value *newCall =
          updateCall(*call, *funcMap[call->getCalledFunction()], *newFunc);

      if (newCall != call) {
        builder.SetInsertPoint(call);
        builder.Insert(newCall);
        call->replaceAllUsesWith(newCall);
        call->eraseFromParent();
      }
    }
  }
  // Replace all uses of old functions with new functions and restore the
  // original name
  for (auto [func, newFunc] : funcMap) {
    if (newFunc == func)
      continue;
    func->replaceAllUsesWith(newFunc);
    newFunc->takeName(func);
    func->eraseFromParent();
  }
  // Clear all cached analyses to force their recomputation later.
  mam.clear();
}
