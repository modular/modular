//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "MetalVerifier.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"

using namespace llvm;
using namespace M::KGEN;

static llvm::Type *getUnsupportedType(llvm::Type *ty) {
  if (ty->isFloatingPointTy())
    return ty->getPrimitiveSizeInBits() > 32 ? ty : nullptr;
  if (auto *vty = dyn_cast<llvm::VectorType>(ty))
    return getUnsupportedType(vty->getElementType());
  if (auto *aty = dyn_cast<llvm::ArrayType>(ty))
    return getUnsupportedType(aty->getElementType());
  if (auto *sty = dyn_cast<llvm::StructType>(ty)) {
    for (llvm::Type *elemTy : sty->elements())
      if (llvm::Type *badTy = getUnsupportedType(elemTy))
        return badTy;
  }
  return nullptr;
}

static LogicalResult verifyFunction(const Function &func) {
  if (llvm::Type *badTy = getUnsupportedType(func.getReturnType())) {
    llvm::errs() << "function's return type '" << *badTy
                 << "' is not supported\n";
    return failure();
  }

  for (const Argument &arg : func.args()) {
    if (llvm::Type *badTy = getUnsupportedType(arg.getType())) {
      llvm::errs() << "argument '" + arg.getName() + "' has unsupported type '"
                   << *badTy << "'\n";
      return failure();
    }
  }

  for (const BasicBlock &bb : func) {
    for (const Instruction &inst : bb) {
      if (llvm::Type *badTy = getUnsupportedType(inst.getType())) {
        llvm::errs() << "instruction '" << inst
                     << "\' returns unsupported type '" << *badTy << "'\n";
        return failure();
      }
      for (const Use &op : inst.operands()) {
        if (llvm::Type *badTy = getUnsupportedType(op->getType())) {
          llvm::errs() << "instruction '" << inst
                       << "\' uses unsupported type '" << *badTy << "'\n";
          return failure();
        }
      }
    }
  }
  return success();
}

PreservedAnalyses MetalVerifierPass::run(Module &M, ModuleAnalysisManager &AM) {
  bool verificationFailed = false;
  for (Function &func : M) {
    if (failed(verifyFunction(func))) {
      llvm::errs() << "Function '" << func.getName()
                   << "' has Metal-unsupported instructions\n";
      verificationFailed = true;
    }
  }
  if (verificationFailed) {
    report_fatal_error("Failed to verify LLVM IR for Metal",
                       /*gen_crash_diag=*/false);
  }

  return PreservedAnalyses::all();
}
