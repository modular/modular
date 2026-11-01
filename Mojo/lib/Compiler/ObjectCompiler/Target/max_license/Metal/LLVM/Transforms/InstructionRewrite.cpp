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
//
// LLVM IR Instruction Rewrite Pass performs LLVM IR to LLVM IR transformation
// by replacing instructions or intrinsics that are not supported by the target
// to instructions or target-specific intrinsics.
//
//===----------------------------------------------------------------------===//

#include "InstructionRewrite.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetFolder.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/Local.h"

#define DEBUG_TYPE "instruction-rewrite"
#include "llvm/Transforms/Utils/InstructionWorklist.h"

#define KGEN_DEBUG_TYPE "instruction-rewrite"

namespace {
// IRBuilder that feeds newly created instructions back into the worklist (so
// they get visited too) and propagates !annotation metadata from the
// instruction currently being rewritten onto every instruction it creates.
// This mirrors LLVM's own ReplacementIRBuilder in AtomicExpandPass.
struct AnnotatingBuilder
    : llvm::IRBuilder<llvm::TargetFolder, llvm::IRBuilderCallbackInserter> {
  AnnotatingBuilder(llvm::Function &func, llvm::InstructionWorklist &worklist)
      : IRBuilder(func.getContext(), llvm::TargetFolder(func.getDataLayout()),
                  llvm::IRBuilderCallbackInserter(
                      [this, &worklist](llvm::Instruction *inst) {
                        worklist.add(inst);
                        // TODO: Revisit and copy other metadata if needed.
                        if (annotationSource)
                          inst->copyMetadata(*annotationSource,
                                             llvm::LLVMContext::MD_annotation);
                      })) {}

  // Set the instruction currently being rewritten. Instructions created
  // afterwards inherit its !annotation metadata.
  void SetAnnotationSource(llvm::Instruction *inst) { annotationSource = inst; }

private:
  llvm::Instruction *annotationSource = nullptr;
};

class InstructionRewrite final
    : public llvm::InstVisitor<InstructionRewrite, llvm::Instruction *> {
public:
  explicit InstructionRewrite(llvm::Function &func,
                              llvm::InstructionWorklist &worklist,
                              AnnotatingBuilder &builder)
      : func(func), worklist(worklist), builder(builder),
        triple(func.getParent()->getTargetTriple()) {}
  ~InstructionRewrite() {}

  // Main entry point of the instruction rewrite pass.
  bool run();

  // Rewrite any intrinsic to a target specific intrinsic or a set of
  // instructions.
  llvm::Instruction *visitIntrinsicInst(llvm::IntrinsicInst &intrin);

  // Do nothing for all other instructions.
  llvm::Instruction *visitInstruction(llvm::Instruction &inst) {
    return nullptr;
  }

  // Prepare the worklist by adding all instructions to the worklist.
  void prepareWorklist(llvm::Function &func);

  // Replace calls to undefined libc functions (like free) with
  // unreachable, enabling LLVM to DCE the dead branches that lead to
  // them. Metal GPU has no libc allocator.
  bool stubUndefinedLibcCalls();

private:
  llvm::Function &func;
  llvm::InstructionWorklist &worklist;
  AnnotatingBuilder &builder;
  llvm::Triple triple;

  // Return true if target is Metal.
  bool isMetalTriple() const { return triple.getArchName() == "air64"; }

  // Replace all uses of an instruction with a value.
  // NOTE: This function is needed, because after change, result value could be
  // a constant, therefore to this function always returns original
  // instruction.
  llvm::Instruction *replaceInstUsesWith(llvm::Instruction &inst,
                                         llvm::Value *value);

  // Remove instruction without any uses from a worklist and function. This
  // function is especially useful to remove stores.
  llvm::Instruction *eraseInstruction(llvm::Instruction &inst);

  // Helper function to rewrite reduce intrinsics
  template <typename BinOp>
  llvm::Value *rewriteReduceIntrinsic(llvm::IntrinsicInst &intrin, BinOp binOp);
};

/// Main entry point of the Instruction Selection, which is responsible for
/// visiting all instructions in the worklist and replace them with
/// instructions/// that are supported by the target. This logic is similar to
/// the InstCombine pass, except that some of it is not needed (yet).
bool InstructionRewrite::run() {
  // For now this pass only works for Metal
  if (!isMetalTriple())
    return false;

  // Replace calls to undefined libc functions before the main worklist loop.
  bool changed = stubUndefinedLibcCalls();

  prepareWorklist(func);
  while (!worklist.isEmpty()) {
    // Walk deferred instructions in reverse order, and push them to the
    // worklist, which means they'll end up popped from the worklist in-order.
    while (llvm::Instruction *inst = worklist.popDeferred()) {
      worklist.push(inst);
    }

    llvm::Instruction *inst = worklist.removeOne();
    builder.SetInsertPoint(inst);
    // Instructions created while visiting `inst` inherit its !annotation
    // metadata (see AnnotatingBuilder).
    builder.SetAnnotationSource(inst);

    if (llvm::Instruction *newInst = visit(*inst)) {
      if (newInst == inst)
        if (isInstructionTriviallyDead(inst, /*TLI=*/nullptr))
          inst->eraseFromParent();
      changed = true;
    }
  }
  worklist.zap(); // make sure everything has been processed and cleaned up.
  return changed;
}

llvm::Instruction *
InstructionRewrite::visitIntrinsicInst(llvm::IntrinsicInst &intrin) {
  switch (intrin.getIntrinsicID()) {
  default:
    return nullptr;
  case llvm::Intrinsic::vector_reduce_add:
    return replaceInstUsesWith(
        intrin,
        rewriteReduceIntrinsic(intrin, [&](llvm::Value *a, llvm::Value *b) {
          return builder.CreateAdd(a, b);
        }));
  case llvm::Intrinsic::vector_reduce_mul:
    return replaceInstUsesWith(
        intrin,
        rewriteReduceIntrinsic(intrin, [&](llvm::Value *a, llvm::Value *b) {
          return builder.CreateMul(a, b);
        }));
  case llvm::Intrinsic::vector_reduce_and:
    return replaceInstUsesWith(
        intrin,
        rewriteReduceIntrinsic(intrin, [&](llvm::Value *a, llvm::Value *b) {
          return builder.CreateAnd(a, b);
        }));
  case llvm::Intrinsic::vector_reduce_or:
    return replaceInstUsesWith(
        intrin,
        rewriteReduceIntrinsic(intrin, [&](llvm::Value *a, llvm::Value *b) {
          return builder.CreateOr(a, b);
        }));
  case llvm::Intrinsic::vector_reduce_xor:
    return replaceInstUsesWith(
        intrin,
        rewriteReduceIntrinsic(intrin, [&](llvm::Value *a, llvm::Value *b) {
          return builder.CreateXor(a, b);
        }));
  case llvm::Intrinsic::vector_reduce_umax:
    return replaceInstUsesWith(
        intrin,
        rewriteReduceIntrinsic(intrin, [&](llvm::Value *a, llvm::Value *b) {
          return builder.CreateIntrinsic(llvm::Intrinsic::umax, a->getType(),
                                         {a, b});
        }));
  case llvm::Intrinsic::vector_reduce_umin:
    return replaceInstUsesWith(
        intrin,
        rewriteReduceIntrinsic(intrin, [&](llvm::Value *a, llvm::Value *b) {
          return builder.CreateIntrinsic(llvm::Intrinsic::umin, a->getType(),
                                         {a, b});
        }));
  case llvm::Intrinsic::vector_reduce_smax:
    return replaceInstUsesWith(
        intrin,
        rewriteReduceIntrinsic(intrin, [&](llvm::Value *a, llvm::Value *b) {
          return builder.CreateIntrinsic(llvm::Intrinsic::smax, a->getType(),
                                         {a, b});
        }));
  case llvm::Intrinsic::vector_reduce_smin:
    return replaceInstUsesWith(
        intrin,
        rewriteReduceIntrinsic(intrin, [&](llvm::Value *a, llvm::Value *b) {
          return builder.CreateIntrinsic(llvm::Intrinsic::smin, a->getType(),
                                         {a, b});
        }));
  case llvm::Intrinsic::vector_reduce_fmax:
    return replaceInstUsesWith(
        intrin,
        rewriteReduceIntrinsic(intrin, [&](llvm::Value *a, llvm::Value *b) {
          return builder.CreateMaxNum(a, b);
        }));
  case llvm::Intrinsic::vector_reduce_fmin:
    return replaceInstUsesWith(
        intrin,
        rewriteReduceIntrinsic(intrin, [&](llvm::Value *a, llvm::Value *b) {
          return builder.CreateMinNum(a, b);
        }));
  case llvm::Intrinsic::vector_reduce_fadd:
    return replaceInstUsesWith(
        intrin,
        rewriteReduceIntrinsic(intrin, [&](llvm::Value *a, llvm::Value *b) {
          return builder.CreateFAdd(a, b);
        }));
  case llvm::Intrinsic::vector_reduce_fmul: {
    return replaceInstUsesWith(
        intrin,
        rewriteReduceIntrinsic(intrin, [&](llvm::Value *a, llvm::Value *b) {
          return builder.CreateFMul(a, b);
        }));
  }
  case llvm::Intrinsic::stepvector: {
    auto *vty = cast<llvm::FixedVectorType>(intrin.getType());
    unsigned numElements = vty->getNumElements();
    unsigned intWidth = vty->getElementType()->getIntegerBitWidth();
    llvm::SmallVector<llvm::Constant *, 8> args;
    for (unsigned i = 0; i < numElements; ++i)
      args.push_back(builder.getIntN(intWidth, i));
    return replaceInstUsesWith(intrin, llvm::ConstantVector::get(args));
  }
  case llvm::Intrinsic::vector_interleave2: {
    llvm::Value *op0 = intrin.getOperand(0);
    llvm::Value *op1 = intrin.getOperand(1);
    llvm::Value *result = llvm::PoisonValue::get(intrin.getType());
    unsigned numElements =
        cast<llvm::FixedVectorType>(op0->getType())->getNumElements();
    for (unsigned i = 0; i < numElements; ++i) {
      llvm::Value *op0I = builder.CreateExtractElement(op0, i);
      llvm::Value *op1I = builder.CreateExtractElement(op1, i);
      llvm::Value *interleavedI =
          builder.CreateInsertElement(result, op0I, 2 * i);
      result = builder.CreateInsertElement(interleavedI, op1I, 2 * i + 1);
    }
    return replaceInstUsesWith(intrin, result);
  }
  case llvm::Intrinsic::vector_deinterleave2: {
    auto *stTy = cast<llvm::StructType>(intrin.getType());
    llvm::Value *op = intrin.getOperand(0);
    auto *vt = cast<llvm::FixedVectorType>(stTy->getElementType(0));
    unsigned numElements = 2 * vt->getNumElements();

    llvm::Value *result0 = llvm::PoisonValue::get(vt);
    llvm::Value *result1 = llvm::PoisonValue::get(vt);
    for (unsigned i = 0; i < numElements; i += 2) {
      llvm::Value *opI0 = builder.CreateExtractElement(op, i);
      llvm::Value *opI1 = builder.CreateExtractElement(op, i + 1);
      result0 = builder.CreateInsertElement(result0, opI0, i / 2);
      result1 = builder.CreateInsertElement(result1, opI1, i / 2);
    }
    llvm::Value *result = llvm::PoisonValue::get(stTy);
    result = builder.CreateInsertValue(result, result0, 0);
    result = builder.CreateInsertValue(result, result1, 1);
    return replaceInstUsesWith(intrin, result);
  }
  case llvm::Intrinsic::is_fpclass: {
    llvm::Value *val = intrin.getOperand(0);
    auto *maskCI = llvm::dyn_cast<llvm::ConstantInt>(intrin.getOperand(1));
    if (!maskCI)
      return nullptr;
    uint64_t mask = maskCI->getZExtValue();
    llvm::Value *result = nullptr;

    // NaN classes: qnan(1) | snan(2) = 0x3
    if (mask == 0x3 || mask == 0x1 || mask == 0x2) {
      result = builder.CreateFCmpUNO(val, val);
    }
    // Inf classes: posinf(0x200) | neginf(0x4) = 0x204
    else if (mask == 0x204 || mask == 0x200 || mask == 0x4) {
      llvm::Value *absVal =
          builder.CreateIntrinsic(llvm::Intrinsic::fabs, val->getType(), {val});
      llvm::Constant *posInf = llvm::ConstantFP::getInfinity(val->getType());
      result = builder.CreateFCmpOEQ(absVal, posInf);
    }
    // Finite classes: 0x1F8 (normal | subnormal | zero, positive and negative)
    else if (mask == 0x1F8) {
      llvm::Value *isOrd = builder.CreateFCmpORD(val, val);
      llvm::Value *absVal =
          builder.CreateIntrinsic(llvm::Intrinsic::fabs, val->getType(), {val});
      llvm::Constant *posInf = llvm::ConstantFP::getInfinity(val->getType());
      llvm::Value *notInf = builder.CreateFCmpONE(absVal, posInf);
      result = builder.CreateAnd(isOrd, notInf);
    }
    // Zero classes: poszero(0x40) | negzero(0x20) = 0x60
    else if (mask == 0x60) {
      llvm::Constant *zero = llvm::ConstantFP::get(val->getType(), 0.0);
      result = builder.CreateFCmpOEQ(val, zero);
    }
    // For any other mask, do a general decomposition
    else {
      // Fallback: treat as "not NaN" (ordered) for safety
      result = builder.CreateFCmpORD(val, val);
    }

    return replaceInstUsesWith(intrin, result);
  }
  case llvm::Intrinsic::masked_scatter:
    llvm::Value *values = intrin.getOperand(0);
    llvm::Value *ptrs = intrin.getOperand(1);
    llvm::Value *masks = intrin.getOperand(2);
    auto *vt = cast<llvm::FixedVectorType>(values->getType());
    unsigned numElements = vt->getNumElements();
    llvm::BasicBlock *origBB = intrin.getParent();

    // Scalarize llvm.masked.scatter:
    //
    //  %mask0 = extractvalue %masks, 0
    //  br %mask0, label %store.0, label %cont.0
    //  store.0:
    //    %ptr0 = extractvalue %ptrs, 0
    //    %value0 = extractvalue %values, 0
    //    store %value0, %ptr0
    //    br label %cont.0
    //  cont.0:
    //
    //  %mask1 = extractvalue %masks, 1
    //  br %mask1, label %store.1, label %cont.1
    //  store.1:
    //    %ptr1 = extractvalue %ptrs, 1
    //    %value1 = extractvalue %values, 1
    //    store %value1, %ptr1
    //    br label %cont.1
    //
    //  cont.1:
    //  ...
    for (unsigned i = 0; i < numElements; ++i) {
      llvm::Value *mask = builder.CreateExtractElement(masks, i);
      // First split current BasicBlock into two and move all instructions after
      // `intrin` into newBB.
      llvm::BasicBlock *contBB =
          llvm::BasicBlock::Create(func.getContext(), "cont." + llvm::Twine(i),
                                   &func, origBB->getNextNode());
      contBB->splice(contBB->end(), origBB, intrin.getIterator(),
                     origBB->end());

      // Create new block that will have a store
      llvm::BasicBlock *storeBB =
          llvm::BasicBlock::Create(func.getContext(), "store." + llvm::Twine(i),
                                   &func, contBB->getNextNode());

      builder.SetInsertPoint(origBB->end());
      (void)builder.CreateCondBr(mask, storeBB, contBB);

      builder.SetInsertPoint(storeBB);
      llvm::Value *value = builder.CreateExtractElement(values, i);
      llvm::Value *ptr = builder.CreateExtractElement(ptrs, i);

      (void)builder.CreateStore(value, ptr);

      (void)builder.CreateBr(contBB);
      // Repeat sequence above of splitting BB, moving and adding instructions
      origBB = contBB;
      builder.SetInsertPoint(origBB->begin());
    }
    return eraseInstruction(intrin);
  }
}

/// Helper function to rewrite reduce intrinsics. To be compliant with
/// current Mojo FP model (which is vaguely defined as 'precise'), this
/// function does in-order reduction.
/// TODO: Support fast math by doing tree-reduction.
template <typename BinOp>
llvm::Value *
InstructionRewrite::rewriteReduceIntrinsic(llvm::IntrinsicInst &intrin,
                                           BinOp binOp) {
  bool hasStartValue = intrin.arg_size() == 2;
  llvm::Value *result = hasStartValue ? intrin.getOperand(0) : nullptr;
  llvm::Value *vec =
      hasStartValue ? intrin.getOperand(1) : intrin.getOperand(0);

  unsigned numElements =
      cast<llvm::FixedVectorType>(vec->getType())->getNumElements();

  for (unsigned i = 0; i < numElements; ++i) {
    llvm::Value *value = builder.CreateExtractElement(vec, i);
    result = result ? binOp(result, value) : value;
  }
  return result;
}

bool InstructionRewrite::stubUndefinedLibcCalls() {
  // Metal GPU has no libc allocator. Remove calls to free() and malloc()
  // from this function and replace with no-ops / null returns.
  if (!isMetalTriple())
    return false;

  llvm::Module &module = *func.getParent();

  bool changed = false;

  llvm::SmallVector<llvm::CallInst *, 4> callsToRemove;
  for (llvm::BasicBlock &bb : func) {
    for (llvm::Instruction &inst : bb) {
      if (auto *call = llvm::dyn_cast<llvm::CallInst>(&inst)) {
        llvm::Function *callee = call->getCalledFunction();
        if (!callee)
          continue;
        llvm::StringRef name = callee->getName();
        if (name == "free") {
          callsToRemove.push_back(call);
        } else if (name == "malloc") {
          callsToRemove.push_back(call);
        }
      }
    }
  }

  for (llvm::CallInst *call : callsToRemove) {
    llvm::Function *callee = call->getCalledFunction();
    if (callee->getName() == "free") {
      // free(ptr) → no-op: just erase the call.
      call->eraseFromParent();
    } else if (callee->getName() == "malloc") {
      // malloc(size) → return null. Callers should check for null or
      // never actually reach this path on GPU.
      call->replaceAllUsesWith(llvm::ConstantPointerNull::get(
          llvm::PointerType::get(module.getContext(), 0)));
      call->eraseFromParent();
    }
    changed = true;
  }

  // Remove the declarations if they have no remaining uses.
  for (const char *name : {"free", "malloc"}) {
    if (llvm::Function *fn = module.getFunction(name)) {
      if (fn->use_empty() && fn->isDeclaration())
        fn->eraseFromParent();
    }
  }

  return changed;
}

/// Prepare the worklist by adding all instructions to the worklist.
/// Unlike InstCombiner, this does not try to optimize list by ignoring
/// unreachable instructions.
void InstructionRewrite::prepareWorklist(llvm::Function &fun) {
  llvm::ReversePostOrderTraversal<llvm::BasicBlock *> rpot(&func.front());
  for (llvm::BasicBlock *bb : rpot)
    for (llvm::Instruction &inst : *bb)
      worklist.add(&inst);
}

/// Use-aware replacement of instruction with a new value.
llvm::Instruction *
InstructionRewrite::replaceInstUsesWith(llvm::Instruction &inst,
                                        llvm::Value *value) {
  if (inst.use_empty())
    return nullptr;

  // Add all modified instrs to worklist.
  worklist.pushUsersToWorkList(inst);
  // If value is a new unnamed instruction, take the name from the old one.
  if (value->use_empty() && isa<llvm::Instruction>(value) &&
      !value->hasName() && inst.hasName())
    value->takeName(&inst);

  inst.replaceAllUsesWith(value);
  return &inst;
}

/// Remove instruction without any uses from a worklist and function. This
/// function is especially useful to remove stores.
llvm::Instruction *
InstructionRewrite::eraseInstruction(llvm::Instruction &inst) {
  assert(inst.use_empty() && "Cannot erase instruction that is used!");
  salvageDebugInfo(inst);
  worklist.remove(&inst);
  inst.eraseFromParent();
  return nullptr;
}

} // namespace

namespace M::KGEN {

llvm::PreservedAnalyses
InstructionRewritePass::run(llvm::Function &func,
                            llvm::FunctionAnalysisManager &am) {
  llvm::InstructionWorklist worklist;
  AnnotatingBuilder builder(func, worklist);

  InstructionRewrite rewrite(func, worklist, builder);

  if (rewrite.run())
    return llvm::PreservedAnalyses::none();

  return llvm::PreservedAnalyses::all();
}
} // namespace M::KGEN
