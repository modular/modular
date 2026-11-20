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

#include "MetalModuleRewritePass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Alignment.h"

namespace M::KGEN {

namespace {
class MetalModuleRewritePassImpl {
public:
  explicit MetalModuleRewritePassImpl(llvm::Module &module)
      : module(module), triple(module.getTargetTriple()) {}

  bool isMetalTriple() const { return triple.getArchName() == "air64"; }

  bool run() {
    bool changed = false;
    changed |= transformGlobalConstantsOnMetal();
    changed |= transformMetalPrintBufferSentinels();
    changed |= removeNoInlineAttribute();
    return changed;
  }

private:
  using AllocaMap = llvm::DenseMap<llvm::GlobalVariable *, llvm::AllocaInst *>;

  /// Returns a `static_string_*` global that can be copied to a stack alloca,
  /// or nullptr. Accepts both ConstantDataArray and ConstantAggregateZero
  /// (all-zero / empty strings canonicalize to the latter).
  static llvm::GlobalVariable *asInlinableStringGlobal(llvm::Value *value) {
    auto *gv = llvm::dyn_cast<llvm::GlobalVariable>(value);
    if (!gv || !gv->getName().starts_with("static_string_") ||
        !gv->hasInitializer())
      return nullptr;

    llvm::Constant *init = gv->getInitializer();
    if (!init->getType()->isArrayTy() ||
        !llvm::isa<llvm::ConstantDataArray, llvm::ConstantAggregateZero>(init))
      return nullptr;

    return gv;
  }

  /// True for ConstantExpr/ConstantAggregate. Excludes GlobalValue: a global's
  /// initializer is not a rewritable use.
  static bool hasInspectableOperands(llvm::Constant *c) {
    return llvm::isa<llvm::ConstantExpr, llvm::ConstantAggregate>(c);
  }

  /// True if `c` is or contains a string global. Globals are usually embedded
  /// in a `{ptr, i64}` struct or a GEP expression, not used as bare operands.
  static bool referencesStringGlobal(llvm::Constant *c) {
    if (asInlinableStringGlobal(c))
      return true;
    if (!hasInspectableOperands(c))
      return false;

    for (llvm::Value *op : c->operand_values())
      if (auto *opC = llvm::dyn_cast<llvm::Constant>(op))
        if (referencesStringGlobal(opC))
          return true;
    return false;
  }

  /// Appends inlinable string globals reachable from `c` to `out`, deduped via
  /// `seen`.
  static void
  collectStringGlobals(llvm::Constant *c,
                       llvm::SmallVectorImpl<llvm::GlobalVariable *> &out,
                       llvm::SmallPtrSetImpl<llvm::GlobalVariable *> &seen) {
    if (auto *gv = asInlinableStringGlobal(c)) {
      if (seen.insert(gv).second)
        out.push_back(gv);
      return;
    }
    if (!hasInspectableOperands(c))
      return;

    for (llvm::Value *op : c->operand_values())
      if (auto *opC = llvm::dyn_cast<llvm::Constant>(op))
        collectStringGlobals(opC, out, seen);
  }

  /// Emits instructions at `b` that replace string globals in `c` with their
  /// allocas. Constants wrapping the globals are expanded to instructions
  /// because a constant cannot reference an alloca.
  static llvm::Value *materialize(llvm::Constant *c, llvm::IRBuilder<> &b,
                                  const AllocaMap &allocas) {
    if (auto *gv = asInlinableStringGlobal(c))
      return allocas.lookup(gv);

    if (!referencesStringGlobal(c))
      return c;

    if (auto *ce = llvm::dyn_cast<llvm::ConstantExpr>(c)) {
      llvm::Instruction *inst = ce->getAsInstruction();
      for (llvm::Use &op : inst->operands())
        if (auto *opC = llvm::dyn_cast<llvm::Constant>(op.get()))
          op.set(materialize(opC, b, allocas));
      b.Insert(inst);
      return inst;
    }

    // Structs and arrays are rebuilt element by element; vectors are not used.
    auto *agg = llvm::dyn_cast<llvm::ConstantAggregate>(c);
    if (!agg || !(agg->getType()->isStructTy() || agg->getType()->isArrayTy()))
      return c;

    // Start from undef so the original aggregate (and its global reference)
    // isn't kept alive as a base operand. undef rather than poison: Mojo emits
    // undef and xcrun metal's older bitcode reader is picky about it.
    llvm::Value *acc = llvm::UndefValue::get(agg->getType());
    for (unsigned i = 0, e = agg->getNumOperands(); i != e; ++i)
      acc = b.CreateInsertValue(
          acc, materialize(agg->getOperand(i), b, allocas), {i});
    return acc;
  }

  /// Inline static_string_* globals into per-function alloca + store pairs.
  /// Metal's JIT can't resolve addrspace(0) globals: one left behind fails
  /// pipeline creation with "Undefined symbols: static_string_<hash>".
  ///
  /// Alloca rather than addrspace(2): promoting the GV to addrspace(2) and
  /// inserting addrspacecast(as2→as0) at each use sites satisfies the linker,
  /// but MetalAIRPass's struct address-space propagation then misidentifies the
  /// i64 length field of the {ptr, i64} slice as a pointer and corrupts the
  /// va-arg buffer layout. Alloca keeps the struct opaque to MetalAIRPass.
  bool transformGlobalConstantsOnMetal() {
    if (!isMetalTriple())
      return false;

    bool changed = false;
    for (llvm::Function &func : module) {
      if (func.isDeclaration())
        continue;

      llvm::SmallVector<llvm::Use *, 4> uses;
      llvm::SmallVector<llvm::GlobalVariable *, 4> globals;
      llvm::SmallPtrSet<llvm::GlobalVariable *, 4> seen;
      for (llvm::BasicBlock &bb : func) {
        for (llvm::Instruction &inst : bb) {
          for (llvm::Use &use : inst.operands()) {
            auto *c = llvm::dyn_cast<llvm::Constant>(use.get());
            if (!c || !referencesStringGlobal(c))
              continue;
            uses.push_back(&use);
            collectStringGlobals(c, globals, seen);
          }
        }
      }

      if (uses.empty())
        continue;

      llvm::IRBuilder<> entryB(func.getEntryBlock().getFirstInsertionPt());
      AllocaMap allocas;
      for (llvm::GlobalVariable *gv : globals) {
        llvm::Constant *init = gv->getInitializer();
        llvm::AllocaInst *alloca =
            entryB.CreateAlloca(init->getType(), nullptr);
        alloca->setAlignment(llvm::Align(16));
        entryB.CreateStore(init, alloca);
        allocas[gv] = alloca;
      }

      for (llvm::Use *use : uses) {
        auto *c = llvm::cast<llvm::Constant>(use->get());
        auto *user = llvm::cast<llvm::Instruction>(use->getUser());
        llvm::IRBuilder<> b(user);
        // PHI operands are live in the predecessor, not at the phi itself.
        if (auto *phi = llvm::dyn_cast<llvm::PHINode>(user)) {
          llvm::BasicBlock *pred = phi->getIncomingBlock(
              llvm::PHINode::getIncomingValueNumForOperand(
                  use->getOperandNo()));
          b.SetInsertPoint(pred->getTerminator());
        }
        use->set(materialize(c, b, allocas));
      }

      changed = true;
    }
    return changed;
  }

  /// Replace __mojo_metal_os_log_64 sentinel calls with air.os_log calls.
  /// Each call site gets a [64 x i32] alloca populated with the 64-byte
  /// source buffer (bytes zero-extended to i32) for the va-arg convention.
  bool transformMetalPrintBufferSentinels() {
    if (!isMetalTriple())
      return false;

    llvm::Function *sentinel = module.getFunction("__mojo_metal_os_log_64");
    if (!sentinel)
      return false;

    llvm::SmallVector<std::pair<llvm::Function *, llvm::CallInst *>, 8> calls;
    for (llvm::Function &func : module) {
      if (func.isDeclaration())
        continue;
      for (llvm::BasicBlock &bb : func)
        for (llvm::Instruction &inst : bb)
          if (auto *ci = llvm::dyn_cast<llvm::CallInst>(&inst))
            if (ci->getCalledFunction() == sentinel)
              calls.push_back({&func, ci});
    }

    if (calls.empty())
      return false;

    llvm::LLVMContext &ctx = module.getContext();
    constexpr unsigned chunk = 64;
    constexpr unsigned asConstant = 2; // Metal constant address space.

    llvm::IntegerType *i8Ty = llvm::Type::getInt8Ty(ctx);
    llvm::IntegerType *i32Ty = llvm::Type::getInt32Ty(ctx);
    auto *i8PtrAS2 = llvm::PointerType::get(ctx, asConstant);
    auto *i8Ptr = llvm::PointerType::get(ctx, 0);

    auto getOrCreateStringGlobal =
        [&](const char *name, llvm::StringRef value) -> llvm::GlobalVariable * {
      if (llvm::GlobalVariable *gv = module.getNamedGlobal(name))
        return gv;
      auto *init =
          llvm::ConstantDataArray::getString(ctx, value, /*AddNull=*/true);
      return new llvm::GlobalVariable(
          module, init->getType(),
          /*isConstant=*/true, llvm::GlobalValue::PrivateLinkage, init, name,
          nullptr, llvm::GlobalVariable::NotThreadLocal, asConstant);
    };

    // Format string is "%c" * 64 — one slot per va-arg byte.
    std::string fmtStr;
    for (unsigned i = 0; i < chunk; ++i)
      fmtStr += "%c";

    llvm::GlobalVariable *fmtGV =
        getOrCreateStringGlobal(".mojo.log.fmt", fmtStr);
    llvm::GlobalVariable *subsysGV =
        getOrCreateStringGlobal(".mojo.log.subsystem", "mojo");
    llvm::GlobalVariable *catGV =
        getOrCreateStringGlobal(".mojo.log.category", "print");

    // Generate following signature of `air.os_log`:
    // void @air.os_log(ptr addrspace(2), ptr addrspace(2), i32,
    //                  ptr addrspace(2), ptr, i64)
    auto *osLogFTy =
        llvm::FunctionType::get(llvm::Type::getVoidTy(ctx),
                                {i8PtrAS2, i8PtrAS2, i32Ty, i8PtrAS2, i8Ptr,
                                 llvm::Type::getInt64Ty(ctx)},
                                /*isVarArg=*/false);
    llvm::FunctionCallee osLogFn =
        module.getOrInsertFunction("air.os_log", osLogFTy);

    // Strip attributes: the older LLVM bitcode reader in xcrun metal rejects
    // non-empty attribute lists on air.os_log.
    if (auto *osLogFunc = llvm::dyn_cast<llvm::Function>(osLogFn.getCallee()))
      osLogFunc->setAttributes(llvm::AttributeList());

    auto *vaBufTy = llvm::ArrayType::get(i32Ty, chunk);

    for (auto &[func, ci] : calls) {
      llvm::IRBuilder<> builder(ci);

      // Alloca in entry block for proper stack management.
      llvm::IRBuilder<> entryB(func->getEntryBlock().getFirstInsertionPt());
      llvm::AllocaInst *vaBuf =
          entryB.CreateAlloca(vaBufTy, nullptr, "mojo.log.va");
      vaBuf->setAlignment(llvm::Align(4));

      llvm::Value *srcPtr = ci->getArgOperand(0);
      for (unsigned i = 0; i < chunk; ++i) {
        llvm::Value *bytePtr =
            builder.CreateInBoundsGEP(i8Ty, srcPtr, builder.getInt64(i));
        llvm::LoadInst *byte = builder.CreateLoad(i8Ty, bytePtr);
        // zext, not sext: bytes are unsigned.
        llvm::Value *wide = builder.CreateZExt(byte, i32Ty);
        llvm::Value *slot = builder.CreateInBoundsGEP(
            vaBufTy, vaBuf, {builder.getInt32(0), builder.getInt32(i)});
        builder.CreateStore(wide, slot);
      }

      llvm::Value *fmtPtr =
          builder.CreateInBoundsGEP(fmtGV->getValueType(), fmtGV,
                                    {builder.getInt64(0), builder.getInt64(0)});
      llvm::Value *subsysPtr =
          builder.CreateInBoundsGEP(subsysGV->getValueType(), subsysGV,
                                    {builder.getInt64(0), builder.getInt64(0)});
      llvm::Value *catPtr =
          builder.CreateInBoundsGEP(catGV->getValueType(), catGV,
                                    {builder.getInt64(0), builder.getInt64(0)});
      builder.CreateCall(osLogFn, {subsysPtr, catPtr,
                                   builder.getInt32(1), // log_type=info
                                   fmtPtr, vaBuf,
                                   builder.getInt64(chunk * 4)}); // va_size=256

      ci->eraseFromParent();
    }

    if (sentinel->use_empty())
      sentinel->eraseFromParent();

    return true;
  }

  /// Force all internal (non-kernel) functions to be inlineable by stripping
  /// noinline and adding alwaysinline. Apple's air-lld crashes on functions
  /// with struct types containing opaque pointers (e.g., { ptr, i64 }).
  /// AlwaysInlinerPass runs after this to actually inline them.
  bool removeNoInlineAttribute() {
    if (!isMetalTriple())
      return false;

    bool changed = false;
    for (llvm::Function &func : module) {
      if (func.hasInternalLinkage() && !func.hasFnAttribute("metal.kernel")) {
        func.removeFnAttr(llvm::Attribute::NoInline);
        func.addFnAttr(llvm::Attribute::AlwaysInline);
        changed = true;
      }
    }
    return changed;
  }

private:
  llvm::Module &module;
  llvm::Triple triple;
};

} // namespace

llvm::PreservedAnalyses
MetalModuleRewritePass::run(llvm::Module &module,
                            llvm::ModuleAnalysisManager &) {
  MetalModuleRewritePassImpl impl(module);
  return impl.run() ? llvm::PreservedAnalyses::none()
                    : llvm::PreservedAnalyses::all();
}

} // namespace M::KGEN
