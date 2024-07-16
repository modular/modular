//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Compiler/LLVMIRUtils.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "Support/Buffer.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/GlobalStatus.h"
#include "llvm/Transforms/Utils/SplitModule.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace M;
using namespace KGEN;

#define DEBUG_TYPE "llvm-module-split"

//===----------------------------------------------------------------------===//
// LLVMModuleAndContext
//===----------------------------------------------------------------------===//

ErrorOrSuccess LLVMModuleAndContext::create(
    function_ref<ErrorOr<std::unique_ptr<llvm::Module>>(llvm::LLVMContext &)>
        createModule) {
  assert(!module && "already have a module");
  auto moduleOr = createModule(*ctx);
  if (moduleOr.isError())
    return moduleOr.takeError();
  module = moduleOr.takeValue();
  return success();
}

//===----------------------------------------------------------------------===//
// Module Splitter
//===----------------------------------------------------------------------===//

namespace {
class LLVMModuleSplitterImpl {
public:
  explicit LLVMModuleSplitterImpl(LLVMModuleAndContext module)
      : mainModule(std::move(module)) {}

  /// Split the LLVM module into multiple modules using the provided process
  /// function.
  void split(LLVMSplitProcessFn processFn);

private:
  struct ValueInfo {
    /// The immediate global value dependencies of a value.
    SmallVector<const llvm::GlobalValue *> dependencies;
    /// Map each global value to its index in the module. We will use this to
    /// materialize global values from bitcode.
    unsigned gvIdx;
  };

  struct TransitiveDeps {
    /// The transitive dependencies.
    llvm::MapVector<const llvm::GlobalValue *, unsigned> deps;
    /// True if computation is complete.
    bool complete = false;
    /// The assigned module index.
    std::optional<unsigned> mutIdx;
  };

  /// Collect the immediate global value dependencies of `value`. `orig` is the
  /// original transitive value, which is not equal to `value` when it is used
  /// in a constant.
  void collectImmediateDependencies(const llvm::Value *value,
                                    const llvm::GlobalValue *orig);

  /// The main LLVM module being split.
  LLVMModuleAndContext mainModule;

  /// The value info for each global value in the module.
  llvm::DenseMap<const llvm::Value *, ValueInfo> infos;

  /// The transitive dependencies of each global value.
  llvm::MapVector<const llvm::GlobalValue *, TransitiveDeps> transitiveDeps;

  /// Users of split "anchors". These are global values where we don't want
  /// their users to be split into different modules because it will cause the
  /// symbol to be duplicated.
  llvm::MapVector<const llvm::GlobalValue *, llvm::SetVector<TransitiveDeps *>>
      splitAnchorUsers;
};
} // namespace

static LLVMModuleAndContext readAndMaterializeDependencies(
    BufferRef buf,
    const llvm::MapVector<const llvm::GlobalValue *, unsigned> &set) {
  CompilerTimeTraceScope traceScope("readAndMaterializeDependencies");

  // First, create a lazy module with an internal bitcode materializer.
  // TODO: Not sure how to make lazy loading metadata work.
  LLVMModuleAndContext result;
  (void)result.create([&](llvm::LLVMContext &ctx) {
    return llvm::cantFail(llvm::getLazyBitcodeModule(
        llvm::MemoryBufferRef(buf->getBuffer(), "<split-module>"), ctx,
        /*ShouldLazyLoadMetadata=*/false));
  });
  result->setModuleInlineAsm("");

  SmallVector<unsigned> sortIndices =
      llvm::to_vector(llvm::make_second_range(set));
  llvm::sort(sortIndices, std::less<unsigned>());
  auto idxIt = sortIndices.begin();
  auto idxEnd = sortIndices.end();

  // The global value indices go from globals, functions, then aliases. This
  // mirrors the order in which global values are deleted by LLVM's GlobalDCE.
  unsigned curIdx = 0;
  // We need to keep the IR "valid" for the verifier because `materializeAll`
  // may invoke it. It doesn't matter since we're deleting the globals anyway.
  for (llvm::GlobalVariable &global : result->globals()) {
    if (idxIt != idxEnd && curIdx == *idxIt) {
      ++idxIt;
      llvm::cantFail(global.materialize());
    } else {
      global.setInitializer(nullptr);
      global.setComdat(nullptr);
      global.setLinkage(llvm::GlobalValue::ExternalLinkage);
    }
    ++curIdx;
  }
  for (llvm::Function &func : result->functions()) {
    if (idxIt != idxEnd && curIdx == *idxIt) {
      ++idxIt;
      llvm::cantFail(func.materialize());
    } else {
      func.deleteBody();
      func.setComdat(nullptr);
      func.setLinkage(llvm::GlobalValue::ExternalLinkage);
    }
    ++curIdx;
  }

  // Finalize materialization of the module.
  llvm::cantFail(result->materializeAll());

  // Now that the module is materialized, we can start deleting stuff. Just
  // delete declarations with no uses.
  for (llvm::GlobalVariable &global :
       llvm::make_early_inc_range(result->globals())) {
    if (global.isDeclaration() && global.use_empty())
      global.eraseFromParent();
  }
  for (llvm::Function &func : llvm::make_early_inc_range(result->functions())) {
    if (func.isDeclaration() && func.use_empty())
      func.eraseFromParent();
  }
  return result;
}

/// support for splitting an LLVM module into multiple parts using exported
/// functions as anchors, and pull in all dependency on the call stack into one
/// module.
void KGEN::splitPerExported(LLVMModuleAndContext module,
                            LLVMSplitProcessFn processFn) {
  CompilerTimeTraceScope traceScope("splitPerExported");
  LLVMModuleSplitterImpl impl(std::move(module));
  impl.split(processFn);
}

void LLVMModuleSplitterImpl::split(LLVMSplitProcessFn processFn) {
  // The use-def list is sparse. Use it to build a sparse dependency graph
  // between global values.
  unsigned gvIdx = 0;
  auto computeDeps = [&](const llvm::GlobalValue &value) {
    infos[&value].gvIdx = gvIdx++;
    collectImmediateDependencies(&value, &value);
  };
  // NOTE: The visitation of globals then functions has to line up with
  // `readAndMaterializeDependencies`.
  for (const llvm::GlobalVariable &global : mainModule->globals()) {
    computeDeps(global);
    if (!global.hasInternalLinkage())
      transitiveDeps[&global];
  }
  for (const llvm::Function &fn : mainModule->functions()) {
    computeDeps(fn);
    if (!fn.isDeclaration() && (fn.hasExternalLinkage() || fn.hasWeakLinkage()))
      transitiveDeps[&fn];
  }

  // If there is only one (or fewer) exported functions, forward the main
  // module.
  if (transitiveDeps.size() <= 1)
    return processFn(forwardModule(std::move(mainModule)), std::nullopt);

  // Now for each export'd global value, compute the transitive set of
  // dependencies using DFS.
  SmallVector<const llvm::GlobalValue *> worklist;
  for (auto &[value, deps] : transitiveDeps) {
    worklist.clear();
    worklist.push_back(value);
    while (!worklist.empty()) {
      const llvm::GlobalValue *it = worklist.pop_back_val();

      auto [iter, inserted] = deps.deps.insert({it, -1});
      if (!inserted) {
        // Already visited.
        continue;
      }
      // Pay the cost of the name lookup only on a miss.
      const ValueInfo &info = infos.at(it);
      iter->second = info.gvIdx;

      // If this value depends on another value that is going to be split, we
      // don't want to duplicate the symbol. Keep all the users together.
      if (it != value) {
        if (auto depIt = transitiveDeps.find(it);
            depIt != transitiveDeps.end()) {
          auto &users = splitAnchorUsers[it];
          users.insert(&deps);
          // Make sure to include the other value in its own user list.
          users.insert(&depIt->second);
          // We don't have to recurse since the subgraph will get processed.
          continue;
        }
      }

      // If this value depends on a mutable global, keep track of it. We have to
      // put all users of a mutable global in the same module.
      if (auto *global = dyn_cast<llvm::GlobalVariable>(it);
          global && !global->isConstant())
        splitAnchorUsers[global].insert(&deps);

      // Recursive on dependencies.
      llvm::append_range(worklist, info.dependencies);
    }

    deps.complete = true;
  }

  // For each mutable global, grab all the transitive users and put them in one
  // module. If global A has user set A* and global B has user set B* where
  // A* and B* have an empty intersection, all values in A* will be assigned 0
  // and all values in B* will be assigned 1. If global C has user set C* that
  // overlaps both A* and B*, it will overwrite both to 2.
  SmallVector<SmallVector<TransitiveDeps *>> bucketing(splitAnchorUsers.size());
  for (auto [curMutIdx, bucket, users] :
       llvm::enumerate(bucketing, llvm::make_second_range(splitAnchorUsers))) {
    for (TransitiveDeps *deps : users) {
      if (deps->mutIdx && *deps->mutIdx != curMutIdx) {
        auto &otherBucket = bucketing[*deps->mutIdx];
        for (TransitiveDeps *other : otherBucket) {
          bucket.push_back(other);
          other->mutIdx = curMutIdx;
        }
        otherBucket.clear();
        assert(*deps->mutIdx == curMutIdx);
      } else {
        bucket.push_back(deps);
        deps->mutIdx = curMutIdx;
      }
    }
  }

  // Now that we have assigned buckets to each value, merge the transitive
  // dependency sets of all values belonging to the same set.
  SmallVector<llvm::MapVector<const llvm::GlobalValue *, unsigned>> buckets(
      bucketing.size());
  for (auto [deps, bucket] : llvm::zip(bucketing, buckets)) {
    for (TransitiveDeps *dep : deps) {
      for (auto &namedValue : dep->deps)
        bucket.insert(namedValue);
    }
  }

  SmallVector<llvm::MapVector<const llvm::GlobalValue *, unsigned> *>
      setsToProcess;
  setsToProcess.reserve(buckets.size() + transitiveDeps.size());

  // Clone each mutable global bucket into its own module.
  for (auto &bucket : buckets) {
    if (bucket.empty())
      continue;
    setsToProcess.push_back(&bucket);
  }

  for (auto &[root, deps] : transitiveDeps) {
    // Skip values included in another transitive dependency set and values
    // included in mutable global sets.
    if (!deps.mutIdx)
      setsToProcess.push_back(&deps.deps);
  }

  if (setsToProcess.size() <= 1)
    return processFn(forwardModule(std::move(mainModule)), std::nullopt);

  // Sort the sets by to schedule the larger modules first.
  llvm::sort(setsToProcess,
             [](auto *lhs, auto *rhs) { return lhs->size() > rhs->size(); });

  // Prepare to materialize slices of the module by first writing the main
  // module as bitcode to a shared buffer.
  auto buf = WriteableBuffer::get();
  {
    CompilerTimeTraceScope traceScope("writeMainModuleBitcode");
    llvm::WriteBitcodeToFile(*mainModule, *buf);
  }

  for (auto [idx, set] : llvm::enumerate(setsToProcess)) {
    auto makeModule = [set = std::move(*set),
                       buf = BufferRef(buf.copy())]() mutable {
      return readAndMaterializeDependencies(std::move(buf), set);
    };
    processFn(std::move(makeModule), idx);
  }
}

void LLVMModuleSplitterImpl::collectImmediateDependencies(
    const llvm::Value *value, const llvm::GlobalValue *orig) {
  for (const llvm::Value *user : value->users()) {
    // Recurse into pure constant users.
    if (isa<llvm::Constant>(user) && !isa<llvm::GlobalValue>(user)) {
      collectImmediateDependencies(user, orig);
      continue;
    }

    if (auto *inst = dyn_cast<llvm::Instruction>(user)) {
      const llvm::Function *func = inst->getParent()->getParent();
      infos[func].dependencies.push_back(orig);
    } else if (auto *globalVal = dyn_cast<llvm::GlobalValue>(user)) {
      infos[globalVal].dependencies.push_back(orig);
    } else {
      llvm_unreachable("unexpected user of global value");
    }
  }
}

namespace {
/// This class provides support for splitting an LLVM module into multiple
/// parts.
/// TODO: Clean up the splitters here (some code duplication) when we can move
/// to per function llvm compilation.
class LLVMModulePerFunctionSplitterImpl {
public:
  LLVMModulePerFunctionSplitterImpl(LLVMModuleAndContext module)
      : mainModule(std::move(module)) {}

  /// Split the LLVM module into multiple modules using the provided process
  /// function.
  void split(LLVMSplitProcessFn processFn);

private:
  struct ValueInfo {
    const llvm::Value *value = nullptr;
    bool canBeSplit = true;
    llvm::SmallPtrSet<const llvm::GlobalValue *, 4> dependencies;
    llvm::SmallPtrSet<const llvm::GlobalValue *, 4> users;
    /// Map each global value to its index in the module. We will use this to
    /// materialize global values from bitcode.
    unsigned gvIdx;
  };

  /// Collect all of the immediate global value users of `value`.
  void collectValueUsers(const llvm::GlobalValue *value);

  /// Propagate use information through the module.
  void propagateUseInfo();

  /// The main LLVM module being split.
  LLVMModuleAndContext mainModule;

  /// The value info for each global value in the module.
  llvm::MapVector<const llvm::GlobalValue *, ValueInfo> valueInfos;
};
} // namespace

/// support for splitting an LLVM module into multiple parts with each part
/// contains only one function (with exception for coroutine related functions.)
void KGEN::splitPerFunction(LLVMModuleAndContext module,
                            LLVMSplitProcessFn processFn) {
  CompilerTimeTraceScope traceScope("splitPerFunction");
  LLVMModulePerFunctionSplitterImpl impl(std::move(module));
  impl.split(processFn);
}

/// Split the LLVM module into multiple modules using the provided process
/// function.
void LLVMModulePerFunctionSplitterImpl::split(LLVMSplitProcessFn processFn) {
  // Compute the value info for each global in the module.
  // NOTE: The visitation of globals then functions has to line up with
  // `readAndMaterializeDependencies`.
  unsigned gvIdx = 0;
  auto computeUsers = [&](const llvm::GlobalValue &value) {
    valueInfos[&value].gvIdx = gvIdx++;
    collectValueUsers(&value);
  };
  llvm::for_each(mainModule->globals(), computeUsers);
  llvm::for_each(mainModule->functions(), computeUsers);

  // With use information collected, propagate it to the dependencies.
  propagateUseInfo();

  // Now we can split the module.
  // We split the module per function and cloning any necessary dependencies:
  // - For function dependencies, only clone the declaration unless its
  //   coroutine related.
  // - For other internal values, clone as is.
  // This is much fine-grained splitting, which enables significantly higher
  // levels of parallelism (and smaller generated artifacts).
  // LLVM LTO style optimization may suffer a bit here since we don't have
  // the full callstack present anymore in each cloned module.
  llvm::DenseSet<const llvm::Value *> splitValues;
  SmallVector<llvm::MapVector<const llvm::GlobalValue *, unsigned>>
      setsToProcess;

  // Hoist these collections to re-use memory allocations.
  llvm::ValueToValueMapTy valueMap;
  SmallPtrSet<const llvm::Value *, 4> splitDeps;
  auto splitValue = [&](const llvm::GlobalValue *root) {
    // If the function is already split, e.g. if it was a dependency of
    // another function, skip it.
    if (splitValues.count(root))
      return;

    auto &valueInfo = valueInfos[root];
    valueMap.clear();
    splitDeps.clear();

    auto shouldSplit = [&](const llvm::GlobalValue *globalVal,
                           const ValueInfo &info) {
      // Only clone root and the declaration of its dependencies.
      if (globalVal == root) {
        splitDeps.insert(globalVal);
        return true;
      }
      if ((info.canBeSplit || info.users.empty()) &&
          isa_and_nonnull<llvm::Function>(globalVal))
        return false;

      if (valueInfo.dependencies.contains(globalVal)) {
        splitDeps.insert(globalVal);
        return true;
      }
      return false;
    };

    auto &set = setsToProcess.emplace_back();
    for (auto &[globalVal, info] : valueInfos) {
      if (shouldSplit(globalVal, info))
        set.insert({globalVal, info.gvIdx});
    }
    if (set.empty())
      setsToProcess.pop_back();

    // Record the split values.
    splitValues.insert(splitDeps.begin(), splitDeps.end());
  };

  int64_t count = 0;
  SmallVector<const llvm::GlobalValue *> toSplit;
  for (auto &global : mainModule->globals()) {
    if (global.hasInternalLinkage()) {
      global.setLinkage(llvm::GlobalValue::WeakAnyLinkage);
      continue;
    }
    // TODO: Add special handling for `llvm.global_ctors` and
    // `llvm.global_dtors`, because otherwise they end up tying almost all
    // symbols into the same split.
    LLVM_DEBUG(llvm::dbgs()
                   << (count++) << ": split global: " << global << "\n";);
    toSplit.emplace_back(&global);
  }

  for (auto &fn : mainModule->functions()) {
    if (!fn.isDeclaration() &&
        (valueInfos[&fn].canBeSplit || valueInfos[&fn].users.empty())) {
      if (fn.hasInternalLinkage())
        fn.setLinkage(llvm::Function::LinkageTypes::WeakAnyLinkage);
      LLVM_DEBUG(llvm::dbgs()
                     << (count++) << ": split fn: " << fn.getName() << "\n";);
      toSplit.emplace_back(&fn);
    }
  }

  // Run this now since we just changed the linkages.
  for (const llvm::GlobalValue *value : toSplit)
    splitValue(value);

  if (setsToProcess.size() <= 1)
    return processFn(forwardModule(std::move(mainModule)), std::nullopt);

  // Prepare to materialize slices of the module by first writing the main
  // module as bitcode to a shared buffer.
  auto buf = WriteableBuffer::get();
  {
    CompilerTimeTraceScope traceScope("writeMainModuleBitcode");
    llvm::WriteBitcodeToFile(*mainModule, *buf);
  }

  for (auto [idx, set] : llvm::enumerate(setsToProcess)) {
    auto makeModule = [set = std::move(set),
                       buf = BufferRef(buf.copy())]() mutable {
      return readAndMaterializeDependencies(std::move(buf), set);
    };
    processFn(std::move(makeModule), idx);
  }
}

/// Collect all of the immediate global value users of `value`.
void LLVMModulePerFunctionSplitterImpl::collectValueUsers(
    const llvm::GlobalValue *value) {
  SmallVector<const llvm::User *> worklist(value->users());

  while (!worklist.empty()) {
    const llvm::User *userIt = worklist.pop_back_val();

    // Recurse into pure constant users.
    if (isa<llvm::Constant>(userIt) && !isa<llvm::GlobalValue>(userIt)) {
      worklist.append(userIt->user_begin(), userIt->user_end());
      continue;
    }

    if (const auto *inst = dyn_cast<llvm::Instruction>(userIt)) {
      const llvm::Function *func = inst->getParent()->getParent();
      valueInfos[value].users.insert(func);
      valueInfos[func];
    } else if (const auto *globalVal = dyn_cast<llvm::GlobalValue>(userIt)) {
      valueInfos[value].users.insert(globalVal);
      valueInfos[globalVal];
    } else {
      llvm_unreachable("unexpected user of global value");
    }
  }

  // If the current value is a mutable global variable, then it can't be
  // split.
  if (auto *global = dyn_cast<llvm::GlobalVariable>(value))
    valueInfos[value].canBeSplit = global->isConstant();
}

/// Propagate use information through the module.
void LLVMModulePerFunctionSplitterImpl::propagateUseInfo() {
  std::vector<ValueInfo *> worklist;
  // Each value depends on itself. Seed the iteration with that.
  for (auto &[value, info] : valueInfos) {
    if (auto func = llvm::dyn_cast<llvm::Function>(value)) {
      if (func->isDeclaration())
        continue;
    }

    info.dependencies.insert(value);
    info.value = value;
    worklist.push_back(&info);
    // If a value cannot be split, its users are also its dependencies.
    if (!info.canBeSplit)
      llvm::set_union(info.dependencies, info.users);
  }

  while (!worklist.empty()) {
    ValueInfo *info = worklist.back();
    worklist.pop_back();

    // Propagate the dependencies of this value to its users.
    for (const llvm::GlobalValue *user : info->users) {
      ValueInfo &userInfo = valueInfos.find(user)->second;
      if (info == &userInfo)
        continue;
      bool changed = false;

      // Merge dependency to user if current value is not a function that will
      // be split into a separate module.
      bool mergeToUserDep = true;
      if (llvm::isa_and_nonnull<llvm::Function>(info->value)) {
        mergeToUserDep = !info->canBeSplit;
      }

      // If there is a change, add the user info to the worklist.
      if (mergeToUserDep) {
        if (llvm::set_union(userInfo.dependencies, info->dependencies))
          changed = true;
      }

      // If the value cannot be split, its users cannot be split either.
      if (!info->canBeSplit && userInfo.canBeSplit) {
        userInfo.canBeSplit = false;
        changed = true;
        // If a value cannot be split, its users are also its dependencies.
        llvm::set_union(userInfo.dependencies, userInfo.users);
      }

      if (changed) {
        userInfo.value = user;
        worklist.push_back(&userInfo);
      }
    }

    if (info->canBeSplit || isa_and_nonnull<llvm::GlobalValue>(info->value))
      continue;

    // If a value cannot be split, propagate its dependencies up to its
    // dependencies.
    for (const llvm::GlobalValue *dep : info->dependencies) {
      ValueInfo &depInfo = valueInfos.find(dep)->second;
      if (info == &depInfo)
        continue;
      if (llvm::set_union(depInfo.dependencies, info->dependencies)) {
        depInfo.value = dep;
        worklist.push_back(&depInfo);
      }
    }
  }
}
