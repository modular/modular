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

  // The global value indices go from aliases, functions, then globals.
  unsigned curIdx = 0;
  auto materializeGlobals = [&](auto origRange) {
    for (auto &value : llvm::make_early_inc_range(origRange)) {
      if (idxIt != idxEnd && curIdx == *idxIt) {
        ++idxIt;
        llvm::cantFail(value.materialize());
      } else {
        if constexpr (std::is_same_v<std::decay_t<decltype(value)>,
                                     llvm::GlobalVariable>)
          value.setInitializer(nullptr);
        else
          value.deleteBody();
        value.setComdat(nullptr);
        value.setLinkage(llvm::GlobalValue::ExternalLinkage);
      }
      ++curIdx;
    }
  };
  materializeGlobals(result->functions());
  materializeGlobals(result->globals());

  // Finalize materialization of the module.
  llvm::cantFail(result->materializeAll());
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
  for (const llvm::Function &fn : mainModule->functions()) {
    computeDeps(fn);
    if (!fn.isDeclaration() && (fn.hasExternalLinkage() || fn.hasWeakLinkage()))
      transitiveDeps[&fn];
  }
  for (const llvm::GlobalVariable &global : mainModule->globals()) {
    computeDeps(global);
    if (!global.hasInternalLinkage())
      transitiveDeps[&global];
  }

  // If there is only one (or fewer) exported functions, forward the main
  // module.
  if (transitiveDeps.size() <= 1) {
    processFn(std::move(mainModule), std::nullopt);
    return;
  }

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

  SmallVector<const llvm::MapVector<const llvm::GlobalValue *, unsigned> *>
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

  // Sort the sets by to schedule the larger modules first.
  llvm::sort(setsToProcess,
             [](auto *lhs, auto *rhs) { return lhs->size() > rhs->size(); });

  // Prepare to materialize slices of the module by first writing the main
  // module as bitcode to a shared buffer.
  auto buf = WriteableBuffer::get();
  llvm::WriteBitcodeToFile(*mainModule, *buf);

  for (auto [idx, set] : llvm::enumerate(setsToProcess))
    processFn(readAndMaterializeDependencies(buf.copy(), *set), idx);
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
    llvm::SmallPtrSet<const llvm::Value *, 4> dependencies;
    llvm::SmallPtrSet<const llvm::Value *, 4> users;
  };

  /// Collect all of the immediate global value users of `value`.
  void collectValueUsers(const llvm::Value *value);

  /// Propagate use information through the module.
  void propagateUseInfo();

  /// The main LLVM module being split.
  LLVMModuleAndContext mainModule;

  /// The value info for each global value in the module.
  llvm::DenseMap<const llvm::Value *, ValueInfo> valueInfos;
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
  auto computeUsers = [&](auto &value) { collectValueUsers(&value); };
  llvm::for_each(mainModule->functions(), computeUsers);
  llvm::for_each(mainModule->globals(), computeUsers);

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
  auto splitValue =
      [&](const llvm::Value *root) -> std::unique_ptr<llvm::Module> {
    // If the function is already split, e.g. if it was a dependency of
    // another function, skip it.
    if (splitValues.count(root))
      return nullptr;

    auto &valueInfo = valueInfos[root];
    llvm::ValueToValueMapTy valueMap;
    SmallPtrSet<const llvm::Value *, 4> splitdDeps;

    // llvm::CloneModule is not thread safe if the cloning is from the same
    // original module because new cloned things will be added to the same
    // llvm::LLVMContext which can have race condition.
    // This is also true that cloned modules (with the same origin) cannot be
    // destroyed in multi-threading without explict mutex because erasing things
    // from the same llvm::LLVMContext is not protected.
    std::unique_ptr<llvm::Module> splitModule(llvm::CloneModule(
        *mainModule, valueMap, [&](const llvm::GlobalValue *globalVal) {
          // Only clone root and the declaration of its dependencies.
          if (globalVal == root) {
            splitdDeps.insert(globalVal);
            return true;
          }

          auto iter = valueInfos.find(globalVal);
          if (iter == valueInfos.end())
            return false;
          if ((iter->second.canBeSplit || iter->second.users.empty()) &&
              isa_and_nonnull<llvm::Function>(globalVal))
            return false;

          if (valueInfo.dependencies.contains(globalVal)) {
            splitdDeps.insert(globalVal);
            return true;
          }
          return false;
        }));

    if (splitModule->empty())
      splitModule->setModuleInlineAsm("");

    // Module cloning creates stubs for every function and global in the
    // original module, even if they aren't used in this slice. Kill all of
    // these off to make the module more self-contained.
    for (auto &func : llvm::make_early_inc_range(*splitModule)) {
      if (func.isDeclaration() &&
          (func.use_empty() && !valueInfo.dependencies.count(&func)))
        func.eraseFromParent();
    }

    for (auto &globalVar : llvm::make_early_inc_range(splitModule->globals())) {
      if (globalVar.isDeclaration() && globalVar.use_empty())
        globalVar.eraseFromParent();
    }

    // Record the split values.
    splitValues.insert(splitdDeps.begin(), splitdDeps.end());

    return splitModule;
  };

  int64_t totalSplit = 0;
  int64_t count = 0;
  SmallVector<llvm::Value *> toSplit;
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

  std::string bitcodeData;
  llvm::raw_string_ostream bitcodeOs(bitcodeData);
  for (auto [idx, value] : llvm::enumerate(toSplit)) {
    // Synchronize incrementally to reduce pressuring memory to much for holding
    // the cloned llvm modules.
    if (std::unique_ptr<llvm::Module> splitModule = splitValue(value)) {
      LLVMModuleAndContext bitcloned;
      // FIXME: We shouldn't be cloning and then roundtripping through bitcode
      // here! We can lazy load symbols out of bitcode to make this faster.
      (void)bitcloned.create([&](llvm::LLVMContext &ctx) {
        llvm::WriteBitcodeToFile(*splitModule, bitcodeOs);
        auto moduleOr = llvm::parseBitcodeFile(
            llvm::MemoryBufferRef(bitcodeOs.str(), "<split>"), ctx);
        bitcodeData.clear();
        // We don't expect roundtripping LLVM bitcode to fail.
        if (!moduleOr)
          llvm::report_fatal_error("LLVM bitcode roundtrip failed");
        return std::move(*moduleOr);
      });
      processFn(std::move(bitcloned), totalSplit++);
    }
  }

  // If we had no functions to split, just process the main module.
  if (totalSplit == 0)
    processFn(std::move(mainModule), std::nullopt);
}

/// Collect all of the immediate global value users of `value`.
void LLVMModulePerFunctionSplitterImpl::collectValueUsers(
    const llvm::Value *value) {
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
    for (const llvm::Value *user : info->users) {
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
    for (const llvm::Value *dep : info->dependencies) {
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
