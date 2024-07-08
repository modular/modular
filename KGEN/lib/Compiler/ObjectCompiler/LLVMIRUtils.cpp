//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Compiler/LLVMIRUtils.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/StringExtras.h"
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

namespace {
class LLVMModuleSplitterImpl {
public:
  explicit LLVMModuleSplitterImpl(llvm::Module &module);

  /// Split the LLVM module into multiple modules using the provided process
  /// function.
  void split(SmallVectorImpl<std::unique_ptr<llvm::Module>> &splitModules);

private:
  struct ValueInfo {
    bool canBeSplit = true;
    llvm::SmallPtrSet<const llvm::Value *, 4> dependencies;
    llvm::SmallPtrSet<const llvm::Value *, 4> users;
  };

  /// Collect all of the immediate global value users of `value`.
  void collectValueUsers(const llvm::Value *value);

  /// Propagate use information through the module.
  void propagateUseInfo();

  /// The main LLVM module being split.
  llvm::Module &mainModule;

  /// The value info for each global value in the module.
  llvm::DenseMap<const llvm::Value *, ValueInfo> valueInfos;
};

/// This class provides support for splitting an LLVM module into multiple
/// parts.
/// TODO: Clean up the splitters here (some code duplication) when we can move
/// to per function llvm compilation.
class LLVMModulePerFunctionSplitterImpl {
public:
  explicit LLVMModulePerFunctionSplitterImpl(
      std::unique_ptr<llvm::Module> module, size_t parallelismLevel);

  /// Split the LLVM module into multiple modules using the provided process
  /// function.
  void split(
      llvm::function_ref<void(std::unique_ptr<llvm::Module>, size_t idx, bool)>
          processFn);

private:
  struct ValueInfo {
    const llvm::Value *value = nullptr;
    bool canBeSplit = true;
    llvm::SmallPtrSet<const llvm::Value *, 4> dependencies;
    llvm::SmallPtrSet<const llvm::Value *, 4> users;
  };

  // TODO(#33945) Special functions to handle wrt coroutines.
  llvm::DenseSet<llvm::StringRef> coroCalledNamesCache{
      "llvm.coro.id.async", "llvm.coro.end.async", "llvm.coro.prepare.async",
      "llvm.coro.async.resume", "llvm.coro.suspend.async.sl_p0p0p0s"};
  llvm::DenseSet<llvm::StringRef> kgenCoroNamesCache{
      "__kgen_coro_end_fn", "__kgen_coro_ctxt_proj_fn"};

  /// Collect all of the immediate global value users of `value`.
  void collectValueUsers(const llvm::Value *value);

  /// Propagate use information through the module.
  void propagateUseInfo();

  /// The main LLVM module being split.
  std::unique_ptr<llvm::Module> mainModule;

  /// The value info for each global value in the module.
  llvm::DenseMap<const llvm::Value *, ValueInfo> valueInfos;

  /// Parallelism level to help guide control splitting concurrency.
  size_t parallelismLevel;
};
} // namespace

namespace M::KGEN {
/// support for splitting an LLVM module into multiple parts using exported
/// functions as anchors, and pull in all dependency on the call stack into one
/// module.
SmallVector<std::unique_ptr<llvm::Module>>
splitPerExported(llvm::Module &module) {
  LLVMModuleSplitterImpl impl(module);
  SmallVector<std::unique_ptr<llvm::Module>> results;
  impl.split(results);
  return results;
}

/// support for splitting an LLVM module into multiple parts with each part
/// contains only one function (with exception for coroutine related functions.)
void splitPerFunction(
    std::unique_ptr<llvm::Module> module, size_t parallelismLevel,
    function_ref<void(std::unique_ptr<llvm::Module>, int64_t idx, bool)>
        processFn) {
  LLVMModulePerFunctionSplitterImpl impl(std::move(module), parallelismLevel);
  impl.split(processFn);
}
} // namespace M::KGEN

LLVMModuleSplitterImpl::LLVMModuleSplitterImpl(llvm::Module &module)
    : mainModule(module) {}

/// Split the LLVM module into multiple modules using the provided process
/// function.
void LLVMModuleSplitterImpl::split(
    SmallVectorImpl<std::unique_ptr<llvm::Module>> &splitModules) {
  // Compute the value info for each global in the module.
  auto computeUsers = [&](auto &value) { collectValueUsers(&value); };
  llvm::for_each(mainModule.functions(), computeUsers);
  llvm::for_each(mainModule.globals(), computeUsers);
  llvm::for_each(mainModule.aliases(), computeUsers);

  // With use information collected, propagate it to the dependencies.
  propagateUseInfo();

  // Now we can split the module. We do this using this by anchoring on the
  // exports of the module, and cloning any necessary dependencies.
  // Realistically we shouldn't be cloning, but we currently depend on LLVM to
  // do various LTO style optimizations for us, which means that each export
  // needs its full callstack present. When this isn't necessary, we should be
  // to define much more fine grained splitting, which would enable
  // significantly higher levels of parallelism (and smaller generated
  // artifacts).
  llvm::DenseSet<const llvm::Value *> splitValues;
  auto splitValue = [&](const llvm::Value *root) {
    // If the function is already split, e.g. if it was a dependency of
    // another function, skip it.
    if (splitValues.count(root))
      return;

    auto &valueInfo = valueInfos[root];
    llvm::ValueToValueMapTy valueMap;
    std::unique_ptr<llvm::Module> splitModule(llvm::CloneModule(
        mainModule, valueMap, [&](const llvm::GlobalValue *globalVal) {
          return globalVal == root || valueInfo.dependencies.count(globalVal);
        }));
    if (splitModule->empty())
      splitModule->setModuleInlineAsm("");

    // Module cloning creates stubs for every function and global in the
    // original module, even if they aren't used in this slice. Kill all of
    // these off to make the module more self-contained.
    for (auto &func : llvm::make_early_inc_range(*splitModule))
      if (func.isDeclaration() && func.use_empty())
        func.eraseFromParent();
    for (auto &globalVar : llvm::make_early_inc_range(splitModule->globals())) {
      if (globalVar.isDeclaration() && globalVar.use_empty())
        globalVar.eraseFromParent();
    }

    splitModules.emplace_back(std::move(splitModule));

    // Record the split values.
    splitValues.insert(root);
    splitValues.insert(valueInfo.dependencies.begin(),
                       valueInfo.dependencies.end());
  };

  for (auto &global : mainModule.globals()) {
    if (global.hasInternalLinkage())
      continue;
    // TODO: Add special handling for `llvm.global_ctors` and
    // `llvm.global_dtors`, because otherwise they end up tying almost all
    // symbols into the same split.
    splitValue(&global);
  }
  for (auto &fn : mainModule.functions())
    if (!fn.isDeclaration() && (fn.hasExternalLinkage() || fn.hasWeakLinkage()))
      splitValue(&fn);

  // If we had no functions to split, just process the main module.
  if (splitModules.empty())
    return;

  // Order the split modules by size. This allows for other threads to start
  // processing the longer compilations first.
  llvm::sort(splitModules, [](const std::unique_ptr<llvm::Module> &lhs,
                              const std::unique_ptr<llvm::Module> &rhs) {
    return lhs->size() > rhs->size();
  });
}

/// Collect all of the immediate global value users of `value`.
void LLVMModuleSplitterImpl::collectValueUsers(const llvm::Value *value) {

  llvm::SmallVector<const llvm::User *> worklist(value->users());

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
    if (!global->isConstant())
      valueInfos[value].canBeSplit = false;
}

/// Propagate use information through the module.
void LLVMModuleSplitterImpl::propagateUseInfo() {
  std::vector<ValueInfo *> worklist;
  // Each value depends on itself. Seed the iteration with that.
  for (auto &[value, info] : valueInfos) {
    info.dependencies.insert(value);
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
      // If there is a change, add the user info to the worklist.
      if (llvm::set_union(userInfo.dependencies, info->dependencies))
        changed = true;

      // If the value cannot be split, its users cannot be split either.
      if (!info->canBeSplit && userInfo.canBeSplit) {
        userInfo.canBeSplit = false;
        changed = true;
        // If a value cannot be split, its users are also its dependencies.
        llvm::set_union(userInfo.dependencies, userInfo.users);
      }

      if (changed)
        worklist.push_back(&userInfo);
    }

    if (info->canBeSplit)
      continue;
    // If a value cannot be split, propagate its dependencies up to its
    // dependencies.
    for (const llvm::Value *dep : info->dependencies) {
      ValueInfo &depInfo = valueInfos.find(dep)->second;
      if (info == &depInfo)
        continue;
      if (llvm::set_union(depInfo.dependencies, info->dependencies))
        worklist.push_back(&depInfo);
    }
  }
}

LLVMModulePerFunctionSplitterImpl::LLVMModulePerFunctionSplitterImpl(
    std::unique_ptr<llvm::Module> module, size_t parallelismLevel)
    : mainModule(std::move(module)), parallelismLevel(parallelismLevel) {}

/// Split the LLVM module into multiple modules using the provided process
/// function.
void LLVMModulePerFunctionSplitterImpl::split(
    llvm::function_ref<void(std::unique_ptr<llvm::Module>, size_t idx, bool)>
        processFn) {
  // Compute the value info for each global in the module.
  auto computeUsers = [&](auto &value) { collectValueUsers(&value); };
  llvm::for_each(mainModule->functions(), computeUsers);
  llvm::for_each(mainModule->globals(), computeUsers);
  llvm::for_each(mainModule->aliases(), computeUsers);

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

  for (auto [idx, value] : llvm::enumerate(toSplit)) {
    // Synchronize incrementally to reduce pressuring memory to much for holding
    // the cloned llvm modules.
    bool sync = ((idx + 1) % (2 * parallelismLevel) == 0);
    if (std::unique_ptr<llvm::Module> splitModule = splitValue(value))
      processFn(std::move(splitModule), totalSplit++, sync);
  }

  // Make sure sync happens when all values are processed.
  processFn(nullptr, totalSplit, true);

  // If we had no functions to split, just process the main module.
  if (totalSplit == 0)
    return processFn(std::move(mainModule), -1, true);
}

// TODO(#33945) Special handling of coroutine related globals (hack, hack).
// They and their users cannot split and duplicate into different modules,
// otherwise coroutine related pointer offset will be wrong and causes crash.
// These globals have names like:
// xxx_async_closure_0_af, or xxx_async_closure_1_afp.
// Hardcode the name matching pattern here as a hack.
// Resolve this properly with #33945.
static bool isAsyncClosureDependency(llvm::StringRef name) {
  std::pair<StringRef, StringRef> p1 = name.rsplit('_');
  if (p1.second != "af" || p1.second != "afp")
    return false;
  p1 = p1.first.rsplit('_');
  size_t asyncClosureNo = 0;
  if (!llvm::to_integer<size_t>(p1.second, asyncClosureNo))
    return false;
  p1 = p1.first.rsplit('_');
  if (p1.second != "closure")
    return false;
  p1 = p1.first.rsplit('_');
  if (p1.second != "async")
    return false;
  return true;
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
      if (const auto *callBase = dyn_cast<llvm::CallBase>(userIt)) {
        if (!callBase->getCalledFunction())
          continue;
        StringRef calledFuncName = callBase->getCalledFunction()->getName();
        if (coroCalledNamesCache.contains(calledFuncName)) {
          for (unsigned i = 0; i < callBase->arg_size(); i++) {
            if (auto argFunc =
                    dyn_cast<llvm::Function>(callBase->getArgOperand(i))) {

              // Add these functions as true dependency so that
              // llvm::CoroSplit won't fail.
              valueInfos[argFunc].canBeSplit = false;
            }
          }
        } else if (isAsyncClosureDependency(calledFuncName)) {
          valueInfos[userIt].canBeSplit = false;
        } else if (kgenCoroNamesCache.contains(calledFuncName)) {
          valueInfos[callBase->getCalledFunction()].canBeSplit = false;
          valueInfos[userIt].dependencies.insert(callBase->getCalledFunction());
        }
      }

    } else if (const auto *globalVal = dyn_cast<llvm::GlobalValue>(userIt)) {
      valueInfos[value].users.insert(globalVal);
      valueInfos[globalVal];
    } else {
      llvm_unreachable("unexpected user of global value");
    }
  }

  // If the current value is a mutable global variable, then it can't be
  // split.
  if (auto *global = dyn_cast<llvm::GlobalVariable>(value)) {
    if (!global->isConstant()) {
      valueInfos[value].canBeSplit = false;
    } else {
      bool isAsyncDep = isAsyncClosureDependency(global->getName());
      valueInfos[value].canBeSplit = !isAsyncDep;
    }
  }
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
