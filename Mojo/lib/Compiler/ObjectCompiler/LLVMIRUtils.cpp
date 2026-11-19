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

#include "Mojo/Compiler/LLVMIRUtils.h"
#include "Mojo/Compiler/Target/TargetBackend.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Mojo/TransformUtils/SplitPlanner.h"
#include "Support/Buffer.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/GlobalStatus.h"
#include "llvm/Transforms/Utils/SplitModule.h"

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

void LLVMModuleAndContext::reset() {
  module.reset();
  ctx.reset();
}

//===----------------------------------------------------------------------===//
// StringConstantTable
//===----------------------------------------------------------------------===//

namespace {
/// Large strings are very inefficiently encoded in LLVM bitcode (each `char` is
/// encoded as a `uint64_t`). The LLVM bitcode reader is also very inefficiently
/// reads strings back, performing 3 ultimate copies of the data. This is made
/// worse by the fact the `getLazyBitcodeModule` does not lazily parse constants
/// from the LLVM bitcode. Thus, when per-function splitting a module with N
/// functions and M large string constants, we form 3*M*N copies of the large
/// strings.
///
/// This class is part of a workaround of this inefficiency. When processing a
/// module for splitting, we track any string global constants and their indices
/// in this table. If a module is going to be roundtripped through bitcode to be
/// lazily loaded, we externalize the strings by setting the corresponding
/// constants to `zeroinitializer` in the module before it is written to
/// bitcode. As we materialize constants on the other side, we check for a
/// materialized global variable that matches an entry in the string table and
/// directly copy the data over into the new LLVM context.
///
/// We can generalize this optimization to other large data types as necessary.
///
/// This class is used in an `RCRef` to be shared across multiple threads.
class StringConstantTable : public ReferenceCounted<StringConstantTable> {
  /// An entry in the string table consists of a global variable, its module
  /// index, and the a reference to the string data. Because the string data is
  /// owned by the original LLVM context, we have to ensure it stays alive.
  struct Entry {
    unsigned idx;
    const llvm::GlobalVariable *var;
    StringRef value;
  };

public:
  /// If `value` denotes a string constant, record the data at index `gvIdx`.
  void recordIfStringConstant(unsigned gvIdx, const llvm::GlobalValue &value) {
    auto var = dyn_cast<llvm::GlobalVariable>(&value);
    if (var && var->isConstant() && var->hasInternalLinkage()) {
      auto *init =
          dyn_cast<llvm::ConstantDataSequential>(var->getInitializer());
      if (init && init->isCString())
        stringConstants.push_back(Entry{gvIdx, var, init->getAsString()});
    }
  }

  /// Before writing the main module to bitcode, externalize large string
  /// constants by stubbing out their values. Take ownership of the main module
  /// so the string data stays alive.
  llvm::Module &externalizeStrings(LLVMModuleAndContext &&module) {
    mainModule = std::move(module);
    // Stub the initializers. The global variable is an internal constant, so it
    // must have an initializer.
    for (Entry &e : stringConstants) {
      auto *stub =
          llvm::Constant::getNullValue(e.var->getInitializer()->getType());
      // `const_cast` is OK because we own the module now.
      const_cast<llvm::GlobalVariable *>(e.var)->setInitializer(stub);
    }
    return *mainModule;
  }

  /// This is an iterator over the entries in the string table.
  class Injector {
    using const_iterator = std::vector<Entry>::const_iterator;

  public:
    /// Given a global variable in a materialized module and its index, if it is
    /// a string constant found in the table, copy the data over into the new
    /// LLVM context and set the initializer.
    void materializeIfStringConstant(unsigned gvIdx,
                                     llvm::GlobalVariable &var) {
      while (it != e && it->idx < gvIdx)
        ++it;
      if (it == e || it->idx != gvIdx)
        return;
      var.setInitializer(llvm::ConstantDataArray::getString(
          var.getType()->getContext(), it->value, /*AddNull=*/false));
    }

  private:
    explicit Injector(const_iterator it, const_iterator e) : it(it), e(e) {}

    const_iterator it, e;

    friend class StringConstantTable;
  };

  Injector begin() const {
    return Injector(stringConstants.begin(), stringConstants.end());
  }

private:
  std::vector<Entry> stringConstants;
  LLVMModuleAndContext mainModule;
};

//===----------------------------------------------------------------------===//
// Module Splitter
//===----------------------------------------------------------------------===//

/// The global values of an LLVM module as `SplitSymbol`s. Each splitter
/// classifies the symbols for its planner.
struct LLVMIRSplitGraph {
  /// Numbers the global values in the order that
  /// `readAndMaterializeDependencies` visits them, and records the string
  /// constants in `strtab`.
  LLVMIRSplitGraph(const llvm::Module &module, StringConstantTable &strtab);

  std::vector<const llvm::GlobalValue *> values;
  std::vector<SplitSymbol> symbols;
};

/// The module that the slices materialize from, written to bitcode once.
class SliceMaterializer {
public:
  SliceMaterializer(LLVMModuleAndContext module,
                    RCRef<StringConstantTable> strtab);

  /// Returns a producer of the module that defines `members`, which are
  /// sorted indices into `LLVMIRSplitGraph::values`.
  llvm::unique_function<LLVMModuleAndContext()>
  materialize(std::vector<unsigned> members) const;

private:
  WriteableBufferRef buf;
  RCRef<StringConstantTable> strtab;
};
} // namespace

LLVMIRSplitGraph::LLVMIRSplitGraph(const llvm::Module &module,
                                   StringConstantTable &strtab) {
  llvm::DenseMap<const llvm::GlobalValue *, unsigned> indexOf;
  auto add = [&](const llvm::GlobalValue &value) {
    indexOf[&value] = values.size();
    values.push_back(&value);
  };
  llvm::for_each(module.globals(), add);
  llvm::for_each(module.functions(), add);
  // Aliases and ifuncs come after the values that
  // `readAndMaterializeDependencies` materializes. They get an index because
  // they use other global values.
  llvm::for_each(module.aliases(), add);
  llvm::for_each(module.ifuncs(), add);
  symbols.resize(values.size());

  // The use-def list is sparse. Walk it from each global value to the global
  // values that use it.
  unsigned numMaterialized = module.global_size() + module.size();
  SmallVector<const llvm::User *> worklist;
  for (unsigned idx = 0; idx != numMaterialized; ++idx) {
    const llvm::GlobalValue *value = values[idx];
    strtab.recordIfStringConstant(idx, *value);
    worklist.assign(value->user_begin(), value->user_end());
    while (!worklist.empty()) {
      const llvm::User *user = worklist.pop_back_val();
      if (isa<llvm::Constant>(user) && !isa<llvm::GlobalValue>(user)) {
        worklist.append(user->user_begin(), user->user_end());
        continue;
      }
      const llvm::GlobalValue *dependent;
      if (const auto *inst = dyn_cast<llvm::Instruction>(user))
        dependent = inst->getFunction();
      else
        dependent = cast<llvm::GlobalValue>(user);
      symbols[indexOf.at(dependent)].dependencies.push_back(idx);
    }
  }
  // Each list is already sorted, because `idx` grows.
  for (SplitSymbol &symbol : symbols)
    symbol.dependencies.erase(llvm::unique(symbol.dependencies),
                              symbol.dependencies.end());
}

static LLVMModuleAndContext
readAndMaterializeDependencies(BufferRef buf, ArrayRef<unsigned> members,
                               const StringConstantTable &strtab) {
  assert(llvm::is_sorted(members) && "slice members are not in index order");

  // First, create a lazy module with an internal bitcode materializer.
  // TODO: Not sure how to make lazy loading metadata work.
  LLVMModuleAndContext result;
  {
    CompilerTimeTraceScope traceScope("createLLVMContext");
    (void)result.create([&](llvm::LLVMContext &ctx) {
      CompilerTimeTraceScope traceScope("getLazyBitcodeModule");
      return llvm::cantFail(llvm::getLazyBitcodeModule(
          llvm::MemoryBufferRef(buf->getBuffer(), "<split-module>"), ctx,
          /*ShouldLazyLoadMetadata=*/false));
    });
    result->removeModuleInlineAsm();
  }

  auto idxIt = members.begin();
  auto idxEnd = members.end();

  // The global value indices go from globals, functions, then aliases. This
  // mirrors the order in which global values are deleted by LLVM's GlobalDCE.
  unsigned curIdx = 0;
  StringConstantTable::Injector it = strtab.begin();
  for (llvm::GlobalVariable &global : result->globals()) {
    if (idxIt != idxEnd && curIdx == *idxIt) {
      ++idxIt;
      llvm::cantFail(global.materialize());
      it.materializeIfStringConstant(curIdx, global);
    } else {
      global.setInitializer(nullptr);
      global.setComdat(nullptr);
      global.setLinkage(llvm::GlobalValue::ExternalLinkage);
      // External link should not be DSOLocal anymore,
      // otherwise position independent code generates
      // `R_X86_64_PC32` instead of `R_X86_64_REX_GOTPCRELX`
      // for these symbols and building shared library from
      // a static archive of this module will error with an `fPIC` confusion.
      global.setDSOLocal(false);
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
      // External link should not be DSOLocal anymore,
      // otherwise position independent code generates
      // `R_X86_64_PC32` instead of `R_X86_64_REX_GOTPCRELX`
      // for these symbols and building shared library from
      // a static archive of this module will error with an `fPIC` confusion.
      // External link should not be DSOLocal anymore,
      // otherwise position independent code generation get confused.
      func.setDSOLocal(false);
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

SliceMaterializer::SliceMaterializer(LLVMModuleAndContext module,
                                     RCRef<StringConstantTable> strtab)
    : buf(WriteableBuffer::get()), strtab(std::move(strtab)) {
  CompilerTimeTraceScope traceScope("writeMainModuleBitcode");
  llvm::Module &main = this->strtab->externalizeStrings(std::move(module));
  llvm::WriteBitcodeToFile(main, *buf);
}

llvm::unique_function<LLVMModuleAndContext()>
SliceMaterializer::materialize(std::vector<unsigned> members) const {
  return [members = std::move(members), buf = BufferRef(buf.copy()),
          strtab = strtab.copy()]() mutable {
    return readAndMaterializeDependencies(std::move(buf), members, *strtab);
  };
}

/// support for splitting an LLVM module into multiple parts using exported
/// functions as anchors, and pull in all dependency on the call stack into one
/// module.
void KGEN::splitPerExported(
    LLVMModuleAndContext module, LLVMSplitProcessFn processFn,
    llvm::StringMap<llvm::GlobalValue::LinkageTypes> &symbolLinkageTypes,
    bool outlineSharedSubtrees, unsigned threads) {
  CompilerTimeTraceScope traceScope("splitPerExported");
  auto strtab = RCRef<StringConstantTable>::create();
  LLVMIRSplitGraph graph(*module, *strtab);

  // The target backend, used to recognize shared-memory globals (which aren't
  // split anchors). Null for unregistered targets, in which case nothing is
  // treated as shared memory.
  ErrorOr<const TargetBackend *> backendOr =
      TargetBackendRegistry::get().lookup(module->getTargetTriple());
  const TargetBackend *backend = backendOr.isError() ? nullptr : *backendOr;

  for (auto [value, symbol] : llvm::zip_equal(graph.values, graph.symbols)) {
    symbol.name = value->getName();
    if (const auto *fn = dyn_cast<llvm::Function>(value)) {
      symbol.isFunction = true;
      symbol.isDeclaration = fn->isDeclaration();
      symbol.isAnchor = !fn->isDeclaration() &&
                        (fn->hasExternalLinkage() || fn->hasWeakLinkage());
      // A slice that holds only the declaration of an `alwaysinline` function
      // has an inline that the AlwaysInliner cannot satisfy.
      symbol.canOutline = !fn->isDeclaration() && fn->hasInternalLinkage() &&
                          !fn->hasFnAttribute(llvm::Attribute::AlwaysInline);
      if (outlineSharedSubtrees)
        symbol.instructionCount = fn->getInstructionCount();
    } else if (const auto *global = dyn_cast<llvm::GlobalVariable>(value)) {
      // Each slice can hold its own copy of a global in shared memory (see
      // `TargetBackend::isSharedMemoryGlobal`).
      bool isSharedMemory = backend && backend->isSharedMemoryGlobal(*global);
      symbol.isDeclaration = global->isDeclaration();
      symbol.isAnchor = !global->hasLocalLinkage() && !isSharedMemory;
      symbol.tiesUsers = !global->isConstant() && !isSharedMemory;
    }
  }

  SplitPlan plan = planSplitPerExported(
      graph.symbols, outlineSharedSubtrees ? std::max(2u, threads) : 0);
  if (plan.slices.size() <= 1)
    return processFn(forwardModule(std::move(module)), std::nullopt,
                     /*numFunctionBase=*/0);

  // An outlined root needs a definition the others can resolve. Record the
  // original linkage first; `MCLinker` restores it, so nothing reaches the
  // final object as anything but internal.
  //
  // WeakAny, which the per-function splitter uses, is interposable and blocks
  // inlining even in the module that owns the definition. WeakODR does not.
  for (unsigned idx : plan.outlineRoots) {
    auto *fn =
        const_cast<llvm::Function *>(cast<llvm::Function>(graph.values[idx]));
    symbolLinkageTypes.insert({fn->getName().str(), fn->getLinkage()});
    fn->setLinkage(llvm::GlobalValue::WeakODRLinkage);
  }

  SliceMaterializer materializer(std::move(module), std::move(strtab));
  unsigned numFunctions = 0;
  for (auto [idx, slice] : llvm::enumerate(plan.slices)) {
    unsigned next = numFunctions + slice.size();
    processFn(materializer.materialize(std::move(slice)), idx, numFunctions);
    numFunctions = next;
  }
}

/// Drops from `slice` the functions that an earlier slice defines and the
/// functions in `duplicatedFns`.
static void removeDuplicates(std::vector<unsigned> &slice,
                             ArrayRef<const llvm::GlobalValue *> values,
                             llvm::StringSet<> &seenFns,
                             const llvm::StringSet<> &duplicatedFns) {
  llvm::erase_if(slice, [&](unsigned idx) {
    const auto *fn = dyn_cast<llvm::Function>(values[idx]);
    if (!fn)
      return false;
    StringRef name = fn->getName();
    return !seenFns.insert(name).second || duplicatedFns.contains(name);
  });
}

static unsigned
getNumFunctionsInSlice(ArrayRef<unsigned> slice,
                       ArrayRef<const llvm::GlobalValue *> values) {
  return llvm::count_if(slice, [&](unsigned idx) {
    const auto *fn = dyn_cast<llvm::Function>(values[idx]);
    return fn && !fn->isDeclaration();
  });
}

/// support for splitting an LLVM module into multiple parts with each part
/// contains only one function (with exception for coroutine related functions.)
void KGEN::splitPerFunction(
    LLVMModuleAndContext module, LLVMSplitProcessFn processFn,
    llvm::StringMap<llvm::GlobalValue::LinkageTypes> &symbolLinkageTypes,
    int64_t inputModuleIdx, unsigned numFunctionBase) {
  CompilerTimeTraceScope traceScope("splitPerFunction");
  auto strtab = RCRef<StringConstantTable>::create();
  LLVMIRSplitGraph graph(*module, *strtab);

  // We split the module per function and cloning any necessary dependencies:
  // - For function dependencies, only clone the declaration unless its
  //   coroutine related.
  // - For other internal values, clone as is.
  // This is much fine-grained splitting, which enables significantly higher
  // levels of parallelism (and smaller generated artifacts).
  // LLVM LTO style optimization may suffer a bit here since we don't have
  // the full callstack present anymore in each cloned module.
  for (auto [value, symbol] : llvm::zip_equal(graph.values, graph.symbols)) {
    if (const auto *fn = dyn_cast<llvm::Function>(value)) {
      symbol.isFunction = true;
      symbol.isDeclaration = fn->isDeclaration();
      symbol.isAnchor = !fn->isDeclaration();
      continue;
    }
    const auto *global = dyn_cast<llvm::GlobalVariable>(value);
    if (!global)
      continue;
    symbol.isDeclaration = global->isDeclaration();
    // TODO: Add special handling for `llvm.global_ctors` and
    // `llvm.global_dtors`, because otherwise they end up tying almost all
    // symbols into the same split.
    symbol.isAnchor =
        !global->hasLocalLinkage() && !global->hasExternalLinkage();

    // A mutable global ties its users into one slice.
    // [MOCO-1680] So does a private symbol. A slice that declares it external
    // may reach it through the GOT (global offset table). The GOT has no entry
    // for a private symbol, so the link fails with "undefined symbol".
    // Internal linkage would fix this, but the number of slices that use the
    // symbol is not known yet.
    symbol.tiesUsers = !global->isConstant() || global->hasPrivateLinkage();
    LLVM_DEBUG(llvm::dbgs()
               << "global " << global->getName()
               << (symbol.tiesUsers ? " cannot be split" : " can be split")
               << '\n');
  }
  SplitPlan plan = planSplitPerFunction(graph.symbols);

  // This is a set of special private symbol name that need to be renamed (e.g.,
  // those inserted by llvm optimizer) in order to avoid symbol override during
  // linking. Since each split goes through its own LLVM optimization, private
  // symbol with the same name (but different values) could be inserted by LLVM
  // (e.g., when detecting the same optimization pattern the transformation
  // leads to an insertion of global).
  // We have to do the renaming now before codegen because we need to ensure
  // name consistency between MCSymbols and global symbol names.
  //
  // Explanation on the entries:
  // - ".memset_pattern": MacOS specific, introduced by `LoopIdiomRecognize`.
  const llvm::StringSet<> namesToRename{".memset_pattern"};

  unsigned unnamedGlobal = numFunctionBase;
  for (llvm::GlobalVariable &global : module->globals()) {
    if (!global.hasInternalLinkage() && !global.hasPrivateLinkage())
      continue;
    if (!global.hasName()) {
      // A unique name keeps MCLink on X86 from confusing the unnamed globals
      // of different slices, which can hold different values. ASan builds
      // insert such globals.
      global.setName("__mojo_unnamed" + Twine(unnamedGlobal++));
    }

    for (StringRef prefix : namesToRename.keys()) {
      if (global.getName().starts_with(prefix)) {
        global.setName(global.getName() + "_" + Twine(inputModuleIdx));
        break;
      }
    }

    symbolLinkageTypes.insert({global.getName().str(), global.getLinkage()});
    global.setLinkage(llvm::GlobalValue::WeakAnyLinkage);
  }

  for (llvm::Function &fn : module->functions()) {
    // Avoid renaming when linking in MCLink.
    if (!fn.isDeclaration() &&
        (fn.hasInternalLinkage() || fn.hasPrivateLinkage())) {
      symbolLinkageTypes.insert({fn.getName().str(), fn.getLinkage()});
      fn.setLinkage(llvm::Function::LinkageTypes::WeakAnyLinkage);
    }
  }

  if (plan.slices.size() <= 1)
    return processFn(forwardModule(std::move(module)), std::nullopt,
                     numFunctionBase);

  llvm::StringSet<> duplicatedFns = std::move(module.duplicatedFns);
  SliceMaterializer materializer(std::move(module), std::move(strtab));
  unsigned numFunctions = numFunctionBase;
  llvm::StringSet<> seenFns;
  for (auto [idx, slice] : llvm::enumerate(plan.slices)) {
    // Giving each function a unique ID across all splits for proper MC level
    // linking and codegen into one object file where duplicated functions
    // in each split will be deduplicated (with the linking).
    removeDuplicates(slice, graph.values, seenFns, duplicatedFns);
    if (slice.empty())
      continue;

    unsigned next = numFunctions + getNumFunctionsInSlice(slice, graph.values);

    LLVM_DEBUG(llvm::dbgs() << "split function base id: " << numFunctions
                            << " set size: " << slice.size() << "\n";);

    processFn(materializer.materialize(std::move(slice)), idx, numFunctions);
    numFunctions = next;
  }
}
