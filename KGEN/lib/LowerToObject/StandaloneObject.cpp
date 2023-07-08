//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LowerToObject.h"

#include "KGEN/KGENVersion/KGENVersion.h"
#include "KGEN/POPDialect/POPOps.h"
#include "LLCL/CompilerSupport/MLIRLocationDecoder.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LowerToObjectImpl.h"
#include "Support/Compiler/MLIRDenseAttrStorage.h"
#include "Support/FileSystemExtras.h"
#include "Support/TimeProfiler.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/SplitModule.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <utility>

#define DEBUG_TYPE "standalone-object"

using namespace M;
using namespace KGEN;
using namespace Cache;

//===----------------------------------------------------------------------===//
// produceStandaloneModule
//===----------------------------------------------------------------------===//

/// Slice the dependencies of an operation out of the existing module into the
/// self-contained slice module.
static void sliceDependencies(Operation *op, SymbolTable &sliceSymtab,
                              const SymbolTable &symtab) {
  // Extract a dependency from the IR parent module and place it into the slice
  // module if it does not already exist. If a symbol was copied, return it.
  auto extractDependency = [&](StringAttr name) -> Operation * {
    // Don't copy the symbol if it is already copied.
    if (sliceSymtab.lookup(name))
      return nullptr;

    Operation *symbol = symtab.lookup(name);
    // If the symbol reference attribute doesn't reference a symbol, somehow
    // invalid IR made it to the ObjectCompiler.
    assert(symbol && "invalid IR?");

    // Clone the symbol into the new symbol table.
    Operation *copy = symbol->clone();
    sliceSymtab.insert(copy);
    return copy;
  };

  mlir::AttrTypeWalker walker;
  walker.addWalk([&](FlatSymbolRefAttr ref) {
    if (Operation *decl = extractDependency(ref.getAttr()))
      sliceDependencies(decl, sliceSymtab, symtab);
  });
  auto extractDependencies = [&](Operation *op) {
    // Extract references to type declarations.
    walker.walk(op->getAttrDictionary());
    for (Type type : op->getResultTypes())
      walker.walk(type);
    for (Region &region : op->getRegions())
      for (Type type : region.getArgumentTypes())
        walker.walk(type);
  };
  op->walk(extractDependencies);
}

OwningOpRef<ModuleOp>
ObjectCompiler::produceStandaloneModule(const SymbolTable &symtab,
                                        const ExportMap &exportedSymbols) {
  auto module = cast<ModuleOp>(symtab.getOp());
  // Create a new module for these funcs. This will go away at the end
  // of this function.
  OwningOpRef<ModuleOp> singleModule = ModuleOp::create(module->getLoc());

  // Propagate the target info.
  TargetInfoAttr target = getTargetInfo(module);
  assert(target && "module to compile is missing target specification");
  setTargetInfo(*singleModule, target);

  // Create a new symbol table for the sliced module.
  SymbolTable sliceSymtab(*singleModule);

  // Re-export exported functions.
  auto builder = OpBuilder::atBlockBegin(singleModule->getBody());

  for (auto [sym, exportVal] : exportedSymbols) {
    builder.create<ExportOp>(module->getLoc(), FlatSymbolRefAttr::get(sym),
                             exportVal.alias, exportVal.isCExport);
    auto func = symtab.lookup<FuncOp>(sym);
    assert(func && "Unknown exported symbol");

    // Traverse the call graph and clone all the callees into this module.
    sliceDependencies(func, sliceSymtab, symtab);

    // Clone the func into this new module. We don't want to remove it from
    // the current module.
    if (!sliceSymtab.lookup(sym))
      sliceSymtab.insert(func.clone());
  }

  return singleModule;
}

//===----------------------------------------------------------------------===//
// Module Splitter
//===----------------------------------------------------------------------===//

namespace {
/// This class provides support for splitting an LLVM module into multiple
/// parts.
class LLVMModuleSplitter {
public:
  LLVMModuleSplitter(llvm::Module &module) : mainModule(module) {}

  /// Split the LLVM module into multiple modules using the provided process
  /// function.
  void split(function_ref<void(llvm::Module &)> processFn) {
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
    DenseSet<const llvm::Value *> splitValues;
    SmallVector<std::unique_ptr<llvm::Module>> splitModules;
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
      for (auto &globalVar :
           llvm::make_early_inc_range(splitModule->globals())) {
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
    for (auto &fn : mainModule.functions()) {
      if (fn.isDeclaration() || !fn.hasExternalLinkage())
        continue;
      splitValue(&fn);
    }

    // If we had no functions to split, just process the main module.
    if (splitModules.empty())
      return processFn(mainModule);

    // Order the split modules by size. This allows for other threads to start
    // processing the longer compilations first.
    llvm::sort(splitModules, [](const std::unique_ptr<llvm::Module> &lhs,
                                const std::unique_ptr<llvm::Module> &rhs) {
      return lhs->size() > rhs->size();
    });
    for (auto &splitModule : splitModules)
      processFn(*splitModule);
  }

private:
  struct ValueInfo {
    bool canBeSplit = true;
    llvm::SmallPtrSet<const llvm::Value *, 4> dependencies;
    llvm::SmallPtrSet<const llvm::Value *, 4> users;
  };

  /// Collect all of the immediate global value users of `value`.
  void collectValueUsers(const llvm::Value *value) {
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
      if (!global->isConstant())
        valueInfos[value].canBeSplit = false;
  }

  /// Propagate use information through the module.
  void propagateUseInfo() {
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

  /// The main LLVM module being split.
  llvm::Module &mainModule;

  /// The value info for each global value in the module.
  DenseMap<const llvm::Value *, ValueInfo> valueInfos;
};
} // namespace

/// For each external call, map the link directive to the set of functions it
/// imports. This allows us to pull out only the objects we need from the linked
/// library when it's an archive, for example.
static void
collectLinksAndUsers(ModuleOp theModule, const SymbolTable &symtab,
                     DenseMap<Attribute, llvm::StringSet<>> &linksAndUsers) {
  auto addLinkOp = [&](SymbolRefAttr linkRef, StringRef user) {
    auto link =
        symtab.lookup<LinkOp>(cast<FlatSymbolRefAttr>(linkRef).getAttr());
    assert(link && "There wasn't a valid LinkOp?");
    if (auto path = link.getLinkPathAttr())
      linksAndUsers[path].insert(user);
    else if (auto bytes = link.getLinkBytesAttr())
      linksAndUsers[bytes].insert(user);
    else
      llvm::report_fatal_error("unknown link directive");
  };

  // Get all LinkOps that were actually used. They're always referenced from
  // kgen.extern.func.
  for (auto func : theModule.getOps<ExternFuncOp>()) {
    SymbolRefAttr linkRef = func.getImportedFrom();
    assert(linkRef && "kgen.extern.func must have an importedFrom");
    addLinkOp(linkRef, func.getSymName());
  }
}

/// Given a binary in `bufferRef`, add all the required pieces of it to the list
/// of archive members. This is effectively a very dumb static linker.
static ErrorOrSuccess
addBinaryToArchive(llvm::MemoryBufferRef bufferRef,
                   const llvm::StringSet<> &users,
                   SmallVectorImpl<llvm::NewArchiveMember> &archiveMembers) {
  // Create the binary.
  auto binaryOr = llvm::object::createBinary(bufferRef);
  if (!binaryOr)
    return Error(llvm::toString(binaryOr.takeError()));
  std::unique_ptr<llvm::object::Binary> binary = std::move(*binaryOr);

  // TODO: Ensure the binary is the correct type.

  // If the binary is an object file, add it to the archive members directly.
  if (binary->isObject()) {
    archiveMembers.emplace_back(bufferRef);
    return success();
  }

  // Otherwise, expand the archive and add its children to the new member list.
  auto *archive = cast<llvm::object::Archive>(binary.get());

  // In theory, we could inspect the archive to find the members we actually
  // need and only add those, but that's a pretty big chunk of the job of a
  // linker. We do correctness checking here, to ensure the symbols we expect to
  // find are in the archive. Don't use `findSym` in a loop here because it
  // basically just iterates all the symbols in the archive for each call, so we
  // inline the logic here.
  llvm::StringSet<> foundSymbols;
  for (auto symbol : archive->symbols()) {
    StringRef symName = symbol.getName().ltrim('_');
    if (users.contains(symName))
      foundSymbols.insert(symName);
  }

  // Set difference computes A - B, so put `users` first since `foundSymbols` is
  // a strict subset.
  llvm::StringSet<> missingSymbols = llvm::set_difference(users, foundSymbols);
  if (!missingSymbols.empty()) {
    std::string tmp;
    llvm::raw_string_ostream stream(tmp);
    stream << "could not find definitions for symbols: [";
    llvm::interleaveComma(missingSymbols, stream,
                          [&](const auto &p) { stream << p.getKey(); });
    stream << "]";
    return Error(tmp);
  }

  // Add all the children in the archive to the new output archive.
  llvm::Error err = llvm::Error::success();
  for (auto &child : archive->children(err)) {
    if (err)
      return Error(llvm::toString(std::move(err)));

    auto refOr = child.getMemoryBufferRef();
    if (!refOr)
      return Error(llvm::toString(refOr.takeError()));

    LLVM_DEBUG(llvm::dbgs() << "Adding object to archive: '"
                            << refOr->getBufferIdentifier() << "'\n");
    archiveMembers.emplace_back(*refOr);
  }
  // Handle all errors - we didn't hit anything.
  llvm::handleAllErrors(std::move(err));

  return success();
}

/// Open the static archive or object specified by the link directive, and add
/// relevant archive members to the provided list.
static ErrorOrSuccess handleLinkPathDirective(
    StringAttr linkPath, const llvm::StringSet<> &users,
    llvm::SourceMgr &linkMgr, DenseSet<Attribute> &processed,
    SmallVectorImpl<llvm::NewArchiveMember> &archiveMembers) {
  // Only pull in each library exactly once.
  if (!processed.insert(linkPath).second)
    return success();

  // Pull in the library as an 'include' file. This has the lovely side effect
  // of holding onto the file for as long as we need it, so we don't have to do
  // anything after opening the file.
  std::string filepath;
  unsigned fileIdx =
      linkMgr.AddIncludeFile(linkPath.str(), llvm::SMLoc{}, filepath);
  if (fileIdx == 0)
    return Error("could not find file at " + linkPath.getValue());
  llvm::MemoryBufferRef bufferRef =
      linkMgr.getMemoryBuffer(fileIdx)->getMemBufferRef();

  return addBinaryToArchive(bufferRef, users, archiveMembers);
}

/// Take the bytes provided, interpret them as a static archive, and add the
/// archive members to the provided list.
static ErrorOrSuccess handleLinkBytesDirective(
    DenseResourceElementsAttr linkBytes, const llvm::StringSet<> &users,
    DenseSet<Attribute> &processed,
    SmallVectorImpl<llvm::NewArchiveMember> &archiveMembers) {
  // Only pull in each library exactly once.
  if (!processed.insert(linkBytes).second)
    return success();

  // Create the llvm memory buffer ref.
  ArrayRef<char> rawBytes = linkBytes.getRawHandle().getBlob()->getData();
  llvm::MemoryBufferRef byteBuffer(StringRef(rawBytes.begin(), rawBytes.size()),
                                   linkBytes.getRawHandle().getKey());

  return addBinaryToArchive(byteBuffer, users, archiveMembers);
}

//===----------------------------------------------------------------------===//
// produceStandaloneArchive
//===----------------------------------------------------------------------===//

ErrorOr<BufferRef> ObjectCompiler::produceStandaloneArchive(
    const SymbolTable &symtab, const ExportMap &exportedSymbols, bool isJIT) {
  TimeTraceScope<> traceScope("produce-standalone-archive");

  // Slice out a standalone module for the exported symbols.
  OwningOpRef<ModuleOp> slicedModule =
      produceStandaloneModule(symtab, exportedSymbols);

  // Pull out the link/external_call map. Do this while we still have the IR in
  // a recognizable state to make things simple. This map contains a mapping
  // from library import path to all the symbols imported from that library.
  DenseMap<Attribute, llvm::StringSet<>> linksAndUsers;
  collectLinksAndUsers(*slicedModule, symtab, linksAndUsers);

  // Set up a SourceMgr that we can use to find link files.
  llvm::SourceMgr linkMgr;
  linkMgr.setIncludeDirs(options.linkDirs);

  // Perform a cache aware transformation to translate the module to an archive
  // file.
  auto runTransformation = [&](Operation *op, WriteableBufferRef buf,
                               LLCL::AnyAsyncValueRef chain) {
    auto output = LLCL::AsyncValueRef<BufferRef>::allocate(runtime);
    chain.andThenSync([this, op, isJIT,
                       linksAndUsers = std::move(linksAndUsers),
                       linkMgr = std::move(linkMgr), output = output.copy(),
                       buf = buf.copy()]() mutable {
      // Lower the module to LLVM.
      llvm::LLVMContext ctx;
      auto llvmModule = lowerAllFuncsToLLVM(ctx, cast<ModuleOp>(op), isJIT);
      if (!llvmModule) {
        return std::move(output).setToError(LLCL::getMLIRDiagnostic(
            "failed to lower module to LLVM IR for archive compilation",
            op->getLoc()));
      }
      TimeTraceScope<> traceScope("split-input-module");

      // If we are saving the temp files we don't want to split.
      bool savingTemps = !options.saveTempsPrefix.empty();

      // Split the module into multiple slices and compile each in parallel.
      SmallVector<LLCL::AnyAsyncValueRef> cacheResults;
      if (runtime.getWorkQueue()->getParallelismLevel() < 2 || savingTemps) {
        cacheResults.push_back(
            lowerLLVMModuleToObject(*llvmModule, op->getLoc(), isJIT));
      } else {
        LLVMModuleSplitter splitter(*llvmModule);
        splitter.split([&](llvm::Module &inputModule) {
          cacheResults.push_back(
              lowerLLVMModuleToObject(inputModule, op->getLoc(), isJIT));
        });
      }
      andThenSyncMoving(
          cacheResults, [op, linksAndUsers = std::move(linksAndUsers),
                         linkMgr = std::move(linkMgr), buf = buf.copy(),
                         output = output.copy()](
                            MutableArrayRef<AnyAsyncValueRef> values) mutable {
            // If any of the cache results failed, propagate the error.
            for (auto &result : values)
              if (result.isError())
                return std::move(output).setToError(result.takeDiagnostic());
            TimeTraceScope<> traceScope("concatenate-object-files");

            SmallVector<llvm::NewArchiveMember> archiveMembers;

            // Process all the link directives now. We keep a set of
            // already-processed link directives, so we don't re-process
            // libraries.
            DenseSet<Attribute> processedLinks;
            for (const auto &lu : linksAndUsers) {
              Attribute link = lu.first;
              const llvm::StringSet<> &users = lu.second;
              auto err =
                  llvm::TypeSwitch<Attribute, ErrorOrSuccess>(link)
                      .Case([&](StringAttr path) {
                        return handleLinkPathDirective(path, users, linkMgr,
                                                       processedLinks,
                                                       archiveMembers);
                      })
                      .Case([&](DenseResourceElementsAttr bytes) {
                        return handleLinkBytesDirective(
                            bytes, users, processedLinks, archiveMembers);
                      })
                      .Default([](Attribute attr) {
                        llvm::report_fatal_error(
                            ("kgen.link expected bytes or path, not " +
                             mlir::debugString(attr))
                                .c_str());
                        return Error("kgen.link expected bytes or path, not " +
                                     mlir::debugString(attr));
                      });
              if (err.isError()) {
                return std::move(output).setToError(
                    LLCL::getMLIRDiagnostic(err.takeError(), op->getLoc()));
              }
            }

            // Now that all the object files have been compiled, merge them
            // all into a single archive.
            SmallVector<std::string> archiveMemberNames(values.size());
            for (auto [index, result] : llvm::enumerate(values)) {
              auto &resultBuf = result.get<BufferRef>();
              archiveMemberNames[index] = (Twine(index) + ".o").str();
              archiveMembers.emplace_back(llvm::MemoryBufferRef(
                  resultBuf->getBuffer(), archiveMemberNames[index]));
            }
            auto result = llvm::writeArchiveToBuffer(
                archiveMembers, /*WriteSymtab=*/true,
                archiveMembers.front().detectKindFromObject(),
                /*Deterministic=*/false, /*Thin=*/false);
            if (!result) {
              return std::move(output).setToError(LLCL::getMLIRDiagnostic(
                  "failed to concatenate object files into archive",
                  op->getLoc()));
            }

            // Copy the result into the output buffer.
            *buf << (*result)->getBuffer();
            std::move(output).emplace(buf.copy());
          });
    });
    return output;
  };
  auto onCacheHit = [](Operation *op, BufferRef buf) { return buf.copy(); };

  WriteableBufferRef produceStandaloneArchiveKey = WriteableBuffer::get();
  options.print(*produceStandaloneArchiveKey << "produceStandaloneArchive(");
  *produceStandaloneArchiveKey << ")";

  auto output = cachedTransform(
      *slicedModule, transformCache.copy(),
      LLCL::AsyncValueRef<Chain>::createReady(runtime),
      std::move(produceStandaloneArchiveKey), runTransformation, onCacheHit);
  await(output);

  if (output.isError())
    return {std::move(output.takeDiagnostic().getMessage())};
  return {std::move(output.get<BufferRef>())};
}

LLCL::AnyAsyncValueRef
ObjectCompiler::lowerLLVMModuleToObject(llvm::Module &module, Location loc,
                                        bool isJIT) {
  WriteableBufferRef keyBuf = WriteableBuffer::get();
  options.print(*keyBuf << "lowerLLVMModuleToObject(");
  *keyBuf << ")";
  size_t nonBitcodeKeySize = keyBuf->getBufferSize();

  // Serialize the module to bitcode to both allow for transferring to a new
  // context (LLVM isn't threadsafe), and to use in the cachedTransform key.
  {
    TimeTraceScope<> traceScope("serialize-input-module");
    llvm::WriteBitcodeToFile(module, *keyBuf);
  }

  // Perform a cached transform to compile this module slice to an object file.
  // This will enable some bare bones incremental compilation, as we will be
  // able to reuse object files for previously compiled slices.
  auto runTransformation = [this, nonBitcodeKeySize, loc, isJIT,
                            keyBuf = keyBuf.copy()](
                               WriteableBufferRef buf,
                               LLCL::AnyAsyncValueRef chain) mutable {
    auto output = LLCL::AsyncValueRef<BufferRef>::allocate(runtime);
    chain.andThenAsync([this, nonBitcodeKeySize, loc, isJIT,
                        output = output.copy(), keyBuf = std::move(keyBuf),
                        buf = buf.copy()]() mutable {
      // Extract out the bitcode from the key, as LLVM bitcode dies if the
      // buffer contains other data.
      StringRef bitcodeBuffer = ((Cache::BufferRef &)(keyBuf))->getBuffer();
      bitcodeBuffer = bitcodeBuffer.drop_front(nonBitcodeKeySize);

      // Load the cached bytecode into a new context. This is necessary to
      // avoid data races during multi-threading.
      llvm::LLVMContext ctx;
      llvm::Expected<std::unique_ptr<llvm::Module>> moduleOr =
          llvm::parseBitcodeFile(
              llvm::MemoryBufferRef(bitcodeBuffer, "<split-module>"), ctx);
      if (!moduleOr) {
        return std::move(output).setToError(
            LLCL::getMLIRDiagnostic("failed to load LLVM IR bitcode", loc));
      }
      std::unique_ptr<llvm::Module> module = std::move(*moduleOr);

      // Create the target machine.
      auto machineOr = createTargetMachine(options, isJIT);
      if (failed(machineOr)) {
        return std::move(output).setToError(
            LLCL::getMLIRDiagnostic(machineOr.takeError(), loc));
      }

      // Set the data layout on the module.
      module->setDataLayout((*machineOr)->createDataLayout());

      // Lower the LLVM to an object file.
      if (failed(compileLLVMToObject(*module, **machineOr, *buf, options))) {
        return std::move(output).setToError(LLCL::getMLIRDiagnostic(
            "failed to lower LLVM IR to object file", loc));
      }
      std::move(output).emplace(buf.copy());
    });
    return output;
  };
  auto onCacheHit = [](BufferRef buf) { return buf.copy(); };

  return cachedTransform(
      LLCL::MLIRLocationDecoder::getEncodedLocation(loc), transformCache.copy(),
      LLCL::AsyncValueRef<Chain>::createReady(runtime), keyBuf.copy(),
      std::move(runTransformation), onCacheHit);
}

ErrorOr<ElementsAttr> ObjectCompiler::produceStandaloneArchiveAttr(
    const SymbolTable &symtab, const ExportMap &exportedSymbols,
    TargetInfoAttr target, bool isJIT) {
  auto bufferOr = produceStandaloneArchive(symtab, exportedSymbols, isJIT);
  if (bufferOr.isError())
    return bufferOr.takeError();
  BufferRef buffer = bufferOr.takeValue();

  auto module = cast<ModuleOp>(symtab.getOp());

  // Get the standalone archive key to use as the archive name.
  WriteableBufferRef produceStandaloneArchiveKey = WriteableBuffer::get();
  options.print(*produceStandaloneArchiveKey << "produceStandaloneArchive(");
  *produceStandaloneArchiveKey << ")";
  if (failed(mlir::writeBytecodeToFile(module.getOperation(),
                                       *produceStandaloneArchiveKey)))
    return Error("failed to write bytecode file");
  // Hash it so the name isn't enormous.
  auto hash = llvm::BLAKE3::hash(
      ArrayRef((const uint8_t *)produceStandaloneArchiveKey->getBufferStart(),
               produceStandaloneArchiveKey->getBufferSize()));

  // Produce a DenseResourceElementsAttr from the file.
  auto resourceManager =
      DenseResourceElementsHandle::getManagerInterface(target.getContext());

  // Pretend this is a "tensor" of data.
  // TODO (#6986) It would be much nicer if we didn't have to clone this data
  //   and we could just reference the data already in the CAS. That would also
  //   prevent us from having to hash the module above.
  auto attrType = RankedTensorType::get(
      {(int64_t)buffer->getBufferSize()},
      IntegerType::get(target.getContext(), 8, IntegerType::Unsigned));
  auto attrName = "archive_" + llvm::toHex(hash, /*LowerCase=*/true);
  ArrayRef<char> blobData(buffer->getBufferStart(), buffer->getBufferSize());
  auto blob = mlir::HeapAsmResourceBlob::allocateAndCopyWithAlign(blobData,
                                                                  /*align=*/8);
  return DenseResourceElementsAttr::get(
      attrType, resourceManager.insert(attrName, std::move(blob)));
}

//===----------------------------------------------------------------------===//
// produceStandaloneAssembly
//===----------------------------------------------------------------------===//

ErrorOrSuccess
ObjectCompiler::produceStandaloneAssembly(const SymbolTable &symtab,
                                          const ExportMap &exportedSymbols,
                                          llvm::raw_pwrite_stream &os) {
  TimeTraceScope<> traceScope("produce-standalone-assembly");

  OwningOpRef<ModuleOp> slicedModule =
      produceStandaloneModule(symtab, exportedSymbols);
  llvm::LLVMContext ctx;
  auto llvmModule = lowerAllFuncsToLLVM(ctx, *slicedModule, /*isJIT=*/false);
  if (!llvmModule)
    return Error("failed to lower module to LLVM IR");

  auto machineOr = createTargetMachine(options, /*isJIT=*/false);
  if (failed(machineOr))
    return machineOr.takeError();

  // Set the data layout on the module.
  llvmModule->setDataLayout((*machineOr)->createDataLayout());

  // Emit the assembly.
  if (failed(compileLLVMToObject(*llvmModule, **machineOr, os, options,
                                 /*emitAssembly=*/true))) {
    return Error("failed to lower LLVM IR to assembly");
  }
  return success();
}
