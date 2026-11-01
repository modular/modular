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

#include "Mojo/Precompile/Precompile.h"

#include "Mojo/LITDialect/LITAttrs.h"
#include "Support/Compiler/MLIRDenseAttr.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"

#include "mlir/IR/AttrTypeSubElements.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/xxhash.h"

#include <stack>
#include <utility>

using namespace M;
using namespace M::KGEN;

/// This function takes a parsed `lit.package` op and creates  a new
/// `lit.package` op. This new `lit.package` op is one that can be serialized
/// into MLIR bytecode and written to disk, as a `.mojoc` file. The generated
/// package retains the full post-parse IR, including function bodies, in its
/// regions, filtering out ops unneeded once packaged (import ops under a
/// package, doc-string source locations). It is suitable for importing into
/// other Mojo programs, where bodies are materialized lazily on demand.
static std::pair<OwningOpRef<ModuleOp>, LIT::PackageOp>
buildPackageModule(ModuleOp theModule, LIT::PackageOp parsedPackageOp) {
  OwningOpRef<ModuleOp> packageModule =
      ModuleOp::create(parsedPackageOp->getLoc());
  auto b = OpBuilder::atBlockEnd(packageModule->getBody());

  // Clone the relevant operations into the package.
  std::stack<SmartVariant<Operation *, OpBuilder::InsertPoint>> worklist;

  // Clone an op without its regions, and ensure that once that op is finished
  // processing, we reset the OpBuilder's insert point to where it was before we
  // walked the ops inside `op`.
  auto cloneWithoutRegions = [&](auto op) {
    // Save the insert point on the worklist stack first.
    worklist.push(b.saveInsertionPoint());

    auto clonedOp = b.cloneWithoutRegions(op);
    clonedOp.getBodyRegion().push_back(new Block);
    b.setInsertionPointToStart(clonedOp.getBody());
    return clonedOp;
  };

  // Push the ops in `opList` onto the worklist, while preserving their original
  // order.
  auto pushOpsOntoWorklist = [&](auto opList) {
    // Push onto a temporary stack so we can reverse it when we put it onto the
    // worklist - this ensures the ops keep their original order.
    std::stack<Operation *> tmp;
    for (Operation &op : opList)
      tmp.push(&op);

    while (!tmp.empty()) {
      worklist.emplace(tmp.top());
      tmp.pop();
    }
  };

  // Include any generated function thunks at the top-level. These are
  // deduplicated when the package is imported.
  for (auto func : theModule.getOps<LIT::FnOp>()) {
    assert(func.getThunkKeyAttr() && "top-level function must be a thunk");
    pushOpsOntoWorklist(MutableArrayRef(*func));
  }
  for (auto trait : theModule.getOps<LIT::TraitDeclOp>()) {
    if (trait.getSymName() == "##__mojo_closure__##")
      pushOpsOntoWorklist(MutableArrayRef(*trait));
  }
  for (auto structOp : theModule.getOps<LIT::StructDeclOp>())
    pushOpsOntoWorklist(MutableArrayRef(*structOp));

  // Clone the parsed package operation and push its ops onto the worklist.
  LIT::PackageOp thePackage = cloneWithoutRegions(parsedPackageOp);
  pushOpsOntoWorklist(parsedPackageOp.getOps());

  while (!worklist.empty()) {
    auto listFront = worklist.top();
    worklist.pop();
    if (isa<OpBuilder::InsertPoint>(listFront)) {
      b.restoreInsertionPoint(cast<OpBuilder::InsertPoint>(listFront));
      continue;
    }
    TypeSwitch<Operation *>(cast<Operation *>(listFront))
        // Always clone and recurse in the case of a package, module, or struct.
        .Case<LIT::PackageOp, LIT::FileModuleOp, LIT::StructDeclOp>(
            [&](auto op) {
              cloneWithoutRegions(op);
              pushOpsOntoWorklist(op.getOps());
            })
        // None of the cases matched? Just clone the op directly.
        .Default([&](auto op) { b.clone(*op); });
  }

  // Process the package to strip out various information from the package.
  thePackage.walk([&](LIT::ASTDeclInterface astDeclOp) {
    // Strip out locations from doc strings, these aren't needed anymore.
    auto docAttr = astDeclOp.getDocStringAttr();
    if (docAttr && docAttr.getLocation())
      astDeclOp.setDocStringAttr(LIT::DocStringAttr::get(docAttr.getString()));
  });

  return std::make_pair(std::move(packageModule), thePackage);
}

/// Reads an LLVM bitcode file and returns it as a DenseResourceElementsAttr.
/// Returns nullptr on failure.
static DenseResourceElementsAttr
writeLLVMBitcodeToDenseAttr(MLIRContext *ctx, StringRef bitcodeFile) {
  // Read the bitcode file
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> bufferOr =
      llvm::MemoryBuffer::getFile(bitcodeFile);
  if (!bufferOr)
    return {};

  // Get the buffer contents
  llvm::MemoryBuffer &buffer = **bufferOr;
  StringRef data = buffer.getBuffer();

  // Hash the bitcode to generate a unique name
  std::string resourceName =
      "llvm_bitcode_" + llvm::utohexstr(xxh3_64bits(data),
                                        /*LowerCase=*/true, /*Width=*/16);

  // Create the resource attribute
  return createResourceAttr(ctx, ArrayRef<char>(data.data(), data.size()),
                            resourceName);
}

/// Collects the link dependencies of \p parsedPackageOp among the packages in
/// \p scope, whose import path is \p scopePath. Every package at an import root
/// or within a namespace, other than the one being compiled, is an imported
/// precompiled dependency (in the precompile flow, dependencies are always
/// provided as `.mojoc` files), so each is recorded by its import path. A
/// namespace only groups packages, so it is never a dependency itself.
static void
collectLinkDependencies(Region &scope, ArrayRef<StringAttr> scopePath,
                        LIT::PackageOp parsedPackageOp, StringRef sourceName,
                        StringRef outputName,
                        SmallVectorImpl<SymbolRefAttr> &dependencies) {
  for (LIT::PackageOp package : scope.getOps<LIT::PackageOp>()) {
    SmallVector<StringAttr> path(scopePath);
    path.push_back(package.getSymNameAttr());
    if (package.getIsNamespace()) {
      collectLinkDependencies(package.getBodyRegion(), path, parsedPackageOp,
                              sourceName, outputName, dependencies);
      continue;
    }
    // A package is never its own link dependency, under either its parse-time
    // (source directory) name or its output name: a recorded self-dependency
    // dangles once the package is renamed for output.
    if (package == parsedPackageOp ||
        (path.size() == 1 && (package.getSymName() == sourceName ||
                              package.getSymName() == outputName)))
      continue;
    SmallVector<FlatSymbolRefAttr> nested;
    for (StringAttr component : ArrayRef(path).drop_front())
      nested.push_back(FlatSymbolRefAttr::get(component));
    dependencies.push_back(SymbolRefAttr::get(path.front(), nested));
  }
}

ErrorOr<OwningOpRef<ModuleOp>> M::KGEN::buildPrecompiledPackage(
    ModuleOp theModule, LIT::PackageOp parsedPackageOp, StringRef sourceName,
    StringRef outputName, ArrayRef<std::string> bitcodeLibs) {
  SmallVector<SymbolRefAttr> dependencies;
  collectLinkDependencies(theModule.getBodyRegion(), /*scopePath=*/{},
                          parsedPackageOp, sourceName, outputName,
                          dependencies);
  if (!dependencies.empty()) {
    parsedPackageOp.setDependenciesAttr(
        LinkDependencyArrayAttr::get(theModule.getContext(), dependencies));
  }

  auto [packageModule, thePackage] =
      buildPackageModule(theModule, parsedPackageOp);

  // The package parses under its source directory's name so that absolute
  // self-imports resolve to the package under compilation; the output name
  // applies from here on: to symbol references and the recorded import paths
  // that name the package (both re-resolved when the package is imported,
  // under the output file's name), and to the debug-info source names that
  // snapshot the package lineage.
  if (outputName != sourceName) {
    MLIRContext *ctx = theModule.getContext();
    auto newName = StringAttr::get(ctx, outputName);
    auto sourceNameAttr = StringAttr::get(ctx, sourceName);
    // Qualified references rooted at the package resolve through the
    // parser's scoping rather than MLIR symbol tables, so rewrite them
    // directly rather than through SymbolTable::replaceAllSymbolUses.
    //
    // Only a reference's root names a package; its nested components name
    // declarations within one, and a module nested in another package may
    // share the source directory's name. Skip, or the replacer walks into
    // those components and revisits each as a reference rooted at itself.
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement(
        [&](SymbolRefAttr ref) -> std::pair<Attribute, WalkResult> {
          if (ref.getRootReference() != sourceNameAttr)
            return {ref, WalkResult::skip()};
          return {SymbolRefAttr::get(newName, ref.getNestedReferences()),
                  WalkResult::skip()};
        });
    // Debug-info source names snapshot the symbol lineage as plain strings at
    // parse time, so the package appears in them as a name, not a symbol
    // reference. Rewriting the parentless package root rebuilds every lineage
    // chained from it; these live in scoped locations, hence replaceLocs.
    // Source names for types and parameter values are rendered text, so
    // package-rooted references inside them are substrings: rewrite the
    // root token (its `::` terminator keeps longer names unmatched) wherever
    // it roots a reference. Preceded by `::` it is instead a nested component
    // of a reference rooted elsewhere, and names a module, not the package.
    std::string sourceToken = ("@" + sourceName + "::").str();
    std::string newToken = ("@" + outputName + "::").str();
    auto rewriteNameText = [&](StringAttr str) -> StringAttr {
      StringRef text = str.getValue();
      if (!text.contains(sourceToken))
        return str;
      std::string rewritten;
      for (size_t pos = text.find(sourceToken); pos != StringRef::npos;
           pos = text.find(sourceToken)) {
        StringRef prefix = text.take_front(pos);
        rewritten += prefix;
        rewritten += prefix.ends_with("::") ? sourceToken : newToken;
        text = text.drop_front(pos + sourceToken.size());
      }
      rewritten += text;
      return StringAttr::get(ctx, rewritten);
    };
    replacer.addReplacement([&](DebugInfo::SourceNameAttr attr)
                                -> std::pair<Attribute, WalkResult> {
      bool isPackageRoot =
          attr.getKind() == DebugInfo::SourceNameKind::Package &&
          !attr.getParent() && attr.getName() == sourceNameAttr;
      StringAttr name =
          isPackageRoot ? newName : rewriteNameText(attr.getName());
      SmallVector<StringAttr> paramValues;
      for (StringAttr value : attr.getParamValues())
        paramValues.push_back(rewriteNameText(value));
      if (name == attr.getName() &&
          ArrayRef<StringAttr>(paramValues) == attr.getParamValues())
        return {attr, WalkResult::advance()};
      return {DebugInfo::SourceNameAttr::get(
                  name, attr.getParamTypes(), attr.getArgTypes(), paramValues,
                  attr.getParent(), attr.getKind(), attr.getDecorators()),
              WalkResult::advance()};
    });
    // Recorded import paths are raw name components, and any import op form
    // may carry one rooted at the package (a resolved import kept inside a
    // file module, an unresolved import, a wildcard import). All re-resolve
    // under the output file's name when the package is imported.
    replacer.addReplacement(
        [&](LIT::ImportPathAttr path) -> std::pair<Attribute, WalkResult> {
          auto components = path.getComponents();
          if (path.getRelativeLevel() != 0 || components.empty() ||
              components.front() != sourceNameAttr)
            return {path, WalkResult::advance()};
          SmallVector<StringAttr> newComponents(components);
          newComponents.front() = newName;
          return {
              LIT::ImportPathAttr::get(ctx, /*relativeLevel=*/0, newComponents),
              WalkResult::advance()};
        });
    replacer.recursivelyReplaceElementsIn(*packageModule,
                                          /*replaceAttrs=*/true,
                                          /*replaceLocs=*/true,
                                          /*replaceTypes=*/true);
    mlir::SymbolTable::setSymbolName(thePackage, newName);
  }

  // Process bitcode libraries if any were specified
  if (!bitcodeLibs.empty()) {
    SmallVector<DenseResourceElementsAttr> bitcodeAttrs;

    for (const std::string &bitcodeFile : bitcodeLibs) {
      DenseResourceElementsAttr bitcodeAttr =
          writeLLVMBitcodeToDenseAttr(theModule.getContext(), bitcodeFile);
      if (!bitcodeAttr)
        return Error("failed to load bitcode library: " + bitcodeFile);
      bitcodeAttrs.push_back(bitcodeAttr);
    }

    // Set the bitcode libraries on the package
    thePackage.setExternLLVMBitcodeModulesAttr(
        LIT::DenseResourceElementsArrayAttr::get(theModule.getContext(),
                                                 bitcodeAttrs));
  }

  return std::move(packageModule);
}
