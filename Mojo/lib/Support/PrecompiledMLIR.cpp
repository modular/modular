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

#include "Mojo/Support/PrecompiledMLIR.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace M;
using namespace M::KGEN;
using namespace mlir;

PrecompiledMLIRRegistry::~PrecompiledMLIRRegistry() = default;

void PrecompiledMLIRRegistry::registerKind(Kind kind, StringRef symbolName,
                                           StringRef mlirText) {
  std::lock_guard<std::mutex> lock(mutex);
  Entry &entry = table[kind];
  entry.symbolName = symbolName.str();
  entry.mlirText = mlirText.str();
}

Operation *
PrecompiledMLIRRegistry::getFunction(Kind kind, MLIRContext *ctx,
                                     OwningOpRef<Operation *> &storage) {
  std::lock_guard<std::mutex> lock(mutex);
  auto it = table.find(kind);
  if (it == table.end())
    return nullptr;

  storage = parseSourceString(it->second.mlirText, ParserConfig(ctx));
  if (!storage)
    return nullptr;
  return SymbolTable::lookupSymbolIn(
      storage.get(), StringAttr::get(ctx, it->second.symbolName));
}

PrecompiledMLIRCache::~PrecompiledMLIRCache() = default;

Operation *
PrecompiledMLIRCache::getFunction(PrecompiledMLIRRegistry &registry,
                                  PrecompiledMLIRRegistry::Kind kind) {
  std::lock_guard<std::mutex> lock(mutex);
  auto [it, inserted] = parsed.try_emplace({&registry, kind});
  if (inserted)
    it->second.function = registry.getFunction(kind, ctx, it->second.module);
  return it->second.function;
}

FailureOr<Operation *>
PrecompiledMLIRRegistry::insertIntoModule(Kind kind, ModuleOp module) {
  OwningOpRef<Operation *> storage;
  Operation *fn = getFunction(kind, module.getContext(), storage);
  if (!fn)
    return failure();

  SymbolTable snippetSymTab(fn->getParentOp());
  SymbolTable targetSymTab(module);

  // Clone the function and its transitive symbol dependencies into `module`,
  // skipping any symbol already present (deduplicated by name).
  Operation *result = nullptr;
  SmallVector<Operation *> worklist{fn};
  llvm::SmallPtrSet<Operation *, 8> visited;
  while (!worklist.empty()) {
    Operation *sym = worklist.pop_back_val();
    if (!visited.insert(sym).second)
      continue;

    Operation *inTarget = targetSymTab.lookup(SymbolTable::getSymbolName(sym));
    if (!inTarget) {
      inTarget = sym->clone();
      targetSymTab.insert(inTarget);
    }
    if (sym == fn)
      result = inTarget;

    AttrTypeWalker walker;
    walker.addWalk([&](FlatSymbolRefAttr ref) {
      if (Operation *dep = snippetSymTab.lookup(ref.getAttr()))
        worklist.push_back(dep);
    });
    sym->walk(
        [&](Operation *nested) { walker.walk(nested->getAttrDictionary()); });
  }
  return result;
}

LogicalResult PrecompiledMLIRRegistry::inlineInPlace(Kind kind, Operation *op,
                                                     ValueRange inputs,
                                                     RewriterBase &rewriter) {
  OwningOpRef<Operation *> storage;
  Operation *fn = getFunction(kind, op->getContext(), storage);
  if (!fn || fn->getNumRegions() != 1)
    return failure();

  // Only single-block bodies are inlined here; a snippet with control flow
  // should be reached through `insertIntoModule` (a call) instead.
  Region &body = fn->getRegion(0);
  if (!body.hasOneBlock())
    return failure();
  Block &block = body.front();
  if (block.getNumArguments() != inputs.size())
    return failure();

  // Clone the body's non-terminator ops at `op`, binding the block arguments
  // to `inputs`, then replace `op`'s results with the terminator's operands.
  IRMapping map;
  for (auto [arg, input] : llvm::zip(block.getArguments(), inputs))
    map.map(arg, input);

  rewriter.setInsertionPoint(op);
  for (Operation &bodyOp : block.without_terminator())
    rewriter.clone(bodyOp, map);

  SmallVector<Value> results;
  for (Value operand : block.getTerminator()->getOperands())
    results.push_back(map.lookupOrDefault(operand));
  rewriter.replaceOp(op, results);
  return success();
}
