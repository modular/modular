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

#include "Support/ML/DType.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <cassert>
#include <optional>
#include <utility>

using namespace M;
using namespace M::KGEN;
using namespace mlir;

DenseMap<PrecompiledMLIRRegistry::Kind, PrecompiledMLIRRegistry::Entry>
PrecompiledMLIRRegistry::buildTable(ArrayRef<Snippet> snippets) {
  DenseMap<Kind, Entry> table;
  table.reserve(snippets.size());
  for (const Snippet &snippet : snippets) {
    Entry entry{snippet.symbolName.str(), snippet.mlirText.str()};
    bool inserted = table.try_emplace(snippet.kind, std::move(entry)).second;
    (void)inserted;
    assert(inserted && "duplicate snippet kind");
  }
  return table;
}

PrecompiledMLIRRegistry::PrecompiledMLIRRegistry(ArrayRef<Snippet> snippets)
    : table(buildTable(snippets)) {}

PrecompiledMLIRRegistry::~PrecompiledMLIRRegistry() = default;

Operation *
PrecompiledMLIRRegistry::getFunction(Kind kind, MLIRContext *ctx,
                                     OwningOpRef<Operation *> &storage) const {
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
PrecompiledMLIRCache::getFunction(const PrecompiledMLIRRegistry &registry,
                                  PrecompiledMLIRRegistry::Kind kind) {
  std::lock_guard<std::mutex> lock(mutex);
  auto [it, inserted] = parsed.try_emplace({&registry, kind});
  if (inserted)
    it->second.function = registry.getFunction(kind, ctx, it->second.module);
  return it->second.function;
}

FailureOr<Operation *>
PrecompiledMLIRRegistry::insertIntoModule(Kind kind, ModuleOp module) const {
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

LogicalResult PrecompiledMLIRRegistry::inlineInPlace(
    Kind kind, Operation *op, ValueRange inputs, RewriterBase &rewriter) const {
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

//===----------------------------------------------------------------------===//
// Float conversions
//===----------------------------------------------------------------------===//

// Regenerating the embedded snippets (manual; no generator script is
// committed):
//
//  1. In oss/modular/mojo/stdlib/std/builtin/simd.mojo, temporarily change
//     the conversion's `@always_inline` to `@no_inline` so each instantiation
//     survives as a distinct symbol (`_convert_float8_to_f32_scalar` for
//     fp8 -> f32; `_convert_f32_to_float8_scalar` and
//     `_convert_f32_to_float8_ue8m0_scalar` for f32 -> fp8), and restore the
//     pre-snippet software body from git history. Capture the two directions
//     in separate runs: `f32 -> fp8` compares against a widened `max_finite`,
//     so with the widening uninlined that constant becomes a call.
//  2. Build the stdlib: ./bazelw build //oss/modular/mojo/stdlib/std
//  3. Write a capture.mojo that casts every variant (forcing the software
//     path), then dump the LLVM-dialect MLIR:
//       kgen --elaborate -I <built-std> capture.mojo -o elab.mlirbc
//       kgen-opt elab.mlirbc -lower-to-llvm -o llvmdialect.mlir
//  4. For each captured function: rename it to its synthetic
//     `__kgen_pop_cast_*` symbol (distinct from the stdlib shim, to avoid a
//     self-recursive call), strip the host-specific `attributes {...}` block
//     and argument attributes, and set `internal` linkage (the snippet must
//     inherit the target of the module it is inserted into).
//
//     The capture yields the `Branching` form: `-lower-to-llvm` keeps the
//     conversion's control flow, and so does `kgen-opt`. Nothing here
//     reproduces the `Flat` entries -- neither pass flattens, and neither
//     does simplifycfg or `-O3` over the result -- so carry those entries
//     over unchanged.
//
//  5. Canonicalize and embed: paste the `kgen-opt snippet.mlir` output into
//     the .inc as the snippet's raw string literal.
//  6. Revert the stdlib changes from step 1.
//
#include "ConversionSnippets.inc"

// The two tables hold the same conversions in the same order, so a `Kind` is
// an index into either one.
static ArrayRef<ConversionSnippet>
snippetTable(PrecompiledMLIRRegistry::FloatConversionForm form) {
  using Form = PrecompiledMLIRRegistry::FloatConversionForm;
  return form == Form::Flat ? ArrayRef(kFloatConversionSnippets)
                            : ArrayRef(kBranchingFloatConversionSnippets);
}

static const PrecompiledMLIRRegistry *
buildRegistry(ArrayRef<ConversionSnippet> table) {
  SmallVector<PrecompiledMLIRRegistry::Snippet> snippets;
  for (auto [kind, snippet] : llvm::enumerate(table))
    snippets.push_back({static_cast<PrecompiledMLIRRegistry::Kind>(kind),
                        snippet.symbolName, snippet.mlirText});
  return new PrecompiledMLIRRegistry(snippets);
}

const PrecompiledMLIRRegistry &
PrecompiledMLIRRegistry::getFloatConversions(FloatConversionForm form) {
  // Leaked, process-lifetime singletons.
  static const PrecompiledMLIRRegistry *flat =
      buildRegistry(snippetTable(FloatConversionForm::Flat));
  static const PrecompiledMLIRRegistry *branching =
      buildRegistry(snippetTable(FloatConversionForm::Branching));
  return form == FloatConversionForm::Flat ? *flat : *branching;
}

std::optional<PrecompiledMLIRRegistry::Kind>
PrecompiledMLIRRegistry::getFloatConversionKind(DType from, DType to,
                                                FloatConversionForm form) {
  for (auto [kind, snippet] : llvm::enumerate(snippetTable(form)))
    if (snippet.from == from.getValue() && snippet.to == to.getValue())
      return kind;
  return std::nullopt;
}

StringRef PrecompiledMLIRRegistry::getFloatConversionSymbolName(
    DType from, DType to, FloatConversionForm form) {
  std::optional<Kind> kind = getFloatConversionKind(from, to, form);
  if (!kind)
    return {};
  return snippetTable(form)[*kind].symbolName;
}

FailureOr<Operation *> PrecompiledMLIRRegistry::insertFloatConversion(
    DType from, DType to, FloatConversionForm form, ModuleOp module) {
  std::optional<Kind> kind = getFloatConversionKind(from, to, form);
  if (!kind)
    return failure();
  return getFloatConversions(form).insertIntoModule(*kind, module);
}

Operation *PrecompiledMLIRRegistry::getFloatConversionFunction(
    DType from, DType to, FloatConversionForm form, MLIRContext *ctx,
    OwningOpRef<Operation *> &storage, PrecompiledMLIRCache *cache) {
  std::optional<Kind> kind = getFloatConversionKind(from, to, form);
  if (!kind)
    return nullptr;
  const PrecompiledMLIRRegistry &registry = getFloatConversions(form);
  if (cache)
    return cache->getFunction(registry, *kind);
  return registry.getFunction(*kind, ctx, storage);
}
