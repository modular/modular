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

#include "Mojo/TransformUtils/EliminateDeadSymbolUtils.h"

#include "Mojo/KGENDialect/KGENOps.h"
#include "Support/Compiler/Threading.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"

#include <memory>
#include <vector>

using namespace M;
using namespace KGEN;

namespace {
/// The attribute walker of one chunk and the symbols it found in the current
/// round.
struct ChunkWalker {
  ChunkWalker() {
    walker.addWalk(
        [this](FlatSymbolRefAttr ref) { refs.push_back(ref.getAttr()); });
  }

  mlir::AttrTypeWalker walker;
  SmallVector<StringAttr, 0> refs;
};
} // namespace

DenseSet<StringAttr> KGEN::getUsedSymbols(mlir::SymbolTableAnalysis &analysis,
                                          ModuleOp theModule) {
  DenseSet<StringAttr> usedSymbols;
  for (auto symbol : theModule.getOps<ExportInterface>()) {
    if (symbol.isExported())
      usedSymbols.insert(symbol.getSymNameAttr());
  }

  mlir::SymbolTable &symtab = analysis.getTopLevelSymbolTable();
  MLIRContext *ctx = theModule->getContext();

  // A chunk index keeps its walker across rounds, so that walker walks each
  // attribute only once. The number of chunks follows the size of the round,
  // so the vector has a slot for every chunk index that any round can use.
  std::vector<std::unique_ptr<ChunkWalker>> chunkWalkers(kMaxParallelChunks);

  // Walk the used symbols in rounds. A round walks the symbols that the round
  // before it found, in parallel, and collects the symbols that they use.
  SmallVector<StringAttr, 0> round(usedSymbols.begin(), usedSymbols.end());
  while (!round.empty()) {
    parallelForEachIndexedChunk(
        ctx, round, [&](size_t chunk, ArrayRef<StringAttr> symbols) {
          if (!chunkWalkers[chunk])
            chunkWalkers[chunk] = std::make_unique<ChunkWalker>();
          mlir::AttrTypeWalker &walker = chunkWalkers[chunk]->walker;
          for (StringAttr symbol : symbols) {
            Operation *callee = symtab.lookup(symbol);
            if (!callee)
              continue;
            callee->walk([&](Operation *op) {
              walker.walk(op->getAttrDictionary());
              // Also walk the location, because a FusedLoc<DISubprogramAttr>
              // can embed symbol references in the argument and result types
              // of its DISubroutineType.
              walker.walk(op->getLoc());
              for (Type type : op->getResultTypes())
                walker.walk(type);
              for (Region &region : op->getRegions()) {
                for (Type type : region.getArgumentTypes())
                  walker.walk(type);
              }
            });
          }
        });

    round.clear();
    for (std::unique_ptr<ChunkWalker> &chunkWalker : chunkWalkers) {
      if (!chunkWalker)
        continue;
      for (StringAttr ref : chunkWalker->refs) {
        if (usedSymbols.insert(ref).second)
          round.push_back(ref);
      }
      chunkWalker->refs.clear();
    }
  }
  return usedSymbols;
}
