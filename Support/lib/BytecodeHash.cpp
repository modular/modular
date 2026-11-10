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

#include "Support/BytecodeHash.h"

#include "Support/Compiler/Threading.h"
#include "Support/HashUtils.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <iterator>
#include <string>
#include <utility>

using namespace mlir;

static LogicalResult writeBytecode(Operation *op, llvm::raw_ostream &os) {
  OwningOpRef<Operation *> cloned = op->clone();

  auto unknownLoc = UnknownLoc::get(op->getContext());

  // Strip the debug info from all operations.
  cloned->walk([&](Operation *op) {
    op->setLoc(unknownLoc);
    // Strip block arguments debug info.
    for (Region &region : op->getRegions()) {
      for (Block &block : region.getBlocks()) {
        for (BlockArgument &arg : block.getArguments())
          arg.setLoc(unknownLoc);
      }
    }
  });

  return mlir::writeBytecodeToFile(*cloned, os);
}

FailureOr<std::string> M::getBytecodeHash(Operation *op) {
  raw_xxhash128_stream ostream;
  if (failed(writeBytecode(op, ostream)))
    return op->emitError("Failed to write bytecode");
  return ostream.hashString();
}

struct ModuleHashCache {
  ModuleHashCache() = default;

  LogicalResult computeHash(Operation *op) {
    auto result = M::getBytecodeHash(op);
    if (failed(result))
      return failure();

    hashes.push_back(std::move(*result));
    return success();
  }

  void join(ModuleHashCache &other) {
    llvm::move(other.hashes, std::back_inserter(hashes));
    other = ModuleHashCache();
  }

  llvm::SmallVector<std::string> hashes;
};

FailureOr<std::string> M::getModuleBytecodeHash(mlir::ModuleOp module) {
  auto context = module.getContext();
  auto ops =
      llvm::map_to_vector(module.getOps(), [](Operation &op) { return &op; });

  auto workFunc = [](ModuleHashCache &cache, Operation *op) -> LogicalResult {
    return cache.computeHash(op);
  };

  auto consolidateFn = [](ModuleHashCache &original,
                          MutableArrayRef<ModuleHashCache> caches) {
    for (ModuleHashCache &cache : caches)
      original.join(cache);
  };

  ModuleHashCache resultCache;
  auto result = failableParallelForEach(
      /*ctx=*/context,
      /*range=*/ops,
      /*func=*/workFunc,
      /*cache=*/resultCache,
      /*consolidate=*/consolidateFn);

  if (failed(result))
    return failure();

  // Sort results to ensure determinism. The multi-threaded processing of
  // failableParallelForEach does not ensure an particular ordering w.r.t. what
  // input elements are associated to which cache.
  llvm::sort(resultCache.hashes);

  raw_xxhash128_stream hasher;
  llvm::interleave(resultCache.hashes, hasher, "");
  return hasher.hashString();
}
