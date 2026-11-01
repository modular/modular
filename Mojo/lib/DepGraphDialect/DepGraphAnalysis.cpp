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

#include "Mojo/DepGraphDialect/DepGraphAnalysis.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/xxhash.h"

using namespace M;
using namespace KGEN;
using namespace DepGraph;

std::string DepGraph::hashModuleBuffer(llvm::StringRef contents) {
  return llvm::utohexstr(
      llvm::xxh3_64bits(llvm::arrayRefFromStringRef(contents)),
      /*LowerCase=*/true, /*Width=*/16);
}

std::optional<std::string> DepGraph::defaultFileHasher(llvm::StringRef path) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFile(path);
  if (!fileOrErr)
    return std::nullopt;
  return hashModuleBuffer((*fileOrErr)->getBuffer());
}

SourceHashes DepGraph::hashModuleSources(GraphOp graph, HashFn hasher) {
  SourceHashes hashes;
  for (ModuleNodeOp mod : getModuleNodes(graph)) {
    std::optional<llvm::StringRef> path = mod.getPath();
    if (!path)
      continue;
    if (std::optional<std::string> hash = hasher(*path))
      hashes.try_emplace(mod.getOperation(), std::move(*hash));
  }
  return hashes;
}

bool DepGraph::isModuleDirty(ModuleNodeOp mod, const SourceHashes &hashes) {
  // A missing stored hash, path, or file cannot be verified: dirty.
  std::optional<llvm::StringRef> storedHash = mod.getContentHash();
  if (!storedHash)
    return true;
  auto it = hashes.find(mod.getOperation());
  if (it == hashes.end())
    return true;
  return llvm::StringRef(it->second) != *storedHash;
}

llvm::DenseSet<mlir::Operation *>
DepGraph::computeDirtyModules(GraphOp graph, const SourceHashes &hashes) {
  llvm::DenseSet<mlir::Operation *> dirty;
  llvm::SmallVector<ModuleNodeOp> worklist;

  // Mark every node whose own content changed.
  for (ModuleNodeOp mod : getModuleNodes(graph)) {
    if (!isModuleDirty(mod, hashes))
      continue;
    dirty.insert(mod.getOperation());
    worklist.push_back(mod);
  }

  // Propagate to transitive dependents via reverse-edge BFS; `dirty` doubles
  // as the visited set, guaranteeing termination on cycles.
  while (!worklist.empty()) {
    ModuleNodeOp mod = worklist.pop_back_val();
    for (ModuleNodeOp dependent : getDependents(mod))
      if (dirty.insert(dependent.getOperation()).second)
        worklist.push_back(dependent);
  }

  return dirty;
}
