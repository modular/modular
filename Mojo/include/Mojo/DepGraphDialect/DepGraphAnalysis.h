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
//
// Dirty-propagation analysis over the `depgraph` dialect: a node whose content
// hash changed is seed-dirty, and dirtiness propagates to transitive importers.
//
//===----------------------------------------------------------------------===//

#ifndef MOJO_DEPGRAPHDIALECT_DEPGRAPHANALYSIS_H
#define MOJO_DEPGRAPHDIALECT_DEPGRAPHANALYSIS_H

#include "Mojo/DepGraphDialect/DepGraphOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>

namespace M::KGEN::DepGraph {

/// Hash function: returns the current content hash for a path, or nullopt if
/// the path cannot be read.
using HashFn =
    llvm::function_ref<std::optional<std::string>(llvm::StringRef path)>;

/// xxh3-64 lowercase-hex digest of a buffer; shared with the parser's
/// per-module content hash so the two sites cannot drift.
std::string hashModuleBuffer(llvm::StringRef contents);

/// Default `HashFn`: `hashModuleBuffer` of the file at `path`, or nullopt if
/// it cannot be read.
std::optional<std::string> defaultFileHasher(llvm::StringRef path);

/// Current content hash of each module node's source, keyed by the
/// `ModuleNodeOp` operation. Absent entry: no `path`, or unreadable file.
using SourceHashes = llvm::DenseMap<mlir::Operation *, std::string>;

/// Hash every module node's source once; the dirtiness queries below read
/// from the returned map.
SourceHashes hashModuleSources(GraphOp graph,
                               HashFn hasher = defaultFileHasher);

/// True if the node itself is dirty: no stored `content_hash`, no entry in
/// `hashes`, or the two hashes differ.
bool isModuleDirty(ModuleNodeOp mod, const SourceHashes &hashes);

/// Full analysis: seed scan plus cycle-correct reverse-BFS propagation to
/// transitive dependents. Returns the set of dirty `ModuleNodeOp` operations.
llvm::DenseSet<mlir::Operation *>
computeDirtyModules(GraphOp graph, const SourceHashes &hashes);

} // namespace M::KGEN::DepGraph

#endif // MOJO_DEPGRAPHDIALECT_DEPGRAPHANALYSIS_H
