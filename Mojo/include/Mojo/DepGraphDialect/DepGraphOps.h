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

#ifndef MOJO_DEPGRAPHDIALECT_DEPGRAPHOPS_H
#define MOJO_DEPGRAPHDIALECT_DEPGRAPHOPS_H

#include "Mojo/DepGraphDialect/DepGraphTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Mojo/DepGraphDialect/DepGraph.h.inc"

//===----------------------------------------------------------------------===//
// Graph utilities — declared here so all op types are complete
//===----------------------------------------------------------------------===//

namespace M::KGEN::DepGraph {

/// Returns all module nodes in the graph.
llvm::SmallVector<ModuleNodeOp> getModuleNodes(GraphOp graph);
/// Returns all modules that `mod` directly depends on.
llvm::SmallVector<ModuleNodeOp> getDependencies(ModuleNodeOp mod);
/// Returns all modules that directly depend on `mod`.
llvm::SmallVector<ModuleNodeOp> getDependents(ModuleNodeOp mod);

} // namespace M::KGEN::DepGraph

#endif // MOJO_DEPGRAPHDIALECT_DEPGRAPHOPS_H
