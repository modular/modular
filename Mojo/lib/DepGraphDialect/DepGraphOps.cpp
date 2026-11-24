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

#include "Mojo/DepGraphDialect/DepGraphOps.h"
#include "Mojo/DepGraphDialect/DepGraphDialect.h"
#include "Mojo/DepGraphDialect/DepGraphTypes.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace M;
using namespace KGEN;
using namespace DepGraph;

//===----------------------------------------------------------------------===//
// DepGraphDialect - op registration
//===----------------------------------------------------------------------===//

void DepGraphDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "Mojo/DepGraphDialect/DepGraph.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// GraphOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult GraphOp::verify() {
  for (auto &op : getBody().front()) {
    if (!isa<ModuleNodeOp>(op))
      return op.emitOpError(
          "only 'depgraph.module' ops are allowed inside a 'depgraph.graph'");
  }
  if (auto root = getRootAttr(); root && !getRootModule())
    return emitOpError("'root' symbol '")
           << root.getValue()
           << "' does not name a 'depgraph.module' in this graph";
  return mlir::success();
}

ModuleNodeOp GraphOp::lookupModule(llvm::StringRef name) {
  // Module nodes produce SSA results, which the `SymbolTable` trait forbids
  // for symbols, so this op is not a symbol table: scan the body directly.
  for (auto mod : getBody().front().getOps<ModuleNodeOp>())
    if (mod.getSymName() == name)
      return mod;
  return {};
}

ModuleNodeOp GraphOp::getRootModule() {
  auto root = getRootAttr();
  if (!root)
    return {};
  return lookupModule(root.getValue());
}

//===----------------------------------------------------------------------===//
// Graph utilities (free functions)
//===----------------------------------------------------------------------===//

llvm::SmallVector<ModuleNodeOp> DepGraph::getModuleNodes(GraphOp graph) {
  llvm::SmallVector<ModuleNodeOp> result;
  for (auto op : graph.getBody().front().getOps<ModuleNodeOp>())
    result.push_back(op);
  return result;
}

//===----------------------------------------------------------------------===//
// ModuleNodeOp
//===----------------------------------------------------------------------===//

void ModuleNodeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                         llvm::StringRef name, llvm::StringRef path,
                         mlir::ValueRange deps) {
  build(builder, state, NodeType::get(builder.getContext()),
        builder.getStringAttr(name), builder.getStringAttr(path),
        /*content_hash=*/mlir::StringAttr{}, deps);
}

mlir::LogicalResult ModuleNodeOp::verify() {
  if (llvm::any_of(getDeps(), [](mlir::Value dep) {
        return !dep.getDefiningOp<ModuleNodeOp>();
      })) {
    return emitOpError("dependency operand must be defined by a "
                       "'depgraph.module'");
  }
  return mlir::success();
}

llvm::SmallVector<ModuleNodeOp> DepGraph::getDependencies(ModuleNodeOp mod) {
  llvm::SmallVector<ModuleNodeOp> result;
  for (mlir::Value dep : mod.getDeps())
    result.push_back(dep.getDefiningOp<ModuleNodeOp>());
  return result;
}

llvm::SmallVector<ModuleNodeOp> DepGraph::getDependents(ModuleNodeOp mod) {
  llvm::SmallVector<ModuleNodeOp> result;
  for (mlir::Operation *user : mod.getHandle().getUsers())
    if (auto dependent = mlir::dyn_cast<ModuleNodeOp>(user))
      result.push_back(dependent);
  return result;
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Mojo/DepGraphDialect/DepGraph.cpp.inc"
