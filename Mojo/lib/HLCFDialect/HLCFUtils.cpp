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

#include "Mojo/HLCFDialect/HLCFUtils.h"
#include "Mojo/HLCFDialect/HLCFOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"

using namespace M;
using namespace HLCF;

/// Return true if the operation is a loop and has a matching label.
bool HLCF::isMatchingLoop(Operation *op, StringAttr label) {
  if (auto loop = dyn_cast<LoopOp>(op))
    return !label || loop.getLabelAttr() == label;
  return false;
}

/// Return the nearest enclosing matching loop or nullptr if nothing found.
LoopOp HLCF::getParentLoop(Operation *op, StringAttr label) {
  LoopOp loop = op->getParentOfType<LoopOp>();
  while (!isMatchingLoop(loop, label))
    loop = loop->getParentOfType<LoopOp>();
  return loop;
}

/// Check if the child loop is nested in the parentToCheck loop.
bool HLCF::isParentLoop(LoopOp child, LoopOp parentToCheck) {
  LoopOp parent = child;
  while (parent && parent != parentToCheck)
    parent = parent->getParentOfType<LoopOp>();
  return parent == parentToCheck;
}

/// Get the parent operation of a terminator.
Operation *HLCF::getParentNode(HLCF::ControlFlowTerminator term) {
  Operation *op = term->getParentOp();
  while (!term.isParentNode(op))
    op = op->getParentOp();
  return op;
}
