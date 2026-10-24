//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// Compile-time utilities for working with MGP buffers operands and operations.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPUTILS_H
#define GENERICML_MGPDIALECT_MGPUTILS_H

#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Operation.h"

namespace M::MGP {

/// Lower a dimension to an mlir index.
Value dimensionToIndex(OpBuilder &builder, Location loc, Value dim);
Value indexToDimension(OpBuilder &builder, Location loc, Value index);

/// Returns the !mgp.context in use in block, from:
///  - a block argument (e.g. the second 'execute' argument, or a control flow
///    sub-block argument).
///  - an mgp.context.create in the block (e.g. in 'init').
///  - the parent block, for a sub-block that didn't carry the !mgp.context
///    down. Using it there is illegal until fixupIsolatedFromAbove repairs
///    the model.
TypedValue<MGP::ContextType> getContextForBlock(Block *block);

/// Returns the 'root' !mef.chain for block, from the same sources as
/// getContextForBlock: a block argument (the first 'execute' argument), a
/// nullary grt.chain.create (in 'init'), or the parent block.
TypedValue<MEF::ChainType> getRootChainForBlock(Block *block);

/// Record the given op has the given in/out semantics.
///
/// Only to be used for ops without an ODS definition. All others should
/// implement the MGPInOutSemanticsOpInterface interface.
void setInOutSignature(Operation *op, InOutSignatureAttr inOutSig);

/// Removes any attribute for in/out semantics added by setInOutSignature.
///
/// Ops without ODS definitions which will be interpreted by MEF cannot
/// retain the attribute without modifying the primitive's implementation.
void removeInOutSignature(Operation *op);

/// Returns the in/out semantics for operand.
InOutSignatureAttr::InOutSemantics getInOutSemantics(OpOperand &opOperand);

/// Returns true if operand is mutated by its op (ie it has either an 'out'
/// or an 'in/out' semantics).
bool isMutated(OpOperand &opOperand);

/// Returns true if operand is read by its op (ie it has either an 'in'
/// or 'in/out' semantics).
///
/// CAUTION: This does not mean 'read only'.
bool isRead(OpOperand &opOperand);

} // namespace M::MGP

#endif // GENERICML_MGPDIALECT_MGPUTILS_H
