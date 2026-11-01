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

#include "GraphCompiler/MGPDialect/Dialect/MGPUtils.h"

#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Support/AssertStream.h"
#include "Support/Compiler/MLIRDType.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/ErrorHandling.h"

using namespace M;
using namespace MGP;

namespace {
/// Name of attribute holding the 'temporary' #M.inout_sig.
constexpr StringRef kInOutSigAttribute = "tempInOutSig";
} // namespace

Value MGP::dimensionToIndex(OpBuilder &builder, Location loc, Value dim) {
  if (dim.getType().isIndex())
    return dim;

  // Shape dimensions are `!kgen.scalar<si64>`; `index.cast` only accepts
  // builtin integers or `index`, so lower through `si64` first.
  if (MOSH::ShapeAttr::isDimType(dim.getType())) {
    auto intTy = M::getEquivalentIntegerType(
        dim.getContext(),
        cast<KGEN::SIMDType>(dim.getType()).getResolvedDType().value());
    dim = builder.createOrFold<KGEN::POP::CastToBuiltinOp>(loc, intTy, dim);
  }

  return builder.createOrFold<mlir::index::CastSOp>(loc, builder.getIndexType(),
                                                    dim);
}

Value MGP::indexToDimension(OpBuilder &builder, Location loc, Value index) {
  if (MOSH::ShapeAttr::isDimType(index.getType()))
    return index;

  auto si64Ty = mlir::IntegerType::get(builder.getContext(), 64,
                                       mlir::IntegerType::Signed);
  Value asSi64 = builder.createOrFold<mlir::index::CastSOp>(loc, si64Ty, index);
  auto dimTy = MOSH::ShapeAttr::getDimType(builder.getContext());
  return builder.createOrFold<KGEN::POP::CastFromBuiltinOp>(loc, dimTy, asSi64);
}

TypedValue<MGP::ContextType> MGP::getContextForBlock(Block *block) {
  while (true) {
    for (Value arg : block->getArguments()) {
      if (isa<MGP::ContextType>(arg.getType()))
        return cast<TypedValue<MGP::ContextType>>(arg);
    }
    for (auto createOp : block->getOps<MGP::ContextCreateOp>()) {
      return createOp.getResult();
    }
    ASSERT_STREAM(!isa<ModelOp>(block->getParentOp()),
                  << "unable to locate an !mgp.context");
    block = block->getParentOp()->getBlock();
  }
  ASSERT_STREAM(false, << "unable to locate an !mgp.context");
}

TypedValue<MEF::ChainType> MGP::getRootChainForBlock(Block *block) {
  while (true) {
    for (Value arg : block->getArguments()) {
      if (isa<MEF::ChainType>(arg.getType()))
        return cast<TypedValue<MEF::ChainType>>(arg);
    }
    for (auto createOp : block->getOps<GRT::ChainCreateOp>()) {
      if (createOp->getNumOperands() == 0)
        return createOp.getResult();
    }
    ASSERT_STREAM(!isa<ModelOp>(block->getParentOp()),
                  << "unable to locate a !mef.chain");
    block = block->getParentOp()->getBlock();
  }
  ASSERT_STREAM(false, << "unable to locate a !mef.chain");
}

void MGP::setInOutSignature(Operation *op, InOutSignatureAttr inOutSig) {
  ASSERT_STREAM(!isa<MGPInOutSemanticsOpInterface>(op),
                << "op already implements MGPInOutSemanticsOpInterface");
  ASSERT_STREAM(inOutSig.size() == op->getNumOperands(),
                << "mismatched operation and in/out signature arity");
  op->setAttr(kInOutSigAttribute, inOutSig);
}

void MGP::removeInOutSignature(Operation *op) {
  op->removeAttr(kInOutSigAttribute);
}

InOutSignatureAttr::InOutSemantics
MGP::getInOutSemantics(OpOperand &opOperand) {
  Operation *op = opOperand.getOwner();

  if (auto inOutSemanticsOp = dyn_cast<MGPInOutSemanticsOpInterface>(op))
    return inOutSemanticsOp.getInOutSemantics(opOperand.getOperandNumber());

  // Ops without the interface may carry an #M.InOutSignature attribute.
  if (auto inOutAttr =
          op->getAttrOfType<InOutSignatureAttr>(kInOutSigAttribute)) {
    ASSERT_STREAM(inOutAttr.size() == op->getNumOperands(),
                  << "mismatched operation and in/out signature arity");
    return inOutAttr[opOperand.getOperandNumber()];
  }

  return InOutSignatureAttr::kIn;
}

bool MGP::isMutated(OpOperand &opOperand) {
  InOutSignatureAttr::InOutSemantics inOut = getInOutSemantics(opOperand);
  return inOut == InOutSignatureAttr::kOut || inOut == InOutSignatureAttr::kMut;
}

bool MGP::isRead(OpOperand &opOperand) {
  InOutSignatureAttr::InOutSemantics inOut = getInOutSemantics(opOperand);
  return inOut == InOutSignatureAttr::kIn || inOut == InOutSignatureAttr::kMut;
}
