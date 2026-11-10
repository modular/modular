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

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MODialect/MOUtils.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/Support/AsmSyntax.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Support/AssertStream.h"
#include "Support/MDialect/ParserUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"

using namespace M;
using namespace mlir;
using namespace M::MAP;

// The declarative formats name these unqualified.
using M::MOGG::parseKernelParamsBody;
using M::MOGG::printKernelParamsBody;

using namespace M;
using namespace mlir;
using namespace M::MAP;
using M::MOGG::parseKernelParamsBody;
using M::MOGG::printKernelParamsBody;

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// map.call.opaque
//===----------------------------------------------------------------------===//

LogicalResult
CallOpaqueOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto decl = symbolTable.lookupNearestSymbolFrom<M::MOGG::KernelDeclOp>(
      *this, getCalleeAttr());
  if (!decl)
    return emitOpError("'")
           << getCallee() << "' does not refer to a kernel.decl op";
  return mlir::success();
}

LogicalResult CallOpaqueOp::verify() {
  // `$outputs` is `AnyType`; a by-value `!mo.opaque` belongs in
  // `$opaqueResult`.
  for (Value output : getOutputs())
    if (isa<MO::OpaqueType>(output.getType()))
      return emitOpError("an opaque is returned through its result, not passed "
                         "as a destination in 'outputs'");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// map.call.elementwise
//===----------------------------------------------------------------------===//

LogicalResult
CallElementwiseOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto decl = symbolTable.lookupNearestSymbolFrom<M::MOGG::KernelDeclOp>(
      *this, getCalleeAttr());
  if (!decl)
    return emitOpError("'")
           << getCallee() << "' does not refer to a kernel.decl op";
  if (!decl.getElementwise())
    return emitOpError("kernel '")
           << getCallee() << "' has no 'elementwise' function";
  return mlir::success();
}

LogicalResult IterOpaqueOp::verify() {
  // The block-arg accessors and the epilogue fusers (result `i` <-> output
  // block arg `i`) rely on [DPS results..., by-value opaque?, chain?]; any
  // other order would silently misalign them.
  bool seenByValue = false;
  for (Value result : getResults()) {
    Type ty = result.getType();
    if (isa<MO::OpaqueType>(ty) || isa<MO::ChainType>(ty)) {
      seenByValue = true;
      continue;
    }
    if (seenByValue)
      return emitOpError("destination-passing results must precede a by-value "
                         "opaque result and the output chain");
  }

  // A by-value result needs a `map.yield` of something other than a DPS
  // output block arg (yielding only those is legal but redundant).
  if (getOpaqueResult()) {
    auto yieldOp = dyn_cast<YieldOp>(getBody().front().back());
    Block::BlockArgListType outputArgs = getOutputBlockArgs();
    bool yieldsByValue =
        yieldOp && llvm::any_of(yieldOp.getOperands(), [&](Value operand) {
          return !llvm::is_contained(outputArgs, operand);
        });
    if (!yieldsByValue)
      return emitOpError("has a by-value result but its body does not "
                         "map.yield a value for it");
  }

  return mlir::success();
}

LogicalResult IterElementwiseOp::verify() {
  Block &body = getBody().front();
  unsigned numInputs = getInputs().size();

  // A chain result must be last and has no block argument.
  ResultRange results = getResults();
  for (unsigned i = 0; i < results.size(); ++i) {
    bool isChain = llvm::isa<MO::ChainType>(results[i].getType());
    if (isChain && i + 1 != results.size())
      return emitOpError() << "only the last result may be a chain";
  }
  bool hasOutChain = getOutputChain() != nullptr;
  unsigned numOutputs = results.size() - (hasOutChain ? 1 : 0);
  unsigned expectedArgs = numInputs + numOutputs + 1; // +1 for index

  if (body.getNumArguments() != expectedArgs)
    return emitOpError() << "body expects " << expectedArgs
                         << " block arguments (" << numInputs << " inputs + "
                         << numOutputs << " outputs + 1 index), " << "got "
                         << body.getNumArguments();

  for (unsigned i = 0; i < numInputs; ++i) {
    if (body.getArgument(i).getType() != getInputs()[i].getType())
      return emitOpError() << "input block argument " << i
                           << " type mismatch with operand";
  }

  for (unsigned i = 0; i < numOutputs; ++i) {
    if (body.getArgument(numInputs + i).getType() != results[i].getType())
      return emitOpError() << "output block argument " << i
                           << " type mismatch with result";
  }

  BlockArgument indexArg = body.getArgument(numInputs + numOutputs);
  auto indexTy = llvm::dyn_cast<IndexType>(indexArg.getType());
  if (!indexTy)
    return emitOpError() << "index block argument must be map.index<N>";

  if (indexTy.getRank() != getShape().getRank())
    return emitOpError() << "index rank " << indexTy.getRank()
                         << " does not match shape rank "
                         << getShape().getRank();

  return mlir::success();
}

LogicalResult IterElementwiseBodyOp::verify() {
  Block &body = getBody().front();
  unsigned numCaptures = getCaptures().size();
  if (body.getNumArguments() != 1 + numCaptures)
    return emitOpError() << "body expects exactly " << (1 + numCaptures)
                         << " block arguments (the index, then one per "
                            "capture), got "
                         << body.getNumArguments();

  auto indexTy = llvm::dyn_cast<IndexType>(body.getArgument(0).getType());
  if (!indexTy)
    return emitOpError() << "index block argument must be map.index<N>";

  if (indexTy.getRank() != getShape().getRank())
    return emitOpError() << "index rank " << indexTy.getRank()
                         << " does not match shape rank "
                         << getShape().getRank();

  return mlir::success();
}

static LogicalResult verifyRowIndex(Operation *op, MOSH::ShapeAttr shape,
                                    int64_t axis, BlockArgument row) {
  int64_t rank = shape.getRank();
  if (axis < 0 || axis >= rank)
    return op->emitOpError()
           << "axis " << axis << " is out of range for rank " << rank;
  auto rowTy = llvm::dyn_cast<MAP::IndexType>(row.getType());
  if (!rowTy)
    return op->emitOpError() << "row block argument must be map.index<N>";
  if (static_cast<int64_t>(rowTy.getRank()) != rank)
    return op->emitOpError() << "row index rank " << rowTy.getRank()
                             << " does not match shape rank " << rank;
  return mlir::success();
}

// Whether `shape` is `full` with dimension `axis` set to 1.
static bool isPerRowShape(MOSH::ShapeAttr shape, MOSH::ShapeAttr full,
                          int64_t axis) {
  if (shape.getRank() != full.getRank() ||
      MOSH::ShapeAttr::getStaticDim(shape[axis]) != 1)
    return false;
  for (int64_t d = 0, e = full.getRank(); d < e; ++d)
    if (d != axis && shape[d] != full[d])
      return false;
  return true;
}

// A store in a map phase writes a per-element tensor; one outside the phases
// writes a per-row tensor at the row index. Folds check their own bodies.
static LogicalResult verifyRowwiseStores(RowwiseOpInterface rowwise) {
  Operation *op = rowwise.getOperation();
  MOSH::ShapeAttr shape = rowwise.getShape();
  int64_t axis = rowwise.getAxis();
  BlockArgument row = rowwise.getRowArg();
  WalkResult result = op->getRegion(0).walk([&](StoreOp store) {
    Operation *phase = store->getParentOp();
    while (phase != op && !isa<RowwisePhaseOpInterface>(phase))
      phase = phase->getParentOp();
    if (isa<IterReduceOp>(phase))
      return WalkResult::advance();
    MOSH::ShapeAttr target =
        MO::castFromTensorLike(store.getTensor().getType()).getShape();
    if (isa<IterMapOp>(phase)) {
      if (target == shape)
        return WalkResult::advance();
      store.emitOpError() << "in a map phase must write a tensor of the "
                             "rowwise shape";
      return WalkResult::interrupt();
    }
    if (target == shape) {
      store.emitOpError() << "of a tensor of the rowwise shape must be in a "
                             "map phase";
      return WalkResult::interrupt();
    }
    if (!isPerRowShape(target, shape, axis) || store.getIndex() != row) {
      store.emitOpError() << "outside a map phase must write a per-row "
                             "tensor at the row index";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return mlir::failure(result.wasInterrupted());
}

LogicalResult IterRowwiseOp::verify() {
  Block &body = getBody().front();
  unsigned numInputs = getInputs().size();

  // A chain result must be last and has no block argument.
  ResultRange results = getResults();
  for (auto [i, result] : llvm::enumerate(results))
    if (llvm::isa<MO::ChainType>(result.getType()) && i + 1 != results.size())
      return emitOpError() << "only the last result may be a chain";
  unsigned numOutputs = results.size() - (getOutputChain() ? 1 : 0);
  unsigned expectedArgs = numInputs + numOutputs + 1;
  if (body.getNumArguments() != expectedArgs)
    return emitOpError() << "body expects " << expectedArgs
                         << " block arguments (" << numInputs << " inputs + "
                         << numOutputs << " outputs + 1 row), got "
                         << body.getNumArguments();

  for (auto [i, pair] :
       llvm::enumerate(llvm::zip(getInputs(), getInputBlockArgs())))
    if (std::get<0>(pair).getType() != std::get<1>(pair).getType())
      return emitOpError() << "input block argument " << i
                           << " type mismatch with operand";
  for (auto [i, pair] :
       llvm::enumerate(llvm::zip(results, getOutputBlockArgs())))
    if (std::get<0>(pair).getType() != std::get<1>(pair).getType())
      return emitOpError() << "output block argument " << i
                           << " type mismatch with result";

  MOSH::ShapeAttr shape = getShape();
  int64_t axis = getAxis();
  if (failed(verifyRowIndex(*this, shape, axis, getRowArg())))
    return mlir::failure();

  // Each result has the full shape (per element) or the shape with `axis`
  // set to 1 (per row).
  for (auto [i, result] : llvm::enumerate(results.take_front(numOutputs))) {
    MOSH::ShapeAttr resultShape =
        llvm::cast<MO::TensorType>(result.getType()).getShape();
    if (resultShape != shape && !isPerRowShape(resultShape, shape, axis))
      return emitOpError() << "result " << i
                           << " must have the rowwise shape, or that shape "
                              "with dimension "
                           << axis << " set to 1; got " << result.getType();
  }
  return mlir::success();
}

LogicalResult IterRowwiseOp::verifyRegions() {
  return verifyRowwiseStores(*this);
}

LogicalResult IterRowwiseBodyOp::verify() {
  Block &body = getBody().front();
  unsigned numCaptures = getCaptures().size();
  if (body.getNumArguments() != 1 + numCaptures)
    return emitOpError() << "body expects exactly " << (1 + numCaptures)
                         << " block arguments (the row, then one per "
                            "capture), got "
                         << body.getNumArguments();
  return verifyRowIndex(*this, getShape(), getAxis(), getRowArg());
}

LogicalResult IterRowwiseBodyOp::verifyRegions() {
  return verifyRowwiseStores(*this);
}

//===----------------------------------------------------------------------===//
// map.iter.reduce
//===----------------------------------------------------------------------===//

static bool isScalarType(Type type) {
  return llvm::isa<FloatType, IntegerType>(type);
}

// A phase's body has one `map.index` argument of the rowwise's rank.
static LogicalResult verifyPhaseIndex(RowwisePhaseOpInterface phase) {
  Operation *op = phase.getOperation();
  Block &body = op->getRegion(0).front();
  if (body.getNumArguments() != 1)
    return op->emitOpError() << "body expects exactly one block argument (the "
                                "index), got "
                             << body.getNumArguments();
  MOSH::ShapeAttr shape =
      llvm::cast<RowwiseOpInterface>(op->getParentOp()).getShape();
  auto indexTy = llvm::dyn_cast<MAP::IndexType>(phase.getIndexArg().getType());
  if (!indexTy)
    return op->emitOpError() << "index block argument must be map.index<N>";
  if (indexTy.getRank() != shape.getRank())
    return op->emitOpError()
           << "index rank " << indexTy.getRank()
           << " does not match the rowwise shape rank " << shape.getRank();
  return mlir::success();
}

LogicalResult IterReduceOp::verify() {
  if (getNumResults() == 0)
    return emitOpError() << "must have at least one result";
  for (auto [i, type] : llvm::enumerate(getResultTypes()))
    if (!isScalarType(type))
      return emitOpError() << "result " << i << " must be a scalar, got "
                           << type;

  if (failed(verifyPhaseIndex(*this)))
    return mlir::failure();

  Block &body = getBody().front();
  if (!getYield())
    return emitOpError() << "body must end with a map.yield of the "
                            "contribution";
  if (getYield().getOperands().empty())
    return emitOpError() << "must yield at least one contribution";
  for (auto [i, type] : llvm::enumerate(getYield().getOperandTypes()))
    if (!isScalarType(type))
      return emitOpError() << "contribution " << i << " must be a scalar, got "
                           << type;

  WalkResult stores =
      body.walk([](StoreOp) { return WalkResult::interrupt(); });
  if (stores.wasInterrupted())
    return emitOpError() << "must not store; a fold only reads the row";
  return mlir::success();
}

LogicalResult
IterReduceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto decl = symbolTable.lookupNearestSymbolFrom<M::MOGG::KernelDeclOp>(
      *this, getMonoidAttr());
  if (!decl)
    return emitOpError("'")
           << getMonoid() << "' does not refer to a kernel.decl op";
  M::MOGG::KernelReductionAttr reduction = decl.getReductionAttr();
  if (!reduction)
    return emitOpError("kernel '") << getMonoid() << "' is not a reduction";

  ArrayRef<Attribute> outputs = reduction.getOutputs().getValue();
  if (outputs.size() != getNumResults())
    return emitOpError("monoid '") << getMonoid() << "' has " << outputs.size()
                                   << " output fields, but the fold has "
                                   << getNumResults() << " results";

  // Outputs whose dtype is the same struct parameter share one type. Fixed
  // dtypes are checked when the monoid's parameters are inferred.
  llvm::StringMap<std::pair<unsigned, Type>> paramTypes;
  for (auto [i, attr] : llvm::enumerate(outputs)) {
    auto output = llvm::cast<M::MOGG::KernelReductionOutputAttr>(attr);
    auto ref = llvm::dyn_cast<KGEN::ParamDeclRefAttr>(
        KGEN::SugarAttr::strip(output.getDtype()));
    if (!ref)
      continue;
    Type type = getResult(i).getType();
    auto [it, inserted] =
        paramTypes.try_emplace(ref.getName().strref(), i, type);
    if (!inserted && it->second.second != type)
      return emitOpError("results ")
             << it->second.first << " and " << i << " are both of dtype "
             << "parameter " << ref.getName() << " but have types "
             << it->second.second << " and " << type;
  }
  return mlir::success();
}

LogicalResult IterCacheOp::verify() {
  Block &body = getBody().front();
  if (body.getNumArguments() != 1 ||
      !llvm::isa<MAP::IndexType>(body.getArgument(0).getType()))
    return emitOpError() << "body takes exactly one map.index argument";
  auto yield = body.empty() ? YieldOp() : llvm::dyn_cast<YieldOp>(body.back());
  if (!yield || yield.getNumOperands() != 1 ||
      yield.getOperand(0).getType() != getType())
    return emitOpError() << "body must yield one value of the result type";
  if (body.walk([](StoreOp) { return WalkResult::interrupt(); })
          .wasInterrupted())
    return emitOpError() << "must not store; it only reads the row";
  for (Operation *op = getOperation()->getPrevNode(); op;
       op = op->getPrevNode())
    if (!isa<ConstantScalarOp>(op))
      return emitOpError() << "must come first in the row, after constants "
                              "only";
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// map.iter.cache.load
//===----------------------------------------------------------------------===//

LogicalResult IterCacheLoadOp::verify() {
  if (!getCache().getDefiningOp<IterCacheOp>())
    return emitOpError() << "reads a value that isn't a map.iter.cache";
  if (getIndex() != cast<RowwisePhaseOpInterface>(getOperation()->getParentOp())
                        .getIndexArg())
    return emitOpError() << "must read at its phase's index";
  return mlir::success();
}

LogicalResult IterMapOp::verify() {
  if (failed(verifyPhaseIndex(*this)))
    return mlir::failure();
  WalkResult stores = getBody().walk([&](StoreOp store) {
    return store.getIndex() == getIndexArg() ? WalkResult::advance()
                                             : WalkResult::interrupt();
  });
  if (stores.wasInterrupted())
    return emitOpError() << "must store at its own index";
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// map.constant.scalar
//===----------------------------------------------------------------------===//

LogicalResult ConstantScalarOp::verify() {
  if (getValue().getType() != getResult().getType())
    return emitOpError("value attribute type ")
           << getValue().getType() << " does not match result type "
           << getResult().getType();
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// map.store
//===----------------------------------------------------------------------===//

LogicalResult StoreOp::verify() {
  if (!M::MO::dynCastFromTensorLike(getTensor().getType()))
    return emitOpError()
           << "tensor operand must be !mo.tensor or !mo.buffer type";
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// map.index.slice
//===----------------------------------------------------------------------===//

static LogicalResult verifySliceBoundsOperandsAreRank1(
    Operation *op, ArrayRef<std::pair<StringRef, Value>> bounds) {
  for (auto [name, value] : bounds) {
    auto tensorTy = llvm::dyn_cast<M::MO::TensorType>(value.getType());
    if (!tensorTy || tensorTy.getRank() != 1)
      return op->emitOpError()
             << name << " operand must be a rank-1 !mo.tensor";
  }
  return mlir::success();
}

LogicalResult SliceIndexOp::verify() {
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  auto indexTy = llvm::cast<IndexType>(getIndex().getType());
  if (indexTy.getRank() != signature.getOut().getRank())
    return emitOpError() << "index rank " << indexTy.getRank()
                         << " does not match spec's out-shape rank "
                         << signature.getOut().getRank();

  auto resultTy = llvm::cast<IndexType>(getResult().getType());
  if (resultTy.getRank() != signature.getIn().getRank())
    return emitOpError() << "result rank " << resultTy.getRank()
                         << " does not match spec's in-shape rank "
                         << signature.getIn().getRank();

  return verifySliceBoundsOperandsAreRank1(getOperation(),
                                           {std::pair{"starts", getStarts()},
                                            {"stops", getStops()},
                                            {"steps", getSteps()}});
}

//===----------------------------------------------------------------------===//
// map.index.slice.select
//===----------------------------------------------------------------------===//

// The shape `index` addresses, where that is known locally: at an epilogue's
// own index, the tensor the kernel hands the epilogue; at a foreach's loop
// index (the last argument, or the first once outlined), its loop shape; at a
// select body's argument, that select's out-shape. Null elsewhere.
static MOSH::ShapeAttr getIndexedShape(Value index) {
  auto arg = llvm::dyn_cast<BlockArgument>(index);
  if (!arg)
    return {};
  Operation *owner = arg.getOwner()->getParentOp();
  if (auto select = llvm::dyn_cast<SliceSelectOp>(owner))
    return select.getSpec().getSignature().getOut();
  if (auto foreach = llvm::dyn_cast<IterElementwiseOp>(owner))
    return arg == arg.getOwner()->getArguments().back() ? foreach.getShape()
                                                        : MOSH::ShapeAttr();
  if (auto body = llvm::dyn_cast<IterElementwiseBodyOp>(owner))
    return arg.getArgNumber() == 0 ? body.getShape() : MOSH::ShapeAttr();
  auto epilogue = llvm::dyn_cast<IterEpilogueOp>(owner);
  if (!epilogue || arg.getArgNumber() != 1)
    return {};
  auto handedTy =
      llvm::dyn_cast<M::MO::TensorType>(epilogue.getResult().getType());
  return handedTy ? handedTy.getShape() : MOSH::ShapeAttr();
}

LogicalResult
SliceSelectOp::verifySpecAt(SliceSpecAttr spec, Value index,
                            function_ref<InFlightDiagnostic()> emitError) {
  auto reject = [&](const auto &...message) {
    if (emitError)
      (emitError() << ... << message);
    return failure();
  };
  MAP::ViewSignatureAttr signature = spec.getSignature();
  unsigned rank = signature.getIn().getRank();
  if (llvm::cast<IndexType>(index.getType()).getRank() != rank)
    return reject("index rank does not match spec's rank ", rank);
  if (rank == 0)
    return reject("cannot select from a slice of a rank-0 tensor");

  // There are no bounds operands to read a runtime start or step from.
  ArrayRef<int64_t> starts = spec.getStarts().asArrayRef();
  ArrayRef<int64_t> steps = spec.getSteps().asArrayRef();
  for (auto [d, start, step] : llvm::enumerate(starts, steps)) {
    if (ShapedType::isDynamic(start) || ShapedType::isDynamic(step))
      return reject("dim ", d, ": start and step must be static");
    if (step <= 0)
      return reject("dim ", d, ": step must be positive, got ", step);
  }
  // The body runs once per stored vector, which spans the innermost dim, so
  // selecting within that dim would split the vector.
  unsigned inner = rank - 1;
  if (starts[inner] != 0 || steps[inner] != 1 ||
      signature.getIn().getValues()[inner] !=
          signature.getOut().getValues()[inner])
    return reject("the innermost dim must not be sliced");

  // The body runs at the stored vector's first index, so the index must run
  // along the in-shape's innermost dim with the vector. A transformed index,
  // such as a reshape's, can carry one vector across rows of the in-shape.
  MOSH::ShapeAttr indexed = getIndexedShape(index);
  if (!indexed)
    return reject("index must be its kernel's loop index or an enclosing "
                  "select's argument");
  if (indexed != signature.getIn())
    return reject("spec's in-shape ", signature.getIn(),
                  " does not match the shape ", indexed,
                  " its index addresses");
  return success();
}

LogicalResult SliceSelectOp::verify() {
  if (failed(
          verifySpecAt(getSpec(), getIndex(), [&] { return emitOpError(); })))
    return failure();
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  BlockArgument outIdx = getBody().front().getNumArguments() == 1
                             ? getBody().front().getArgument(0)
                             : BlockArgument();
  if (!outIdx || outIdx.getType() != getIndex().getType())
    return emitOpError() << "body must take one index of the spec's rank";

  // A load or store at the body's argument addresses the slice's output.
  // Nested ops are not verified yet, hence the checked cast.
  for (Operation *user : outIdx.getUsers()) {
    Value tensor;
    if (auto load = llvm::dyn_cast<LoadOp>(user))
      tensor = load.getTensor();
    else if (auto store = llvm::dyn_cast<StoreOp>(user))
      tensor = store.getTensor();
    M::MO::TensorType tensorTy =
        tensor ? M::MO::dynCastFromTensorLike(tensor.getType())
               : M::MO::TensorType();
    if (tensorTy && tensorTy.getShape() != signature.getOut())
      return emitOpError() << "tensor shape " << tensorTy.getShape()
                           << " at the body's index does not match the "
                              "slice's out-shape "
                           << signature.getOut();
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// map.index.slice.stride
//===----------------------------------------------------------------------===//

LogicalResult SliceStrideOp::verify() {
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  auto inputTy = M::MO::castFromTensorLike(getInput().getType());
  if (inputTy.getShape() != signature.getIn())
    return emitOpError() << "input shape " << inputTy.getShape()
                         << " does not match spec's in-shape "
                         << signature.getIn();

  auto resultTy = llvm::cast<M::MO::TensorType>(getResult().getType());
  if (resultTy.getShape() != signature.getOut())
    return emitOpError() << "result shape " << resultTy.getShape()
                         << " does not match spec's out-shape "
                         << signature.getOut();

  if (resultTy.getElementType() != inputTy.getElementType())
    return emitOpError() << "result element type " << resultTy.getElementType()
                         << " does not match input element type "
                         << inputTy.getElementType();

  return verifySliceBoundsOperandsAreRank1(getOperation(),
                                           {std::pair{"starts", getStarts()},
                                            {"stops", getStops()},
                                            {"steps", getSteps()}});
}

//===----------------------------------------------------------------------===//
// map.index.broadcast
//===----------------------------------------------------------------------===//

LogicalResult BroadcastIndexOp::verify() {
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  auto indexTy = llvm::cast<IndexType>(getIndex().getType());
  if (indexTy.getRank() != signature.getOut().getRank())
    return emitOpError() << "index rank " << indexTy.getRank()
                         << " does not match spec's out-shape rank "
                         << signature.getOut().getRank();

  auto resultTy = llvm::cast<IndexType>(getResult().getType());
  if (resultTy.getRank() != signature.getIn().getRank())
    return emitOpError() << "result rank " << resultTy.getRank()
                         << " does not match spec's in-shape rank "
                         << signature.getIn().getRank();

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// map.index.broadcast.stride
//===----------------------------------------------------------------------===//

LogicalResult BroadcastStrideOp::verify() {
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  auto inputTy = M::MO::castFromTensorLike(getInput().getType());
  if (inputTy.getShape() != signature.getIn())
    return emitOpError() << "input shape " << inputTy.getShape()
                         << " does not match spec's in-shape "
                         << signature.getIn();

  auto resultTy = llvm::cast<M::MO::TensorType>(getResult().getType());
  if (resultTy.getShape() != signature.getOut())
    return emitOpError() << "result shape " << resultTy.getShape()
                         << " does not match spec's out-shape "
                         << signature.getOut();

  if (resultTy.getElementType() != inputTy.getElementType())
    return emitOpError() << "result element type " << resultTy.getElementType()
                         << " does not match input element type "
                         << inputTy.getElementType();

  return mlir::success();
}

LogicalResult ReshapeIndexOp::verify() {
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  auto indexTy = llvm::cast<IndexType>(getIndex().getType());
  if (indexTy.getRank() != signature.getOut().getRank())
    return emitOpError() << "index rank " << indexTy.getRank()
                         << " does not match spec's out-shape rank "
                         << signature.getOut().getRank();

  auto resultTy = llvm::cast<IndexType>(getResult().getType());
  if (resultTy.getRank() != signature.getIn().getRank())
    return emitOpError() << "result rank " << resultTy.getRank()
                         << " does not match spec's in-shape rank "
                         << signature.getIn().getRank();

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// map.index.reshape.stride
//===----------------------------------------------------------------------===//

LogicalResult ReshapeStrideOp::verify() {
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  auto inputTy = M::MO::castFromTensorLike(getInput().getType());
  if (inputTy.getShape() != signature.getIn())
    return emitOpError() << "input shape " << inputTy.getShape()
                         << " does not match spec's in-shape "
                         << signature.getIn();

  auto resultTy = llvm::cast<M::MO::TensorType>(getResult().getType());
  if (resultTy.getShape() != signature.getOut())
    return emitOpError() << "result shape " << resultTy.getShape()
                         << " does not match spec's out-shape "
                         << signature.getOut();

  if (resultTy.getElementType() != inputTy.getElementType())
    return emitOpError() << "result element type " << resultTy.getElementType()
                         << " does not match input element type "
                         << inputTy.getElementType();

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// map.index.transpose
//===----------------------------------------------------------------------===//

LogicalResult TransposeIndexOp::verify() {
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  auto indexTy = llvm::cast<IndexType>(getIndex().getType());
  if (indexTy.getRank() != signature.getOut().getRank())
    return emitOpError() << "index rank " << indexTy.getRank()
                         << " does not match spec's out-shape rank "
                         << signature.getOut().getRank();

  auto resultTy = llvm::cast<IndexType>(getResult().getType());
  if (resultTy.getRank() != signature.getIn().getRank())
    return emitOpError() << "result rank " << resultTy.getRank()
                         << " does not match spec's in-shape rank "
                         << signature.getIn().getRank();

  return verifySliceBoundsOperandsAreRank1(
      getOperation(), {std::pair{"permutations", getPermutations()}});
}

//===----------------------------------------------------------------------===//
// map.index.transpose.stride
//===----------------------------------------------------------------------===//

LogicalResult TransposeStrideOp::verify() {
  MAP::ViewSignatureAttr signature = getSpec().getSignature();
  auto inputTy = M::MO::castFromTensorLike(getInput().getType());
  if (inputTy.getShape() != signature.getIn())
    return emitOpError() << "input shape " << inputTy.getShape()
                         << " does not match spec's in-shape "
                         << signature.getIn();

  auto resultTy = llvm::cast<M::MO::TensorType>(getResult().getType());
  if (resultTy.getShape() != signature.getOut())
    return emitOpError() << "result shape " << resultTy.getShape()
                         << " does not match spec's out-shape "
                         << signature.getOut();

  if (resultTy.getElementType() != inputTy.getElementType())
    return emitOpError() << "result element type " << resultTy.getElementType()
                         << " does not match input element type "
                         << inputTy.getElementType();

  return verifySliceBoundsOperandsAreRank1(
      getOperation(), {std::pair{"permutations", getPermutations()}});
}
