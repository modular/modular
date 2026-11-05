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
// Fills in what a `mogg.call.rowwise` needs to print as a Row API body. Each
// step is its own pass so it can be tested alone;
// `mogg-legalize-rowwise-kernels` runs them in order.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOInterfaces.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/LITDialect/LITAttrs.h"
#include "Support/AssertStream.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/StringMap.h"

using namespace M;
using namespace mlir;

namespace M::MOGG {

#define GEN_PASS_DEF_LEGALIZEROWWISEKERNELS
#define GEN_PASS_DEF_ROWWISESELECTPRIMARYINPUT
#define GEN_PASS_DEF_ROWWISEBINDMONOIDS
#define GEN_PASS_DEF_ROWWISEGROUPROWSTORES
#define GEN_PASS_DEF_ROWWISESETLAUNCHPARAMS
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

Block::BlockArgListType captureArgs(CallRowwiseOp op) {
  return op.getBody().front().getArguments().take_front(
      op.getCapturing().size());
}

BlockArgument rowArg(CallRowwiseOp op) {
  return op.getBody().front().getArguments().back();
}

auto folds(CallRowwiseOp op) {
  return op.getBody().front().getOps<RowwiseReduceOp>();
}

auto phases(CallRowwiseOp op) {
  return op.getBody().front().getOps<RowwisePhaseOpInterface>();
}

// Loads of `tensor` at `phase`'s own index.
SmallVector<TensorLoadOp> elementLoads(RowwisePhaseOpInterface phase,
                                       Value tensor) {
  Value index = phase.getIndexArg();
  SmallVector<TensorLoadOp> loads;
  phase->getRegion(0).walk([&](TensorLoadOp load) {
    if (load.getTensor() == tensor && load.getIndex() == index)
      loads.push_back(load);
  });
  return loads;
}

LogicalResult selectPrimaryInput(CallRowwiseOp op) {
  if (op.getPrimary())
    return success();
  if (folds(op).empty())
    return op.emitOpError("has no fold");

  // The first capture the first fold loads at its own index.
  RowwiseReduceOp first = *folds(op).begin();
  Value index = first.getBody().front().getArgument(0);
  std::optional<unsigned> primary;
  first.getBody().walk([&](TensorLoadOp load) {
    auto arg = dyn_cast<BlockArgument>(load.getTensor());
    if (!arg || load.getIndex() != index)
      return WalkResult::advance();
    auto it = llvm::find(captureArgs(op), arg);
    if (it == captureArgs(op).end())
      return WalkResult::advance();
    primary = it - captureArgs(op).begin();
    return WalkResult::interrupt();
  });
  if (!primary)
    return op.emitOpError("has no input its first fold loads at the element "
                          "index");

  Value tensor = captureArgs(op)[*primary];
  Type dtype;
  unsigned phasesReadingPrimary = 0;
  for (RowwisePhaseOpInterface phase : phases(op)) {
    SmallVector<TensorLoadOp> loads = elementLoads(phase, tensor);
    if (loads.empty())
      continue;
    ++phasesReadingPrimary;
    for (TensorLoadOp load : loads) {
      OpBuilder builder(load);
      auto value =
          RowwisePrimaryOp::create(builder, load.getLoc(), load.getType());
      dtype = value.getType().getElementType();
      load.replaceAllUsesWith(value.getResult());
      load.erase();
    }
  }
  op.setPrimary(*primary);
  op.setDtype(dtype);
  op.setIsCached(phasesReadingPrimary > 1);
  return success();
}

// The output field dtype `attr` declares, when it is a constant.
Attribute fixedDType(Attribute attr) {
  auto structAttr =
      dyn_cast<KGEN::LIT::LITStructAttr>(KGEN::SugarAttr::strip(attr));
  if (!structAttr || structAttr.getValues().size() != 1)
    return nullptr;
  return std::get<1>(structAttr.getValues().front());
}

LogicalResult bindMonoid(RowwiseReduceOp fold) {
  auto decl = SymbolTable::lookupNearestSymbolFrom<KernelDeclOp>(
      fold, fold.getMonoidAttr());
  if (!decl || !decl.getReductionAttr())
    return fold.emitOpError("'") << fold.getMonoid() << "' is not a reduction";
  ArrayRef<Attribute> outputs = decl.getReductionAttr().getOutputs().getValue();
  if (outputs.size() != fold.getNumResults())
    return fold.emitOpError("has ")
           << fold.getNumResults() << " results, but the monoid has "
           << outputs.size() << " output fields";

  MLIRContext *ctx = fold.getContext();
  llvm::StringMap<Type> bound;
  SmallVector<StringRef> fields;
  for (auto [attr, resultType] : llvm::zip(outputs, fold.getResultTypes())) {
    Type type = cast<SimdType>(resultType).getElementType();
    auto output = cast<KernelReductionOutputAttr>(attr);
    fields.push_back(output.getName().getValue());
    Attribute dtype = output.getDtype();
    if (auto ref =
            dyn_cast<KGEN::ParamDeclRefAttr>(KGEN::SugarAttr::strip(dtype))) {
      auto [it, inserted] = bound.try_emplace(ref.getName().strref(), type);
      if (!inserted && it->second != type)
        return fold.emitOpError("binds dtype parameter ")
               << ref.getName() << " to both " << it->second << " and " << type;
      continue;
    }
    if (fixedDType(dtype) !=
        KGEN::DTypeConstantAttr::get(ctx, MO::getDTypeOrInvalid(type)))
      return fold.emitOpError("has a result of type ")
             << type << ", but the monoid's field is " << dtype;
  }

  SmallVector<NamedAttribute> params;
  for (auto &entry : bound)
    params.emplace_back(StringAttr::get(ctx, entry.getKey()),
                        TypeAttr::get(entry.getValue()));
  fold.setParamsAttr(DictionaryAttr::get(ctx, params));
  fold.setFieldsAttr(OpBuilder(fold).getStrArrayAttr(fields));
  return success();
}

LogicalResult bindMonoids(CallRowwiseOp op) {
  Type accum;
  for (RowwiseReduceOp fold : folds(op)) {
    if (failed(bindMonoid(fold)))
      return failure();
    auto yield = cast<YieldOp>(fold.getBody().front().getTerminator());
    for (Type yieldType : yield.getOperandTypes()) {
      Type type = cast<SimdType>(yieldType).getElementType();
      if (!accum ||
          type.getIntOrFloatBitWidth() > accum.getIntOrFloatBitWidth())
        accum = type;
    }
  }
  if (accum)
    op.setAccum(accum);
  return success();
}

// Moves the stores at the row index, and the row-level ops only they read, into
// one emit at the end of the body, which prints them one lane per output. A
// load at the row index left in the body feeds a value a phase reads, so it
// becomes a `mogg.rowwise.load_lane` spread by a `mogg.rowwise.splat_lane`.
LogicalResult groupRowStores(CallRowwiseOp op) {
  Block &body = op.getBody().front();
  SmallVector<TensorStoreOp> stores;
  for (TensorStoreOp store : body.getOps<TensorStoreOp>())
    if (store.getIndex() == rowArg(op))
      stores.push_back(store);
  if (!stores.empty()) {
    OpBuilder builder = OpBuilder::atBlockEnd(&body);
    auto emit = RowwiseEmitOp::create(builder, op.getLoc());
    Block *emitBody = builder.createBlock(&emit.getBody());
    for (TensorStoreOp store : stores)
      store->moveBefore(emitBody, emitBody->end());
    auto readByEmit = [&](Operation *user) { return emit->isAncestor(user); };
    for (Operation &o : llvm::make_early_inc_range(llvm::reverse(body)))
      if (!o.use_empty() && !isa<RowwisePhaseOpInterface>(o) &&
          llvm::all_of(o.getUsers(), readByEmit))
        o.moveBefore(emitBody, emitBody->begin());
  }
  for (TensorLoadOp load :
       llvm::make_early_inc_range(body.getOps<TensorLoadOp>())) {
    if (load.getIndex() != rowArg(op))
      continue;
    OpBuilder builder(load);
    auto lane =
        RowwiseLoadLaneOp::create(builder, load.getLoc(), load.getType(),
                                  load.getTensor(), load.getIndex());
    load.replaceAllUsesWith(
        RowwiseSplatLaneOp::create(builder, load.getLoc(), lane).getResult());
    load.erase();
  }
  return success();
}

// An op that divides or calls a transcendental costs about 16 cheap ones.
constexpr unsigned kCheapOpCost = 1;
constexpr unsigned kExpensiveOpCost = 16;

unsigned cost(CallElementwiseOp call) {
  std::optional<RegisteredOperationName> name =
      RegisteredOperationName::lookup(call.getMoOpName(), call.getContext());
  return name && name->hasTrait<MO::MOExpensiveElementwise>() ? kExpensiveOpCost
                                                              : kCheapOpCost;
}

unsigned stateBits(RowwiseReduceOp fold) {
  unsigned bits = 0;
  for (Type type : fold.getResultTypes())
    bits += cast<SimdType>(type).getElementType().getIntOrFloatBitWidth();
  return bits;
}

LogicalResult setLaunchParams(CallRowwiseOp op,
                              unsigned expensiveCostThreshold) {
  SmallVector<RowwiseReduceOp> rowFolds = llvm::to_vector(folds(op));
  if (rowFolds.empty())
    return op.emitOpError("has no fold");

  Block &body = op.getBody().front();
  RowwiseReduceOp widest = rowFolds.front();
  for (RowwiseReduceOp fold : rowFolds)
    if (stateBits(fold) > stateBits(widest))
      widest = fold;

  // Per-element work only: row-level ops run once per row.
  unsigned elementCost = 0;
  for (RowwiseReduceOp fold : rowFolds)
    fold.getBody().walk(
        [&](CallElementwiseOp call) { elementCost += cost(call); });

  // The Row API's launch parameters, set as the hand-written kernels set
  // them (rowwise_reductions.mojo, rowwise_normalization.mojo and
  // rowwise_softmax.mojo):
  //
  //   body writes              num_phases  dtype_size       expensive
  //   one value per row (sum)  1           0                cost >= threshold
  //   every element (rms_norm) folds + 1   row bytes or 0   true
  //
  // `dtype_size` > 0 lets a long row split across blocks. rms_norm and
  // softmax, whose folds keep one field, opt in; layer_norm, whose Welford
  // fold keeps three, doesn't, so neither does a body with a multi-field
  // fold. Every hand-written body that writes every element is expensive:
  // it reads the row once per fold and writes it once.
  bool writesEveryElement = !body.getOps<RowwiseMapOp>().empty();
  bool oneFieldFolds = llvm::all_of(
      rowFolds, [](RowwiseReduceOp fold) { return fold.getNumResults() == 1; });
  op.setNumPhases(writesEveryElement ? rowFolds.size() + 1 : 1);
  op.setDtypeSize(0);
  if (writesEveryElement && oneFieldFolds) {
    ASSERT_STREAM(op.getDtype(), << "the primary input is selected first");
    op.setDtypeSize((op.getDtype()->getIntOrFloatBitWidth() + 7) / 8);
  }
  op.setComputationallyExpensive(writesEveryElement ||
                                 elementCost >= expensiveCostThreshold);
  op.setSimdMonoidAttr(widest.getMonoidAttr());
  if (auto shape = op.getShape().getDefiningOp<ShapeCreateOp>())
    if (std::optional<int64_t> cols =
            MOSH::ShapeAttr::getStaticDim(shape.getShape()[op.getAxis()]))
      op.setCols(*cols);
  return success();
}

LogicalResult forEachRowwise(ModuleOp module,
                             function_ref<LogicalResult(CallRowwiseOp)> fn) {
  SmallVector<CallRowwiseOp> ops;
  module.walk([&](CallRowwiseOp op) {
    ops.push_back(op);
    return WalkResult::skip();
  });
  bool ok = true;
  for (CallRowwiseOp op : ops)
    ok &= succeeded(fn(op));
  return success(ok);
}

struct RowwiseSelectPrimaryInputPass
    : public impl::RowwiseSelectPrimaryInputBase<
          RowwiseSelectPrimaryInputPass> {
  void runOnOperation() override {
    if (failed(forEachRowwise(getOperation(), selectPrimaryInput)))
      signalPassFailure();
  }
};

struct RowwiseBindMonoidsPass
    : public impl::RowwiseBindMonoidsBase<RowwiseBindMonoidsPass> {
  void runOnOperation() override {
    if (failed(forEachRowwise(getOperation(), bindMonoids)))
      signalPassFailure();
  }
};

struct RowwiseGroupRowStoresPass
    : public impl::RowwiseGroupRowStoresBase<RowwiseGroupRowStoresPass> {
  void runOnOperation() override {
    if (failed(forEachRowwise(getOperation(), groupRowStores)))
      signalPassFailure();
  }
};

struct RowwiseSetLaunchParamsPass
    : public impl::RowwiseSetLaunchParamsBase<RowwiseSetLaunchParamsPass> {
  using Base::Base;

  void runOnOperation() override {
    auto set = [&](CallRowwiseOp op) {
      return setLaunchParams(op, expensiveCostThreshold);
    };
    if (failed(forEachRowwise(getOperation(), set)))
      signalPassFailure();
  }
};

struct LegalizeRowwiseKernelsPass
    : public impl::LegalizeRowwiseKernelsBase<LegalizeRowwiseKernelsPass> {
  using Base::Base;

  void runOnOperation() override {
    OpPassManager pm(ModuleOp::getOperationName());
    pm.addPass(createRowwiseSelectPrimaryInput());
    pm.addPass(createRowwiseBindMonoids());
    pm.addPass(createRowwiseGroupRowStores());
    pm.addPass(createRowwiseSetLaunchParams(
        RowwiseSetLaunchParamsOptions{expensiveCostThreshold}));
    if (failed(runPipeline(pm, getOperation())))
      signalPassFailure();
  }
};

} // namespace

} // namespace M::MOGG
