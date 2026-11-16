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
// Outlines a device-graph build region into a function, so a recording that
// repeats is emitted once and each build site holds a call instead.
//
// A `mgp.device_graph.dependencies` body records against a builder and
// references enclosing SSA directly:
//
//   %node = mgp.device_graph.dependencies builder(%b) dependencies() {
//   ^bb0(%builder: !mgp.device_graph.builder):
//     %sub = "kernel.a"(...)
//     %dial = "kernel.b"(...)
//     mgp.device_graph.dependencies.yield
//   }
//
// The body moves into a module-level `mogg.func`, and the op keeps one call.
// The values it captured from the enclosing scope become parameters, so
// identical recordings share one function. The op's own node result stays
// with the op, which synthesizes it:
//
//   %node = mgp.device_graph.dependencies builder(%b) dependencies() {
//   ^bb0(%builder: !mgp.device_graph.builder):
//     mogg.call @__device_graph_dependencies(%builder, ...)
//     mgp.device_graph.dependencies.yield
//   }
//
//   mogg.func @__device_graph_dependencies(%builder: builder, ...) {
//     %sub = "kernel.a"(...)
//     %dial = "kernel.b"(...)
//     mogg.output
//   }
//
// A `mgp.device_graph.collective.create` region outlines the same way, one
// function per device region.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"

using namespace M;
using namespace M::MGP;
using namespace M::MOGG;
using namespace mlir;

namespace M::MGP {

#define GEN_PASS_DEF_OUTLINEDEVICEGRAPHDEPENDENCIESPASS
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

} // namespace M::MGP

namespace {

/// The device whose graph region `op` records into, when the enclosing
/// device-graph op names it.
M::DeviceRefAttr enclosingGraphDevice(Operation *op) {
  Region *region = op->getParentRegion();

  if (auto create = op->getParentOfType<DeviceGraphCollectiveCreateOp>()) {
    ValueRange deviceContexts = create.getDeviceContexts();
    for (auto &&[idx, graph] : llvm::enumerate(create.getGraphs()))
      if (graph.isAncestor(region) && idx < deviceContexts.size())
        return cast<DeviceContextType>(deviceContexts[idx].getType())
            .getDevice();
  } else if (auto collective = op->getParentOfType<DeviceGraphCollectiveOp>()) {
    for (auto graph : collective.graphs())
      if (graph.bodyBlock->getParent()->isAncestor(region))
        return graph.device;
  }

  return {};
}

/// The device one build region of `create` records into.
M::DeviceRefAttr collectiveRegionDevice(DeviceGraphCollectiveCreateOp create,
                                        Region &region) {
  ValueRange deviceContexts = create.getDeviceContexts();
  unsigned idx = region.getRegionNumber();
  if (idx < deviceContexts.size())
    return cast<DeviceContextType>(deviceContexts[idx].getType()).getDevice();
  return {};
}

/// Whether `body` has more than one op to separate into a function. A single-op
/// body (including a body that is already one call) is not worth the
/// indirection.
bool hasBodyToOutline(Block &body) {
  return llvm::hasNItemsOrMore(body.without_terminator(), 2);
}

/// Whether `op` may move into a body. A buffer constant cannot: it reads the
/// enclosing compiled region's constant-resource table, which only the MGP ABI
/// provides, so the outliner holds it back outside the function.
bool canMoveIntoBody(Operation *op) {
  return !isa<MGP::BufferConstantOp, MOGG::BufferConstantOp>(op);
}

/// Phase 1 of outlining: hold back what cannot move into a `mogg.func`.
///
/// A `mogg.buffer.constant` reads the enclosing compiled region's
/// constant-resource table, which only the MGP ABI provides, so a top-level
/// constant stays in the source body and crosses as a call argument. A nested
/// constant, a MOGG-level one, or a body that uses a value of a type that
/// cannot cross the boundary leaves the whole body inline.
///
/// Returns the ops that stay, or failure when the body cannot be outlined.
FailureOr<SetVector<Operation *>> holdBackUnmovableOps(Region &region) {
  Block &block = region.front();
  SetVector<Operation *> stay;
  bool outlineable = true;
  block.walk([&](Operation *op) {
    if (canMoveIntoBody(op))
      return;
    if (isa<MGP::BufferConstantOp>(op) && op->getBlock() == &block)
      stay.insert(op);
    else
      outlineable = false;
  });

  if (!outlineable)
    return failure();
  return stay;
}

/// A function holding an outlined body, and the values its call passes, in
/// parameter order.
struct OutlinedBody {
  MOGG::FuncOp func;
  SmallVector<Value> arguments;
};

/// Closes `region` over the values its body uses from outside it: appends one
/// entry-block argument per value, rewrites the uses within the region, and
/// returns the values in argument order.
SmallVector<Value> closeOverCapturedValues(Region &region) {
  llvm::SetVector<Value> captured;
  region.walk([&](Operation *op) {
    for (Value operand : op->getOperands())
      if (!region.isAncestor(operand.getParentRegion()))
        captured.insert(operand);
  });

  Block &body = region.front();
  SmallVector<Value> arguments;
  for (Value value : captured) {
    // Break captured chains: all execution ordering is now encoded as
    // dependency nodes. Chains are not present in the generated Mojo code and
    // can inhibit sinking.
    if (isa<MEF::ChainType>(value.getType())) {
      OpBuilder builder(value.getContext());
      builder.setInsertionPointToStart(&body);
      Value chain = MOGG::ChainDummyOp::create(builder, value.getLoc());
      value.replaceUsesWithIf(chain, [&](OpOperand &use) {
        return region.isAncestor(use.getOwner()->getParentRegion());
      });
      continue;
    }

    BlockArgument arg = body.addArgument(value.getType(), value.getLoc());
    value.replaceUsesWithIf(arg, [&](OpOperand &use) {
      return region.isAncestor(use.getOwner()->getParentRegion());
    });
    arguments.push_back(value);
  }

  return arguments;
}

/// Phase 2 of outlining: move `block`'s body, except the `stay` ops and the
/// terminator, into a new module-level `mogg.func`, closed over the values it
/// uses from outside. The old terminator's operands become its results.
OutlinedBody synthesizeFunc(OpBuilder &builder, SymbolTable &symtab,
                            Operation *owner, Block &block,
                            const SetVector<Operation *> &stay,
                            M::DeviceRefAttr device) {
  MLIRContext *ctx = owner->getContext();
  Location loc = owner->getLoc();

  Operation *terminator = block.getTerminator();

  // What the source body returned is what the function returns.
  SmallVector<Value> yielded(terminator->getOperands());

  // A detached region holds the new body until the function takes it over.
  Region body;
  Block *bodyBlock = new Block();
  body.push_back(bodyBlock);

  // Moving an op out of `block` invalidates the iteration, so step early.
  for (Operation &op : llvm::make_early_inc_range(block.without_terminator())) {
    if (!llvm::is_contained(stay, &op))
      op.moveBefore(bodyBlock, bodyBlock->end());
  }

  OpBuilder bodyBuilder(ctx);
  bodyBuilder.setInsertionPointToEnd(bodyBlock);
  MOGG::OutputOp::create(bodyBuilder, loc, yielded);

  SmallVector<Value> arguments = closeOverCapturedValues(body);

  // The function is named after the op; identical bodies share a preferred name
  // and the symbol table suffixes the rest.
  MOGG::FusionInfoAttr fusionInfo =
      MOGG::FusionInfoAttr::get(ctx, MOGG::createFusionLeaf(owner));

  auto func = MOGG::FuncOp::create(
      builder, loc, MOGG::getFuncName(fusionInfo),
      TypeAttr::get(
          FunctionType::get(ctx, body.getArgumentTypes(), TypeRange(yielded))),
      fusionInfo,
      /*dispatch=*/nullptr,
      /*constantResources=*/builder.getArrayAttr({}),
      MOGG::ABIKindAttr::get(ctx, MOGG::ABIKind::MOGG),
      builder.getStringAttr(device.getLabel()),
      /*trait_stub_buffer_capture=*/nullptr, /*trait_stub_arg_bounds=*/nullptr);

  symtab.insert(func);
  func.getBody().takeBody(body);

  return {func, std::move(arguments)};
}

/// Phase 3 of outlining: replace `block`'s body with one call to `outlined`,
/// and rebuild the terminator over the call's results.
void createCall(OpBuilder &builder, Block &block, OutlinedBody &outlined,
                llvm::function_ref<void(OpBuilder &, Location, ValueRange)>
                    rebuildTerminator) {
  Location loc = block.getTerminator()->getLoc();
  block.getTerminator()->erase();
  builder.setInsertionPointToEnd(&block);
  auto call = MOGG::CallOp::create(
      builder, loc, outlined.func.getResultTypes(),
      FlatSymbolRefAttr::get(outlined.func.getSymNameAttr()),
      outlined.arguments, /*affinity_device_id=*/nullptr);
  rebuildTerminator(builder, loc, call.getResults());
}

/// Outlines the recording body of a `mgp.device_graph.dependencies` op, which
/// references enclosing SSA directly. The builder block argument becomes a
/// parameter only when the body records through it, so the op's
/// `producesNewNodes` answer is unaffected.
void outlineDependenciesBody(OpBuilder &builder, SymbolTable &symtab,
                             DeviceGraphDependenciesOp op) {
  FailureOr<SetVector<Operation *>> stay = holdBackUnmovableOps(op.getBody());
  if (failed(stay))
    return;

  M::DeviceRefAttr device = enclosingGraphDevice(op);
  if (!device)
    device = MO::getDeviceOrDefaultAny(op);

  OutlinedBody outlined =
      synthesizeFunc(builder, symtab, op, op.getBodyBlock(), *stay, device);
  createCall(builder, op.getBodyBlock(), outlined,
             [](OpBuilder &bodyBuilder, Location loc, ValueRange) {
               DeviceGraphDependenciesYieldOp::create(bodyBuilder, loc);
             });
}

/// Outlines one build region of an `mgp.device_graph.collective.create`. The
/// op is `IsolatedFromAbove`, so the region's block arguments are its whole
/// interface and nothing is captured from an enclosing scope.
void outlineCollectiveRegion(OpBuilder &builder, SymbolTable &symtab,
                             DeviceGraphCollectiveCreateOp op, Region &region) {
  FailureOr<SetVector<Operation *>> stay = holdBackUnmovableOps(region);
  if (failed(stay))
    return;

  M::DeviceRefAttr device = collectiveRegionDevice(op, region);
  if (!device)
    device = MO::getDeviceOrDefaultAny(op);

  OutlinedBody outlined =
      synthesizeFunc(builder, symtab, op, region.front(), *stay, device);
  createCall(builder, region.front(), outlined,
             [](OpBuilder &bodyBuilder, Location loc, ValueRange results) {
               DeviceGraphYieldOp::create(bodyBuilder, loc, results);
             });
}

struct OutlineDeviceGraphDependenciesPass
    : public MGP::impl::OutlineDeviceGraphDependenciesPassBase<
          OutlineDeviceGraphDependenciesPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Register through the cached `SymbolTableAnalysis` so names stay unique
    // and the table consistent.
    SymbolTable &symtab =
        getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();
    OpBuilder builder(&getContext());

    module.walk<WalkOrder::PreOrder>([&](DeviceGraphDependenciesOp op) {
      // A body that yields values stays inline: re-yielding its results through
      // the op's `bodyResults` is not enabled yet. A body of one op is not
      // worth a function.
      if (!op.getBodyResults().empty() || !hasBodyToOutline(op.getBodyBlock()))
        return WalkResult::skip();

      builder.setInsertionPointToEnd(module.getBody());
      outlineDependenciesBody(builder, symtab, op);
      return WalkResult::skip();
    });

    module.walk<WalkOrder::PreOrder>([&](DeviceGraphCollectiveCreateOp op) {
      for (Region &region : op.getGraphs()) {
        if (!hasBodyToOutline(region.front()))
          continue;
        builder.setInsertionPointToEnd(module.getBody());
        outlineCollectiveRegion(builder, symtab, op, region);
      }
      // Regions are outlined whole; nothing inside them is collected here.
      return WalkResult::skip();
    });
  }
};

} // namespace
