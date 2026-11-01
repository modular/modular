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

#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

using namespace M::MGP;
using namespace M::MOGG;
using namespace mlir;

namespace {

/// The producer dispatch in a kernel stub body and its launch context. Any
/// enclosing `mogg.with_trace` is optional and returned separately. MAP ops
/// share no interface with the legacy ones, so the match is by op.
struct ProducerDispatch {
  Operation *op = nullptr;
  Value context;

  explicit operator bool() const { return op != nullptr; }
};

ProducerDispatch matchProducerDispatch(Operation *op) {
  if (auto opaque = dyn_cast<CallOpaqueOp>(op)) {
    ValueRange contexts = opaque.getDeviceContexts();
    return {op, contexts.empty() ? Value() : contexts.front()};
  }

  if (auto foreachRegion = dyn_cast<CallForeachOp>(op))
    return {op, foreachRegion.getContext()};

  if (auto foreachTile = dyn_cast<CallForeachTileOp>(op))
    return {op, foreachTile.getContext()};

  if (auto viewCopy = dyn_cast<CallViewCopyOp>(op))
    return {op, viewCopy.getContext()};

  return {};
}

ProducerDispatch findProducerDispatch(FuncOp stub, WithTraceOp &withTrace) {
  ProducerDispatch result;
  stub.walk([&](Operation *op) {
    ProducerDispatch match = matchProducerDispatch(op);
    if (!match)
      return WalkResult::advance();

    result = match;
    withTrace = op->getParentOfType<WithTraceOp>();
    return WalkResult::interrupt();
  });

  return result;
}

/// Re-stamps the variadic ABI of each `mogg.tuple.create` feeding the
/// recording dispatch from the `build_graph` parameter it satisfies.
/// `ConformCallOperands` packed the run against `execute`, whose signature
/// cannot declare a slot-backed (`StableVariadicTensors`) pack; `build_graph`
/// can, and this stub is what calls it. Packed operands sit 1:1 with the
/// signature's arguments, as `CallBuildGraphOp::emitMojo` already relies on.
void stampBuildGraphVariadicAbi(KernelDeclOp decl, ValueRange inputs,
                                ValueRange outputs) {
  KernelDeclAdaptor adaptor(decl);
  if (!adaptor.buildGraphFunction)
    return;
  auto stamp = [](ValueRange operands,
                  ArrayRef<KernelDeclOperandAdaptor> parameters) {
    for (auto [operand, parameter] : llvm::zip(operands, parameters)) {
      auto tuple = operand.getDefiningOp<TupleCreateOp>();
      if (tuple && parameter.isVariadicTensorType() &&
          parameter.isStableTensorType())
        tuple.setVariadicAbi(VariadicABI::Stable);
    }
  };
  stamp(inputs, adaptor.buildGraphFunction->inputArguments);
  stamp(outputs, adaptor.buildGraphFunction->outputArguments);
}

/// A generated recording sibling stub; `contextArgIndex` is the call operand
/// the builder replaces.
struct RecordingStub {
  StringAttr name;
  unsigned contextArgIndex = 0;
};

/// Builds the recording stub `@<original>_build_graph`, a clone of `original`
/// with its launch arg retyped from `!mogg.context` to
/// `!mgp.device_graph.builder`. The producer dispatch records as follows:
///   - A `mogg.call.execute` whose kernel has a node-free `build_graph` method:
///     the dispatch becomes a `mogg.call.build_graph` recording into the
///     builder directly (the `mogg.with_trace` is dropped).
///   - An execute-only kernel (`mogg.call.execute` with no `build_graph`) or a
///     `mogg.call.view_copy` view copy: the producer is kept, but its launch
///     context is repointed to a recording device context viewed off the
///     builder via `mgp.device_graph.recording_context`, so the ordinary
///     dispatch records into the graph.
/// The stub returns nothing; the enclosing `mgp.device_graph.dependencies` op
/// carries node ordering.
FailureOr<RecordingStub> createRecordingStub(SymbolTable &symtab,
                                             FuncOp original) {
  MLIRContext *ctx = original.getContext();
  OwningOpRef<FuncOp> clone = cast<FuncOp>(original->clone());

  WithTraceOp withTrace;
  ProducerDispatch producer = findProducerDispatch(*clone, withTrace);
  if (!producer) {
    return original.emitError(
        "device-graph recording stub generation: kernel stub has no producer "
        "dispatch");
  }

  // The builder takes over the launch context's block argument.
  Block &body = clone->getBodyBlock();
  auto context = dyn_cast<BlockArgument>(producer.context);
  if (!context || context.getOwner() != &body) {
    return original.emitError(
        "device-graph recording stub generation: expected the producer launch "
        "context to be an argument of the kernel stub");
  }

  auto opaque = dyn_cast<CallOpaqueOp>(producer.op);
  KernelDeclOp decl;
  if (opaque)
    decl = symtab.lookup<KernelDeclOp>(opaque.getCallee());
  bool useBuildGraph = decl && decl.getBuildGraph().has_value();

  // Snapshot before retyping the launch arg: `getKernelInputs()` drops the
  // context by its `!mogg.context` type.
  SmallVector<Value> kernelInputs, outputs;
  StringAttr kernelSymName, moOpName;
  UnitAttr isThrows;
  DictionaryAttr kernelParams;
  bool hasEpilogueFusion = false;
  if (useBuildGraph) {
    // `useBuildGraph` implies a decl, which only an opaque resolves.
    assert(opaque && "a build_graph producer is an opaque");
    // `_call.opaque` keeps contexts in their own operand group. `moOpName` is
    // unread provenance, so the decl's registration stands in.
    kernelInputs = llvm::to_vector(opaque.getInputs());
    outputs = llvm::to_vector(opaque.getOutputs());
    kernelSymName = StringAttr::get(ctx, opaque.getCallee());
    moOpName = StringAttr::get(ctx, decl.getRegistration());
    // Raises comes from `build_graph`, which this dispatch calls.
    // `CallOpaqueOp::isThrows()` would look up the decl by symbol, which fails
    // silently (false) inside a detached clone.
    auto buildGraphSignature = decl.getBuildGraph()->getSignature();
    isThrows = buildGraphSignature.isThrows() ? UnitAttr::get(ctx) : UnitAttr();
    kernelParams = opaque.getKernelParamAttr();
    hasEpilogueFusion = opaque.getHasEpilogueFusion();
  }
  if (useBuildGraph)
    stampBuildGraphVariadicAbi(decl, kernelInputs, outputs);

  BlockArgument launchArg = context;
  launchArg.setType(DeviceGraphBuilderType::get(ctx));

  Operation &insertionPoint = withTrace ? *withTrace : *producer.op;

  if (useBuildGraph) {
    OpBuilder builder(&insertionPoint);
    // `call.build_graph` is not a kernel terminator, so `with_trace` is
    // dropped.
    CallBuildGraphOp::create(builder,
                             /*location=*/producer.op->getLoc(),
                             /*kernelSymName=*/kernelSymName,
                             /*moOpName=*/moOpName,
                             /*isThrows=*/isThrows,
                             /*inputs=*/kernelInputs,
                             /*outputs=*/outputs,
                             /*graphBuilder=*/launchArg,
                             /*kernel_param=*/kernelParams,
                             /*hasEpilogueFusion=*/
                             builder.getBoolAttr(hasEpilogueFusion));
    // Erasing a legacy `with_trace` also erases its producer; a MAP producer
    // stands alone.
    if (withTrace)
      withTrace.erase();
    else
      producer.op->erase();
  } else {
    auto builder = OpBuilder::atBlockBegin(&clone->getBodyBlock());
    // Repoint the dispatch at a recording context so it records instead of
    // running eagerly.
    auto recordingContext = DeviceGraphRecordingContextOp::create(
        builder, producer.op->getLoc(), M::MOGG::ContextType::get(ctx),
        launchArg);
    launchArg.replaceUsesWithIf(
        recordingContext.getContext(),
        [&](OpOperand &use) { return use.getOwner() == producer.op; });
  }

  // Node-free: the stub returns void and only its launch arg type changes.
  clone->setFunctionTypeAttr(
      TypeAttr::get(FunctionType::get(ctx, body.getArgumentTypes(), {})));

  // The original stays for the direct-execution / init path. The name is
  // unique, so `insert` preserves it.
  clone->setSymNameAttr(
      StringAttr::get(ctx, original.getSymName() + "_build_graph"));

  StringAttr newName =
      symtab.insert(clone.release(), std::next(original->getIterator()));
  return RecordingStub{newName, context.getArgNumber()};
}

/// Returns the builder recording `op`'s region: the innermost
/// `mgp.device_graph.dependencies` block arg, else the container's builder.
Value enclosingGraphBuilder(Operation *op) {
  if (auto deps = op->getParentOfType<DeviceGraphDependenciesOp>())
    return deps.getBuilderBlockArgument();

  if (auto staging = op->getParentOfType<DeviceGraphStagingOp>())
    return staging.getBodyBuilder();

  auto collective = op->getParentOfType<DeviceGraphCollectiveOp>();
  ASSERT_STREAM(collective,
                "reached an operation not inside a device graph region");

  Region *opRegion = op->getParentRegion();
  for (DeviceGraphCollectiveOp::CollectiveRegion region : collective.graphs()) {
    auto blockRegion = region.bodyBlock->getParent();
    if (blockRegion->isAncestor(opRegion))
      return region.builder;
  }

  llvm_unreachable("unable to determine body region");
}

/// Retargets a `mogg.call @stub_N` to `@stub_N_build_graph`, passing the
/// builder in the launch-context position; other operands pass through.
LogicalResult rewriteGraphCall(M::MOGG::CallOp call,
                               const RecordingStub &sibling) {
  SmallVector<Value> operands = llvm::to_vector(call.getInputs());
  if (sibling.contextArgIndex >= operands.size() ||
      !isa<DeviceContextType, M::MOGG::ContextType>(
          operands[sibling.contextArgIndex].getType())) {
    return call.emitError("device-graph recording stub call: expected operand ")
           << sibling.contextArgIndex << " to be the launch context";
  }

  IRRewriter rewriter(call);

  // Take the builder directly, as `rewriteBuilderAllocs` does; the launch
  // context chain this leaves dead is canonicalized away before finalization.
  operands[sibling.contextArgIndex] = enclosingGraphBuilder(call);

  auto newCall = M::MOGG::CallOp::create(
      rewriter, call.getLoc(), call.getResultTypes(), sibling.name, operands,
      call.getAffinityDeviceIdAttr());

  rewriter.replaceAllUsesWith(call.getResults(), newCall.getResults());

  call.erase();

  return success();
}

/// Replaces each `mgp.buffer.alloc` in `container` with
/// `mgp.device_graph.alloc`, so recorded addresses come from the builder's
/// pool and stay reserved for the graph's lifetime.
void rewriteBuilderAllocs(Operation *container) {
  container->walk([&](BufferAllocOp alloc) {
    IRRewriter rewriter(alloc);
    rewriter.replaceOpWithNewOp<DeviceGraphAllocOp>(
        alloc,
        /*outBuffer=*/alloc.getOutBuffer().getType(),
        /*inChain=*/alloc.getInChain(),
        /*runtimeSize=*/alloc.getRuntimeSize(),
        /*graphBuilder=*/enclosingGraphBuilder(alloc),
        /*rawAlign=*/alloc.getRawAlign());
  });
}

/// Retargets each `mogg.call` in `container` to a generated recording
/// stub.
LogicalResult generateBuildGraphStubs(SymbolTable &symtab,
                                      Operation *container) {
  // TODO: Support `CallAsyncOp`.
  SmallVector<M::MOGG::CallOp> calls;
  container->walk([&](M::MOGG::CallOp call) { calls.push_back(call); });

  if (calls.empty())
    return success();

  // One sibling per distinct callee; all calls to it are retargeted.
  llvm::DenseMap<StringAttr, RecordingStub> calleeToSibling;
  for (M::MOGG::CallOp call : calls) {
    StringAttr callee = call.getNameAttr().getRootReference();

    RecordingStub &sibling = calleeToSibling[callee];
    if (!sibling.name) {
      auto original = symtab.lookup<FuncOp>(callee.getValue());
      if (!original)
        return call.emitError("device-graph recording stub call: '")
               << callee.getValue() << "' does not name a kernel stub";

      FailureOr<RecordingStub> recording =
          createRecordingStub(symtab, original);
      if (failed(recording))
        return failure();

      sibling = std::move(*recording);
    }

    if (failed(rewriteGraphCall(call, sibling)))
      return failure();
  }
  return success();
}

} // namespace

namespace M::MGP {

#define GEN_PASS_DEF_GENERATEBUILDGRAPHSTUBSPASS
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

namespace {

class GenerateBuildGraphStubsPass final
    : public impl::GenerateBuildGraphStubsPassBase<
          GenerateBuildGraphStubsPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // The cached `SymbolTableAnalysis` is current (`mogg-to-mgp` registers the
    // stubs through it) and `symtab.insert` keeps it so, hence preserve.
    SymbolTable &symtab =
        getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();

    // Collect containers first: generating stubs mutates the module's
    // top-level op list.
    SmallVector<DeviceGraphCollectiveOp> graphContainers;
    for (ModelOp model : module.getOps<ModelOp>()) {
      if (!model.getIsDeviceGraph())
        continue;

      model.getExecuteBlock()->walk<WalkOrder::PreOrder>(
          [&](DeviceGraphCollectiveOp collective) {
            graphContainers.push_back(collective);
            // Device-graph containers do not nest.
            return WalkResult::skip();
          });
    }

    for (Operation *container : graphContainers)
      if (failed(generateBuildGraphStubs(symtab, container)))
        return signalPassFailure();

    for (Operation *container : graphContainers)
      rewriteBuilderAllocs(container);
  }
};

} // namespace
} // namespace M::MGP
